//! Independent local HTTP API, reusable sequence pool and bounded FIFO admission.
mod control;
mod hub;
mod insight;
mod progress;
mod tuning;
use crate::runtime::LoadStep;
use crate::{
    model::Step,
    protocol::{ChatRequest, ChatTemplate, Sampler, parse_output},
    runtime::{BackendArgs, RuntimeModel, RuntimeSequence},
};
use anyhow::{Context, Result, bail, ensure};
use clap::Args;
use progress::LoadProgress;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, VecDeque},
    io::{Read, Write},
    net::{IpAddr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    sync::{
        Arc, Condvar, Mutex, RwLock,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Args, Clone, Debug)]
pub struct ServeArgs {
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// Optional directory containing switchable local models; selection uses exact catalogue IDs.
    #[arg(long)]
    pub models_dir: Option<PathBuf>,
    #[command(flatten)]
    pub backend: BackendArgs,
    #[arg(long, default_value_t = 8192)]
    pub ctx: usize,
    #[arg(long, default_value_t = 2048)]
    pub max_tokens: usize,
    #[arg(long, default_value = "127.0.0.1")]
    pub host: IpAddr,
    #[arg(long, default_value_t = 9070)]
    pub port: u16,
    /// Maximum active requests. State is allocated on demand and reused after completion.
    #[arg(long, default_value_t = 1)]
    pub parallel: usize,
    #[arg(long, default_value_t = 32)]
    pub max_queued: usize,
    /// Waiting prompt-token budget; zero derives a bounded budget from context and parallelism.
    #[arg(long, default_value_t = 0)]
    pub max_queued_prompt_tokens: usize,
    /// Prefill tokens per model call; tune using measured prefill and first-token latency.
    #[arg(long, default_value_t = 32)]
    pub prefill_chunk_tokens: usize,
    #[arg(long, default_value_t = 120)]
    pub queue_timeout_secs: u64,
    #[arg(long, default_value_t = 128)]
    pub max_connections: usize,
    #[arg(long)]
    pub api_key: Option<String>,
    /// Open the native service configuration page after the HTTP listener starts.
    #[arg(long)]
    pub open_ui: bool,
    /// Optional JSON profile. An existing profile overrides model/runtime CLI settings;
    /// successful UI changes atomically replace it. Host, port and key remain CLI-only.
    #[arg(long)]
    pub settings_file: Option<PathBuf>,
}
#[derive(Clone, Copy)]
struct QueuedRequest {
    ticket: u64,
    prompt_tokens: usize,
}

struct GateState {
    active: usize,
    queue: VecDeque<QueuedRequest>,
    queued_prompt_tokens: usize,
    queued_prompt_token_limit: usize,
    changing: bool,
    high_water: usize,
    capacity: usize,
    queue_limit: usize,
    timeout: Duration,
}
impl GateState {
    /// Maintain queue token accounting on cancellation, timeout and disconnect.
    fn remove_waiter(&mut self, ticket: u64) {
        if let Some(index) = self.queue.iter().position(|entry| entry.ticket == ticket) {
            if let Some(entry) = self.queue.remove(index) {
                self.queued_prompt_tokens -= entry.prompt_tokens;
            }
        }
    }
}
struct Admission {
    state: Mutex<GateState>,
    changed: Condvar,
    next: AtomicU64,
}
struct Permit(Arc<Admission>);
impl Drop for Permit {
    fn drop(&mut self) {
        let mut s = self.0.state.lock().unwrap();
        s.active -= 1;
        self.0.changed.notify_all();
    }
}
impl Admission {
    fn new(
        capacity: usize,
        queue_limit: usize,
        timeout: Duration,
        queued_prompt_token_limit: usize,
    ) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(GateState {
                active: 0,
                queue: VecDeque::new(),
                queued_prompt_tokens: 0,
                queued_prompt_token_limit,
                changing: false,
                high_water: 0,
                capacity,
                queue_limit,
                timeout,
            }),
            changed: Condvar::new(),
            next: AtomicU64::new(1),
        })
    }
    fn acquire(
        self: &Arc<Self>,
        stream: &TcpStream,
        prompt_tokens: usize,
    ) -> std::result::Result<Permit, (u16, &'static str)> {
        let mut s = self.state.lock().unwrap();
        if s.changing {
            return Err((503, "model is changing"));
        }
        if s.active < s.capacity && s.queue.is_empty() {
            s.active += 1;
            s.high_water = s.high_water.max(s.active);
            return Ok(Permit(self.clone()));
        }
        if s.queue.len() >= s.queue_limit {
            return Err((429, "generation queue is full"));
        }
        let Some(queued_tokens) = s.queued_prompt_tokens.checked_add(prompt_tokens) else {
            return Err((429, "generation queued prompt token budget exceeded"));
        };
        if queued_tokens > s.queued_prompt_token_limit {
            return Err((429, "generation queued prompt token budget exceeded"));
        }
        let ticket = self.next.fetch_add(1, Ordering::Relaxed);
        s.queue.push_back(QueuedRequest { ticket, prompt_tokens });
        s.queued_prompt_tokens = queued_tokens;
        let deadline = Instant::now() + s.timeout;
        loop {
            if disconnected(stream) {
                s.remove_waiter(ticket);
                self.changed.notify_all();
                return Err((499, "client disconnected"));
            }
            if Instant::now() >= deadline {
                s.remove_waiter(ticket);
                self.changed.notify_all();
                return Err((504, "generation queue deadline exceeded"));
            }
            if !s.changing
                && s.active < s.capacity
                && s.queue.front().is_some_and(|entry| entry.ticket == ticket)
            {
                let admitted = s.queue.pop_front().expect("front ticket exists");
                s.queued_prompt_tokens -= admitted.prompt_tokens;
                s.active += 1;
                s.high_water = s.high_water.max(s.active);
                self.changed.notify_all();
                return Ok(Permit(self.clone()));
            }
            s = self
                .changed
                .wait_timeout(s, Duration::from_millis(25))
                .unwrap()
                .0;
        }
    }
    fn snapshot(&self) -> Value {
        let s = self.state.lock().unwrap();
        json!({"active":s.active,"queued":s.queue.len(),"queued_prompt_tokens":s.queued_prompt_tokens,"queued_prompt_token_limit":s.queued_prompt_token_limit,"capacity":s.capacity,"queue_limit":s.queue_limit,"queue_timeout_secs":s.timeout.as_secs(),"active_high_water":s.high_water,"model_changing":s.changing,"policy":"fifo","allocation":"state on demand; reusable per-model sequence pool"})
    }
}

struct ServedModel {
    id: String,
    model: RuntimeModel,
    default_max_tokens: usize,
    template: ChatTemplate,
    idle: Mutex<Vec<CachedSequence>>,
}
impl ServedModel {
    fn load(
        id: String,
        path: &Path,
        args: &ServeArgs,
        progress: impl Fn(LoadStep),
    ) -> Result<Arc<Self>> {
        let model = RuntimeModel::load_with_progress(path, args.ctx, &args.backend, &progress)?;
        progress(LoadStep::Stage("template", "Preparing the chat template"));
        let template = ChatTemplate::new(&model.source)?;
        Ok(Arc::new(Self {
            id,
            model,
            default_max_tokens: args.max_tokens,
            template,
            idle: Mutex::new(vec![]),
        }))
    }
    fn lease(self: &Arc<Self>, prompt: &[u32], cache_salt: Option<&str>) -> Result<SequenceLease> {
        let mut idle = self.idle.lock().unwrap();
        let best = idle
            .iter()
            .enumerate()
            .filter(|(_, s)| reusable_prefix(&s.history, s.cache_salt.as_deref(), prompt, cache_salt))
            .max_by_key(|(_, s)| s.history.len())
            .map(|(i, _)| i);
        let cached = best.map(|i| idle.swap_remove(i));
        let cached = match cached {
            Some(s) => s,
            None => match idle.pop() {
                Some(mut s) => {
                    s.state.reset()?;
                    s.history.clear();
                    s.output = Step::default();
                    s
                }
                None => CachedSequence {
                    state: self.model.new_sequence()?,
                    history: vec![],
                    output: Step::default(),
                    cache_salt: None,
                },
            },
        };
        Ok(SequenceLease {
            owner: self.clone(),
            state: Some(cached.state),
            history: cached.history,
            output: cached.output,
            cache_salt: cache_salt.map(str::to_owned),
            cacheable: true,
        })
    }
}
/// Reuse only complete executed histories inside the same cache namespace.
fn reusable_prefix(
    history: &[u32],
    history_salt: Option<&str>,
    prompt: &[u32],
    requested_salt: Option<&str>,
) -> bool {
    !history.is_empty() && history_salt == requested_salt && prompt.starts_with(history)
}
struct CachedSequence {
    state: RuntimeSequence,
    history: Vec<u32>,
    output: Step,
    cache_salt: Option<String>,
}
struct SequenceLease {
    owner: Arc<ServedModel>,
    state: Option<RuntimeSequence>,
    history: Vec<u32>,
    output: Step,
    cache_salt: Option<String>,
    cacheable: bool,
}
impl Drop for SequenceLease {
    fn drop(&mut self) {
        if let Some(mut state) = self.state.take() {
            let valid = self.cacheable
                && state.position() == self.history.len()
                && !self.output.logits.is_empty();
            if valid || state.reset().is_ok() {
                self.owner.idle.lock().unwrap().push(CachedSequence {
                    state,
                    history: if valid {
                        std::mem::take(&mut self.history)
                    } else {
                        vec![]
                    },
                    output: if valid {
                        std::mem::take(&mut self.output)
                    } else {
                        Step::default()
                    },
                    cache_salt: if valid { self.cache_salt.take() } else { None },
                });
            }
        }
    }
}
struct Server {
    args: ServeArgs,
    profile: RwLock<control::ServiceProfile>,
    threads: Option<usize>,
    administration: Mutex<()>,
    scanning: Mutex<()>,
    hub: hub::Hub,
    tuning: tuning::Tuning,
    load_progress: Mutex<LoadProgress>,
    startup_error: RwLock<Option<String>>,
    desired_model: RwLock<Option<String>>,
    loaded: RwLock<Option<Arc<ServedModel>>>,
    catalogue: RwLock<BTreeMap<String, PathBuf>>,
    admission: Arc<Admission>,
    pool: RwLock<Arc<rayon::ThreadPool>>,
    connections: AtomicUsize,
    ids: AtomicU64,
    insight: insight::LiveTelemetry,
}

fn worker_pool(threads: Option<usize>) -> Result<Arc<rayon::ThreadPool>> {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if let Some(threads) = threads { builder = builder.num_threads(threads); }
    Ok(Arc::new(builder.build()?))
}

pub fn serve(mut args: ServeArgs, threads: Option<usize>) -> Result<()> {
    ensure!(
        args.host.is_loopback() || args.api_key.as_ref().is_some_and(|k| k.len() >= 16),
        "non-loopback binding requires an API key of at least 16 characters"
    );
    if let Some(n) = threads {
        ensure!((1..=1024).contains(&n), "invalid CPU worker count");
    }
    ensure!(
        (1..=4096).contains(&args.prefill_chunk_tokens),
        "prefill_chunk_tokens must be in 1..=4096"
    );
    ensure!(
        args.max_queued_prompt_tokens <= 16_777_216,
        "max_queued_prompt_tokens must be at most 16777216 (0 means automatic)"
    );
    let mut progress = LoadProgress::default();
    progress.begin();
    let (profile, catalogue, loaded, startup_error, desired_model) = control::startup(&mut args)?;
    let pending_restore = desired_model.clone().filter(|_| startup_error.is_none());
    if let Some(error) = &startup_error {
        progress.finish("failed", error.clone());
    } else if loaded.is_some() {
        progress.finish("ready", "Model ready");
    } else if pending_restore.is_none() {
        progress = LoadProgress::default();
    }
    let pool = worker_pool(profile.settings.threads.or(threads))?;
    let listener = TcpListener::bind((args.host, args.port))?;
    // Record the actual port, including an OS-assigned port requested with --port 0.
    args.port = listener.local_addr()?.port();
    let browser_host = if args.host.is_unspecified() {
        IpAddr::from([127, 0, 0, 1])
    } else {
        args.host
    };
    let ui_url = format!(
        "http://{}/",
        std::net::SocketAddr::new(browser_host, args.port)
    );
    eprintln!(
        "Forge native service listening on {} | model {} | {} active / {} queued",
        listener.local_addr()?,
        loaded
            .as_ref()
            .map(|m| m.id.as_str())
            .unwrap_or("none; configure in the browser"),
        profile.settings.parallel,
        profile.settings.max_queued
    );
    eprintln!("Configuration UI: {ui_url}");
    if args.open_ui {
        if let Err(error) = control::open_browser(&ui_url) {
            eprintln!("Could not open a browser ({error}); open {ui_url} manually.");
        }
    }
    let state = Arc::new(Server {
        admission: Admission::new(
            profile.settings.parallel,
            profile.settings.max_queued,
            Duration::from_secs(profile.settings.queue_timeout_secs),
            profile.settings.queue_token_budget(&args),
        ),
        args,
        profile: RwLock::new(profile),
        threads,
        administration: Mutex::new(()),
        scanning: Mutex::new(()),
        hub: hub::Hub::default(),
        tuning: tuning::Tuning::new(),
        load_progress: Mutex::new(progress),
        startup_error: RwLock::new(startup_error),
        desired_model: RwLock::new(desired_model),
        loaded: RwLock::new(loaded),
        catalogue: RwLock::new(catalogue),
        pool: RwLock::new(pool),
        connections: AtomicUsize::new(0),
        ids: AtomicU64::new(1),
        insight: insight::LiveTelemetry::new(),
    });
    if let Some(id) = pending_restore {
        control::restore_in_background(&state, id)?;
    }
    // Monitor memory even when no browser or inference client is connected.
    // The controller owns no model; each sample looks up the current one.
    let weak = Arc::downgrade(&state);
    std::thread::spawn(move || {
        let mut healthy_samples = 0u8;
        loop {
            std::thread::sleep(Duration::from_secs(1));
            let Some(server) = weak.upgrade() else { break };
            let loaded = match server.loaded.read() {
                Ok(loaded) => loaded,
                Err(_) => break,
            };
            let Some(model) = loaded.as_ref() else {
                healthy_samples = 0;
                continue;
            };
            let source = &model.model.source;
            let Ok(stats) = source.expert_cache_stats() else { continue };
            let Ok(available) = crate::memory::available_memory() else { continue };
            let report = model.model.report();
            let initial_ceiling = report["host_memory_plan"]
                ["expert_cache_capacity_bytes"].as_u64().unwrap_or(0);
            let future_context = report["future_ram_reserve_bytes"].as_u64().unwrap_or(0);
            let margin = report["host_memory_plan"]
                ["os_margin_bytes"].as_u64().unwrap_or(0)
                .max(512 * 1024 * 1024);
            // MemAvailable already includes current cached payload effects.
            // Add that payload back before subtracting the OS margin and
            // RAM context capacity promised to future sequence growth.
            let sustainable = available.saturating_add(stats.resident_bytes as u64)
                .saturating_sub(margin)
                .saturating_sub(future_context)
                .min(initial_ceiling);
            let current = stats.capacity_bytes as u64;
            let target = if sustainable < current {
                healthy_samples = 0;
                sustainable
            } else {
                healthy_samples = healthy_samples.saturating_add(1);
                if healthy_samples >= 3 {
                    healthy_samples = 0;
                    current.saturating_add(64 * 1024 * 1024).min(sustainable)
                } else {
                    current
                }
            };
            if let Err(error) = apply_expert_cache_target(source, &stats, target) {
                eprintln!("Forge expert-cache controller: {error:#}");
            }
            // No warmup is scheduled on growth; demand fetches fill the cache.
        }
    });
    for incoming in listener.incoming() {
        let mut stream = incoming?;
        if state
            .connections
            .try_update(Ordering::AcqRel, Ordering::Relaxed, |n| {
                if n < state.args.max_connections {
                    Some(n + 1)
                } else {
                    None
                }
            })
            .is_err()
        {
            let _ = json_response(&mut stream, 503, &error_value("connection limit reached"));
            continue;
        }
        let state = state.clone();
        std::thread::spawn(move || {
            struct Connection(Arc<Server>);
            impl Drop for Connection {
                fn drop(&mut self) {
                    self.0.connections.fetch_sub(1, Ordering::Release);
                }
            }
            let _guard = Connection(state.clone());
            let _ = stream.set_read_timeout(Some(Duration::from_secs(15)));
            let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
            let _ = stream.set_nodelay(true);
            if let Err(e) = handle(&state, &mut stream) {
                eprintln!("native request ended: {e}");
            }
        });
    }
    Ok(())
}

fn apply_expert_cache_target(
    source: &crate::gguf::Gguf,
    stats: &crate::gguf::ExpertCacheStats,
    target: u64,
) -> Result<()> {
    // A shrink can leave active leases above the target. Retry on later samples
    // even when the target is unchanged, so released leases are reclaimed while
    // idle instead of waiting for another capacity change or cache admission.
    if target != stats.capacity_bytes as u64 || stats.resident_bytes as u64 > target {
        source.resize_expert_cache(
            usize::try_from(target).context("expert cache target exceeds address range")?,
        )?;
    }
    Ok(())
}

struct Request {
    method: String,
    path: String,
    host: String,
    authorization: String,
    content_type: String,
    origin: Option<String>,
    body: Vec<u8>,
}
fn read_request(stream: &mut TcpStream) -> Result<Request> {
    let mut bytes = Vec::with_capacity(4096);
    let header_end;
    loop {
        if let Some(i) = bytes.windows(4).position(|x| x == b"\r\n\r\n") {
            header_end = i + 4;
            break;
        }
        ensure!(bytes.len() < 65536, "HTTP headers exceed64KiB");
        let mut chunk = [0u8; 4096];
        let n = stream.read(&mut chunk)?;
        ensure!(n > 0, "incomplete HTTP headers");
        bytes.extend_from_slice(&chunk[..n]);
    }
    ensure!(header_end <= 65536, "HTTP headers exceed64KiB");
    let mut headers = [httparse::EMPTY_HEADER; 128];
    let mut r = httparse::Request::new(&mut headers);
    ensure!(
        r.parse(&bytes[..header_end])?.is_complete(),
        "incomplete request"
    );
    let mut length = None;
    let mut host = None;
    let mut authorization = None;
    let mut content_type = String::new();
    let mut origin = None;
    for h in r.headers.iter() {
        let value = std::str::from_utf8(h.value)?.trim();
        match h.name.to_ascii_lowercase().as_str() {
            "content-length" => {
                ensure!(length.is_none(), "duplicate Content-Length");
                length = Some(value.parse::<usize>()?);
            }
            "transfer-encoding" => bail!("chunked request uploads are unsupported"),
            "host" => {
                ensure!(host.is_none(), "duplicate Host");
                host = Some(value.to_owned());
            }
            "authorization" => {
                ensure!(authorization.is_none(), "duplicate Authorization");
                authorization = Some(value.to_owned());
            }
            "content-type" => content_type = value.to_owned(),
            "origin" => {
                ensure!(origin.is_none(), "duplicate Origin");
                origin = Some(value.to_owned());
            }
            _ => {}
        }
    }
    let method = r.method.context("missing method")?.to_owned();
    let path = r.path.context("missing path")?.to_owned();
    let length = length.unwrap_or(0);
    ensure!(length <= 8 * 1024 * 1024, "request body exceeds8MiB");
    let mut body = bytes[header_end..].to_vec();
    ensure!(body.len() <= length, "unexpected pipelined request bytes");
    body.resize(length, 0);
    let received = bytes.len() - header_end;
    if received < length {
        stream.read_exact(&mut body[received..])?;
    }
    Ok(Request {
        method,
        path,
        host: host.context("Host header required")?,
        authorization: authorization.unwrap_or_default(),
        content_type,
        origin,
        body,
    })
}
fn error_value(message: &str) -> Value {
    json!({"error":{"message":message,"type":"invalid_request_error","code":Value::Null}})
}
fn json_response(stream: &mut TcpStream, status: u16, value: &Value) -> Result<()> {
    let body = serde_json::to_vec(value)?;
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        409 => "Conflict",
        429 => "Too Many Requests",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Error",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n\r\n",
        body.len()
    )?;
    stream.write_all(&body)?;
    Ok(())
}
fn disconnected(stream: &TcpStream) -> bool {
    if stream.set_nonblocking(true).is_err() {
        return false;
    }
    let result = stream.peek(&mut [0u8]);
    let _ = stream.set_nonblocking(false);
    match result {
        Ok(0) => true,
        Err(e) => e.kind() != std::io::ErrorKind::WouldBlock,
        Ok(_) => false,
    }
}
fn authorized(value: &str, key: &str) -> bool {
    let candidate = value.strip_prefix("Bearer ").unwrap_or("").as_bytes();
    candidate.len() == key.len()
        && candidate
            .iter()
            .zip(key.bytes())
            .fold(0u8, |n, (a, b)| n | (a ^ b))
            == 0
}
fn host_allowed(host: &str, local: IpAddr) -> bool {
    let name = host
        .rsplit_once(':')
        .filter(|(_, p)| p.parse::<u16>().is_ok())
        .map_or(host, |x| x.0)
        .trim_end_matches('.')
        .to_ascii_lowercase();
    matches!(name.as_str(), "localhost" | "127.0.0.1" | "[::1]")
        || name == local.to_string()
        || name == format!("[{local}]")
}

fn handle(server: &Arc<Server>, stream: &mut TcpStream) -> Result<()> {
    let request = match read_request(stream) {
        Ok(r) => r,
        Err(e) => return json_response(stream, 400, &error_value(&e.to_string())),
    };
    // A wildcard listener (0.0.0.0 or ::) has no client-facing address of its
    // own. Validate against the interface that actually accepted this socket,
    // retaining exact Host checks without rejecting legitimate LAN clients.
    if !host_allowed(&request.host, stream.local_addr()?.ip()) {
        return json_response(stream, 400, &error_value("Host is not allowed"));
    }
    if !control::origin_allowed(request.origin.as_deref(), &request.host, server.args.port) {
        return json_response(stream, 403, &error_value("Origin must match this service"));
    }
    if request.method == "GET" {
        if let Some((mime, body)) = control::asset(&request.path) {
            return control::asset_response(stream, mime, body);
        }
    }
    if server
        .args
        .api_key
        .as_ref()
        .is_some_and(|key| !authorized(&request.authorization, key))
    {
        return json_response(stream, 401, &error_value("valid Bearer API key required"));
    }
    if request.method == "POST"
        && !request
            .content_type
            .split(';')
            .next()
            .is_some_and(|value| value.trim().eq_ignore_ascii_case("application/json"))
    {
        return json_response(
            stream,
            400,
            &error_value("Content-Type must be application/json"),
        );
    }
    if control::handle_control(server, stream, &request)? {
        return Ok(());
    }
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => {
            let ready = server.loaded.read().unwrap().is_some();
            json_response(
                stream,
                if ready { 200 } else { 503 },
                &json!({"status":if ready{"ok"}else{"no_model"},"engine":crate::ENGINE_ID}),
            )
        }
        ("GET", "/api/insight/live") => json_response(stream, 200, &insight::live(server)),
        ("GET", "/v1/models") => {
            let active = server.loaded.read().unwrap().as_ref().map(|m| m.id.clone());
            let data=server.catalogue.read().unwrap().keys().map(|id|json!({"id":id,"object":"model","created":0,"owned_by":"local","loaded":active.as_ref()==Some(id)})).collect::<Vec<_>>();
            json_response(stream, 200, &json!({"object":"list","data":data}))
        }
        ("GET", "/v1/status") => {
            let admission = server.admission.snapshot();
            let progress = server.load_progress.lock().unwrap().snapshot();
            let m = server.loaded.read().unwrap();
            json_response(
                stream,
                200,
                &json!({
                    "load_progress": progress,
                    "model": m.as_ref().map(|m| &m.id),
                    "runtime": m.as_ref().map(|m| m.model.report()),
                    "admission": admission,
                    "idle_sequences": m.as_ref().map(|m| m.idle.lock().unwrap().len()).unwrap_or(0),
                    "prefill_chunk_tokens": server.args.prefill_chunk_tokens,
                    "connections": server.connections.load(Ordering::Relaxed),
                }),
            )
        }
        ("POST", "/v1/chat/completions") | ("POST", "/api/context") => {
            if !request
                .content_type
                .split(';')
                .next()
                .is_some_and(|x| x.trim().eq_ignore_ascii_case("application/json"))
            {
                return json_response(
                    stream,
                    400,
                    &error_value("Content-Type must be application/json"),
                );
            }
            let raw: Value = match serde_json::from_slice(&request.body) {
                Ok(v) => v,
                Err(e) => return json_response(stream, 400, &error_value(&e.to_string())),
            };
            let loaded = {
                // Configuration commits hold the admission barrier. Snapshot the
                // model under that barrier so preparation never observes a partly
                // installed model/configuration pair. The Arc then prevents a swap.
                let gate = server.admission.state.lock().unwrap();
                if gate.changing {
                    return json_response(
                        stream,
                        503,
                        &error_value("service configuration is changing"),
                    );
                }
                match server.loaded.read().unwrap().clone() {
                    Some(model) => model,
                    None => {
                        return json_response(
                            stream,
                            503,
                            &error_value(
                                "no model is loaded; configure the service in the browser",
                            ),
                        );
                    }
                }
            };
            let mut parsed = match ChatRequest::parse(&raw, loaded.default_max_tokens) {
                Ok(r) => r,
                Err(e) => return json_response(stream, 400, &error_value(&e.to_string())),
            };
            if parsed.model != loaded.id {
                return json_response(
                    stream,
                    400,
                    &error_value(
                        "requested model is not loaded; select an exact catalogue ID with POST /api/models/load",
                    ),
                );
            }
            let text = match loaded.template.render(&parsed) {
                Ok(t) => t,
                Err(e) => {
                    return json_response(
                        stream,
                        400,
                        &error_value(&format!("chat template: {e}")),
                    );
                }
            };
            parsed.opens_reasoning = text.trim_end().ends_with("<think>");
            let prompt = match loaded.model.tokenizer.encode(&text) {
                Ok(p) => p,
                Err(e) => return json_response(stream, 400, &error_value(&e.to_string())),
            };
            if prompt.is_empty()
                || prompt
                    .len()
                    .checked_add(parsed.max_tokens)
                    .is_none_or(|n| n > loaded.model.context())
            {
                return json_response(
                    stream,
                    400,
                    &error_value(&format!(
                        "prompt {} + output {} exceeds context {}",
                        prompt.len(),
                        parsed.max_tokens,
                        loaded.model.context()
                    )),
                );
            }
            if request.path == "/api/context" {
                return json_response(
                    stream,
                    200,
                    &json!({"model":loaded.id,"prompt_tokens":prompt.len(),"reserved_output_tokens":parsed.max_tokens,"context":loaded.model.context(),"remaining_tokens":loaded.model.context()-prompt.len()-parsed.max_tokens}),
                );
            }
            // Preparation happens before admission and before any SSE headers.
            let _permit = match server.admission.acquire(stream, prompt.len()) {
                Ok(p) => p,
                Err((code, message)) => {
                    return if code == 499 {
                        Ok(())
                    } else {
                        json_response(stream, code, &error_value(message))
                    };
                }
            };
            if server
                .loaded
                .read()
                .unwrap()
                .as_ref()
                .is_none_or(|m| m.id != loaded.id)
            {
                return json_response(
                    stream,
                    409,
                    &error_value("model changed while request was preparing; retry"),
                );
            }
            let lease = match loaded.lease(&prompt, parsed.cache_salt.as_deref()) {
                Ok(s) => s,
                Err(e) => {
                    return json_response(
                        stream,
                        503,
                        &error_value(&format!("sequence allocation: {e}")),
                    );
                }
            };
            let pool = Arc::clone(&server.pool.read().unwrap());
            pool.install(|| generate(server, stream, &loaded, lease, &parsed, &prompt))
        }
        _ => json_response(stream, 404, &error_value("unknown native API route")),
    }
}

fn sse(stream: &mut TcpStream, value: &Value) -> Result<()> {
    write!(stream, "data: {}\n\n", serde_json::to_string(value)?)?;
    stream.flush()?;
    Ok(())
}
fn chunk(id: &str, created: u64, model: &str, delta: Value, finish: Option<&str>) -> Value {
    json!({"id":id,"object":"chat.completion.chunk","created":created,"model":model,"choices":[{"index":0,"delta":delta,"finish_reason":finish}]})
}

fn generate(
    server: &Server,
    stream: &mut TcpStream,
    loaded: &Arc<ServedModel>,
    mut lease: SequenceLease,
    request: &ChatRequest,
    prompt: &[u32],
) -> Result<()> {
    let created = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let id = format!(
        "chatcmpl-native-{created}-{}",
        server.ids.fetch_add(1, Ordering::Relaxed)
    );
    let cached_prompt_tokens = lease.history.len();
    let mut insight_trace = server.insight.begin(prompt.len(), cached_prompt_tokens);
    let state = lease.state.as_mut().unwrap();
    let start = Instant::now();
    let output = &mut lease.output;
    // Check disconnects during long prefill; final prefix returns actual logits.
    for chunk in prompt[cached_prompt_tokens..].chunks(server.args.prefill_chunk_tokens) {
        if disconnected(stream) {
            return Ok(());
        }
        match loaded.model.prefill(state, chunk) {
            Ok(o) => {
                *output = o;
                lease.history.extend_from_slice(chunk);
            }
            Err(e) => {
                lease.cacheable = false;
                return json_response(stream, 503, &error_value(&format!("prefill failed: {e}")));
            }
        }
    }
    let prefill = start.elapsed();
    insight_trace.prefill_finished(prefill);
    if request.stream {
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nX-Accel-Buffering: no\r\n\r\n"
        )?;
        sse(
            stream,
            &chunk(
                &id,
                created,
                &loaded.id,
                json!({"role":"assistant","content":""}),
                None,
            ),
        )?;
    }
    let mut sampler = Sampler::new(request.seed, prompt);
    let mut tokens = Vec::with_capacity(request.max_tokens);
    let mut text = String::new();
    let mut finish = "length";
    let mut emitted = 0usize;
    let start = Instant::now();
    let mut decode_token = loaded.model.tokenizer.stream_decoder();
    let longest_stop = request.stops.iter().map(String::len).max().unwrap_or(0);
    let hold = longest_stop.max(16);
    let run = (|| -> Result<()> {
        for n in 0..request.max_tokens {
            if disconnected(stream) {
                return Ok(());
            }
            let token = sampler.sample(&output.logits, request)?;
            if loaded.model.tokenizer.is_eos(token) {
                finish = "stop";
                break;
            }
            tokens.push(token);
            let previous = text.len();
            if let Some(next) = decode_token(token)? {
                text.push_str(&next);
            }
            // A new stop can begin only in the new suffix or across its boundary.
            let mut scan = previous.saturating_sub(longest_stop.saturating_sub(1));
            while !text.is_char_boundary(scan) {
                scan -= 1;
            }
            if let Some(at) = request
                .stops
                .iter()
                .filter_map(|s| text[scan..].find(s).map(|at| scan + at))
                .min()
            {
                text.truncate(at);
                finish = "stop";
                break;
            }
            // Tool/reasoning payloads are emitted after complete parsing; plain text can stream immediately.
            if request.stream
                && request.tools.is_empty()
                && !request.opens_reasoning
                && !text.starts_with("<think>")
            {
                let mut end = text.len().saturating_sub(hold);
                while !text.is_char_boundary(end) {
                    end -= 1;
                }
                if end > emitted {
                    sse(
                        stream,
                        &chunk(
                            &id,
                            created,
                            &loaded.id,
                            json!({"content":&text[emitted..end]}),
                            None,
                        ),
                    )?;
                    emitted = end;
                }
            }
            if n + 1 < request.max_tokens {
                if let Err(e) = loaded.model.step_into(state, token, output) {
                    lease.cacheable = false;
                    return Err(e);
                }
                lease.history.push(token);
            }
        }
        // Flush a final incomplete byte sequence exactly once. Streaming keeps a
        // suffix back, so replacement characters cannot revise already-sent text.
        let completed = loaded.model.tokenizer.decode(&tokens)?;
        ensure!(
            completed.starts_with(&text[..emitted]),
            "incremental decoder changed an emitted prefix"
        );
        text = completed;
        if let Some(at) = request.stops.iter().filter_map(|s| text.find(s)).min() {
            text.truncate(at);
            finish = "stop";
        }
        Ok(())
    })();
    if let Err(e) = run {
        if request.stream {
            sse(stream, &error_value(&format!("generation failed: {e}")))?;
            write!(stream, "data: [DONE]\n\n")?;
            return Ok(());
        }
        return json_response(stream, 503, &error_value(&e.to_string()));
    }
    let decode_seconds = start.elapsed().as_secs_f64();
    let timings = json!({"prefill_seconds":prefill.as_secs_f64(),"cached_prompt_tokens":cached_prompt_tokens,"decode_seconds":decode_seconds});
    if disconnected(stream) {
        return Ok(());
    }
    let message = match parse_output(&text, request, &id) {
        Ok(m) => m,
        Err(e) => {
            if request.stream {
                sse(stream, &error_value(&e.to_string()))?;
                write!(stream, "data: [DONE]\n\n")?;
                return Ok(());
            }
            return json_response(stream, 503, &error_value(&e.to_string()));
        }
    };
    if message.get("tool_calls").is_some() {
        finish = "tool_calls";
    }
    insight_trace.complete(tokens.len(), decode_seconds);
    let usage = json!({"prompt_tokens":prompt.len(),"completion_tokens":tokens.len(),"total_tokens":prompt.len()+tokens.len(),"prompt_tokens_details":{"cached_tokens":cached_prompt_tokens}});
    if request.stream {
        if let Some(reasoning) = message.get("reasoning_content") {
            sse(
                stream,
                &chunk(
                    &id,
                    created,
                    &loaded.id,
                    json!({"reasoning_content":reasoning}),
                    None,
                ),
            )?;
        }
        let content = message["content"].as_str().unwrap_or("");
        ensure!(
            content.starts_with(&text[..emitted]),
            "streamed prefix differs from final content"
        );
        if content.len() > emitted {
            sse(
                stream,
                &chunk(
                    &id,
                    created,
                    &loaded.id,
                    json!({"content":&content[emitted..]}),
                    None,
                ),
            )?;
        }
        if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
            for (i, call) in calls.iter().enumerate() {
                let mut call = call.clone();
                call["index"] = json!(i);
                sse(
                    stream,
                    &chunk(&id, created, &loaded.id, json!({"tool_calls":[call]}), None),
                )?;
            }
        }
        sse(
            stream,
            &chunk(&id, created, &loaded.id, json!({}), Some(finish)),
        )?;
        if request.include_usage {
            sse(
                stream,
                &json!({"id":id,"object":"chat.completion.chunk","created":created,"model":loaded.id,"choices":[],"usage":usage,"timings":timings}),
            )?;
        }
        write!(stream, "data: [DONE]\n\n")?;
        stream.flush()?;
        Ok(())
    } else {
        json_response(
            stream,
            200,
            &json!({"id":id,"object":"chat.completion","created":created,"model":loaded.id,"choices":[{"index":0,"message":message,"finish_reason":finish}],"usage":usage,"timings":timings}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_controller_reclaims_released_leases_at_an_unchanged_target() {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-qwen3moe.gguf");
        let capacity = 1024 * 1024;
        let source = crate::gguf::Gguf::open_with_cache(fixture, capacity).unwrap();
        let lease = source.expert_cached("blk.0.ffn_gate_exps.weight", 0).unwrap();
        assert!(lease.is_owned());
        let payload = lease.tensor().bytes().to_vec();
        let before = source.expert_cache_stats().unwrap();
        assert!(before.resident_bytes > 0);

        apply_expert_cache_target(&source, &before, 0).unwrap();
        let pinned = source.expert_cache_stats().unwrap();
        assert_eq!(pinned.capacity_bytes, 0);
        assert_eq!(pinned.resident_bytes, before.resident_bytes);
        assert_eq!(pinned.pinned_bytes, before.resident_bytes);
        assert_eq!(lease.tensor().bytes(), payload);

        // Simulate the next idle controller sample after generation releases
        // its lease: no miss, warmup, new request or target change occurs.
        drop(lease);
        let released = source.expert_cache_stats().unwrap();
        assert_eq!(released.pinned_bytes, 0);
        apply_expert_cache_target(&source, &released, 0).unwrap();
        let trimmed = source.expert_cache_stats().unwrap();
        assert_eq!(trimmed.resident_bytes, 0);
        assert_eq!(trimmed.resident_entries, 0);
        assert_eq!(trimmed.payload_bytes_read, before.payload_bytes_read);

        apply_expert_cache_target(&source, &trimmed, capacity as u64).unwrap();
        let recovered = source.expert_cache_stats().unwrap();
        assert_eq!(recovered.capacity_bytes, capacity);
        assert_eq!(recovered.resident_bytes, 0);
        assert_eq!(recovered.payload_bytes_read, before.payload_bytes_read);
    }

    #[test]
    fn auth_and_hosts_are_exact() {
        assert!(authorized("Bearer abc", "abc"));
        assert!(!authorized("Bearer abC", "abc"));
        assert!(!authorized("abc", "abc"));
        let b = "127.0.0.1".parse().unwrap();
        assert!(host_allowed("localhost:9070", b));
        assert!(!host_allowed("localhost.attacker.example:9070", b));
        let lan = "192.168.1.10".parse().unwrap();
        assert!(host_allowed("192.168.1.10:9070", lan));
        assert!(!host_allowed("192.168.1.11:9070", lan));
        assert!(!host_allowed("192.168.1.10.attacker.example:9070", lan));
        let ipv6 = "fd00::9070".parse().unwrap();
        assert!(host_allowed("[fd00::9070]:9070", ipv6));
        assert!(!host_allowed("[fd00::9071]:9070", ipv6));
    }

    #[test]
    fn cache_namespaces_reuse_only_identical_executed_prefixes() {
        assert!(reusable_prefix(&[1, 2], None, &[1, 2, 3], None));
        assert!(reusable_prefix(&[1, 2], Some("agent-a"), &[1, 2, 3], Some("agent-a")));
        assert!(!reusable_prefix(&[1, 2], Some("agent-a"), &[1, 2, 3], Some("agent-b")));
        assert!(!reusable_prefix(&[1, 2], None, &[1, 2, 3], Some("agent-a")));
        assert!(!reusable_prefix(&[1, 2], Some("agent-a"), &[1, 2, 3], None));
        assert!(!reusable_prefix(&[], None, &[1, 2, 3], None));
        assert!(!reusable_prefix(&[1, 2, 4], None, &[1, 2, 3], None));
    }

    #[test]
    fn queued_token_budget_is_accounted_on_removal() {
        let gate = Admission::new(1, 2, Duration::from_secs(1), 12);
        let mut state = gate.state.lock().unwrap();
        state.queue.push_back(QueuedRequest { ticket: 1, prompt_tokens: 5 });
        state.queue.push_back(QueuedRequest { ticket: 2, prompt_tokens: 7 });
        state.queued_prompt_tokens = 12;
        state.remove_waiter(1);
        assert_eq!(state.queued_prompt_tokens, 7);
        assert_eq!(state.queue.front().map(|x| x.ticket), Some(2));
        state.remove_waiter(2);
        assert_eq!(state.queued_prompt_tokens, 0);
        assert!(state.queue.is_empty());
    }

    #[test]
    fn queued_prompt_budget_rejects_oversized_waiters() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let _client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (socket, _) = listener.accept().unwrap();
        let gate = Admission::new(1, 2, Duration::from_secs(1), 4);
        let permit = gate.acquire(&socket, 3).unwrap();
        assert_eq!(gate.acquire(&socket, 5).err().unwrap().0, 429);
        assert_eq!(gate.snapshot()["queued_prompt_tokens"], 0);
        drop(permit);
    }

    #[test]
    fn generation_permit_releases_and_queue_limits_apply() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let _client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (socket, _) = listener.accept().unwrap();
        let gate = Admission::new(1, 0, Duration::from_secs(1), 4);
        let permit = gate.acquire(&socket, 3).unwrap_or_else(|_| panic!("first"));
        assert_eq!(gate.snapshot()["active"], 1);
        assert_eq!(gate.acquire(&socket, 2).err().unwrap().0, 429);
        drop(permit);
        assert_eq!(gate.snapshot()["active"], 0);
        assert!(gate.acquire(&socket, 2).is_ok());
    }
}
