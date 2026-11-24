//! An explicit per-model tuning job releases serving memory while a child process measures it.
use super::*;
use control::{ServiceProfile, ServiceSettings};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs,
    process::{Child, Command, Stdio},
    sync::atomic::AtomicBool,
};

const MAX_REPORT_BYTES: u64 = 8 * 1024 * 1024;
const JOB_TIMEOUT: Duration = Duration::from_secs(30 * 60);

pub(super) struct Tuning {
    state: Mutex<Job>,
    cancel: AtomicBool,
    machine: String,
}

#[derive(Default)]
struct Job {
    phase: &'static str,
    model: Option<String>,
    directory: Option<PathBuf>,
    preparing: bool,
    activated: bool,
    context: Option<usize>,
    identity: Option<String>,
    progress: Option<Value>,
    result: Option<Value>,
    error: Option<String>,
    worker_pid: Option<u32>,
    cached: bool,
}

impl Job {
    fn active(&self) -> bool {
        matches!(
            self.phase,
            "starting" | "fitting" | "running" | "saving" | "restoring" | "applying"
        )
    }
    fn snapshot(&self) -> Value {
        json!({"phase": if self.phase.is_empty() { "idle" } else { self.phase }, "model":self.model,
            "directory":self.directory,"preparing":self.preparing,"activated":self.activated,"context":self.context,"cached":self.cached,"progress":self.progress,"result":self.result,
            "error":self.error,"worker_pid":self.worker_pid})
    }
}

impl Tuning {
    pub(super) fn new() -> Self {
        let cpu = std::env::var("PROCESSOR_IDENTIFIER").ok().or_else(|| {
            fs::read_to_string("/proc/cpuinfo").ok().and_then(|info| {
                info.lines()
                    .find(|line| line.starts_with("model name"))
                    .map(str::to_owned)
            })
        });
        let executable = std::env::current_exe()
            .ok()
            .and_then(|path| metadata_stamp(&path).ok());
        let description = json!({"engine":crate::ENGINE_VERSION,"os":std::env::consts::OS,"arch":std::env::consts::ARCH,
            "cpu":cpu,"cpus":std::thread::available_parallelism().map(usize::from).unwrap_or(1),
            "hardware":stable_hardware(crate::runtime::hardware_report()),"executable":executable});
        Self {
            state: Mutex::new(Job::default()),
            cancel: AtomicBool::new(false),
            machine: hash(&description),
        }
    }
}

// Runtime heap usage and free-memory snapshots change as processes come and go;
// they are fit evidence, not a machine identity or a cache invalidation signal.
fn stable_hardware(value: Value) -> Value {
    match value {
        Value::Object(mut fields) => {
            fields.retain(|key, _| {
                let key = key.to_ascii_lowercase();
                !["budget", "usage", "available", "free", "allocated"]
                    .iter()
                    .any(|part| key.contains(part))
                    && !key.starts_with("live_")
                    && !matches!(
                        key.as_str(),
                        "compatible" | "rejection_reasons" | "discovery_only" | "compute_verified"
                    )
            });
            for value in fields.values_mut() {
                *value = stable_hardware(value.take());
            }
            Value::Object(fields)
        }
        Value::Array(values) => Value::Array(values.into_iter().map(stable_hardware).collect()),
        value => value,
    }
}
fn hash(value: &Value) -> String {
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(value).expect("JSON value"))
    )
}
fn metadata_stamp(path: &Path) -> Result<Value> {
    let metadata = path.metadata()?;
    let modified = metadata.modified()?.duration_since(UNIX_EPOCH)?;
    Ok(
        json!({"path":path.canonicalize()?,"bytes":metadata.len(),"modified_seconds":modified.as_secs(),"modified_nanos":modified.subsec_nanos()}),
    )
}
fn model_identity(server: &Server, path: &Path) -> Result<String> {
    let source = crate::gguf::Gguf::open(path)?;
    let shards = source
        .shard_paths()
        .iter()
        .map(|p| metadata_stamp(p))
        .collect::<Result<Vec<_>>>()?;
    Ok(hash(
        &json!({"machine":server.tuning.machine,"shards":shards,"engine":"native"}),
    ))
}
fn identity_path(server: &Server, path: &Path, settings: &ServiceSettings) -> Result<String> {
    let mut identity = json!({"model_identity":model_identity(server,path)?,"context":settings.ctx,
        "limits":{"fast_math":settings.fast_math,"kv_precision":settings.kv_precision,
        "kv_placement":settings.kv_placement,"kv_ram_mib":settings.kv_ram_mib,
        "device":settings.device,"vram_mib":settings.vram_mib,"expert_ram_mib":settings.expert_ram_mib,"ram_margin_mib":settings.ram_margin_mib}});
    // Preserve existing profile identities when the new setting was omitted.
    if let Some(mode) = settings.direct_experts { identity["limits"]["direct_experts"] = json!(mode); }
    Ok(hash(&identity))
}
fn identity(server: &Server, model: &ServedModel, settings: &ServiceSettings) -> Result<String> {
    identity_path(server, &model.model.source.path, settings)
}
fn store_root(server: &Server) -> Result<PathBuf> {
    let parent = server
        .args
        .settings_file
        .as_deref()
        .and_then(Path::parent)
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or(std::env::current_dir()?);
    Ok(parent.join(".nextora-tuning"))
}
fn read_json(path: &Path) -> Result<Value> {
    let metadata = fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_file()
            && !metadata.file_type().is_symlink()
            && metadata.len() <= MAX_REPORT_BYTES,
        "invalid tuning report file"
    );
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(MAX_REPORT_BYTES + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_REPORT_BYTES,
        "tuning report too large"
    );
    serde_json::from_slice(&bytes).context("read tuning report")
}
fn write_json(path: &Path, value: &Value) -> Result<()> {
    let parent = path.parent().context("tuning report parent missing")?;
    let serial = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let temporary = parent.join(format!(".report-{}-{serial}.tmp", std::process::id()));
    let result = (|| {
        let mut options = fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temporary)?;
        let bytes = serde_json::to_vec_pretty(value)?;
        ensure!(
            bytes.len() as u64 <= MAX_REPORT_BYTES,
            "tuning report too large"
        );
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}
fn cached(server: &Server, key: &str) -> Option<Value> {
    let record = read_json(&store_root(server).ok()?.join(format!("{key}.json"))).ok()?;
    (record["identity"] == key)
        .then(|| record["result"].clone())
        .filter(|r| r["winner"].is_object())
}

fn status(server: &Server) -> Result<Value> {
    {
        let job = server.tuning.state.lock().unwrap();
        if job.active() || job.preparing {
            return Ok(job.snapshot());
        }
    }
    let profile = server.profile.read().unwrap().clone();
    let loaded = server.loaded.read().unwrap();
    let Some(model) = loaded.as_ref() else {
        return Ok(server.tuning.state.lock().unwrap().snapshot());
    };
    let key = identity(server, model, &profile.settings)?;
    let job = server.tuning.state.lock().unwrap();
    if job.identity.as_deref() == Some(&key) && matches!(job.phase, "cancelled" | "failed") {
        return Ok(job.snapshot());
    }
    if let Some(report) = cached(server, &key) {
        return Ok(Job {
            phase: "complete",
            model: Some(model.id.clone()),
            context: Some(profile.settings.ctx),
            identity: Some(key),
            worker_pid: report["worker_pid"]
                .as_u64()
                .and_then(|n| u32::try_from(n).ok()),
            result: Some(report),
            cached: true,
            ..Job::default()
        }
        .snapshot());
    }
    if job.identity.as_deref() == Some(&key) {
        return Ok(job.snapshot());
    }
    Ok(Job {
        phase: "idle",
        model: Some(model.id.clone()),
        context: Some(profile.settings.ctx),
        ..Job::default()
    }
    .snapshot())
}

#[derive(Clone, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct Target {
    directory: PathBuf,
    model: String,
}
#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct Start {
    force: bool,
    target: Option<Target>,
    mode: Option<String>,
    goal: Option<crate::autotune::SetupGoal>,
    activate: Option<bool>,
}
#[derive(Clone)]
struct RestoreSnapshot {
    profile: ServiceProfile,
    path: Option<PathBuf>,
    desired: Option<String>,
    startup_error: Option<String>,
    catalogue: BTreeMap<String, PathBuf>,
}
struct StudyTarget {
    profile: ServiceProfile,
    path: PathBuf,
    preparing: bool,
    activate: bool,
    using_saved: bool,
    goal: Option<crate::autotune::SetupGoal>,
    model_identity: String,
}
fn resolve_target(target: &Target) -> Result<(PathBuf, PathBuf, control::FoundModel)> {
    let root = target
        .directory
        .canonicalize()
        .context("model directory is unavailable")?;
    ensure!(root.is_dir(), "model directory must be a directory");
    let path = control::selected_path(&root, &target.model)?;
    let found = control::inspect_model(&root, &path)?;
    Ok((root, path, found))
}
fn prepared_record(server: &Server, path: &Path) -> Option<Value> {
    let key = model_identity(server, path).ok()?;
    let record = read_json(&store_root(server).ok()?.join(format!("profile-{key}.json"))).ok()?;
    (record["schema_version"] == 2
        && record["model_identity"] == key
        && record["result"]["winner"].is_object())
    .then_some(record)
}
fn prepared_status(server: &Server, target: &Target) -> Result<Value> {
    let (root, path, _) = resolve_target(target)?;
    let Some(record) = prepared_record(server, &path) else {
        return Ok(Job {
            phase: "idle",
            model: Some(target.model.clone()),
            directory: Some(root),
            preparing: true,
            ..Job::default()
        }
        .snapshot());
    };
    let report = record["result"].clone();
    Ok(Job {
        phase: "complete",
        model: Some(target.model.clone()),
        directory: Some(root),
        preparing: true,
        context: report["context"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok()),
        result: Some(report),
        cached: true,
        ..Job::default()
    }
    .snapshot())
}
fn start(server: &Arc<Server>, input: Start, use_saved: bool) -> Result<(u16, Value)> {
    ensure!(
        input
            .mode
            .as_deref()
            .is_none_or(|m| matches!(m, "prepare" | "tune")),
        "unknown setup mode"
    );
    let preparing = use_saved || input.target.is_some() || input.mode.as_deref() == Some("prepare");
    let goal = if preparing {
        Some(input.goal.clone().unwrap_or_default())
    } else {
        None
    };
    if let Some(goal) = &goal {
        goal.validate()?;
    }
    let Ok(_admin) = server.administration.try_lock() else {
        return Ok((
            409,
            error_value("another model or tuning operation is active"),
        ));
    };
    let original = server.profile.read().unwrap().clone();
    let selection = input.target.clone().or_else(|| {
        Some(Target {
            directory: original.directory.clone()?,
            model: original.model.clone()?,
        })
    });
    let (root, path, found) = if let Some(selection) = selection {
        resolve_target(&selection)?
    } else {
        return Ok((
            409,
            error_value("select a downloaded model to prepare, or load a model before tuning"),
        ));
    };
    let mut profile = original.clone();
    profile.directory = Some(root.clone());
    profile.model = Some(found.id.clone());
    if preparing {
        profile.settings.parallel = 1;
        if goal.as_ref().is_none_or(|goal| goal.context != "fixed") {
            profile.settings.ctx = profile.settings.ctx.min(found.training_context).max(2);
        } else {
            ensure!(
                profile.settings.ctx <= found.training_context,
                "fixed requested context exceeds this model's declared limit"
            );
        }
        profile.settings.max_tokens = profile.settings.max_tokens.min(profile.settings.ctx - 1);
        if !matches!(found.architecture.as_str(), "qwen3" | "qwen3moe") {
            profile.settings.fast_math = false;
            profile.settings.direct_experts = None;
            profile.settings.kv_precision = crate::vulkan::KvPrecision::F32;
            profile.settings.kv_placement = crate::vulkan::KvPlacement::Gpu;
            profile.settings.kv_ram_mib = None;
        }
    }
    let mut gate = server.admission.state.lock().unwrap();
    if gate.active > 0 || !gate.queue.is_empty() || gate.changing {
        return Ok((
            409,
            error_value("wait for active and queued requests before model setup"),
        ));
    }
    let loaded = server.loaded.read().unwrap();
    if !preparing && loaded.is_none() {
        return Ok((409, error_value("load a model before tuning")));
    }
    if loaded
        .as_ref()
        .is_some_and(|model| Arc::strong_count(model) > 1)
    {
        return Ok((
            409,
            error_value("a request is preparing against the model; retry when idle"),
        ));
    }
    let restore = RestoreSnapshot {
        profile: original,
        path: loaded.as_ref().map(|m| m.model.source.path.clone()),
        desired: server.desired_model.read().unwrap().clone(),
        startup_error: server.startup_error.read().unwrap().clone(),
        catalogue: server.catalogue.read().unwrap().clone(),
    };
    let model_key = model_identity(server, &path)?;
    let key = if preparing {
        let mut limits = serde_json::to_value(&profile.settings)?;
        for field in [
            "threads",
            "max_tokens",
            "max_queued",
            "queue_timeout_secs",
            "parallel",
        ] {
            limits.as_object_mut().unwrap().remove(field);
        }
        if goal.as_ref().is_none_or(|g| g.context != "fixed") {
            limits.as_object_mut().unwrap().remove("ctx");
        }
        hash(&json!({"model_identity":model_key,"goal":goal,"settings":limits}))
    } else {
        identity_path(server, &path, &profile.settings)?
    };
    let prepared = if preparing && !input.force {
        prepared_record(server, &path)
    } else {
        None
    };
    let saved = if use_saved {
        Some(
            prepared
                .as_ref()
                .context("prepare this model before using its saved profile")?["result"]
                .clone(),
        )
    } else if preparing {
        prepared
            .filter(|record| record["identity"] == key)
            .map(|record| record["result"].clone())
    } else if input.force {
        None
    } else {
        cached(server, &key)
    };
    if use_saved {
        profile = serde_json::from_value(
            prepared_record(server, &path).context("saved profile disappeared")?["profile"].clone(),
        )?;
        profile.settings.validate(&server.args)?;
    }
    let activate = if use_saved {
        true
    } else {
        input.activate.unwrap_or(!preparing)
    };
    if preparing && !activate {
        if let Some(report) = saved.as_ref() {
            *server.tuning.state.lock().unwrap() = Job {
                phase: "complete",
                model: profile.model.clone(),
                directory: profile.directory.clone(),
                preparing: true,
                context: report["context"]
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok()),
                identity: Some(key),
                result: Some(report.clone()),
                cached: true,
                ..Job::default()
            };
            return Ok((200, server.tuning.state.lock().unwrap().snapshot()));
        }
    }
    let target = StudyTarget {
        profile,
        path,
        preparing,
        activate,
        using_saved: use_saved,
        goal,
        model_identity: model_key,
    };
    let root = store_root(server)?;
    fs::create_dir_all(&root).context("create tuning results folder")?;
    ensure!(
        !fs::symlink_metadata(&root)?.file_type().is_symlink(),
        "tuning results directory must not be a symlink"
    );
    let serial = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let work = root.join(format!("job-{}-{serial}", std::process::id()));
    fs::create_dir(&work)?;
    let mut baseline = target.profile.settings.clone();
    baseline.threads = Some(server.pool.read().unwrap().current_num_threads());
    write_json(
        &work.join("request.json"),
        &json!({"model":target.path,"context":baseline.ctx,"settings":baseline,
        "machine":server.tuning.machine,"prepare":target.goal}),
    )?;
    server.tuning.cancel.store(false, Ordering::Release);
    *server.tuning.state.lock().unwrap() = Job {
        phase: if saved.is_some() {
            if activate { "applying" } else { "saving" }
        } else {
            "starting"
        },
        model: target.profile.model.clone(),
        directory: target.profile.directory.clone(),
        preparing,
        activated: activate,
        context: Some(target.profile.settings.ctx),
        identity: Some(key.clone()),
        cached: saved.is_some(),
        ..Job::default()
    };
    gate.changing = true;
    drop(gate);
    drop(loaded);
    let worker = Arc::clone(server);
    let cleanup = work.clone();
    if let Err(error) = std::thread::Builder::new()
        .name("nextora-tuning-supervisor".into())
        .spawn(move || {
            let _admin = worker.administration.lock().unwrap();
            run(&worker, restore, target, key, work, saved);
        })
    {
        server.admission.state.lock().unwrap().changing = false;
        server.admission.changed.notify_all();
        let mut job = server.tuning.state.lock().unwrap();
        job.phase = "failed";
        job.error = Some(error.to_string());
        let _ = fs::remove_dir_all(cleanup);
        return Err(error.into());
    }
    Ok((202, server.tuning.state.lock().unwrap().snapshot()))
}

struct Work(PathBuf);
impl Drop for Work {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
struct Worker(Child);
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn measure(server: &Server, work: &Path) -> Result<Value> {
    let mut command = Command::new(std::env::current_exe()?);
    command
        .arg("tune-worker")
        .arg("--request")
        .arg(work.join("request.json"))
        .arg("--progress")
        .arg(work.join("progress.json"))
        .arg("--result")
        .arg(work.join("result.json"))
        .arg("--watch-parent-stdin")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    let mut child = Worker(command.spawn().context("launch separate tuning worker")?);
    {
        let mut job = server.tuning.state.lock().unwrap();
        job.phase = "running";
        job.worker_pid = Some(child.0.id());
    }
    let began = Instant::now();
    loop {
        ensure!(
            !server.tuning.cancel.load(Ordering::Acquire),
            "Tuning cancelled"
        );
        ensure!(
            began.elapsed() < JOB_TIMEOUT,
            "tuning exceeded its 30 minute limit"
        );
        if let Ok(progress) = read_json(&work.join("progress.json")) {
            let mut job = server.tuning.state.lock().unwrap();
            job.phase = if progress["stage"].as_str() == Some("fitting") {
                "fitting"
            } else {
                "running"
            };
            job.progress = Some(progress);
        }
        if let Some(exit) = child.0.try_wait()? {
            ensure!(
                exit.success(),
                "tuning worker exited unsuccessfully ({exit})"
            );
            break;
        }
        std::thread::sleep(Duration::from_millis(150));
    }
    let mut result = read_json(&work.join("result.json"))?;
    result["worker_pid"] = json!(child.0.id());
    ensure!(
        result["machine"] == server.tuning.machine
            && matches!(result["schema_version"].as_u64(), Some(1 | 2)),
        "worker returned an incompatible report"
    );
    Ok(result)
}

fn recommended(
    server: &Server,
    original: &ServiceProfile,
    report: &Value,
    preparing: bool,
) -> Result<ServiceProfile> {
    ensure!(
        preparing || report["context"] == original.settings.ctx,
        "worker changed the required context"
    );
    let winner = &report["winner"];
    ensure!(
        winner.is_object(),
        "no benchmark candidate completed successfully"
    );
    let mut settings: ServiceSettings = serde_json::from_value(winner["settings"].clone())?;
    settings.max_tokens = original
        .settings
        .max_tokens
        .min(settings.ctx.saturating_sub(1));
    settings.max_queued = original.settings.max_queued;
    settings.queue_timeout_secs = original.settings.queue_timeout_secs;
    settings.expert_warmup = original.settings.expert_warmup;
    settings.validate(&server.args)?;
    ensure!(
        preparing
            || (settings.ctx == original.settings.ctx
                && settings.max_tokens == original.settings.max_tokens),
        "tuning cannot change the context or output token budget"
    );
    ensure!(
        settings.parallel == 1,
        "tuning winner must use one active request"
    );
    ensure!(
        winner["decode_tps"]
            .as_f64()
            .is_some_and(|n| n.is_finite() && n > 0.0),
        "invalid benchmark rate"
    );
    let mut profile = original.clone();
    profile.settings = settings;
    Ok(profile)
}

fn load_profile(
    server: &Server,
    profile: &ServiceProfile,
    path: &Path,
    persist: bool,
) -> Result<()> {
    let args = profile.settings.args(&server.args)?;
    let pool = super::worker_pool(profile.settings.threads.or(server.threads))?;
    let staged = if persist {
        server
            .args
            .settings_file
            .as_deref()
            .map(|file| control::StagedProfile::new(file, profile))
            .transpose()?
    } else {
        None
    };
    server.load_progress.lock().unwrap().begin();
    let loaded = ServedModel::load(
        profile.model.clone().context("tuning model missing")?,
        path,
        &args,
        |step| {
            server.load_progress.lock().unwrap().update(step);
        },
    )?;
    if let Some(staged) = staged {
        staged.commit()?;
    }
    *server.loaded.write().unwrap() = Some(loaded);
    let directory_changed = server.profile.read().unwrap().directory != profile.directory;
    {
        let mut catalogue = server.catalogue.write().unwrap();
        if directory_changed {
            catalogue.clear();
        }
        if let Some(id) = &profile.model {
            catalogue.insert(id.clone(), path.to_path_buf());
        }
    }
    *server.profile.write().unwrap() = profile.clone();
    *server.pool.write().unwrap() = pool;
    *server.startup_error.write().unwrap() = None;
    *server.desired_model.write().unwrap() = None;
    let mut gate = server.admission.state.lock().unwrap();
    gate.capacity = profile.settings.parallel;
    gate.queue_limit = profile.settings.max_queued;
    gate.queued_prompt_token_limit = profile.settings.queue_token_budget(&server.args);
    gate.timeout = Duration::from_secs(profile.settings.queue_timeout_secs);
    gate.high_water = 0;
    server
        .load_progress
        .lock()
        .unwrap()
        .finish("ready", "Model ready");
    Ok(())
}

fn restore(server: &Server, snapshot: &RestoreSnapshot) -> Result<()> {
    drop(server.loaded.write().unwrap().take());
    if let Some(path) = &snapshot.path {
        load_profile(server, &snapshot.profile, path, false)?;
    } else {
        *server.profile.write().unwrap() = snapshot.profile.clone();
    }
    *server.catalogue.write().unwrap() = snapshot.catalogue.clone();
    *server.desired_model.write().unwrap() = snapshot.desired.clone();
    *server.startup_error.write().unwrap() = snapshot.startup_error.clone();
    Ok(())
}
fn begin_commit(server: &Server) -> Result<()> {
    let mut job = server.tuning.state.lock().unwrap();
    ensure!(
        !server.tuning.cancel.load(Ordering::Acquire),
        "Tuning cancelled"
    );
    job.phase = "saving";
    Ok(())
}
fn run(
    server: &Server,
    original: RestoreSnapshot,
    target: StudyTarget,
    key: String,
    work: PathBuf,
    saved: Option<Value>,
) {
    let work = Work(work);
    drop(server.loaded.write().unwrap().take());
    let result = (|| -> Result<Value> {
        let report = match saved {
            Some(report) => report,
            None => measure(server, &work.0)?,
        };
        ensure!(
            !server.tuning.cancel.load(Ordering::Acquire),
            "Tuning cancelled"
        );
        {
            let mut job = server.tuning.state.lock().unwrap();
            job.result = Some(report.clone());
            job.context = report["context"]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok());
        }
        let profile = recommended(server, &target.profile, &report, target.preparing)?;
        begin_commit(server)?;
        if target.preparing && !target.using_saved {
            server.tuning.state.lock().unwrap().phase = "saving";
            write_json(
                &store_root(server)?.join(format!("profile-{}.json", target.model_identity)),
                &json!({"schema_version":2,"identity":key,"model_identity":target.model_identity,
                    "engine":"native","profile":profile,"result":report}),
            )?;
        }
        if !target.preparing {
            write_json(
                &store_root(server)?.join(format!("{key}.json")),
                &json!({"schema_version":1,"identity":key,"model":profile.model,"context":profile.settings.ctx,"result":report}),
            )?;
            let effective = identity_path(server, &target.path, &profile.settings)?;
            if effective != key {
                write_json(
                    &store_root(server)?.join(format!("{effective}.json")),
                    &json!({"schema_version":1,"identity":effective,"model":profile.model,"context":profile.settings.ctx,"result":report}),
                )?;
            }
        }
        if target.activate {
            server.tuning.state.lock().unwrap().phase = "applying";
            load_profile(server, &profile, &target.path, true)?;
        } else {
            server.tuning.state.lock().unwrap().phase = "restoring";
            restore(server, &original)?;
        }
        Ok(report)
    })();
    match result {
        Ok(report) => {
            let mut job = server.tuning.state.lock().unwrap();
            job.phase = "complete";
            job.result = Some(report);
            job.error = None;
        }
        Err(error) => {
            let cancelled = server.tuning.cancel.load(Ordering::Acquire);
            {
                let mut job = server.tuning.state.lock().unwrap();
                job.phase = "restoring";
                job.worker_pid = None;
                job.error = Some(format!("{error:#}"));
            }
            let restored = restore(server, &original);
            let mut job = server.tuning.state.lock().unwrap();
            job.phase = if cancelled && restored.is_ok() {
                "cancelled"
            } else {
                "failed"
            };
            if let Err(restore) = restored {
                let message =
                    format!("{error:#}; previous service could not be restored: {restore:#}");
                job.error = Some(message.clone());
                let mut visible = original.profile.clone();
                visible.model = None;
                *server.profile.write().unwrap() = visible;
                *server.desired_model.write().unwrap() = original.profile.model.clone();
                *server.startup_error.write().unwrap() = Some(message.clone());
                server
                    .load_progress
                    .lock()
                    .unwrap()
                    .finish("failed", message);
            }
        }
    }
    server.admission.state.lock().unwrap().changing = false;
    server.admission.changed.notify_all();
}

pub(super) fn handle(
    server: &Arc<Server>,
    stream: &mut TcpStream,
    request: &Request,
) -> Result<()> {
    let result = (|| -> Result<(u16, Value)> {
        match (request.method.as_str(), request.path.as_str()) {
            ("GET", "/api/tuning/status") => Ok((200, status(server)?)),
            ("POST", "/api/tuning/start") => {
                let input: Start = serde_json::from_slice(&request.body)?;
                start(server, input, false)
            }
            ("POST", "/api/tuning/profile") => {
                let target: Target = serde_json::from_slice(&request.body)?;
                Ok((200, prepared_status(server, &target)?))
            }
            ("POST", "/api/tuning/use") => {
                let target: Target = serde_json::from_slice(&request.body)?;
                start(
                    server,
                    Start {
                        target: Some(target),
                        ..Start::default()
                    },
                    true,
                )
            }
            ("POST", "/api/tuning/cancel") => {
                let mut job = server.tuning.state.lock().unwrap();
                if matches!(job.phase, "starting" | "fitting" | "running") {
                    server.tuning.cancel.store(true, Ordering::Release);
                    job.progress = Some(
                        json!({"stage":"cancelling","message":"Stopping the worker and restoring your model"}),
                    );
                }
                Ok((200, job.snapshot()))
            }
            _ => Ok((404, error_value("unknown tuning route"))),
        }
    })();
    match result {
        Ok((code, value)) => json_response(stream, code, &value),
        Err(error) => json_response(stream, 400, &error_value(&format!("{error:#}"))),
    }
}

#[cfg(test)]
mod setup_tests {
    use super::*;
    #[test]
    fn dynamic_heap_occupancy_does_not_invalidate_saved_profiles() {
        let a = json!({"adapters":[{"device_id":3,"driver_version":"a","largest_usable_device_local_heap_bytes":16000,
            "heap_budget_bytes":14000,"heap_usage_bytes":100,"available_bytes":13900,"compatible":true}]});
        let mut b = a.clone();
        b["adapters"][0]["heap_budget_bytes"] = json!(13000);
        b["adapters"][0]["heap_usage_bytes"] = json!(3000);
        b["adapters"][0]["available_bytes"] = json!(10000);
        assert_eq!(
            hash(&stable_hardware(a.clone())),
            hash(&stable_hardware(b.clone()))
        );
        b["adapters"][0]["driver_version"] = json!("b");
        assert_ne!(hash(&stable_hardware(a)), hash(&stable_hardware(b)));
    }
}
