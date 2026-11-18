//! The small HTTP/1.1 subset `rustingstrata serve` speaks (std::net, one request per connection) and its chat request.

use crate::chat::{Json, Msg, Template, tool_def};
use crate::tokenizer::Tokenizer;
use anyhow::{Context, bail};
use serde_json::Value;
use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};

pub struct Request {
    pub method: String,
    pub path: String,
    pub host: String,
    pub auth: String,
    pub json: bool,
    pub body: Vec<u8>,
}

/// Read-only token previews carry the same customer-key permission as generation; management stays admin-only.
pub fn customer_route(route: &str) -> bool {
    matches!(route, "/v1/chat/completions" | "/v1/messages" | "/v1/messages/count_tokens" | "/api/context" | "/v1/models" | "/v1/status")
}

/// Reads one request (Content-Length bodies only, no chunked uploads). Headers are capped at 64 KiB and the body at
/// 8 MiB.
pub fn read_request(r: &mut impl BufRead) -> anyhow::Result<Request> {
    let mut h = r.take(64 << 10);
    let mut line = String::new();
    h.read_line(&mut line)?;
    let mut it = line.split_whitespace();
    let (method, path) = (it.next().context("empty request")?.to_string(), it.next().context("no path")?.to_string());
    let (mut len, mut host, mut auth, mut json) = (0, String::new(), String::new(), false);
    loop {
        line.clear();
        if h.read_line(&mut line)? == 0 { bail!("headers cut off or too long"); }
        let l = line.trim_end();
        if l.is_empty() { break; }
        let Some((k, v)) = l.split_once(':') else { continue };
        let (k, v) = (k.to_ascii_lowercase(), v.trim());
        match k.as_str() {
            "content-length" => len = v.parse().context("bad Content-Length")?,
            "host" => host = v.to_string(),
            "authorization" => auth = v.to_string(),
            // The Anthropic API's key header (Claude Code with ANTHROPIC_API_KEY).
            "x-api-key" if auth.is_empty() => auth = format!("Bearer {v}"),
            "content-type" => json = v.starts_with("application/json"),
            _ => {}
        }
    }
    if len > 8 << 20 { bail!("body too large"); }
    let mut body = vec![0; len];
    r.read_exact(&mut body)?;
    Ok(Request { method, path, host, auth, json, body })
}

/// Host header check against DNS rebinding: a loopback name, the host the server was bound to, or a name the user
/// explicitly allowed (`--allowed-hosts` / `STRATA_ALLOWED_HOSTS`).
pub fn host_ok(host: &str, bound: &str, allowed: &[String]) -> bool {
    let name = |s: &str| {
        let s = s.trim();
        s.rsplit_once(':').filter(|(_, p)| p.parse::<u16>().is_ok()).map_or(s, |x| x.0).trim_end_matches('.').to_ascii_lowercase()
    };
    let n = name(host);
    !n.is_empty() && (matches!(n.as_str(), "localhost" | "127.0.0.1" | "[::1]") || n == name(bound) || allowed.iter().any(|a| name(a) == n))
}

/// Whether the client closed the connection (it sends nothing more after its request).
pub fn gone(c: &std::net::TcpStream) -> bool {
    if c.set_nonblocking(true).is_err() { return false; }
    let r = c.peek(&mut [0]);
    let _ = c.set_nonblocking(false);
    match r { Ok(n) => n == 0, Err(e) => e.kind() != std::io::ErrorKind::WouldBlock }
}

/// `Authorization` header check for `--api-key`, without an early exit on the first differing byte.
pub fn key_ok(auth: &str, key: &str) -> bool {
    if key.is_empty() { return false; }
    let a = auth.strip_prefix("Bearer ").unwrap_or("").as_bytes();
    a.len() == key.len() && a.iter().zip(key.as_bytes()).fold(0, |x, (p, q)| x | (p ^ q)) == 0
}

/// The requests each customer key has open (`--max-per-key`), by key id.
pub type Open = std::sync::Arc<std::sync::Mutex<std::collections::HashMap<u64, usize>>>;

/// One request's place in its key's count of open requests; dropping it frees the place.
pub struct Hold(Open, u64);

impl Hold {
    /// A place for key `id`, or None when it already has `cap` requests open.
    pub fn take(open: &Open, id: u64, cap: usize) -> Option<Hold> {
        let mut m = open.lock().unwrap();
        let n = m.entry(id).or_default();
        if *n >= cap { return None; }
        *n += 1;
        Some(Hold(open.clone(), id))
    }
}

impl Drop for Hold {
    fn drop(&mut self) {
        let mut m = self.0.lock().unwrap();
        if let Some(n) = m.get_mut(&self.1) { *n -= 1; if *n == 0 { m.remove(&self.1); } }
    }
}

/// The API keys: `--api-key` (all routes) and the `--api-keys` file (chat and model list only). No key at all
/// leaves the server open, as for local use.
#[derive(Clone, Default)]
pub struct Keys {
    pub admin: Option<String>,
    /// (name, key), the name as the file gives it.
    pub users: Vec<(String, String)>,
}

/// Who sent a request: the key's name for the usage log, a number that keeps each key's cached conversations
/// apart (0 for the admin key or an open server), and whether it may use the admin routes.
#[derive(Clone, Debug, PartialEq)]
pub struct Who { pub name: String, pub id: u64, pub admin: bool }

impl Keys {
    /// Reads a key file: one `name key` pair a line, `#` starts a comment.
    pub fn parse(text: &str) -> anyhow::Result<Vec<(String, String)>> {
        let mut v: Vec<(String, String)> = vec![];
        for (n, line) in text.lines().enumerate() {
            let l = line.split('#').next().unwrap_or("").trim();
            if l.is_empty() { continue; }
            let mut it = l.split_whitespace();
            let (Some(name), Some(key), None) = (it.next(), it.next(), it.next()) else { bail!("line {}: want `name key`", n + 1) };
            if key.len() < 16 { bail!("line {}: key of {name} is shorter than 16 characters", n + 1); }
            if v.iter().any(|(a, k)| a == name || k == key) { bail!("line {}: name or key of {name} repeats", n + 1); }
            v.push((name.into(), key.into()));
        }
        Ok(v)
    }

    pub fn any(&self) -> bool { self.admin.is_some() || !self.users.is_empty() }

    /// The sender of `auth` (an `Authorization` header), or None for a missing or wrong key.
    pub fn who(&self, auth: &str) -> Option<Who> {
        if !self.any() { return Some(Who { name: String::new(), id: 0, admin: true }); }
        if self.admin.as_deref().is_some_and(|k| key_ok(auth, k)) { return Some(Who { name: "admin".into(), id: 0, admin: true }); }
        // The id comes from the key, not its line, so a reloaded file with other lines gives no key another's id.
        self.users.iter().find(|(_, k)| key_ok(auth, k)).map(|(n, k)| {
            use std::hash::{Hash, Hasher};
            let mut h = std::hash::DefaultHasher::new();
            k.hash(&mut h);
            Who { name: n.clone(), id: h.finish().max(1), admin: false }
        })
    }
}

/// GGUF models below `dir`, grouped by complete shards and identified by stable
/// relative paths with `/` separators. Namespaces from different Hugging Face
/// repositories/revisions remain distinct. Symlinks are not traversed or listed;
/// every listed file must still canonicalize inside the model root.
///
/// Traversal is bounded to 16 directory levels so an accidentally recursive or
/// unrelated directory tree does not make the model picker unbounded.
pub fn list_models(dir: &Path) -> Vec<(String, PathBuf, u64)> {
    let Ok(root) = dir.canonicalize() else { return vec![] };
    let (mut files, mut dirs) = (Vec::new(), vec![(root.clone(), 0)]);
    while let Some((d, depth)) = dirs.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for entry in rd.flatten() {
            let Ok(kind) = entry.file_type() else { continue };
            if kind.is_symlink() { continue; }
            let p = entry.path();
            let Ok(real) = p.canonicalize() else { continue };
            if !real.starts_with(&root) { continue; }
            if kind.is_dir() {
                if depth < 16 { dirs.push((real, depth + 1)); }
                continue;
            }
            if !kind.is_file() { continue; }
            let Ok(relative) = real.strip_prefix(&root) else { continue };
            let Some(id) = relative.to_str() else { continue };
            let id = id.replace('\\', "/");
            if id.split('/').any(|part| part.is_empty() || part == "." || part == ".." || part.chars().any(char::is_control)) { continue; }
            files.push((id, std::fs::metadata(&real).map_or(0, |m| m.len())));
        }
    }
    let mut out: Vec<_> = models(&files).into_iter().map(|(id, n)| (id.clone(), root.join(id), n)).collect();
    out.sort();
    out
}

/// Whether an error is a GPU failure no later request can recover from: a lost Vulkan device, or a CUDA error that
/// leaves the context unusable (the CUDA documentation: "the process must be terminated and relaunched"). It looks
/// only at the backend's typed error in the chain, never at message text: an error message can quote request data.
pub fn gpu_dead(err: &anyhow::Error) -> bool {
    use cudarc::driver::sys::CUresult as C;
    err.chain().any(|e| e.is::<GpuLost>() || e.downcast_ref::<ash::vk::Result>() == Some(&ash::vk::Result::ERROR_DEVICE_LOST)
        || e.downcast_ref::<cudarc::driver::DriverError>().is_some_and(|d| matches!(d.0, C::CUDA_ERROR_ILLEGAL_ADDRESS
            | C::CUDA_ERROR_LAUNCH_FAILED | C::CUDA_ERROR_ECC_UNCORRECTABLE | C::CUDA_ERROR_HARDWARE_STACK_ERROR
            | C::CUDA_ERROR_ILLEGAL_INSTRUCTION | C::CUDA_ERROR_MISALIGNED_ADDRESS | C::CUDA_ERROR_INVALID_ADDRESS_SPACE
            | C::CUDA_ERROR_INVALID_PC | C::CUDA_ERROR_CONTEXT_IS_DESTROYED)))
}

/// Marks a copy of an error that was `gpu_dead` (`copy_err`).
#[derive(Debug)]
pub struct GpuLost;
impl std::fmt::Display for GpuLost { fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("GPU device lost") } }
impl std::error::Error for GpuLost {}

/// A copy of `err` for each of several waiting requests (`anyhow::Error` is not `Clone`): its text, and still `gpu_dead`.
pub fn copy_err(err: &anyhow::Error) -> anyhow::Error {
    let text = format!("{err:#}");
    if gpu_dead(err) { anyhow::Error::new(GpuLost).context(text) } else { anyhow::anyhow!(text) }
}

/// After `gpu_dead` the server exits with status 1, so a service manager restarts it (`Restart=on-failure`) instead of
/// every later request failing on a server that looks healthy.
pub fn exit_if_gpu_dead(err: &anyhow::Error) {
    if gpu_dead(err) {
        eprintln!("the GPU device is lost and this process cannot use it again: exiting with status 1 (restart the server)");
        std::process::exit(1);
    }
}

/// A projector filename (`mmproj*`, `*-mmproj.gguf`), matching `vision::is_mmproj` without reading the header.
fn is_mmproj_name(name: &str) -> bool { name.starts_with("mmproj") || name.contains("-mmproj") || name.contains("mmproj-") }

/// The models among files given as (name, bytes): .gguf files, complete split models once by their first shard
/// with the bytes of their shards, vision projectors (`mmproj*`, `*-mmproj.gguf`) left out.
pub fn models(files: &[(String, u64)]) -> Vec<(String, u64)> {
    let sizes: std::collections::HashMap<&str, u64> = files.iter().map(|(name, bytes)| (name.as_str(), *bytes)).collect();
    files.iter().filter(|(n, _)| n.ends_with(".gguf") && !is_mmproj_name(n.rsplit('/').next().unwrap_or(n)))
        .filter_map(|(n, b)| match shard_parts(n) {
            Some((stem, 1, total)) if total > 0 && total as usize <= files.len() => {
                (1..=total).try_fold(0u64, |bytes, i| {
                    sizes.get(format!("{stem}-{i:05}-of-{total:05}.gguf").as_str())
                        .and_then(|part| bytes.checked_add(*part))
                }).map(|bytes| (n.clone(), bytes))
            }
            Some(_) => None,
            None => Some((n.clone(), *b)),
        }).collect()
}

/// "dir/name-00002-of-00003.gguf" gives ("dir/name", 2).
pub fn shard(n: &str) -> Option<(&str, u32)> {
    shard_parts(n).map(|(stem, i, _)| (stem, i))
}

fn shard_parts(n: &str) -> Option<(&str, u32, u32)> {
    let (head, total) = n.strip_suffix(".gguf")?.rsplit_once("-of-")?;
    let (stem, i) = head.rsplit_once('-')?;
    let digits = |s: &str| s.len() == 5 && s.bytes().all(|b| b.is_ascii_digit());
    if !digits(total) || !digits(i) { return None; }
    Some((stem, i.parse().ok()?, total.parse().ok()?))
}

/// The parts of an OpenAI chat completion request that strata uses. Sampling fields left out fall back to the
/// server's flags (greedy by default).
pub struct ChatRequest {
    pub messages: Vec<Msg>,
    /// Tool definitions as the template prints them (`chat::tool_def`).
    pub tools: Vec<Json>,
    /// Whether to look for automatically selected function calls in the reply.
    pub tool_calls: bool,
    pub max_tokens: Option<usize>,
    pub stream: bool,
    /// Send the OpenAI stream's separate final usage chunk when explicitly requested.
    pub include_usage: bool,
    pub think: Option<bool>,
    /// `chat_template_kwargs.preserve_thinking`; the server's `--preserve-thinking` when absent.
    pub preserve_thinking: Option<bool>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    /// The reply ends before the first of these strings (`stop`).
    pub stop: Vec<String>,
}

/// The same rendering is used for token previews and generation. In particular, a final assistant message is
/// an open continuation, not a completed turn followed by another assistant header.
pub struct ChatPrompt {
    pub prompt: String,
    pub history: String,
    pub system: String,
    pub images: Vec<String>,
}

pub fn render_chat(tpl: Template, req: &ChatRequest, think_default: bool, preserve_default: bool) -> ChatPrompt {
    let think = req.think.unwrap_or(think_default);
    let preserve = req.preserve_thinking.unwrap_or(preserve_default);
    let open = tpl.render_open(&req.messages, &req.tools, think, preserve);
    let n_hist = req.messages.len() - open.is_some() as usize;
    let n_sys = req.messages.iter().take_while(|m| m.role == "system").count();
    ChatPrompt {
        prompt: open.unwrap_or_else(|| tpl.render(&req.messages, &req.tools, think, preserve, true)),
        history: tpl.render(&req.messages[..n_hist], &req.tools, think, preserve, false),
        system: tpl.render(&req.messages[..n_sys], &req.tools, think, preserve, false),
        images: req.messages.iter().flat_map(|m| m.images.iter().cloned()).collect(),
    }
}

/// Tokenize once, then use these exact ids both to check the budget and to prefill. The two prefix boundaries
/// are conservative: only a complete, unchanged token prefix can reuse recurrent and attention state.
pub struct ChatTokens {
    pub ids: Vec<u32>,
    pub keep: usize,
    pub sys: usize,
    pub system_tokens: usize,
}

impl ChatTokens {
    pub fn new(tok: &Tokenizer, prompt: &str, history: &str, system: &str) -> anyhow::Result<Self> {
        let ids = tok.encode(prompt)?;
        let h = tok.encode(history)?;
        // Cached prefill needs one final token to produce logits, even if a template's open header is empty.
        let keep = if h.len() < ids.len() && ids.starts_with(&h) { h.len() } else { 0 };
        let s = if system.is_empty() { vec![] } else { tok.encode(system)? };
        let sys = if s.len() <= keep && ids.starts_with(&s) { s.len() } else { 0 };
        Ok(Self { ids, keep, sys, system_tokens: s.len() })
    }
}

/// A request's complete prompt and output reservation. Explicit output requests are never silently clamped;
/// an omitted output limit uses what the server cap and remaining context permit.
#[derive(Debug, serde::Serialize)]
pub struct ContextBudget {
    pub context_tokens: usize,
    pub prompt_tokens: usize,
    pub requested_output_tokens: Option<usize>,
    pub output_tokens: usize,
    pub server_max_tokens: Option<usize>,
    pub available_output_tokens: usize,
    pub fits: bool,
    pub error: Option<ContextError>,
}

#[derive(Debug, serde::Serialize)]
pub struct ContextError {
    pub message: String,
    pub r#type: &'static str,
    pub code: &'static str,
}

impl ContextBudget {
    pub fn new(context: usize, prompt: usize, requested: Option<usize>, server_cap: usize) -> Self {
        let available = context.saturating_sub(prompt);
        let output = requested.unwrap_or_else(|| available.min(server_cap));
        let failure = if prompt == 0 {
            Some(("empty_prompt", "the rendered prompt has no tokens".to_string()))
        } else if prompt >= context {
            Some(("context_length_exceeded", format!(
                "the rendered prompt uses {prompt} tokens but context is {context}; no output space remains. Shorten the prompt/history or reload with a supported larger --ctx.")))
        } else if output == 0 {
            Some(("invalid_output_limit", "output token limit must be at least 1".to_string()))
        } else if requested.is_some_and(|n| n > server_cap) {
            Some(("output_limit_exceeded", format!(
                "requested {output} output tokens exceeds this server's --max-tokens {server_cap}. The prompt uses {prompt} of {context} context tokens, leaving {available}. Lower max_tokens/max_completion_tokens or restart the server with a larger output cap.")))
        } else if output > available {
            Some(("context_length_exceeded", format!(
                "the prompt uses {prompt} of {context} context tokens, leaving {available} for output, but {output} were requested. Shorten the prompt/history, reduce max_tokens/max_completion_tokens to at most {available}, or reload with a supported larger --ctx.")))
        } else { None };
        Self { context_tokens: context, prompt_tokens: prompt, requested_output_tokens: requested,
            output_tokens: output, server_max_tokens: (server_cap != usize::MAX).then_some(server_cap),
            available_output_tokens: available, fits: failure.is_none(),
            error: failure.map(|(code, message)| ContextError { message, r#type: "invalid_request_error", code }) }
    }

    pub fn ensure_fits(&self) -> anyhow::Result<usize> {
        if let Some(e) = &self.error { bail!("{}", e.message); }
        Ok(self.output_tokens)
    }
}

/// Qwen's tool-response templates carry results in call order, without their OpenAI ids. Validate the ids
/// before rendering and reorder a complete result group so a fast tool cannot answer another tool's call.
fn chat_messages(msgs: &[Json]) -> anyhow::Result<Vec<Msg>> {
    let mut out: Vec<Msg> = vec![];
    let mut pending: Vec<(&str, Option<Msg>)> = vec![];
    let mut conversation = false;
    for m in msgs {
        let parsed = Msg::from_json(m)?;
        if parsed.role == "tool" {
            let id = m.get("tool_call_id").as_str().filter(|id| !id.is_empty())
                .context("tool message needs a nonempty tool_call_id")?;
            let (_, result) = pending.iter_mut().find(|(call, _)| *call == id)
                .with_context(|| format!("tool_call_id {id:?} does not match a pending assistant function call"))?;
            if result.is_some() { bail!("duplicate tool result for tool_call_id {id:?}"); }
            *result = Some(parsed);
            if pending.iter().all(|(_, result)| result.is_some()) {
                out.extend(pending.drain(..).map(|(_, result)| result.unwrap()));
            }
            continue;
        }
        if !pending.is_empty() { bail!("every assistant tool call needs a tool result before the next message"); }
        if parsed.role == "system" {
            if conversation { bail!("system/developer instructions must precede user, assistant and tool messages for these model templates"); }
            // Some supported templates only retain one or two leading system turns.
            if let Some(first) = out.first_mut() { first.content.push_str("\n\n"); first.content.push_str(&parsed.content); }
            else { out.push(parsed); }
            continue;
        }
        conversation = true;
        if let Json::Arr(calls) = m.get("tool_calls") {
            if calls.len() > 128 { bail!("an assistant turn supports at most 128 function calls"); }
            for call in calls {
                let id = call.get("id").as_str().filter(|id| !id.is_empty())
                    .context("assistant tool call needs a nonempty id")?;
                if pending.iter().any(|(old, _)| *old == id) { bail!("duplicate assistant tool call id {id:?}"); }
                pending.push((id, None));
            }
        }
        out.push(parsed);
    }
    if !pending.is_empty() { bail!("every assistant tool call needs a tool result before requesting another completion"); }
    Ok(out)
}

pub fn chat_request(body: &[u8]) -> anyhow::Result<ChatRequest> {
    let v: Value = serde_json::from_slice(body)?;
    if !v.is_object() { bail!("chat request must be a JSON object"); }
    if !v["model"].is_null() && !v["model"].is_string() { bail!("model must be a string"); }
    // Messages and tools are read again in key order: the template prints tools and call arguments as given.
    let j: Json = serde_json::from_slice(body)?;
    let Json::Arr(msgs) = j.get("messages") else { bail!("missing messages") };
    let messages = chat_messages(msgs)?;
    if messages.is_empty() { bail!("no messages"); }
    let mut tools = match j.get("tools") {
        Json::Null => vec![],
        Json::Arr(a) => {
            if a.len() > 128 { bail!("tools supports at most 128 functions"); }
            let mut names = std::collections::HashSet::new();
            a.iter().map(|tool| {
                if !matches!(tool.get("function").get("strict"), Json::Null | Json::Bool(false)) {
                    bail!("strict function schemas are not implemented; omit strict or set it to false");
                }
                let tool = tool_def(tool)?;
                let function = tool.get("function");
                let name = function.get("name").as_str().unwrap();
                if name.is_empty() || name.len() > 64 || !name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-')) {
                    bail!("function name must contain 1 to 64 ASCII letters, digits, underscores or hyphens");
                }
                if !matches!(function.get("description"), Json::Str(_)) { bail!("function description must be a string"); }
                if !matches!(function.get("parameters"), Json::Obj(_)) { bail!("function parameters must be a JSON schema object"); }
                if !names.insert(name.to_owned()) { bail!("duplicate function name {name:?}"); }
                Ok(tool)
            }).collect::<anyhow::Result<_>>()?
        }
        _ => bail!("tools must be an array"),
    };
    let tool_calls = match &v["tool_choice"] {
        Value::Null => true,
        Value::String(s) if ["auto", "none"].contains(&s.as_str()) => s != "none",
        Value::String(s) if s == "required" => bail!("tool_choice=required is not implemented; use auto or none (forced function generation needs constrained decoding)"),
        Value::Object(o) if o.get("type").and_then(Value::as_str) == Some("function") =>
            bail!("a forced named tool_choice is not implemented; use auto or none"),
        _ => bail!("tool_choice must be \"auto\", \"required\", \"none\" or a function"),
    };
    if !tool_calls { tools.clear(); }
    // Optional fields: absent or null means the default, anything else must be valid.
    let int = |k: &str, min: u64| match &v[k] {
        Value::Null => Ok(None),
        x => x.as_u64().filter(|&n| n >= min).map(Some).with_context(|| format!("{k} must be an integer >= {min}")),
    };
    let num = |k: &str, ok: fn(f64) -> bool, want: &str| match &v[k] {
        Value::Null => Ok(None),
        x => x.as_f64().filter(|&f| f.is_finite() && (f as f32).is_finite() && ok(f) && ok(f as f32 as f64))
            .map(|f| Some(f as f32)).with_context(|| format!("{k} must be {want}, representable as a finite 32-bit value")),
    };
    let boolean = |x: &Value, key: &str| match x {
        Value::Null => Ok(None),
        x => x.as_bool().map(Some).with_context(|| format!("{key} must be a boolean")),
    };
    let stream = boolean(&v["stream"], "stream")?.unwrap_or(false);
    if !v["stream_options"].is_null() && !v["stream_options"].is_object() { bail!("stream_options must be an object"); }
    if !v["stream_options"].is_null() && !stream { bail!("stream_options requires stream=true"); }
    if let Some(options) = v["stream_options"].as_object() {
        for key in options.keys() {
            if !["include_usage", "include_obfuscation"].contains(&key.as_str()) { bail!("stream_options.{key} is not implemented"); }
        }
    }
    let include_usage = boolean(&v["stream_options"]["include_usage"], "stream_options.include_usage")?.unwrap_or(false);
    if boolean(&v["stream_options"]["include_obfuscation"], "stream_options.include_obfuscation")? == Some(true) {
        bail!("stream_options.include_obfuscation=true is not implemented");
    }
    if boolean(&v["parallel_tool_calls"], "parallel_tool_calls")? == Some(false) {
        bail!("parallel_tool_calls=false is not implemented; omit it or allow automatic function calls (the model may emit several calls)");
    }
    if boolean(&v["logprobs"], "logprobs")? == Some(true) || int("top_logprobs", 0)?.is_some_and(|n| n != 0) {
        bail!("logprobs/top_logprobs are not implemented");
    }
    if !v["response_format"].is_null() && v["response_format"] != serde_json::json!({"type": "text"}) {
        bail!("response_format supports only type=text; constrained JSON/schema output is not implemented");
    }
    if !v["modalities"].is_null() && v["modalities"] != serde_json::json!(["text"]) { bail!("only text output modalities are implemented"); }
    if boolean(&v["store"], "store")? == Some(true) { bail!("store=true and retrieving stored completions are not implemented"); }
    for key in ["functions", "function_call", "audio", "prediction", "reasoning_effort", "verbosity", "web_search_options"] {
        if !v[key].is_null() { bail!("{key} is not implemented on this Chat Completions endpoint"); }
    }
    if !v["chat_template_kwargs"].is_null() && !v["chat_template_kwargs"].is_object() { bail!("chat_template_kwargs must be an object"); }
    if let Some(options) = v["chat_template_kwargs"].as_object() {
        for key in options.keys() {
            if !["enable_thinking", "preserve_thinking"].contains(&key.as_str()) { bail!("chat_template_kwargs.{key} is not implemented"); }
        }
    }
    // These settings change the output distribution. Silently accepting them
    // would misrepresent the requested model/sampling behaviour. Clients may
    // still send their conventional neutral defaults.
    for (key, neutral) in [("presence_penalty", 0.0), ("frequency_penalty", 0.0),
                          ("repetition_penalty", 1.0), ("min_p", 0.0), ("typical_p", 1.0)] {
        if !v[key].is_null() && v[key].as_f64() != Some(neutral) {
            bail!("{key} is not implemented; omit it or use its neutral value {neutral}");
        }
    }
    if !v["logit_bias"].is_null() && !v["logit_bias"].as_object().is_some_and(|map| map.is_empty()) {
        bail!("logit_bias is not implemented; omit it or use an empty object");
    }
    let (modern_limit, legacy_limit) = (int("max_completion_tokens", 1)?, int("max_tokens", 1)?);
    if matches!((modern_limit, legacy_limit), (Some(a), Some(b)) if a != b) {
        bail!("max_tokens and max_completion_tokens disagree; send one output limit or the same value for both");
    }
    let max_tokens = modern_limit.or(legacy_limit).map(|n| n as usize);
    if int("n", 1)?.is_some_and(|n| n != 1) { bail!("only n = 1 is supported"); }
    let stop = match &v["stop"] {
        Value::Null => Some(vec![]),
        Value::String(s) => Some(vec![s.clone()]),
        Value::Array(a) => a.iter().map(|s| s.as_str().map(str::to_owned)).collect(),
        _ => None,
    }.context("stop must be a string or an array of strings")?.into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>();
    // Hold checks every prefix of every stop string at each token, so both are bounded.
    if stop.len() > 16 || stop.iter().any(|s| s.len() > 256) { bail!("stop takes at most 16 strings of at most 256 bytes"); }
    let think = boolean(&v["chat_template_kwargs"]["enable_thinking"], "chat_template_kwargs.enable_thinking")?;
    Ok(ChatRequest { messages, tools, tool_calls, max_tokens, stream, include_usage, think,
        preserve_thinking: boolean(&v["chat_template_kwargs"]["preserve_thinking"], "chat_template_kwargs.preserve_thinking")?,
        temperature: num("temperature", |f| f >= 0.0, "a number >= 0")?, top_k: int("top_k", 0)?.map(|n| n as usize),
        top_p: num("top_p", |f| f > 0.0 && f <= 1.0, "a number in (0, 1]")?, seed: int("seed", 0)?, stop })
}

/// What the chat page's side panel shows about the server, kept by the request loop and answered by the reader
/// threads (`GET /api/stats`), so it is current while a reply is generated or a model loads.
#[derive(Clone, Default, serde::Serialize)]
pub struct Stats {
    /// The loaded model's name, or the one loading.
    pub model: Option<String>,
    pub loading: bool,
    /// "cuda", "vulkan" or "cpu", and the GPU's name.
    pub backend: String,
    pub device: String,
    pub ctx: usize,
    #[serde(skip)]
    pub ctx_train: Option<u64>,
    /// Requests decoded at once (`--parallel`).
    pub slots: usize,
    /// Effective MTP drafts per step; zero when disabled.
    pub mtp_drafts: usize,
    /// Optional approximate expert omissions, separately from routing misses.
    pub expert_tail: crate::expert_tail::Stats,
    /// Device memory in use and in total, measured after the load and after each request.
    pub vram: Option<(u64, u64)>,
    /// Host RAM cap in bytes (`total - --ram-margin`, or `--max-ram`); `None` if undetectable.
    pub ram_cap: Option<u64>,
    /// Owned expert payload and actual I/O/cache counters; not an OS reservation.
    pub expert_ram: Option<crate::expert_ram::ExpertRamStats>,
    /// MoE expert cache hits and misses, and MTP drafts tried and accepted, since the model loaded.
    pub experts: Option<(u64, u64)>,
    pub drafts: Option<(u64, u64)>,
    /// Replies being generated, and their tokens so far.
    pub busy: usize,
    pub live_tokens: usize,
    pub requests: u64,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub completion_tokens: u64,
    /// The latest requests, oldest first (at most 40).
    pub recent: std::collections::VecDeque<Last>,
}

/// One answered chat request.
#[derive(Clone, Default, serde::Serialize)]
pub struct Last {
    /// Unix time in seconds when it started.
    pub at: u64,
    pub model: String,
    /// Prompt tokens: the system part, reused from the previous request (cached, which includes the system part
    /// when it was kept) and in total; reply tokens.
    pub system_tokens: usize,
    pub cached_tokens: usize,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// Prefill (up to the first token) and decode times in milliseconds.
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub finish: String,
}

impl Stats {
    pub fn record(&mut self, l: Last) {
        self.requests += 1;
        self.prompt_tokens += l.prompt_tokens as u64;
        self.cached_tokens += l.cached_tokens as u64;
        self.completion_tokens += l.completion_tokens as u64;
        if self.recent.len() == 40 { self.recent.pop_front(); }
        self.recent.push_back(l);
    }

    /// The stats in Prometheus text format, with `waiting` requests open besides the scrape.
    pub fn prometheus(&self, waiting: usize, uptime_s: u64) -> String {
        let mut s = String::new();
        let mut m = |name: &str, kind: &str, help: &str, v: u64| {
            s += &format!("# HELP strata_{name} {help}\n# TYPE strata_{name} {kind}\nstrata_{name} {v}\n");
        };
        m("up", "gauge", "1 when a model is loaded and not loading.", (self.model.is_some() && !self.loading) as u64);
        m("slots", "gauge", "Requests generated at once (--parallel).", self.slots as u64);
        m("busy", "gauge", "Replies being generated.", self.busy as u64);
        m("waiting", "gauge", "Requests being read, generated or queued.", waiting as u64);
        m("uptime_seconds", "counter", "Seconds since the server started.", uptime_s);
        m("requests_total", "counter", "Chat requests answered since start.", self.requests);
        m("prompt_tokens_total", "counter", "Prompt tokens of answered requests.", self.prompt_tokens);
        m("cached_prompt_tokens_total", "counter", "Prompt tokens reused from earlier requests.", self.cached_tokens);
        m("completion_tokens_total", "counter", "Reply tokens of answered requests.", self.completion_tokens);
        if let Some((u, t)) = self.vram {
            m("vram_used_bytes", "gauge", "Device memory in use.", u);
            m("vram_total_bytes", "gauge", "Device memory in total.", t);
        }
        if let Some(c) = self.ram_cap {
            m("ram_cap_bytes", "gauge", "Host RAM cap (total minus margin, or --max-ram).", c);
        }
        if let Some((h, x)) = self.experts {
            m("expert_hits_total", "counter", "Routed experts found on the GPU since the model loaded.", h);
            m("expert_misses_total", "counter", "Routed experts absent from the GPU cache; approximate mode may omit some.", x);
            m("expert_tail_min_rank", "gauge", "One-based approximate tail threshold; zero disables omissions.", self.expert_tail.min_rank as u64);
            m("expert_tail_skipped_assignments_total", "counter", "Omitted token-expert assignments in approximate decode.", self.expert_tail.skipped_assignments);
            m("expert_tail_skipped_groups_total", "counter", "Omitted distinct expert groups per layer and decode/verify pass.", self.expert_tail.skipped_expert_groups);
        }
        s
    }

    /// The stats with no model loaded (`model` while it loads), keeping the totals and recent requests since start.
    pub fn unload(&mut self, model: Option<String>) {
        let recent = std::mem::take(&mut self.recent);
        *self = Stats { loading: model.is_some(), model, requests: self.requests, prompt_tokens: self.prompt_tokens,
            cached_tokens: self.cached_tokens, completion_tokens: self.completion_tokens, recent, ..Default::default() };
    }
}

/// This machine's memory and CPU use and this process's resident memory, as JSON for the chat page. CPU use is
/// over the time since the previous call. Linux and Windows; elsewhere only the thread count.
pub fn system() -> Value {
    static PREV: std::sync::Mutex<(u64, u64)> = std::sync::Mutex::new((0, 0));
    let mut v = serde_json::json!({"threads": std::thread::available_parallelism().map_or(0, |n| n.get())});
    // (busy, total) CPU time since boot, in any unit.
    let cpu_pct = |busy: u64, total: u64| -> Value {
        let mut p = PREV.lock().unwrap();
        let (db, dt) = (busy.saturating_sub(p.0), total.saturating_sub(p.1));
        *p = (busy, total);
        if dt > 0 { (100.0 * db as f64 / dt as f64).into() } else { Value::Null }
    };
    #[cfg(target_os = "linux")]
    {
        let read = |p: &str| std::fs::read_to_string(p).unwrap_or_default();
        let kb = |s: &str, k: &str| s.lines().find_map(|l| l.strip_prefix(k)?.trim().strip_suffix("kB")?.trim().parse::<u64>().ok()).map(|n| n << 10);
        if let Some(l) = read("/proc/stat").lines().next() {
            let n: Vec<u64> = l.split_whitespace().skip(1).filter_map(|x| x.parse().ok()).collect();
            if n.len() >= 5 { let total = n.iter().take(8).sum::<u64>(); v["cpu_pct"] = cpu_pct(total - n[3] - n[4], total); }
        }
        let m = read("/proc/meminfo");
        if let (Some(t), Some(a)) = (kb(&m, "MemTotal:"), kb(&m, "MemAvailable:")) { (v["ram_total"], v["ram_used"]) = (t.into(), (t - a).into()); }
        if let Some(r) = kb(&read("/proc/self/status"), "VmRSS:") { v["rss"] = r.into(); }
        if let Some(n) = read("/proc/cpuinfo").lines().find_map(|l| l.strip_prefix("model name")?.split_once(':').map(|x| x.1.trim().to_string())) { v["cpu"] = n.into(); }
    }
    #[cfg(windows)]
    {
        #[repr(C)]
        struct MemStatus { len: u32, load: u32, total: u64, avail: u64, rest: [u64; 5] }
        #[repr(C)]
        struct ProcMem { cb: u32, faults: u32, peak: usize, ws: usize, rest: [usize; 6] }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GlobalMemoryStatusEx(m: *mut MemStatus) -> i32;
            fn GetSystemTimes(idle: *mut u64, kernel: *mut u64, user: *mut u64) -> i32;
            fn GetCurrentProcess() -> isize;
            fn K32GetProcessMemoryInfo(p: isize, m: *mut ProcMem, cb: u32) -> i32;
        }
        let mut m = MemStatus { len: size_of::<MemStatus>() as u32, load: 0, total: 0, avail: 0, rest: [0; 5] };
        if unsafe { GlobalMemoryStatusEx(&mut m) } != 0 { (v["ram_total"], v["ram_used"]) = (m.total.into(), (m.total - m.avail).into()); }
        let (mut i, mut k, mut u) = (0, 0, 0);
        // Kernel time includes idle time.
        if unsafe { GetSystemTimes(&mut i, &mut k, &mut u) } != 0 { v["cpu_pct"] = cpu_pct(k + u - i, k + u); }
        let mut p = ProcMem { cb: size_of::<ProcMem>() as u32, faults: 0, peak: 0, ws: 0, rest: [0; 6] };
        if unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut p, size_of::<ProcMem>() as u32) } != 0 { v["rss"] = (p.ws as u64).into(); }
        if let Ok(n) = std::env::var("PROCESSOR_IDENTIFIER") { v["cpu"] = n.into(); }
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    let _ = cpu_pct;
    v
}

/// An Anthropic Messages API request (`POST /v1/messages`, what Claude Code sends) as the OpenAI chat request
/// `chat_request` reads: the system blocks become a system message, `tool_use` blocks tool calls, `tool_result`
/// blocks tool messages, and `thinking: {"type": "enabled"}` turns thinking on. Images and server tools are dropped.
pub fn messages_body(body: &[u8]) -> anyhow::Result<Vec<u8>> {
    let j: Json = serde_json::from_slice(body)?;
    let s = |x: &str| Json::Str(x.into());
    let obj = |v: Vec<(&str, Json)>| Json::Obj(v.into_iter().map(|(k, v)| (k.to_string(), v)).collect());
    // A string, or the text of an array of blocks.
    let text = |c: &Json| match c {
        Json::Str(t) => t.clone(),
        Json::Arr(a) => a.iter().filter(|b| b.get("type").as_str() == Some("text")).filter_map(|b| b.get("text").as_str()).collect::<Vec<_>>().join("\n"),
        _ => String::new(),
    };
    let mut msgs = vec![];
    let sys = text(j.get("system"));
    if !sys.is_empty() { msgs.push(obj(vec![("role", s("system")), ("content", Json::Str(sys))])); }
    let Json::Arr(ms) = j.get("messages") else { bail!("missing messages") };
    for m in ms {
        let role = m.get("role").as_str().context("message without role")?;
        let blocks = match m.get("content") { Json::Str(t) => vec![obj(vec![("type", s("text")), ("text", s(t))])], Json::Arr(a) => a.clone(), _ => bail!("bad content") };
        let (mut texts, mut calls, mut reasoning, mut images) = (vec![], vec![], String::new(), vec![]);
        for b in &blocks {
            match (role, b.get("type").as_str().unwrap_or("")) {
                (_, "text") => texts.push(b.get("text").as_str().unwrap_or("").to_string()),
                (_, "image") => {
                    let src = b.get("source");
                    let url = src.get("url").as_str().map(str::to_owned).or_else(|| {
                        (src.get("type").as_str() == Some("base64")).then(|| {
                            format!("data:{};base64,{}", src.get("media_type").as_str().unwrap_or("image/png"), src.get("data").as_str().unwrap_or(""))
                        })
                    });
                    if let Some(url) = url.filter(|u| !u.is_empty() && !u.ends_with(";base64,")) {
                        images.push(obj(vec![("type", s("image_url")), ("image_url", obj(vec![("url", Json::Str(url))]))]));
                    }
                }
                ("assistant", "thinking") => reasoning += b.get("thinking").as_str().unwrap_or(""),
                ("assistant", "tool_use") => calls.push(obj(vec![("id", b.get("id").clone()), ("type", s("function")),
                    ("function", obj(vec![("name", b.get("name").clone()), ("arguments", b.get("input").clone())]))])),
                ("user", "tool_result") => {
                    let mut c = text(b.get("content"));
                    if matches!(b.get("is_error"), Json::Bool(true)) { c = format!("Error: {c}"); }
                    msgs.push(obj(vec![("role", s("tool")), ("tool_call_id", b.get("tool_use_id").clone()), ("content", Json::Str(c))]));
                }
                _ => {}
            }
        }
        if role == "assistant" {
            let mut a = vec![("role", s("assistant")), ("content", Json::Str(texts.join("\n")))];
            if !reasoning.is_empty() { a.push(("reasoning_content", Json::Str(reasoning))); }
            if !calls.is_empty() { a.push(("tool_calls", Json::Arr(calls))); }
            msgs.push(obj(a));
        } else if !texts.is_empty() || !images.is_empty() {
            let mut content: Vec<Json> = texts.into_iter().map(|t| obj(vec![("type", s("text")), ("text", Json::Str(t))])).collect();
            content.extend(images);
            msgs.push(obj(vec![("role", s(role)), ("content", Json::Arr(content))]));
        }
    }
    let mut out = vec![("messages", Json::Arr(msgs))];
    if let Json::Arr(ts) = j.get("tools") {
        // Server tools (web search and the like) have a type other than "custom" and no input_schema.
        let ts = ts.iter().filter(|t| !matches!(t.get("input_schema"), Json::Null)).map(|t| obj(vec![("type", s("function")),
            ("function", obj(vec![("name", t.get("name").clone()), ("description", t.get("description").clone()), ("parameters", t.get("input_schema").clone())]))]));
        out.push(("tools", Json::Arr(ts.collect())));
    }
    match j.get("tool_choice").get("type").as_str() {
        Some("any") => out.push(("tool_choice", s("required"))),
        Some("none") => out.push(("tool_choice", s("none"))),
        Some("tool") => out.push(("tool_choice", obj(vec![("type", s("function")), ("function", obj(vec![("name", j.get("tool_choice").get("name").clone())]))]))),
        _ => {}
    }
    for (from, to) in [("max_tokens", "max_tokens"), ("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"), ("stop_sequences", "stop"), ("stream", "stream")] {
        if !matches!(j.get(from), Json::Null) { out.push((to, j.get(from).clone())); }
    }
    if j.get("thinking").get("type").as_str() == Some("enabled") { out.push(("chat_template_kwargs", obj(vec![("enable_thinking", Json::Bool(true))]))); }
    Ok(obj(out).dump().into_bytes())
}

#[cfg(test)]
mod context_tests {
    use super::*;

    #[test]
    fn explicit_output_reservation_fits_exactly_or_is_rejected() {
        let b = ContextBudget::new(32768, 24576, Some(8192), 8192);
        assert_eq!(b.ensure_fits().unwrap(), 8192);
        assert_eq!(b.available_output_tokens, 8192);
        let b = ContextBudget::new(32768, 24577, Some(8192), 8192);
        assert!(!b.fits);
        assert_eq!(b.output_tokens, 8192); // Report the request, never a silently reduced reply.
        assert_eq!(b.error.as_ref().unwrap().code, "context_length_exceeded");
        assert!(b.error.as_ref().unwrap().message.contains("8191"));
        let b = ContextBudget::new(32768, 1000, Some(8192), 4096);
        assert_eq!(b.error.as_ref().unwrap().code, "output_limit_exceeded");
        assert!(b.ensure_fits().is_err());
    }

    #[test]
    fn omitted_output_uses_remaining_context_and_server_cap() {
        assert_eq!(ContextBudget::new(32768, 30000, None, 8192).ensure_fits().unwrap(), 2768);
        assert_eq!(ContextBudget::new(32768, 1000, None, 8192).ensure_fits().unwrap(), 8192);
        assert_eq!(ContextBudget::new(32768, 1000, None, usize::MAX).ensure_fits().unwrap(), 31768);
        for prompt in [0, 32768, 32769, usize::MAX] {
            assert!(ContextBudget::new(32768, prompt, Some(1), 8192).ensure_fits().is_err());
        }
        assert!(ContextBudget::new(32768, 1000, Some(usize::MAX), usize::MAX).ensure_fits().is_err());
        assert!(ContextBudget::new(32768, 1000, Some(0), 8192).ensure_fits().is_err());
        assert!(ContextBudget::new(32768, 1000, None, 0).ensure_fits().is_err());
    }

    #[test]
    fn preview_and_generation_share_templates_tools_and_continuation_ids() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe.gguf");
        let gguf = crate::gguf::Gguf::open(&path).unwrap();
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        let body = serde_json::json!({
            "messages": [
                {"role": "system", "content": "Repository: src/main.rs. Keep exact indentation."},
                {"role": "user", "content": "Complete this Rust function."},
                {"role": "assistant", "content": "```rust\nfn answer() {\n    ", "reasoning_content": "Check the return type."}
            ],
            "tools": [{"type":"function", "function":{"name":"read_file", "parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}],
            "max_tokens": 256,
            "chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}
        }).to_string();
        let req = chat_request(body.as_bytes()).unwrap();
        for tpl in [Template::Xml, Template::Json { reasoning: true }] {
            let rendered = render_chat(tpl, &req, false, false);
            assert_eq!(rendered.prompt, tpl.render_open(&req.messages, &req.tools, true, true).unwrap());
            assert_eq!(rendered.history, tpl.render(&req.messages[..2], &req.tools, true, true, false));
            assert!(rendered.prompt.ends_with("fn answer() {\n    "));
            assert!(rendered.prompt.contains("read_file"));
            assert!(rendered.prompt.contains("Check the return type."));
            let checked = ChatTokens::new(&tok, &rendered.prompt, &rendered.history, &rendered.system).unwrap();
            let generated = tok.encode(&rendered.prompt).unwrap();
            assert_eq!(checked.ids, generated);
            assert!(checked.sys <= checked.keep && checked.keep < checked.ids.len());
            assert!(checked.ids.starts_with(&tok.encode(&rendered.history).unwrap()));
            assert_eq!(ContextBudget::new(4096, checked.ids.len(), req.max_tokens, 256).ensure_fits().unwrap(), 256);
        }
        let mut user_only = req;
        user_only.messages.pop();
        let rendered = render_chat(Template::Xml, &user_only, false, false);
        assert_eq!(rendered.prompt, Template::Xml.render(&user_only.messages, &user_only.tools, true, true, true));
    }

    #[test]
    fn open_code_prefix_keeps_whitespace_but_closed_xml_turn_uses_template_rules() {
        let msgs = vec![Msg::new("user", "write code"), Msg::new("assistant", "  value\n\t")];
        let closed = Template::Xml.render(&msgs, &[], false, true, false);
        assert!(closed.ends_with("value<|im_end|>\n"));
        let open = Template::Xml.render_open(&msgs, &[], false, true).unwrap();
        assert!(open.ends_with("  value\n\t"));
        let mut with_inline_thinking = msgs;
        with_inline_thinking[1].content = "<think>inspect</think>\n\n```rust\n    ".into();
        assert!(Template::Xml.render_open(&with_inline_thinking, &[], true, true).unwrap().ends_with("```rust\n    "));
    }

    #[test]
    fn customer_token_preview_does_not_grant_model_management() {
        for route in ["/api/context", "/v1/chat/completions", "/v1/messages/count_tokens", "/v1/status"] { assert!(customer_route(route)); }
        for route in ["/api/load", "/api/unload", "/api/download", "/api/stats", "/api/context/other"] { assert!(!customer_route(route)); }
    }

    #[test]
    fn coding_client_options_are_validated_before_generation() {
        let base = serde_json::json!({"messages":[{"role":"user","content":"read the code"}]});
        let request = |extra: Value| {
            let mut body = base.clone();
            body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            chat_request(body.to_string().as_bytes())
        };
        let good = request(serde_json::json!({"max_completion_tokens":7,"max_tokens":7,"stream":true,
            "stream_options":{"include_usage":true,"include_obfuscation":false},"parallel_tool_calls":true,
            "response_format":{"type":"text"},"logprobs":false,"top_logprobs":0,"store":false})).unwrap();
        assert_eq!(good.max_tokens, Some(7));
        assert!(good.stream && good.include_usage);
        assert!(!request(serde_json::json!({})).unwrap().include_usage);
        for bad in [
            serde_json::json!({"max_tokens":7,"max_completion_tokens":8}),
            serde_json::json!({"stream":"true"}),
            serde_json::json!({"stream_options":{"include_usage":true}}),
            serde_json::json!({"stream":true,"stream_options":{"include_usage":1}}),
            serde_json::json!({"stream":true,"stream_options":{"include_obfuscation":true}}),
            serde_json::json!({"parallel_tool_calls":false}),
            serde_json::json!({"tool_choice":"required"}),
            serde_json::json!({"tool_choice":{"type":"function","function":{"name":"read"}}}),
            serde_json::json!({"response_format":{"type":"json_object"}}),
            serde_json::json!({"logprobs":true}),
            serde_json::json!({"top_logprobs":1}),
            serde_json::json!({"store":true}),
            serde_json::json!({"reasoning_effort":"high"}),
            serde_json::json!({"modalities":["text","audio"]}),
            serde_json::json!({"chat_template_kwargs":{"enable_thinking":"yes"}}),
            serde_json::json!({"chat_template_kwargs":{"unsupported":true}}),
        ] { assert!(request(bad.clone()).is_err(), "silently accepted {bad}"); }
        let q = request(serde_json::json!({"tool_choice":"none","tools":[{"type":"function","function":{"name":"read"}}]})).unwrap();
        assert!(!q.tool_calls && q.tools.is_empty());
        assert!(request(serde_json::json!({"tools":[{"type":"function","function":{"name":"read","strict":true}}]})).is_err());
        assert!(request(serde_json::json!({"tools":[{"type":"function","function":{"name":"café"}}]})).is_err());
    }

    #[test]
    fn tool_history_uses_ids_and_retains_developer_instructions() {
        let body = serde_json::json!({"messages":[
            {"role":"system","content":"System rule."},
            {"role":"developer","content":"Keep formatting."},
            {"role":"user","content":"Read both files."},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"left","type":"function","function":{"name":"read","arguments":"{\"path\":\"a.rs\"}"}},
                {"id":"right","type":"function","function":{"name":"read","arguments":{"path":"b.rs"}}}
            ]},
            {"role":"tool","tool_call_id":"right","content":"right result"},
            {"role":"tool","tool_call_id":"left","content":"left result"}
        ]});
        let q = chat_request(body.to_string().as_bytes()).unwrap();
        assert_eq!(q.messages[0].content, "System rule.\n\nKeep formatting.");
        assert_eq!(q.messages[0].role, "system");
        assert_eq!(q.messages[3].content, "left result");
        assert_eq!(q.messages[4].content, "right result");
        let p = render_chat(Template::Xml, &q, false, true).prompt;
        assert!(p.contains("Keep formatting."));
        assert!(p.find("left result").unwrap() < p.find("right result").unwrap());

        for (index, key, value) in [(4, "tool_call_id", serde_json::json!("unknown")),
                                  (4, "tool_call_id", serde_json::json!("left")),
                                  (4, "tool_call_id", Value::Null)] {
            let mut bad = body.clone();
            bad["messages"][index][key] = value;
            assert!(chat_request(bad.to_string().as_bytes()).is_err());
        }
        let mut missing = body.clone();
        missing["messages"].as_array_mut().unwrap().pop();
        assert!(chat_request(missing.to_string().as_bytes()).is_err());
        let mut later = base_instruction_history();
        later["messages"].as_array_mut().unwrap().push(serde_json::json!({"role":"developer","content":"This would be lost by XML templates."}));
        assert!(chat_request(later.to_string().as_bytes()).is_err());
    }

    fn base_instruction_history() -> Value {
        serde_json::json!({"messages":[{"role":"system","content":"rule"},{"role":"user","content":"question"}]})
    }

    #[test]
    fn unsupported_message_parts_fail_instead_of_losing_input() {
        for part in [serde_json::json!({"type":"input_audio","input_audio":{"data":"AAAA","format":"wav"}}),
                     serde_json::json!({"type":"file","file":{"file_id":"file-1"}}),
                     serde_json::json!({"type":"text","text":12})] {
            let body = serde_json::json!({"messages":[{"role":"user","content":[part]}]});
            assert!(chat_request(body.to_string().as_bytes()).is_err());
        }
    }
}
