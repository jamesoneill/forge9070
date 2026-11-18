//! Hugging Face discovery/download and control of the existing inference server.
//! All model inference still runs in RustingStrata. No repository code is run.

use anyhow::{Context, Result, ensure};
use clap::{Args, Subcommand};
use rusting_strata::{hf, serve};
use serde_json::{Value, json};
use std::{path::{Path, PathBuf}, sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}}, time::Duration};

#[derive(Subcommand)]
pub enum ModelsCommand {
    /// Search Hugging Face GGUF repositories. Community variants are not filtered by their labels.
    Search {
        query: String,
        #[arg(long, default_value_t = 10)] limit: usize,
        #[arg(long)] json: bool,
    },
    /// List downloadable GGUF choices and resolve a branch/tag to an immutable revision.
    Files {
        repo: String,
        #[arg(long, default_value = "main")] revision: String,
        /// Verify only this exact GGUF choice and its shards; does not download weights.
        #[arg(long)] file: Option<String>,
        #[arg(long)] json: bool,
    },
    /// Download one GGUF choice and all its shards, then inspect its actual format.
    Pull {
        repo: String,
        #[arg(long)] file: String,
        #[arg(long, default_value = "main")] revision: String,
        #[arg(long)] models_dir: Option<PathBuf>,
        /// Also switch the running server to this downloaded model (same models-dir required).
        #[arg(long)] load: bool,
        #[command(flatten)] target: ServerTarget,
        #[arg(long)] json: bool,
    },
    /// List the local model catalog, including nested Hugging Face snapshots.
    List {
        #[arg(long)] models_dir: Option<PathBuf>,
        #[arg(long)] json: bool,
    },
    /// Switch the running server to an ID from models list or models status.
    Load {
        id: String,
        #[command(flatten)] target: ServerTarget,
        #[arg(long)] json: bool,
    },
    /// Unload the running model and release its resources without stopping the server.
    Unload {
        #[command(flatten)] target: ServerTarget,
        #[arg(long)] json: bool,
    },
    /// Show the running server's catalog and currently loaded model.
    Status {
        #[command(flatten)] target: ServerTarget,
        #[arg(long)] json: bool,
    },
}

#[derive(Args)]
pub struct ServerTarget {
    /// Server root URL (omit /v1). Authentication comes from STRATA_API_KEY.
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    pub server: String,
    /// Timeout in seconds; cold loading a large model may take several minutes.
    #[arg(long, default_value_t = 600)]
    pub timeout: u64,
}

pub fn default_models_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("FORGE9070_MODELS_DIR") { return PathBuf::from(path); }
    if cfg!(windows) {
        if let Some(path) = std::env::var_os("LOCALAPPDATA") {
            return PathBuf::from(path).join("forge9070").join("models");
        }
    }
    let data = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from)
        .unwrap_or_else(|| std::env::var_os("HOME").map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(".")).join(".local").join("share"));
    data.join("forge9070").join("models")
}

fn endpoint(base: &str, path: &str) -> Result<String> {
    let uri: http::Uri = base.parse().context("--server must be a valid http(s) root URL")?;
    ensure!(matches!(uri.scheme_str(), Some("http" | "https")), "--server requires http:// or https://");
    let authority = uri.authority().context("--server needs a host")?;
    ensure!(!authority.as_str().contains('@'), "Use STRATA_API_KEY for authentication, not credentials in --server");
    ensure!(uri.path() == "/" || uri.path().is_empty(), "--server is the server root URL; omit /v1 and other paths");
    ensure!(uri.query().is_none() && !base.contains('#'), "--server cannot contain a query or fragment");
    Ok(format!("{}{path}", base.trim_end_matches('/')))
}

fn request(target: &ServerTarget, path: &str, body: Option<Value>) -> Result<Value> {
    ensure!((1..=7200).contains(&target.timeout), "--timeout must be from 1 to 7200 seconds");
    let url = endpoint(&target.server, path)?;
    let mut config = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(10)))
        .timeout_global(Some(Duration::from_secs(target.timeout)))
        .http_status_as_error(false)
        .max_redirects(0);
    let uri: http::Uri = url.parse()?;
    let host = uri.host().unwrap_or("").trim_start_matches('[').trim_end_matches(']');
    if host.eq_ignore_ascii_case("localhost") || host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback()) {
        config = config.proxy(None);
    }
    let agent: ureq::Agent = config.build().into();
    let key = std::env::var("STRATA_API_KEY").ok().filter(|s| !s.is_empty());
    let mut response = if let Some(body) = body {
        let mut call = agent.post(&url).header("Content-Type", "application/json");
        if let Some(key) = &key { call = call.header("Authorization", format!("Bearer {key}")); }
        call.send(body.to_string())
    } else {
        let mut call = agent.get(&url);
        if let Some(key) = &key { call = call.header("Authorization", format!("Bearer {key}")); }
        call.call()
    }.with_context(|| format!("contact server at {}", target.server))?;
    let status = response.status().as_u16();
    let text = response.body_mut().read_to_string().context("read server response")?;
    let value: Value = serde_json::from_str(&text).context("server response was not JSON")?;
    ensure!((200..300).contains(&status), "server returned HTTP {status}: {}",
        value.pointer("/error/message").and_then(Value::as_str).unwrap_or(&text));
    Ok(value)
}

fn print_json(value: &impl serde::Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn relative_model_id(dir: &Path, path: &Path) -> Result<String> {
    let root = std::fs::canonicalize(dir).context("resolve models directory")?;
    let file = std::fs::canonicalize(path).context("resolve downloaded model")?;
    let relative = file.strip_prefix(root).context("downloaded model is outside models directory")?;
    Ok(relative.components().map(|c| c.as_os_str().to_string_lossy()).collect::<Vec<_>>().join("/"))
}

pub fn run(command: ModelsCommand) -> Result<()> {
    match command {
        ModelsCommand::Search { query, limit, json } => {
            ensure!((1..=10).contains(&limit), "--limit must be from 1 to 10 (bounded metadata verification)");
            let results = hf::search(&query, limit)?;
            if json { return print_json(&results); }
            for repo in &results { println!("{}", repo.id); }
            eprintln!("{} repository result(s) with a GGUF passing the MoE metadata policy. GPU capacity and numerical correctness still need a load/test.", results.len());
        }
        ModelsCommand::Files { repo, revision, file, json } => {
            let files = match file.as_deref() {
                Some(file) => hf::list_file_at(&repo, &revision, file)?,
                None => hf::list_at(&repo, &revision)?,
            };
            if json { return print_json(&files); }
            println!("Repository: {}\nResolved revision: {}", files.repo, files.commit);
            for model in &files.models {
                println!("{:.2} GiB  {}", model.size as f64 / 1_073_741_824.0, model.file);
            }
            eprintln!("{} incompatible or unverifiable choices excluded. Listed GGUF metadata passed the MoE policy; sizes include all shards.", files.excluded.len());
            if file.is_some() {
                for excluded in &files.excluded { eprintln!("Excluded {}: {}", excluded.file, excluded.reason); }
            }
        }
        ModelsCommand::Pull { repo, file, revision, models_dir, load, target, json } => {
            ensure!(!file.to_ascii_lowercase().contains("iq3_k"), "IQ3_K/IQ3_KS is not supported by this engine; choose another GGUF quantization");
            let dir = models_dir.unwrap_or_else(default_models_dir);
            let progress = Arc::new(Mutex::new(hf::Progress { repo: repo.clone(), file: file.clone(), running: true, ..Default::default() }));
            let done = Arc::new(AtomicBool::new(false));
            let reporter = if json { None } else {
                let (progress, done) = (Arc::clone(&progress), Arc::clone(&done));
                Some(std::thread::spawn(move || {
                    while !done.load(Ordering::Relaxed) {
                        if let Ok(p) = progress.lock() {
                            eprint!("\rDownloading: {:.2} / {:.2} GiB", p.done as f64 / 1_073_741_824.0, p.total as f64 / 1_073_741_824.0);
                        }
                        std::thread::sleep(Duration::from_millis(500));
                    }
                    eprintln!();
                }))
            };
            let downloaded = hf::download_at(&repo, &file, &revision, &dir, &progress);
            done.store(true, Ordering::Relaxed);
            if let Some(reporter) = reporter { let _ = reporter.join(); }
            let path = downloaded?;
            let id = relative_model_id(&dir, &path)?;
            let inspected = super::inspect::inspect(&path).and_then(|report| report.ensure_supported());
            if let Err(error) = inspected {
                if json { print_json(&json!({"path":path,"id":id,"downloaded":true,"compatible":false,"error":format!("{error:#}")}))?; }
                anyhow::bail!("download is complete at {}, but this build cannot load it: {error:#}", path.display());
            }
            let loaded = if load { Some(request(&target, "/api/load", Some(json!({"model":id})))?) } else { None };
            if json { print_json(&json!({"path":path,"id":id,"downloaded":true,"compatible":true,"server":loaded}))?; }
            else {
                println!("Downloaded: {}\nModel ID: {id}\nFormat preflight: accepted", path.display());
                if load { println!("Loaded by {}", target.server); }
                else { println!("Select it in the chat page, or run:\nforge9070 models load '{id}'"); }
            }
        }
        ModelsCommand::List { models_dir, json } => {
            let dir = models_dir.unwrap_or_else(default_models_dir);
            let found = serve::list_models(&dir);
            let count = found.len();
            let catalog: Vec<_> = found.into_iter().filter_map(|(id, path, bytes)| {
                let report = super::inspect::inspect(&path).ok()?;
                report.ensure_supported().ok()?;
                Some((id, path, bytes, report))
            }).collect();
            if json {
                return print_json(&json!({"dir":dir,"policy":"moe_only","excluded_count":count-catalog.len(),"models":catalog.iter().map(|(id,path,bytes,report)|json!({"id":id,"path":path,"bytes":bytes,"compatibility":report.compatibility})).collect::<Vec<_>>()}));
            }
            println!("Model directory: {}", dir.display());
            for (id, _, bytes, report) in &catalog { println!("{:.2} GiB  {id}  [{}; {} of {} experts active]", *bytes as f64 / 1_073_741_824.0, report.architecture, report.active_experts_per_token, report.experts_per_layer); }
            if catalog.is_empty() { println!("No compatible MoE GGUF choices found. Use `models search`, then `models pull`."); }
            if count > catalog.len() { eprintln!("{} incompatible or unreadable local choices excluded.", count-catalog.len()); }
        }
        ModelsCommand::Load { id, target, json } => {
            ensure!(!id.trim().is_empty(), "model ID is empty");
            let result = request(&target, "/api/load", Some(json!({"model":id})))?;
            if json { print_json(&result)?; } else { println!("Loaded: {}", result["loaded"].as_str().unwrap_or(&id)); }
        }
        ModelsCommand::Unload { target, json } => {
            let result = request(&target, "/api/unload", Some(json!({})))?;
            if json { print_json(&result)?; } else { println!("Model unloaded. The server is ready for another selection."); }
        }
        ModelsCommand::Status { target, json } => {
            let result = request(&target, "/api/models", None)?;
            if json { print_json(&result)?; }
            else {
                println!("Server: {}\nDirectory: {}", target.server, result["dir"].as_str().unwrap_or("unknown"));
                let models = result["models"].as_array().context("server returned no model catalog")?;
                let mut loaded = false;
                for model in models {
                    let active = model["loaded"].as_bool().unwrap_or(false);
                    loaded |= active;
                    println!("{} {}", if active { "[loaded]" } else { "[ready] " }, model["id"].as_str().unwrap_or("unknown"));
                }
                if !loaded { println!("No model is currently loaded."); }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn management_url_is_a_root_without_credential_or_path_confusion() {
        assert_eq!(endpoint("http://127.0.0.1:8080/", "/api/models").unwrap(), "http://127.0.0.1:8080/api/models");
        assert_eq!(endpoint("http://[::1]:8080", "/api/models").unwrap(), "http://[::1]:8080/api/models");
        for invalid in ["ftp://example.org", "http://user:pass@localhost", "http://localhost/v1", "http://localhost/?token=x", "http://localhost/#x", "localhost:8080"] {
            assert!(endpoint(invalid, "/api/models").is_err(), "accepted {invalid}");
        }
    }
}
