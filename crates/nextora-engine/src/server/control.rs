//! Local configuration UI, bounded model discovery and transactional reconfiguration.
//!
//! Administration shares the same admission barrier as model switching. No accepted
//! configuration changes are published until the replacement has loaded and its
//! optional profile file has been committed.
use super::*;
use crate::runtime::Backend;
use serde::{Deserialize, Serialize};
use std::{fs, process::Command};

const MAX_PROFILE_BYTES: u64 = 1024 * 1024;
const MAX_SCAN_ENTRIES: usize = 20_000;
const MAX_SCAN_DEPTH: usize = 16;
const MAX_SCAN_MODELS: usize = 1024;
const MAX_SCAN_MESSAGES: usize = 256;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ServiceSettings {
    backend: String,
    device: Option<usize>,
    ctx: usize,
    pub(super) max_tokens: usize,
    pub(super) parallel: usize,
    pub(super) max_queued: usize,
    pub(super) queue_timeout_secs: u64,
    vram_mib: usize,
    expert_ram_mib: Option<usize>,
    ram_margin_mib: usize,
    expert_warmup: bool,
}

impl ServiceSettings {
    fn from_args(args: &ServeArgs) -> Result<Self> {
        let selected =
            if args.model.is_none() && !args.backend.cpu && args.backend.backend.is_none() {
                Backend::Auto
            } else {
                args.backend.selected()?
            };
        Ok(Self {
            backend: match selected {
                Backend::Auto => "auto",
                Backend::Cpu => "cpu",
                Backend::Vulkan => "vulkan",
                Backend::Hybrid => "hybrid",
            }
            .into(),
            device: args.backend.device,
            ctx: args.ctx,
            max_tokens: args.max_tokens,
            parallel: args.parallel,
            max_queued: args.max_queued,
            queue_timeout_secs: args.queue_timeout_secs,
            vram_mib: args.backend.vram_mib,
            expert_ram_mib: args.backend.expert_ram_mib,
            ram_margin_mib: args.backend.ram_margin_mib,
            expert_warmup: args.backend.expert_warmup,
        })
    }

    fn backend(&self) -> Result<Backend> {
        Ok(match self.backend.as_str() {
            "auto" => Backend::Auto,
            "cpu" => Backend::Cpu,
            "vulkan" => Backend::Vulkan,
            "hybrid" => Backend::Hybrid,
            _ => bail!("backend must be auto, cpu, vulkan or hybrid"),
        })
    }

    fn validate(&self, args: &ServeArgs) -> Result<()> {
        let backend = self.backend()?;
        ensure!(
            (2..=1_048_576).contains(&self.ctx),
            "context must be in 2..=1048576; model limits are also enforced on load"
        );
        ensure!(
            self.max_tokens > 0 && self.max_tokens < self.ctx,
            "max_tokens must be positive and less than context"
        );
        ensure!(
            (1..=4).contains(&self.parallel),
            "parallel must be in 1..=4"
        );
        ensure!(self.max_queued <= 4096, "max_queued must be at most 4096");
        ensure!(
            (1..=3600).contains(&self.queue_timeout_secs),
            "queue_timeout_secs must be in 1..=3600"
        );
        ensure!(
            (self.parallel..=4096).contains(&args.max_connections),
            "max_connections must accommodate parallel requests and be at most 4096"
        );
        ensure!(
            self.device.is_none_or(|n| n <= 1023),
            "device index must be at most 1023"
        );
        ensure!(
            backend != Backend::Cpu || (self.device.is_none() && self.vram_mib == 0),
            "CPU backend requires device=null and vram_mib=0"
        );
        for (name, n) in [
            ("vram_mib", self.vram_mib),
            ("expert_ram_mib", self.expert_ram_mib.unwrap_or(0)),
            ("ram_margin_mib", self.ram_margin_mib),
        ] {
            ensure!(
                n <= 16 * 1024 * 1024 && n.checked_mul(1024 * 1024).is_some(),
                "{name} exceeds the memory-setting limit"
            );
        }
        Ok(())
    }

    fn args(&self, base: &ServeArgs) -> Result<ServeArgs> {
        self.validate(base)?;
        let mut args = base.clone();
        let backend = self.backend()?;
        args.backend.cpu = false;
        args.backend.backend = Some(backend);
        args.backend.device = self.device;
        // This permission is supplied only on the CLI, never via a profile or UI.
        args.backend.allow_software_vulkan =
            base.backend.allow_software_vulkan && backend != Backend::Cpu;
        args.backend.vram_mib = self.vram_mib;
        args.backend.expert_ram_mib = self.expert_ram_mib;
        args.backend.ram_margin_mib = self.ram_margin_mib;
        args.backend.expert_warmup = self.expert_warmup;
        args.backend.state_slots = self.parallel;
        args.ctx = self.ctx;
        args.max_tokens = self.max_tokens;
        args.parallel = self.parallel;
        args.max_queued = self.max_queued;
        args.queue_timeout_secs = self.queue_timeout_secs;
        Ok(args)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ServiceProfile {
    directory: Option<PathBuf>,
    model: Option<String>,
    pub(super) settings: ServiceSettings,
}

fn checked_directory(path: &Path) -> Result<PathBuf> {
    ensure!(!path.as_os_str().is_empty(), "model directory is empty");
    ensure!(
        path.as_os_str().len() <= 32768,
        "model directory path is too long"
    );
    let real = path.canonicalize().with_context(|| format!("model directory {} is unavailable; Windows drives use F:\\ in a Windows process and /mnt/f in native WSL", path.display()))?;
    ensure!(real.is_dir(), "{} is not a directory", path.display());
    Ok(real)
}

fn selected_path(root: &Path, id: &str) -> Result<PathBuf> {
    ensure!(
        !id.is_empty() && id.len() <= 32768,
        "a non-empty model catalogue ID is required"
    );
    let relative = Path::new(id);
    ensure!(
        relative
            .components()
            .all(|c| matches!(c, std::path::Component::Normal(_))),
        "model must be a relative catalogue ID without parent traversal"
    );
    let real = root
        .join(relative)
        .canonicalize()
        .with_context(|| format!("model {id} is unavailable"))?;
    ensure!(
        real.starts_with(root) && real.is_file(),
        "model must be a regular file inside the selected directory"
    );
    ensure!(
        real.extension()
            .is_some_and(|x| x.eq_ignore_ascii_case("gguf")),
        "native service requires a complete GGUF checkpoint"
    );
    Ok(real)
}

fn id_for(root: &Path, path: &Path) -> Result<String> {
    Ok(path
        .strip_prefix(root)?
        .to_string_lossy()
        .replace('\\', "/"))
}

fn check_profile_file(path: &Path) -> Result<()> {
    ensure!(
        path.extension()
            .is_some_and(|x| x.eq_ignore_ascii_case("json")),
        "settings-file must have a .json extension"
    );
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            ensure!(
                metadata.is_file() && !metadata.file_type().is_symlink(),
                "settings-file must be a regular, non-symlink file"
            );
            ensure!(
                metadata.len() <= MAX_PROFILE_BYTES,
                "settings-file exceeds 1 MiB"
            );
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).context("inspect settings-file"),
    }
    Ok(())
}

fn read_profile(path: &Path) -> Result<ServiceProfile> {
    check_profile_file(path)?;
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(MAX_PROFILE_BYTES + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_PROFILE_BYTES,
        "settings-file exceeds 1 MiB"
    );
    serde_json::from_slice(&bytes).context(
        "settings-file must contain {directory, model, settings} from the native configuration UI",
    )
}

pub(super) fn startup(
    args: &mut ServeArgs,
) -> Result<(
    ServiceProfile,
    BTreeMap<String, PathBuf>,
    Option<Arc<ServedModel>>,
    Option<String>,
    Option<String>,
)> {
    let saved = if let Some(file) = &args.settings_file {
        check_profile_file(file)?;
        if file.exists() {
            Some(read_profile(file)?)
        } else {
            None
        }
    } else {
        None
    };
    let from_saved_profile = saved.is_some();
    let mut profile = if let Some(saved) = saved {
        eprintln!(
            "Loading model and runtime settings from {}; transport settings remain CLI-only.",
            args.settings_file.as_ref().unwrap().display()
        );
        saved
    } else {
        let settings = ServiceSettings::from_args(args)?;
        let requested = args
            .model
            .as_ref()
            .map(|path| path.canonicalize())
            .transpose()?;
        let root = args.models_dir.clone().or_else(|| {
            requested
                .as_ref()
                .and_then(|p| p.parent().map(Path::to_path_buf))
        });
        let root = if requested.is_some() {
            root.as_deref().map(checked_directory).transpose()?
        } else {
            root
        };
        let id = requested
            .as_ref()
            .map(|p| id_for(root.as_deref().context("model directory missing")?, p))
            .transpose()?;
        ServiceProfile {
            directory: root,
            model: id,
            settings,
        }
    };
    profile.settings.validate(args)?;
    let mut catalogue = BTreeMap::new();
    let mut startup_error = None;
    let mut desired_model = None;
    let loaded = if let Some(id) = &profile.model {
        let attempt =
            (|| -> Result<(PathBuf, String, BTreeMap<String, PathBuf>, Arc<ServedModel>)> {
                let root = checked_directory(
                    profile
                        .directory
                        .as_deref()
                        .context("profile model requires a directory")?,
                )?;
                let path = selected_path(&root, id)?;
                let id = id_for(&root, &path)?;
                let scan = scan_directory(&root)?;
                let mut catalogue = scan.catalogue();
                catalogue.insert(id.clone(), path.clone());
                let loaded = ServedModel::load(id.clone(), &path, &profile.settings.args(args)?)?;
                Ok((root, id, catalogue, loaded))
            })();
        match attempt {
            Ok((root, id, found, loaded)) => {
                profile.directory = Some(root);
                profile.model = Some(id);
                catalogue = found;
                Some(loaded)
            }
            Err(error) if from_saved_profile => {
                let message = format!(
                    "Saved model could not be loaded: {error:#}. The service is running without a model so you can repair the configuration. The saved profile has not been changed."
                );
                eprintln!("{message}");
                startup_error = Some(message);
                desired_model = profile.model.take();
                None
            }
            Err(error) => return Err(error),
        }
    } else {
        // Empty startup deliberately does not scan an entire drive or open Vulkan.
        if let Some(root) = &profile.directory {
            match checked_directory(root) {
                Ok(real) => profile.directory = Some(real),
                Err(error) => {
                    let message =
                        format!("Model folder is unavailable: {error}. Select a folder in the UI.");
                    eprintln!("{message}");
                    startup_error = Some(message);
                }
            }
        }
        None
    };
    Ok((profile, catalogue, loaded, startup_error, desired_model))
}

#[derive(Serialize)]
struct FoundModel {
    id: String,
    path: PathBuf,
    bytes: usize,
    shards: usize,
    architecture: String,
}
#[derive(Serialize)]
struct RejectedModel {
    path: PathBuf,
    reason: String,
}
#[derive(Serialize)]
struct ScanReport {
    root: PathBuf,
    models: Vec<FoundModel>,
    rejected: Vec<RejectedModel>,
    warnings: Vec<String>,
    truncated: bool,
    scanned_entries: usize,
    safetensors_files: usize,
    skipped_symlinks: usize,
}
impl ScanReport {
    fn catalogue(&self) -> BTreeMap<String, PathBuf> {
        self.models
            .iter()
            .map(|m| (m.id.clone(), m.path.clone()))
            .collect()
    }
    fn warning(&mut self, message: String) {
        if self.warnings.len() < MAX_SCAN_MESSAGES {
            self.warnings.push(message);
        }
    }
    fn reject(&mut self, path: PathBuf, reason: String) {
        if self.rejected.len() < MAX_SCAN_MESSAGES {
            self.rejected.push(RejectedModel { path, reason });
        } else {
            self.truncated = true;
        }
    }
}

fn is_later_shard(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|x| x.to_str()) else {
        return false;
    };
    let Some((prefix, tail)) = name.rsplit_once("-of-") else {
        return false;
    };
    tail.len() == 10
        && prefix.len() >= 6
        && prefix.as_bytes()[prefix.len() - 6] == b'-'
        && prefix.as_bytes()[prefix.len() - 5..]
            .iter()
            .all(u8::is_ascii_digit)
        && &prefix[prefix.len() - 5..] != "00001"
}

fn inspect_model(root: &Path, path: &Path) -> Result<FoundModel> {
    let path = path.canonicalize()?;
    ensure!(
        path.starts_with(root),
        "model path escaped the selected root"
    );
    let g = Arc::new(crate::gguf::Gguf::open(&path)?);
    // Validate layout and the small norm vectors, not expert caches or sequence state.
    let model = crate::model::Model::load(g.clone(), 1)?;
    ensure!(
        model.config().expert_count > 0,
        "native coding service accepts supported sparse MoE models; this checkpoint is dense"
    );
    let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&g)?;
    ensure!(
        tokenizer.vocab_size() == model.config().vocab_size,
        "tokenizer/model vocabulary differs"
    );
    ChatTemplate::new(&g)?;
    Ok(FoundModel {
        id: id_for(root, &path)?,
        path,
        bytes: g.mapped_bytes(),
        shards: g.shard_paths().len(),
        architecture: model.config().architecture.clone(),
    })
}

fn scan_directory(directory: &Path) -> Result<ScanReport> {
    let root = checked_directory(directory)?;
    let mut report = ScanReport {
        root: root.clone(),
        models: vec![],
        rejected: vec![],
        warnings: vec![],
        truncated: false,
        scanned_entries: 0,
        safetensors_files: 0,
        skipped_symlinks: 0,
    };
    let mut dirs = vec![(root.clone(), 0usize)];
    let mut visited = std::collections::BTreeSet::new();
    'walk: while let Some((directory, depth)) = dirs.pop() {
        let real_dir = match directory.canonicalize() {
            Ok(real) if real.starts_with(&root) => real,
            Ok(_) => {
                report.warning(format!(
                    "Skipped directory outside selected root: {}",
                    directory.display()
                ));
                continue;
            }
            Err(error) => {
                report.warning(format!("Skipped {}: {error}", directory.display()));
                continue;
            }
        };
        if !visited.insert(real_dir.clone()) {
            continue;
        }
        let entries = match fs::read_dir(&real_dir) {
            Ok(entries) => entries,
            Err(error) => {
                report.warning(format!("Skipped {}: {error}", directory.display()));
                continue;
            }
        };
        for item in entries {
            if report.scanned_entries >= MAX_SCAN_ENTRIES || report.models.len() >= MAX_SCAN_MODELS
            {
                report.truncated = true;
                report.warning(
                    "Scan limit reached. Select a narrower model folder and scan again.".into(),
                );
                break 'walk;
            }
            report.scanned_entries += 1;
            let item = match item {
                Ok(item) => item,
                Err(error) => {
                    report.warning(format!("Skipped unreadable entry: {error}"));
                    continue;
                }
            };
            let path = item.path();
            let kind = match item.file_type() {
                Ok(kind) => kind,
                Err(error) => {
                    report.warning(format!("Skipped {}: {error}", path.display()));
                    continue;
                }
            };
            if kind.is_symlink() {
                report.skipped_symlinks += 1;
                continue;
            }
            if kind.is_dir() {
                if depth < MAX_SCAN_DEPTH {
                    dirs.push((path, depth + 1));
                } else {
                    report.truncated = true;
                    report.warning(format!(
                        "Depth limit reached at {}; select this folder directly if needed",
                        path.display()
                    ));
                }
            } else if kind.is_file() {
                if path
                    .extension()
                    .is_some_and(|x| x.eq_ignore_ascii_case("safetensors"))
                {
                    report.safetensors_files += 1;
                }
                if path
                    .extension()
                    .is_none_or(|x| !x.eq_ignore_ascii_case("gguf"))
                {
                    continue;
                }
                if is_later_shard(&path) {
                    let name = path.file_name().unwrap().to_string_lossy();
                    let (prefix, tail) = name.rsplit_once("-of-").unwrap();
                    let first = path
                        .with_file_name(format!("{}00001-of-{tail}", &prefix[..prefix.len() - 5]));
                    if !first.is_file() {
                        report.reject(path, format!("Later GGUF shard has no first shard {}; download the complete shard set into one directory", first.display()));
                    }
                    continue;
                }
                match inspect_model(&root, &path) {
                    Ok(model) => report.models.push(model),
                    Err(error) => report.reject(path, format!("{error:#}")),
                }
            }
        }
    }
    report.models.sort_by(|a, b| a.id.cmp(&b.id));
    if report.safetensors_files > 0 {
        report.warning("Safetensors files were found. This native service requires a supported GGUF checkpoint; select all shards of a compatible GGUF download.".into());
    }
    if report.skipped_symlinks > 0 {
        report.warning(format!("Skipped {} symbolic links. Hugging Face cache snapshots may use links; place the complete GGUF files in a regular model folder to scan them here.", report.skipped_symlinks));
    }
    Ok(report)
}

fn in_wsl() -> bool {
    cfg!(target_os = "linux")
        && (std::env::var_os("WSL_DISTRO_NAME").is_some()
            || fs::read_to_string("/proc/sys/kernel/osrelease")
                .is_ok_and(|x| x.to_ascii_lowercase().contains("microsoft")))
}

fn suggested_model_dirs() -> Vec<Value> {
    let mut result = vec![];
    if cfg!(windows) {
        result.push(json!({"label":"F: drive", "path":"F:\\"}));
    } else if in_wsl() {
        result.push(json!({"label":"Windows F: drive", "path":"/mnt/f"}));
    }
    if let Some(path) = std::env::var_os("NEXTORA_MODELS_DIR") {
        result.push(json!({"label":"Configured Forge models", "path":PathBuf::from(path)}));
    }
    if cfg!(windows) {
        if let Some(path) = std::env::var_os("LOCALAPPDATA") {
            result.push(json!({"label":"Forge download folder", "path":PathBuf::from(path).join("nextora-cli").join("models")}));
            return result;
        }
    }
    let data = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".local")
                .join("share")
        });
    result.push(
        json!({"label":"Forge download folder", "path":data.join("nextora-cli").join("models")}),
    );
    result
}

pub(super) fn config_value(server: &Server) -> Value {
    let profile = server.profile.read().unwrap();
    json!({
        "engine": crate::ENGINE_ID, "version": crate::ENGINE_VERSION,
        "platform": std::env::consts::OS,
        "in_wsl": in_wsl(),
        "suggested_model_dirs": suggested_model_dirs(),
        "startup_error": *server.startup_error.read().unwrap(),
        "desired_model": *server.desired_model.read().unwrap(),
        "settings": profile.settings, "models_dir": profile.directory, "model": profile.model,
        "settings_file": server.args.settings_file,
        "persistence": if server.args.settings_file.is_some() { "Successful changes are saved to the configured settings file." } else { "Changes apply to this process. Export settings and restart with --settings-file to retain them." },
        "startup": { "host": server.args.host, "port": server.args.port, "threads": server.threads, "max_connections": server.args.max_connections, "api_key_required": server.args.api_key.is_some() }
    })
}

pub(super) fn handle_control(
    server: &Arc<Server>,
    stream: &mut TcpStream,
    request: &Request,
) -> Result<bool> {
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/api/config") => json_response(stream, 200, &config_value(server))?,
        ("GET", "/api/hardware") => json_response(stream, 200, &crate::runtime::hardware_report())?,
        ("POST", "/api/models/scan") => {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct ScanRequest {
                directory: PathBuf,
            }
            let parsed: ScanRequest = match serde_json::from_slice(&request.body) {
                Ok(p) => p,
                Err(e) => {
                    json_response(stream, 400, &error_value(&e.to_string()))?;
                    return Ok(true);
                }
            };
            let _scan = match server.scanning.try_lock() {
                Ok(s) => s,
                Err(_) => {
                    json_response(
                        stream,
                        409,
                        &error_value("another model-folder scan is already running"),
                    )?;
                    return Ok(true);
                }
            };
            match scan_directory(&parsed.directory) {
                Ok(report) => json_response(stream, 200, &serde_json::to_value(report)?)?,
                Err(error) => json_response(stream, 400, &error_value(&format!("{error:#}")))?,
            }
        }
        ("POST", "/api/config") | ("POST", "/api/models/load") => {
            let _admin = match server.administration.try_lock() {
                Ok(a) => a,
                Err(_) => {
                    json_response(
                        stream,
                        409,
                        &error_value("another service configuration change is in progress"),
                    )?;
                    return Ok(true);
                }
            };
            let profile = if request.path == "/api/config" {
                match serde_json::from_slice::<ServiceProfile>(&request.body) {
                    Ok(p) => p,
                    Err(e) => {
                        json_response(stream, 400, &error_value(&e.to_string()))?;
                        return Ok(true);
                    }
                }
            } else {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct LoadRequest {
                    model: String,
                }
                let load = match serde_json::from_slice::<LoadRequest>(&request.body) {
                    Ok(p) => p,
                    Err(e) => {
                        json_response(stream, 400, &error_value(&e.to_string()))?;
                        return Ok(true);
                    }
                };
                if !server.catalogue.read().unwrap().contains_key(&load.model) {
                    json_response(stream, 404, &error_value("model is outside the catalogue"))?;
                    return Ok(true);
                }
                let mut profile = server.profile.read().unwrap().clone();
                profile.model = Some(load.model);
                profile
            };
            match apply_profile(server, profile) {
                Ok(()) => {
                    let mut response = config_value(server);
                    response["runtime"] = server
                        .loaded
                        .read()
                        .unwrap()
                        .as_ref()
                        .map(|m| m.model.report())
                        .unwrap_or(Value::Null);
                    json_response(stream, 200, &response)?;
                }
                Err((status, message)) => json_response(stream, status, &error_value(&message))?,
            }
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn idle(server: &Server) -> std::result::Result<(), (u16, String)> {
    let gate = server.admission.state.lock().unwrap();
    if gate.active > 0 || !gate.queue.is_empty() || gate.changing {
        return Err((409, "configuration changes require an idle server; wait for active and queued requests to finish".into()));
    }
    Ok(())
}

fn apply_profile(
    server: &Server,
    mut profile: ServiceProfile,
) -> std::result::Result<(), (u16, String)> {
    idle(server)?;
    let bad = |error: anyhow::Error| (400, format!("{error:#}"));
    let effective = profile.settings.args(&server.args).map_err(bad)?;
    let previous_profile = server.profile.read().unwrap().clone();
    let (path, mut catalogue) = if let Some(id) = &profile.model {
        let root = checked_directory(
            profile
                .directory
                .as_deref()
                .ok_or_else(|| (400, "model requires a directory".into()))?,
        )
        .map_err(bad)?;
        let path = selected_path(&root, id).map_err(bad)?;
        // Validate the exact selected entry again; no arbitrary absolute path is accepted.
        let found = inspect_model(&root, &path).map_err(bad)?;
        let catalogue = scan_directory(&root).map_err(bad)?.catalogue();
        profile.directory = Some(root);
        profile.model = Some(found.id);
        (Some(path), catalogue)
    } else {
        if let Some(root) = &profile.directory {
            // Unload remains possible if a removable model drive was disconnected.
            if let Ok(real) = checked_directory(root) {
                profile.directory = Some(real);
            }
        }
        let catalogue = if previous_profile.directory == profile.directory {
            server.catalogue.read().unwrap().clone()
        } else {
            BTreeMap::new()
        };
        (None, catalogue)
    };
    if let (Some(path), Some(id)) = (&path, &profile.model) {
        catalogue.insert(id.clone(), path.clone());
    }
    // Stage and fsync the new JSON before dropping a working model. A failed write
    // does not change the model, budgets, queue or the previous on-disk profile.
    let staged = server
        .args
        .settings_file
        .as_deref()
        .map(|p| StagedProfile::new(p, &profile))
        .transpose()
        .map_err(bad)?;
    let previous = {
        let mut gate = server.admission.state.lock().unwrap();
        if gate.active > 0 || !gate.queue.is_empty() || gate.changing {
            return Err((409, "configuration changes require an idle server".into()));
        }
        let mut slot = server.loaded.write().unwrap();
        if slot.as_ref().is_some_and(|m| Arc::strong_count(m) > 1) {
            return Err((
                409,
                "a request is preparing against the current model; retry when idle".into(),
            ));
        }
        gate.changing = true;
        slot.take()
    };
    let previous_identity = previous
        .as_ref()
        .map(|m| (m.id.clone(), m.model.source.path.clone()));
    drop(previous);
    let next = match (profile.model.as_ref(), path.as_ref()) {
        (Some(id), Some(path)) => ServedModel::load(id.clone(), path, &effective).map(Some),
        _ => Ok(None),
    };
    let next = next.and_then(|model| {
        if let Some(staged) = staged {
            staged.commit()?;
        }
        Ok(model)
    });
    let result = match next {
        Ok(next) => {
            *server.loaded.write().unwrap() = next;
            *server.catalogue.write().unwrap() = catalogue;
            *server.profile.write().unwrap() = profile.clone();
            *server.startup_error.write().unwrap() = None;
            *server.desired_model.write().unwrap() = None;
            let mut gate = server.admission.state.lock().unwrap();
            gate.capacity = profile.settings.parallel;
            gate.queue_limit = profile.settings.max_queued;
            gate.timeout = Duration::from_secs(profile.settings.queue_timeout_secs);
            gate.high_water = 0;
            Ok(())
        }
        Err(error) => {
            let restored = previous_identity
                .map(|(id, path)| {
                    ServedModel::load(id, &path, &previous_profile.settings.args(&server.args)?)
                })
                .transpose();
            match restored {
                Ok(previous) => {
                    *server.loaded.write().unwrap() = previous;
                    Err((
                        400,
                        format!(
                            "configuration was not applied; previous service configuration restored: {error:#}"
                        ),
                    ))
                }
                Err(restore) => {
                    server.profile.write().unwrap().model = None;
                    Err((
                        503,
                        format!(
                            "configuration failed ({error:#}); previous model reload also failed ({restore:#}); no model is loaded"
                        ),
                    ))
                }
            }
        }
    };
    let mut gate = server.admission.state.lock().unwrap();
    gate.changing = false;
    server.admission.changed.notify_all();
    result
}

struct StagedProfile {
    temporary: PathBuf,
    destination: PathBuf,
}
impl StagedProfile {
    fn new(destination: &Path, profile: &ServiceProfile) -> Result<Self> {
        check_profile_file(destination)?;
        // Existing files must actually be native profiles, never arbitrary JSON.
        if destination.exists() {
            read_profile(destination)?;
        }
        let parent = destination
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(parent).context("create settings-file parent directory")?;
        let name = destination
            .file_name()
            .context("settings-file needs a filename")?
            .to_string_lossy();
        let serial = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let temporary = parent.join(format!(".{name}.{}.{}.tmp", std::process::id(), serial));
        let staged = Self {
            temporary,
            destination: destination.to_path_buf(),
        };
        let mut options = fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options
            .open(&staged.temporary)
            .context("stage settings-file")?;
        let mut bytes = serde_json::to_vec_pretty(profile)?;
        bytes.push(b'\n');
        ensure!(
            bytes.len() as u64 <= MAX_PROFILE_BYTES,
            "settings profile exceeds 1 MiB"
        );
        file.write_all(&bytes)
            .context("write staged settings-file")?;
        file.sync_all().context("flush staged settings-file")?;
        Ok(staged)
    }
    fn commit(self) -> Result<()> {
        check_profile_file(&self.destination)?;
        // std::fs::rename atomically replaces regular files on both Unix and Windows.
        fs::rename(&self.temporary, &self.destination)
            .context("commit settings-file; runtime changes rolled back")?;
        Ok(())
    }
}
impl Drop for StagedProfile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.temporary);
    }
}

pub(super) fn origin_allowed(origin: Option<&str>, host: &str, port: u16) -> bool {
    let Some(origin) = origin else {
        return true;
    };
    let Some(authority) = origin.strip_prefix("http://") else {
        return false;
    };
    if authority.contains(['/', '?', '#', '@']) {
        return false;
    }
    let expected = format!("http://{}", host.to_ascii_lowercase());
    origin.to_ascii_lowercase() == expected
        || (port == 80
            && origin.eq_ignore_ascii_case(&format!(
                "http://{}",
                host.strip_suffix(":80").unwrap_or(host)
            )))
}

pub(super) fn asset(path: &str) -> Option<(&'static str, &'static str)> {
    match path {
        "/" | "/index.html" => Some(("text/html; charset=utf-8", include_str!("../ui/index.html"))),
        "/ui/app.js" => Some((
            "application/javascript; charset=utf-8",
            include_str!("../ui/app.js"),
        )),
        "/ui/style.css" => Some(("text/css; charset=utf-8", include_str!("../ui/style.css"))),
        _ => None,
    }
}

pub(super) fn asset_response(stream: &mut TcpStream, mime: &str, body: &str) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nReferrer-Policy: no-referrer\r\nContent-Security-Policy: default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'self'; frame-ancestors 'none'\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body.as_bytes())?;
    Ok(())
}

pub(super) fn open_browser(url: &str) -> Result<()> {
    #[cfg(target_os = "windows")]
    let mut command = {
        // A fixed program receives the URL as data; no user path enters a shell command.
        let mut c = Command::new("rundll32.exe");
        c.arg("url.dll,FileProtocolHandler").arg(url);
        c
    };
    #[cfg(target_os = "macos")]
    let mut command = {
        let mut c = Command::new("open");
        c.arg(url);
        c
    };
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    let mut command = {
        let wsl = std::env::var_os("WSL_DISTRO_NAME").is_some()
            || fs::read_to_string("/proc/sys/kernel/osrelease")
                .is_ok_and(|x| x.to_ascii_lowercase().contains("microsoft"));
        if wsl && Path::new("/mnt/c/Windows/System32/rundll32.exe").is_file() {
            let mut c = Command::new("/mnt/c/Windows/System32/rundll32.exe");
            c.arg("url.dll,FileProtocolHandler").arg(url);
            c
        } else {
            let mut c = Command::new("xdg-open");
            c.arg(url);
            c
        }
    };
    let mut child = command
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .context("launch default browser")?;
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn browser_origin_is_exact_and_cli_has_no_origin() {
        assert!(origin_allowed(None, "localhost:9070", 9070));
        assert!(origin_allowed(
            Some("http://localhost:9070"),
            "localhost:9070",
            9070
        ));
        for origin in [
            "null",
            "https://localhost:9070",
            "http://localhost:9071",
            "http://localhost:9070.attacker",
            "http://localhost:9070/",
            "http://user@localhost:9070",
        ] {
            assert!(
                !origin_allowed(Some(origin), "localhost:9070", 9070),
                "{origin}"
            );
        }
    }
    #[test]
    fn first_standard_shard_is_the_selectable_entry() {
        assert!(!is_later_shard(Path::new("Qwen-IQ3_S-00001-of-00002.gguf")));
        assert!(is_later_shard(Path::new("Qwen-IQ3_S-00002-of-00002.gguf")));
        assert!(!is_later_shard(Path::new("model.gguf")));
        assert!(!is_later_shard(Path::new("a-xx💡-of-00002.gguf")));
    }
}
