//! Public, revision-pinned Hugging Face downloads for the native model catalogue.
use super::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    process::{Command, Stdio},
    sync::atomic::AtomicBool,
};

#[derive(Default)]
pub(super) struct Hub {
    search: Mutex<()>,
    verified: Mutex<Vec<Choice>>,
    job: Mutex<Job>,
    cancel: AtomicBool,
}

#[derive(Clone, Default, Serialize)]
struct Job {
    phase: &'static str,
    directory: PathBuf,
    repo: String,
    file: String,
    current_file: String,
    completed_files: usize,
    total_files: usize,
    received_bytes: u64,
    total_bytes: u64,
    model: Option<String>,
    architecture: String,
    native_backends: &'static [&'static str],
    context_length: Option<u64>,
    error: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Search {
    query: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Selection {
    repo: String,
    revision: String,
    file: String,
}

#[derive(Clone)]
struct Blob {
    name: String,
    bytes: u64,
    hash: String,
}

#[derive(Clone, Serialize)]
struct Choice {
    repo: String,
    revision: String,
    architecture: String,
    native_backends: &'static [&'static str],
    context_length: Option<u64>,
    file: String,
    shards: usize,
    bytes: u64,
    #[serde(skip)]
    blobs: Vec<Blob>,
}

fn destination(server: &Server) -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("NEXTORA_MODELS_DIR").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(path));
    }
    if cfg!(windows) {
        return Ok(PathBuf::from(r"F:\Models"));
    }
    if control::in_wsl() {
        return Ok(PathBuf::from("/mnt/f/Models"));
    }
    server
        .args
        .models_dir
        .clone()
        .or_else(|| std::env::var_os("NEXTORA_MODELS_DIR").map(PathBuf::from))
        .context("Choose the service download folder at startup with --models-dir")
}

fn safe_component(s: &str) -> bool {
    let stem = s.split('.').next().unwrap_or("").to_ascii_uppercase();
    !s.is_empty()
        && s.len() <= 180
        && !s.starts_with('.')
        && !s.ends_with('.')
        && !s.ends_with(' ')
        && s.bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"-_.+ ".contains(&c))
        && !matches!(stem.as_str(), "CON" | "AUX" | "PRN" | "NUL")
        && !(stem.len() == 4
            && (stem.starts_with("COM") || stem.starts_with("LPT"))
            && stem.as_bytes()[3].is_ascii_digit())
}

fn valid_repo(s: &str) -> bool {
    s.split('/').count() == 2
        && s.split('/').all(|p| {
            safe_component(p)
                && p.bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))
        })
}

fn valid_sha(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|c| c.is_ascii_hexdigit())
}

fn valid_file(s: &str) -> bool {
    s.len() <= 480 && s.to_ascii_lowercase().ends_with(".gguf") && s.split('/').all(safe_component)
}

fn encode(s: &str) -> String {
    s.bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) {
                char::from(b).to_string()
            } else {
                format!("%{b:02X}")
            }
        })
        .collect()
}

fn curl() -> Command {
    let mut cmd = Command::new(if cfg!(windows) { "curl.exe" } else { "curl" });
    cmd.args([
        "-q",
        "--silent",
        "--show-error",
        "--fail",
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--connect-timeout",
        "10",
        "--user-agent",
        "Nextora-model-catalogue",
    ]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // avoid opening a console for each Hub request
    }
    cmd
}

fn metadata(url: &str) -> Result<Value> {
    let output = curl()
        .args([
            "--max-time",
            "18",
            "--max-filesize",
            "4194304",
            "--url",
            url,
        ])
        .output()
        .context("start curl to contact Hugging Face")?;
    ensure!(
        output.status.success(),
        "Hugging Face metadata request failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(serde_json::from_slice(&output.stdout).context("invalid Hugging Face metadata response")?)
}

fn supported_name(filename: &str) -> bool {
    let lower = filename
        .rsplit('/')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    if lower.contains("mmproj") || lower.contains("imatrix") {
        return false;
    }
    // Names only exclude auxiliary files. Compatibility is determined by the
    // pinned native header checks, including mixed-format checkpoints.
    lower.ends_with(".gguf")
}

fn shard_group(name: &str) -> Option<(&str, usize, usize)> {
    let (first, count) = name.strip_suffix(".gguf")?.rsplit_once("-of-")?;
    let (prefix, index) = first.rsplit_once('-')?;
    if count.len() != 5
        || index.len() != 5
        || !count
            .bytes()
            .chain(index.bytes())
            .all(|c| c.is_ascii_digit())
    {
        return None;
    }
    Some((prefix, index.parse().ok()?, count.parse().ok()?))
}

fn choices(repo: &str, revision: &str, info: &Value) -> Result<Vec<Choice>> {
    ensure!(
        info["sha"] == revision && valid_sha(revision, 40),
        "Hugging Face returned a different revision"
    );
    let arch = info["gguf"]["architecture"].as_str().unwrap_or("");
    if !matches!(arch, "qwen3" | "qwen3moe" | "qwen4exp" | "qwen35") {
        return Ok(vec![]);
    }
    let mut files = BTreeMap::new();
    for s in info["siblings"]
        .as_array()
        .context("model files missing from repository metadata")?
    {
        let Some(name) = s["rfilename"].as_str() else {
            continue;
        };
        let Some(size) = s["size"].as_u64() else {
            continue;
        };
        let Some(hash) = s["lfs"]["sha256"].as_str() else {
            continue;
        };
        if size > 0 && valid_sha(hash, 64) && valid_file(name) && supported_name(name) {
            files.insert(
                name,
                Blob {
                    name: name.into(),
                    bytes: size,
                    hash: hash.to_ascii_lowercase(),
                },
            );
        }
    }
    let mut results = vec![];
    for (name, blob) in &files {
        let group = if name.contains("-of-") {
            let Some((prefix, 1, total)) = shard_group(name) else {
                continue;
            };
            if !(1..=256).contains(&total) {
                continue;
            }
            let Some(parts) = (1..=total)
                .map(|i| {
                    files
                        .get(format!("{prefix}-{i:05}-of-{total:05}.gguf").as_str())
                        .cloned()
                })
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            parts
        } else {
            vec![blob.clone()]
        };
        let bytes = group
            .iter()
            .try_fold(0u64, |sum, b| sum.checked_add(b.bytes))
            .context("model size overflow")?;
        results.push(Choice {
            repo: repo.into(),
            revision: revision.into(),
            architecture: arch.into(),
            native_backends: control::native_backends(arch),
            context_length: info["gguf"]["context_length"].as_u64(),
            file: (*name).into(),
            shards: group.len(),
            bytes,
            blobs: group,
        });
    }
    Ok(results)
}

fn pinned(repo: &str, revision: &str) -> Result<Vec<Choice>> {
    ensure!(
        valid_repo(repo) && valid_sha(revision, 40),
        "Invalid Hugging Face repository or commit"
    );
    choices(
        repo,
        revision,
        &metadata(&format!(
            "https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
        ))?,
    )
}

fn resolve_url(choice: &Choice, blob: &Blob) -> String {
    let encoded = blob
        .name
        .split('/')
        .map(encode)
        .collect::<Vec<_>>()
        .join("/");
    format!(
        "https://huggingface.co/{}/resolve/{}/{encoded}?download=true",
        choice.repo, choice.revision
    )
}

const HEADER_BYTES: u64 = 16 * 1024 * 1024;

fn probe(choice: &Choice, remaining: &mut u64, deadline: Instant) -> Result<(String, usize)> {
    let mut headers = Vec::with_capacity(choice.shards);
    for (idx, blob) in choice.blobs.iter().enumerate() {
        let bytes = blob
            .bytes
            .min(if idx == 0 { HEADER_BYTES } else { 512 * 1024 });
        *remaining = remaining
            .checked_sub(bytes)
            .context("search metadata transfer limit reached")?;
        let timeout = deadline
            .saturating_duration_since(Instant::now())
            .as_secs()
            .min(20);
        ensure!(timeout > 0, "search verification deadline reached");
        let output = curl()
            .args([
                "--location",
                "--max-redirs",
                "5",
                "--header",
                "Accept-Encoding: identity",
                "--max-time",
                &timeout.to_string(),
                "--range",
                &format!("0-{}", bytes - 1),
                "--max-filesize",
                &bytes.to_string(),
                "--write-out",
                "%{stderr}%{http_code} %header{content-range}",
                "--url",
                &resolve_url(choice, blob),
            ])
            .output()
            .context("read pinned GGUF header from Hugging Face")?;
        let received = String::from_utf8_lossy(&output.stderr);
        ensure!(
            output.status.success(),
            "GGUF header request failed ({})",
            output.status
        );
        ensure!(
            received.trim().eq_ignore_ascii_case(&format!(
                "206 bytes 0-{}/{}",
                bytes - 1,
                blob.bytes
            )) || (received.trim() == "200" && blob.bytes == bytes),
            "Hugging Face returned an unexpected byte range"
        );
        ensure!(
            output.stdout.len() as u64 == bytes,
            "truncated GGUF header range"
        );
        headers.push(crate::gguf::RemoteHeader::parse(
            &blob.name,
            &output.stdout,
            blob.bytes,
        )?);
    }
    crate::model::validate_hub_header(Arc::new(crate::gguf::Gguf::from_remote_headers(headers)?))
}

fn search(query: &str, dir: PathBuf, hub: &Hub) -> Result<Value> {
    let query = query.trim();
    ensure!(query.len() <= 160, "Search text is too long");
    let query = if query.is_empty() {
        "Qwen3-30B-A3B"
    } else {
        query
    };
    let result = metadata(&format!(
        "https://huggingface.co/api/models?filter=gguf&search={}&limit=32&expand%5B%5D=sha&expand%5B%5D=gguf",
        encode(query)
    ))?;
    let candidates = result
        .as_array()
        .context("invalid Hugging Face search response")?;
    let cached = hub.verified.lock().unwrap().clone();
    let mut remaining = 96 * 1024 * 1024;
    let deadline = Instant::now() + Duration::from_secs(65);
    let mut models = vec![];
    let mut excluded = 0usize;
    let mut checked = 0usize;
    let mut warnings = vec![];
    for item in candidates
        .iter()
        .filter(|r| {
            matches!(
                r["gguf"]["architecture"].as_str(),
                Some("qwen3" | "qwen3moe" | "qwen4exp" | "qwen35")
            )
        })
        .take(5)
    {
        let (Some(repo), Some(sha)) = (item["id"].as_str(), item["sha"].as_str()) else {
            continue;
        };
        if Instant::now() >= deadline || remaining < HEADER_BYTES || models.len() >= 8 {
            break;
        }
        match pinned(repo, sha) {
            Ok(mut choices) => {
                choices.sort_by_key(|c| {
                    let name = c.file.to_ascii_uppercase();
                    let rank = if name.contains("Q4_K_M") {
                        0
                    } else if name.contains("IQ4_XS") {
                        1
                    } else if name.contains("Q4_K_S") {
                        2
                    } else if name.contains("IQ3_XXS") {
                        3
                    } else if ["Q3_K", "Q4_0", "Q5_K", "Q6_K", "IQ3_S", "IQ4_NL", "Q8_0", "Q2_0", "IQ2_S", "BF16", "F16", "F32"].iter().any(|tag| name.contains(tag)) {
                        4
                    } else {
                        5
                    };
                    (rank, c.bytes)
                });
                for mut choice in choices.into_iter().take(4) {
                    if Instant::now() >= deadline || remaining < HEADER_BYTES || models.len() >= 8 {
                        break;
                    }
                    if let Some(hit) = cached.iter().find(|c| {
                        c.repo == choice.repo
                            && c.revision == choice.revision
                            && c.file == choice.file
                    }) {
                        models.push(hit.clone());
                        continue;
                    }
                    checked += 1;
                    match probe(&choice, &mut remaining, deadline) {
                        Ok((arch, context)) => {
                            choice.native_backends = control::native_backends(&arch);
                            choice.architecture = arch;
                            choice.context_length = Some(context as u64);
                            models.push(choice);
                        }
                        Err(e) => {
                            excluded += 1;
                            if warnings.len() < 4 {
                                warnings.push(format!("{}: {e:#}", choice.file));
                            }
                        }
                    }
                }
            }
            Err(error) => warnings.push(format!("{repo}: {error:#}")),
        }
    }
    let response = json!({"models":models,"directory":dir,"warnings":warnings,"checked":checked,"excluded":excluded,
        "verification":"Pinned GGUF headers, all shard descriptors, architecture, tensor shapes and formats, tokenizer and chat template. Weight integrity is checked after download.",
        "bounded":true});
    let mut cache = hub.verified.lock().unwrap();
    for entry in models {
        if !cache
            .iter()
            .any(|c| c.repo == entry.repo && c.revision == entry.revision && c.file == entry.file)
        {
            if cache.len() >= 64 {
                cache.remove(0);
            }
            cache.push(entry);
        }
    }
    Ok(response)
}

struct Stage(PathBuf);
impl Drop for Stage {
    fn drop(&mut self) {
        if self.0.exists() {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}

impl Hub {
    fn snapshot(&self) -> Value {
        let job = self.job.lock().unwrap();
        if job.phase.is_empty() {
            json!({"phase":"idle"})
        } else {
            json!(&*job)
        }
    }
    fn check_cancel(&self) -> Result<()> {
        ensure!(!self.cancel.load(Ordering::Acquire), "Download cancelled");
        Ok(())
    }
}

fn download_blob(
    dir: &Path,
    choice: &Choice,
    blob: &Blob,
    hub: &Hub,
    completed_bytes: u64,
) -> Result<()> {
    hub.check_cancel()?;
    let filename = blob.name.rsplit('/').next().unwrap();
    let path = dir.join(filename);
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    let url = resolve_url(choice, blob);
    let mut child = curl()
        .args([
            "--location",
            "--max-redirs",
            "6",
            "--speed-limit",
            "1024",
            "--speed-time",
            "25",
            "--max-filesize",
            &blob.bytes.to_string(),
            "--url",
            &url,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("start curl to download GGUF")?;
    let mut hasher = Sha256::new();
    let copy = (|| -> Result<()> {
        let mut input = child.stdout.take().context("download output missing")?;
        let mut chunk = vec![0u8; 256 * 1024];
        let mut received = 0u64;
        let mut last = Instant::now();
        loop {
            hub.check_cancel()?;
            let n = input.read(&mut chunk)?;
            if n == 0 {
                break;
            }
            received += n as u64;
            ensure!(
                received <= blob.bytes,
                "download exceeded advertised file size"
            );
            output.write_all(&chunk[..n])?;
            hasher.update(&chunk[..n]);
            if last.elapsed() >= Duration::from_millis(120) {
                hub.job.lock().unwrap().received_bytes = completed_bytes + received;
                last = Instant::now();
            }
        }
        ensure!(
            received == blob.bytes,
            "Incomplete shard {filename}: received {received} of {} bytes",
            blob.bytes
        );
        Ok(())
    })();
    if copy.is_err() {
        let _ = child.kill();
    }
    let status = child.wait()?;
    copy?;
    ensure!(
        status.success(),
        "Hugging Face transfer failed for {filename} ({status})"
    );
    ensure!(
        format!("{:x}", hasher.finalize()) == blob.hash,
        "SHA-256 verification failed for {filename}"
    );
    output.sync_all()?;
    Ok(())
}

#[cfg(unix)]
fn available_disk_bytes(root: &Path) -> Result<u64> {
    use std::{ffi::CString, mem::MaybeUninit, os::unix::ffi::OsStrExt};
    let path = CString::new(root.as_os_str().as_bytes())?;
    let mut stat = MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: path is NUL-terminated, and stat is writable for the native structure.
    if unsafe { libc::statvfs(path.as_ptr(), stat.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error())
            .context("query free space on the models drive");
    }
    // SAFETY: statvfs initialized the output on success.
    let stat = unsafe { stat.assume_init() };
    Ok((stat.f_bavail as u64).saturating_mul(stat.f_frsize as u64))
}

#[cfg(windows)]
fn available_disk_bytes(root: &Path) -> Result<u64> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetDiskFreeSpaceExW(
            directory: *const u16,
            available: *mut u64,
            capacity: *mut u64,
            total_free: *mut u64,
        ) -> i32;
    }
    let directory: Vec<_> = root.as_os_str().encode_wide().chain([0]).collect();
    let mut available = 0;
    // SAFETY: the UTF-16 path is terminated and available is a writable ULARGE_INTEGER.
    // The two unused outputs are optional. Available reflects this account's disk quota.
    if unsafe {
        GetDiskFreeSpaceExW(
            directory.as_ptr(),
            &mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    } == 0
    {
        return Err(std::io::Error::last_os_error())
            .context("query free space on the models drive");
    }
    Ok(available)
}

#[cfg(not(any(unix, windows)))]
fn available_disk_bytes(_: &Path) -> Result<u64> {
    bail!("Cannot measure free space on this platform")
}

fn ensure_room(available: u64, remaining: u64) -> Result<()> {
    const RESERVE: u64 = 256 * 1024 * 1024;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    ensure!(
        available
            .checked_sub(RESERVE)
            .is_some_and(|bytes| bytes >= remaining),
        "Insufficient space on the models drive. The remaining download needs {:.2} GiB with 256 MiB of headroom. Available: {:.2} GiB.",
        remaining as f64 / GIB,
        available as f64 / GIB
    );
    Ok(())
}

fn install(server: &Server, selected: &Selection, directory: &Path) -> Result<()> {
    let hub = &server.hub;
    let choice = hub
        .verified
        .lock()
        .unwrap()
        .iter()
        .find(|c| {
            c.repo == selected.repo && c.revision == selected.revision && c.file == selected.file
        })
        .cloned()
        .context("Search for this exact GGUF first so its headers and shards can be verified")?;
    hub.check_cancel()?;
    ensure!(
        directory.parent().is_some_and(Path::is_dir),
        "Models drive or parent folder is unavailable"
    );
    fs::create_dir_all(directory).context("create model download folder")?;
    let root = directory.canonicalize()?;
    let fingerprint =
        Sha256::digest(format!("{}/{}/{}", choice.repo, choice.revision, choice.file).as_bytes());
    let hex = format!("{fingerprint:x}");
    let destination = root.join(format!(
        "hf-{}-{}",
        choice.repo.split('/').next_back().unwrap(),
        &hex[..24]
    ));
    ensure!(
        fs::symlink_metadata(&destination).is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound),
        "This model download already exists; scan the folder to load it"
    );
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let stage = root.join(format!(".nextora-partial-{}-{stamp}", std::process::id()));
    fs::create_dir(&stage)?;
    let stage = Stage(stage);
    {
        let mut job = hub.job.lock().unwrap();
        job.architecture = choice.architecture.clone();
        job.native_backends = choice.native_backends;
        job.total_bytes = choice.bytes;
        job.total_files = choice.shards;
        job.phase = "downloading";
    }
    let mut completed = 0u64;
    for (i, blob) in choice.blobs.iter().enumerate() {
        ensure_room(available_disk_bytes(&root)?, choice.bytes - completed)?;
        hub.job.lock().unwrap().current_file = blob.name.clone();
        download_blob(&stage.0, &choice, blob, hub, completed)?;
        completed += blob.bytes;
        let mut job = hub.job.lock().unwrap();
        job.completed_files = i + 1;
        job.received_bytes = completed;
    }
    hub.check_cancel()?;
    hub.job.lock().unwrap().phase = "validating";
    let entry = choice.file.split('/').next_back().unwrap();
    control::inspect_model(&root, &stage.0.join(entry))
        .context("Downloaded GGUF failed Nextora model validation")?;
    let mut job = hub.job.lock().unwrap();
    hub.check_cancel()?;
    fs::rename(&stage.0, &destination).context("publish validated model snapshot")?;
    let id = destination
        .strip_prefix(&root)?
        .join(entry)
        .to_string_lossy()
        .replace('\\', "/");
    job.model = Some(id);
    job.context_length = choice.context_length;
    job.phase = "ready";
    Ok(())
}

fn active(phase: &str) -> bool {
    matches!(phase, "queued" | "downloading" | "validating")
}

pub(super) fn handle(
    server: &Arc<Server>,
    stream: &mut TcpStream,
    request: &Request,
) -> Result<()> {
    let hub = &server.hub;
    let route = (request.method.as_str(), request.path.as_str());
    let result = (|| -> Result<(u16, Value)> {
        match route {
            ("POST", "/api/hub/search") => {
                let input: Search = serde_json::from_slice(&request.body)?;
                let Ok(_guard) = hub.search.try_lock() else {
                    return Ok((409, error_value("A model search is already running")));
                };
                Ok((200, search(&input.query, destination(server)?, hub)?))
            }
            ("GET", "/api/hub/progress") => Ok((200, hub.snapshot())),
            ("POST", "/api/hub/cancel") => {
                let job = hub.job.lock().unwrap();
                if active(job.phase) {
                    hub.cancel.store(true, Ordering::Release);
                }
                drop(job);
                Ok((200, hub.snapshot()))
            }
            ("POST", "/api/hub/download") => {
                let selected: Selection = serde_json::from_slice(&request.body)?;
                ensure!(
                    valid_repo(&selected.repo)
                        && valid_sha(&selected.revision, 40)
                        && valid_file(&selected.file),
                    "Invalid pinned Hugging Face GGUF selection"
                );
                let dir = destination(server)?;
                let mut job = hub.job.lock().unwrap();
                if active(job.phase) {
                    return Ok((409, error_value("Another download is running")));
                }
                hub.cancel.store(false, Ordering::Release);
                *job = Job {
                    phase: "queued",
                    directory: dir.clone(),
                    repo: selected.repo.clone(),
                    file: selected.file.clone(),
                    ..Job::default()
                };
                let queued = json!(&*job);
                let worker = Arc::clone(server);
                if let Err(e) = std::thread::Builder::new()
                    .name("hub-download".into())
                    .spawn(move || {
                        if let Err(e) = install(&worker, &selected, &dir) {
                            let mut job = worker.hub.job.lock().unwrap();
                            job.phase = if worker.hub.cancel.load(Ordering::Acquire) {
                                "cancelled"
                            } else {
                                "failed"
                            };
                            job.error = Some(format!("{e:#}"));
                        }
                    })
                {
                    job.phase = "failed";
                    job.error = Some(e.to_string());
                    return Err(e.into());
                }
                Ok((202, queued))
            }
            _ => Ok((404, error_value("Unknown Hub route"))),
        }
    })();
    match result {
        Ok((status, data)) => json_response(stream, status, &data),
        Err(e) => json_response(stream, 400, &error_value(&format!("{e:#}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn free_space_check_includes_every_remaining_shard_and_headroom() {
        let reserve = 256 * 1024 * 1024;
        assert!(ensure_room(reserve + 1024, 1024).is_ok());
        assert!(ensure_room(reserve + 1023, 1024).is_err());
        assert!(ensure_room(reserve - 1, 0).is_err());
        assert!(ensure_room(u64::MAX, u64::MAX).is_err());
        assert!(available_disk_bytes(&std::env::temp_dir()).is_ok());
    }

    #[test]
    fn pinned_shards_and_candidate_filters() {
        let sha = "a".repeat(40);
        let hash = "b".repeat(64);
        let blob = |name: &str| json!({"rfilename":name,"size":100,"lfs":{"sha256":hash}});
        let mut meta = json!({"sha":sha,"gguf":{"architecture":"qwen3moe","context_length":8192},"siblings":[
            blob("Q4_K_M/model-Q4_K_M-00001-of-00002.gguf"),blob("Q4_K_M/model-Q4_K_M-00002-of-00002.gguf"),
            blob("model-Q4_K_M-00001-of-00003.gguf"),blob("model-Q2_K.gguf"),blob("NUL.gguf"),blob("../m-Q8_0.gguf")]});
        assert!(supported_name("model.gguf") && supported_name("model-Q3_K_M.gguf"));
        assert!(!supported_name("mmproj-model.gguf") && !supported_name("imatrix-model.gguf"));
        let list = choices("org/repo", &sha, &meta).unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].bytes, 200);
        assert_eq!(list[0].shards, 2);
        assert!(choices("org/repo", &"f".repeat(40), &meta).is_err());
        meta["gguf"]["architecture"] = json!("qwen3");
        assert_eq!(choices("org/repo", &sha, &meta).unwrap().len(), 2);
        meta["gguf"]["architecture"] = json!("unsupported");
        assert!(choices("org/repo", &sha, &meta).unwrap().is_empty());
        assert!(!valid_repo("../repo"));
        assert!(!valid_file("COM1.gguf"));
        assert!(!valid_file("a/../b.gguf"));
    }
}
