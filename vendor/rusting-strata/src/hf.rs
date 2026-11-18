//! Hugging Face GGUF discovery and immutable, verified model downloads.
//!
//! Forge9070 modification of RustingStrata by Vasyl Trefilov. Only model data is
//! downloaded: repository Python, custom loaders and remote code are never run.
//! `HF_TOKEN` is read at request time for private/gated repositories; it is never
//! stored in model provenance and is not forwarded to a different redirect host.

use crate::{serve::models, compatibility::{self, Policy, Report}};
use anyhow::{Context, bail, ensure};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Clone, Default, Serialize)]
pub struct Progress {
    pub repo: String,
    pub file: String,
    pub done: u64,
    pub total: u64,
    pub running: bool,
    pub cancel: bool,
    pub error: Option<String>,
    /// Completed first shard, relative to the model directory, with `/` separators.
    pub saved_path: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RepoSummary {
    pub id: String,
    pub downloads: u64,
    pub likes: u64,
    pub gated: bool,
    pub private: bool,
    pub tags: Vec<String>,
    /// At least this immutable-revision GGUF passed the sparse-MoE metadata policy.
    pub verified_file: String,
    pub compatibility: Report,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelFile {
    /// The single file or first shard to pass to `download_at`.
    pub file: String,
    /// Total bytes of every shard; 0 when the Hub omitted sizes.
    pub size: u64,
    pub shards: usize,
    /// Digest of the first file when the Hub supplies an LFS SHA-256.
    pub sha256: Option<String>,
    /// Present and accepted on public listings; None only in private raw metadata parsing.
    pub compatibility: Option<Report>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExcludedFile { pub file: String, pub reason: String }

#[derive(Clone, Debug, Serialize)]
pub struct RepoFiles {
    pub repo: String,
    pub revision: String,
    /// Immutable resolved repository commit.
    pub commit: String,
    pub models: Vec<ModelFile>,
    pub license: Option<String>,
    pub gated: bool,
    pub policy: String,
    pub excluded: Vec<ExcludedFile>,
}

#[derive(Clone, Debug)]
struct Blob {
    file: String,
    size: Option<u64>,
    sha256: Option<String>,
}

struct Snapshot {
    listing: RepoFiles,
    blobs: Vec<Blob>,
    advertised_architecture: Option<String>,
}

const PROBE_BUDGET: usize = 256 * 1024 * 1024;
const MAX_CHOICES: usize = 64;
static PROBE_CACHE: OnceLock<Mutex<BTreeMap<String, Report>>> = OnceLock::new();

struct ProbeBudget { remaining: usize, deadline: Instant }
impl ProbeBudget {
    fn new() -> Self { Self { remaining: PROBE_BUDGET, deadline: Instant::now() + Duration::from_secs(180) } }
    fn timeout(&self) -> anyhow::Result<Duration> {
        let left = self.deadline.saturating_duration_since(Instant::now());
        ensure!(!left.is_zero(), "compatibility verification time budget exhausted; use a more specific repository/query");
        ensure!(self.remaining > 0, "compatibility verification byte budget exhausted; inspect a smaller repository");
        Ok(left.min(Duration::from_secs(60)))
    }
}

struct Hub {
    base: String,
    token: Option<String>,
}

impl Hub {
    fn official() -> Self {
        Self {
            base: "https://huggingface.co".into(),
            token: std::env::var("HF_TOKEN").ok().filter(|s| !s.trim().is_empty()),
        }
    }

    fn get(&self, url: &str, whole: Option<Duration>) -> anyhow::Result<ureq::RequestBuilder<ureq::typestate::WithoutBody>> {
        let timeout = Some(Duration::from_secs(60));
        let mut config = ureq::Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(30)))
            .timeout_recv_response(timeout)
            .timeout_per_read(timeout)
            .timeout_global(whole)
            // Use the configured platform trust store (and SSL_CERT_FILE on
            // Unix), including explicitly installed enterprise proxy roots.
            // Certificate and hostname verification remain enabled.
            .tls_config(ureq::tls::TlsConfig::builder()
                .root_certs(ureq::tls::RootCerts::PlatformVerifier).build())
            // ureq's automatic order prefers ALL_PROXY over HTTPS_PROXY.
            // Match protocol-specific proxy expectations and retain NO_PROXY.
            .proxy(proxy_for_url(url, |key| std::env::var(key).ok())?)
            .redirect_auth_headers(ureq::config::RedirectAuthHeaders::SameHost);
        // This branch is only reached by the local, in-process mock server tests.
        if self.base.starts_with("http://127.0.0.1:") { config = config.proxy(None); }
        let agent: ureq::Agent = config.build().into();
        let request = agent.get(url).header("User-Agent", "Forge9070/0.3 RustingStrata");
        Ok(match &self.token {
            Some(token) => request.header("Authorization", format!("Bearer {token}")),
            None => request,
        })
    }

    fn json_timeout(&self, url: &str, timeout: Duration) -> anyhow::Result<serde_json::Value> {
        let mut response = self.get(url, Some(timeout))?.call()
            .context("Hugging Face request failed; check repository/revision and HF_TOKEN permissions for gated or private models")?;
        let body = response.body_mut().with_config().limit(16 * 1024 * 1024).read_to_string()?;
        serde_json::from_str(&body).context("invalid Hugging Face metadata JSON")
    }

    #[cfg(test)]
    fn snapshot(&self, repo: &str, revision: &str) -> anyhow::Result<Snapshot> {
        self.snapshot_timeout(repo, revision, Duration::from_secs(60))
    }

    fn snapshot_timeout(&self, repo: &str, revision: &str, timeout: Duration) -> anyhow::Result<Snapshot> {
        ensure!(valid_repo(repo), "{repo:?} is not a Hugging Face repository name (owner/name)");
        ensure!(!revision.is_empty() && revision.len() <= 256 && !revision.chars().any(char::is_control), "invalid Hugging Face revision");
        let url = format!("{}/api/models/{repo}/revision/{}?blobs=true", self.base, encode(revision));
        snapshot_from_json(repo, revision, &self.json_timeout(&url, timeout)?)
    }

    fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<RepoSummary>> {
        ensure!((1..=10).contains(&limit), "verified model search limit must be from 1 to 10");
        ensure!(query.len() <= 512 && !query.chars().any(char::is_control), "search query is too long or contains control characters");
        let mut budget = ProbeBudget::new();
        let candidates = (limit * 3).min(30);
        let url = format!("{}/api/models?search={}&filter=gguf&limit={candidates}&sort=downloads&direction=-1", self.base, encode(query));
        let data = self.json_timeout(&url, budget.timeout()?)?;
        let entries = data.as_array().context("Hugging Face search did not return a repository list")?;
        let mut results = Vec::new();
        for v in entries.iter().take(candidates) {
            if results.len() == limit || budget.timeout().is_err() { break; }
            let Some(id) = v["id"].as_str().or_else(|| v["modelId"].as_str()) else { continue };
            if !valid_repo(id) { continue; }
            let Ok(snapshot) = self.snapshot_timeout(id, "main", budget.timeout()?) else { continue };
            if snapshot.advertised_architecture.as_deref().is_some_and(advertised_incompatible) { continue; }
            let mut choices: Vec<_> = snapshot.listing.models.iter().collect();
            choices.sort_by_key(|m| preferred_choice(&m.file));
            let mut verified = None;
            for choice in choices.into_iter().take(3) {
                if budget.timeout().is_err() { break; }
                if let Ok(report) = self.probe_model(id, &snapshot, &choice.file, &mut budget, None) {
                    verified = Some((choice.file.clone(), report)); break;
                }
            }
            let Some((verified_file, compatibility)) = verified else { continue };
            results.push(RepoSummary {
                id: id.to_owned(), downloads: v["downloads"].as_u64().unwrap_or(0),
                likes: v["likes"].as_u64().unwrap_or(0), gated: gated(&v["gated"]),
                private: v["private"].as_bool().unwrap_or(false),
                tags: v["tags"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect()).unwrap_or_default(),
                verified_file, compatibility,
            });
        }
        if results.is_empty() { budget.timeout()?; }
        Ok(results)
    }

    fn list_compatible(&self, repo: &str, revision: &str) -> anyhow::Result<RepoFiles> {
        self.list_selected(repo, revision, None)
    }

    fn list_selected(&self, repo: &str, revision: &str, file: Option<&str>) -> anyhow::Result<RepoFiles> {
        if let Some(file) = file {
            ensure!(safe_relative_file(file), "selected file must be a valid relative .gguf path; raw .gguf.part files are not GGUF shards");
            ensure!(shard_parts(file).is_none_or(|(_, index, _)| index == 1),
                "select the first GGUF shard (-00001-of-xxxxx.gguf), not a later shard");
        }
        let mut budget = ProbeBudget::new();
        let mut snapshot = self.snapshot_timeout(repo, revision, budget.timeout()?)?;
        let mut candidates = std::mem::take(&mut snapshot.listing.models);
        if let Some(file) = file {
            candidates.retain(|model| model.file == file);
            ensure!(!candidates.is_empty(),
                "repository has no selectable GGUF model {file}; specify a single .gguf or the first shard of a complete GGUF shard group");
        }
        // Spend the bounded verification budget on practical quantized choices
        // before very large float/sharded variants or auxiliary files.
        candidates.sort_by_key(|model| preferred_choice(&model.file));
        let mut accepted = Vec::new();
        let mut excluded = Vec::new();
        for (index, mut candidate) in candidates.into_iter().enumerate() {
            let result = if index >= MAX_CHOICES { Err(anyhow::anyhow!("repository exceeds the 64-choice verification limit")) }
                else { self.probe_model(repo, &snapshot, &candidate.file, &mut budget, None) };
            match result {
                Ok(report) => { candidate.compatibility = Some(report); accepted.push(candidate); },
                Err(error) => excluded.push(ExcludedFile { file: candidate.file, reason: format!("{error:#}") }),
            }
        }
        snapshot.listing.models = accepted;
        snapshot.listing.excluded = excluded;
        snapshot.listing.policy = Policy::MoeOnly.name().into();
        Ok(snapshot.listing)
    }

    fn probe_model(&self, repo: &str, snapshot: &Snapshot, file: &str, budget: &mut ProbeBudget,
                   progress: Option<&Mutex<Progress>>) -> anyhow::Result<Report> {
        ensure!(!file.to_ascii_lowercase().contains("iq3_k"), "Avalon/ik_llama IQ3_K and IQ3_KS are unsupported");
        let chosen = select_blobs(&snapshot.blobs, file)?;
        ensure!(chosen.len() <= compatibility::MAX_SHARDS, "model exceeds the 64-shard verification limit");
        let key = format!("{}|{repo}|{}|{}", self.base, snapshot.listing.commit,
            chosen.iter().map(|b| format!("{}:{}:{}", b.file, b.size.unwrap_or(0), b.sha256.as_deref().unwrap_or(""))).collect::<Vec<_>>().join("|"));
        let cache = PROBE_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
        if let Some(report) = cache.lock().map_err(|_| anyhow::anyhow!("compatibility cache poisoned"))?.get(&key).cloned() { return Ok(report); }
        let mut headers = Vec::new();
        for blob in chosen {
            if let Some(progress) = progress { cancelled(progress)?; }
            headers.push(self.probe_header(repo, &snapshot.listing.commit, blob, budget, progress)?);
        }
        let report = compatibility::inspect_headers(&headers, file, Policy::MoeOnly)?;
        report.ensure_supported()?;
        let mut cache = cache.lock().map_err(|_| anyhow::anyhow!("compatibility cache poisoned"))?;
        if cache.len() >= 256 { cache.clear(); }
        cache.insert(key, report.clone());
        Ok(report)
    }

    fn probe_header(&self, repo: &str, commit: &str, blob: &Blob, budget: &mut ProbeBudget,
                    progress: Option<&Mutex<Progress>>) -> anyhow::Result<compatibility::Header> {
        let limit = compatibility::HEADER_LIMIT.min(budget.remaining);
        ensure!(limit >= 24, "compatibility verification byte budget exhausted");
        let path = blob.file.split('/').map(encode).collect::<Vec<_>>().join("/");
        let url = format!("{}/{repo}/resolve/{commit}/{path}?download=true", self.base);
        let response = self.get(&url, Some(budget.timeout()?))?
            .header("Accept-Encoding", "identity").header("Range", format!("bytes=0-{}", limit - 1)).call()
            .with_context(|| format!("verify GGUF header {}", blob.file))?;
        let total = if response.status() == 206 {
            let value = response.headers().get("Content-Range").and_then(|v| v.to_str().ok()).context("header probe requires Content-Range")?;
            validate_range(value, 0, blob.size)?;
            let (range, total) = value.strip_prefix("bytes ").and_then(|s| s.split_once('/')).context("invalid header Content-Range")?;
            let total: u64 = total.parse().context("header probe requires a known total size")?;
            let end: u64 = range.split_once('-').context("invalid header range")?.1.parse()?;
            ensure!(end < limit as u64 && end + 1 == total.min(limit as u64), "server returned an unexpected header range");
            total
        } else {
            ensure!(response.status() == 200, "unexpected header HTTP status {}", response.status());
            let length = response.headers().get("Content-Length").and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<u64>().ok());
            ensure!(length.is_some_and(|n| n <= limit as u64) && blob.size.is_none_or(|n| Some(n) == length),
                "server ignored Range for a large/unknown file; refusing an unbounded model download during compatibility checks");
            length.unwrap()
        };
        let mut reader = response.into_body().into_reader().take(limit as u64);
        let mut bytes = Vec::new();
        let mut chunk = vec![0u8; 1024 * 1024];
        loop {
            ensure!(bytes.len() < limit, "{}: GGUF header exceeds the bounded verification budget", blob.file);
            // TLS/HTTP readers may return one small record at a time. Reparse
            // only after a MiB (or EOF), not after every record: otherwise a
            // multi-MiB tokenizer header incurs quadratic rescanning work.
            let target = bytes.len().saturating_add(chunk.len()).min(limit);
            let mut eof = false;
            while bytes.len() < target {
                budget.timeout()?;
                if let Some(progress) = progress { cancelled(progress)?; }
                let read = reader.read(&mut chunk[..target - bytes.len()])?;
                if read == 0 { eof = true; break; }
                budget.remaining = budget.remaining.saturating_sub(read);
                bytes.extend_from_slice(&chunk[..read]);
            }
            if let Some(header) = compatibility::parse_prefix(&bytes, total)? { return Ok(header); }
            ensure!(!eof, "{}: truncated GGUF header response", blob.file);
        }
    }

    fn download_checked(&self, repo: &str, file: &str, revision: &str, dir: &Path, progress: &Mutex<Progress>) -> anyhow::Result<PathBuf> {
        ensure!(safe_relative_file(file), "invalid GGUF repository path: {file:?}");
        cancelled(progress)?;
        let mut budget = ProbeBudget::new();
        let snapshot = self.snapshot_timeout(repo, revision, budget.timeout()?)?;
        self.probe_model(repo, &snapshot, file, &mut budget, Some(progress))?;
        // The exact inspected snapshot goes to transport; a branch cannot move
        // between verification and download, while provenance keeps the user's revision.
        let result = self.download_snapshot(repo, file, revision, snapshot, dir, progress)?;
        compatibility::inspect_path(&result, Policy::MoeOnly)?.ensure_supported()
            .context("downloaded GGUF failed the sparse-MoE policy; it cannot be loaded")?;
        Ok(result)
    }

    #[cfg(test)]
    fn download(&self, repo: &str, file: &str, revision: &str, dir: &Path, progress: &Mutex<Progress>) -> anyhow::Result<PathBuf> {
        ensure!(safe_relative_file(file), "invalid GGUF repository path: {file:?}");
        cancelled(progress)?;
        let snapshot = self.snapshot(repo, revision)?;
        self.download_snapshot(repo, file, revision, snapshot, dir, progress)
    }

    fn download_snapshot(&self, repo: &str, file: &str, revision: &str, snapshot: Snapshot, dir: &Path, progress: &Mutex<Progress>) -> anyhow::Result<PathBuf> {
        let chosen = select_blobs(&snapshot.blobs, file)?;
        let first = chosen.first().context("no selected model shards")?;
        let relative_root = snapshot_relative(repo, &snapshot.listing.commit);
        std::fs::create_dir_all(dir).with_context(|| format!("create model directory {}", dir.display()))?;
        let root = std::fs::canonicalize(dir).context("resolve model directory")?;
        let snapshot_dir = make_directory(&root, &relative_root)?;
        let first_relative = Path::new(&first.file);
        let first_parent = make_directory(&snapshot_dir, first_relative.parent().unwrap_or_else(|| Path::new("")))?;
        let first_name = first_relative.file_name().context("missing model filename")?.to_string_lossy();
        let lock_path = first_parent.join(format!("{first_name}.download.lock"));
        regular_or_absent(&lock_path)?;
        let lock = OpenOptions::new().create(true).read(true).write(true).truncate(false).open(&lock_path)?;
        lock.try_lock().map_err(|e| anyhow::anyhow!("another download is using this model snapshot: {e}"))?;
        let total = chosen.iter().try_fold(0u64, |n, blob| n.checked_add(blob.size.unwrap_or(0))).context("model size overflow")?;
        update(progress, |p| { p.total = total; p.done = 0; })?;
        let mut done = 0;
        let mut downloaded = Vec::new();
        for blob in &chosen {
            cancelled(progress)?;
            let name = Path::new(&blob.file);
            let parent = make_directory(&snapshot_dir, name.parent().unwrap_or_else(|| Path::new("")))?;
            let filename = name.file_name().context("missing filename")?.to_string_lossy();
            let destination = parent.join(filename.as_ref());
            let partial = parent.join(format!("{filename}.part"));
            regular_or_absent(&destination)?;
            regular_or_absent(&partial)?;
            let actual_size = if destination.exists() {
                verify_file(&destination, blob).context("cached model is corrupt; move it aside and retry")?
            } else {
                self.download_blob(repo, &snapshot.listing.commit, blob, &partial, done, progress)?;
                let size = std::fs::metadata(&partial)?.len();
                // Only complete, verified data becomes visible to model discovery.
                std::fs::rename(&partial, &destination).context("publish verified GGUF file")?;
                size
            };
            done = done.checked_add(actual_size).context("download byte count overflow")?;
            update(progress, |p| p.done = done)?;
            downloaded.push(serde_json::json!({
                "file": blob.file, "bytes": actual_size, "sha256": blob.sha256,
                "sha256_verified": blob.sha256.is_some(),
            }));
        }
        let completed = snapshot_dir.join(&first.file);
        let relative = path_slashes(&relative_root.join(&first.file));
        let provenance = serde_json::json!({
            "schema_version": 1, "repo": repo, "requested_revision": revision,
            "commit": snapshot.listing.commit, "file": first.file, "model_id": relative,
            "license": snapshot.listing.license, "files": downloaded,
            "source": format!("https://huggingface.co/{repo}"),
            "remote_code_executed": false,
        });
        let manifest = first_parent.join(format!("{first_name}.forge9070.json"));
        regular_or_absent(&manifest)?;
        // A model-specific OS lock protects this sidecar and the partial files.
        let temp = first_parent.join(format!("{first_name}.forge9070.json.tmp"));
        regular_or_absent(&temp)?;
        std::fs::write(&temp, serde_json::to_vec_pretty(&provenance)?)?;
        if manifest.exists() { std::fs::remove_file(&manifest)?; }
        std::fs::rename(temp, manifest)?;
        update(progress, |p| p.saved_path = Some(relative))?;
        drop(lock);
        Ok(completed)
    }

    fn download_blob(&self, repo: &str, commit: &str, blob: &Blob, partial: &Path, before: u64, progress: &Mutex<Progress>) -> anyhow::Result<()> {
        let mut have = std::fs::metadata(partial).map_or(0, |m| m.len());
        if blob.size.is_some_and(|size| have > size) {
            OpenOptions::new().write(true).truncate(true).open(partial)?;
            have = 0;
        }
        if blob.size == Some(have) && have > 0 {
            if let Err(error) = verify_file(partial, blob) {
                std::fs::remove_file(partial)?;
                return Err(error.context("corrupt complete partial file removed; retry the download"));
            }
            return Ok(());
        }
        let url_path = blob.file.split('/').map(encode).collect::<Vec<_>>().join("/");
        let url = format!("{}/{repo}/resolve/{commit}/{url_path}?download=true", self.base);
        let mut request = self.get(&url, None)?.header("Accept-Encoding", "identity");
        if have > 0 { request = request.header("Range", format!("bytes={have}-")); }
        let response = request.call().with_context(|| format!("download {} (check HF_TOKEN if access is gated)", blob.file))?;
        let expected_size;
        if response.status() == 206 {
            let content_range = response.headers().get("Content-Range").and_then(|v| v.to_str().ok()).context("partial response has no valid Content-Range")?;
            validate_range(content_range, have, blob.size)?;
            let range_total = content_range.rsplit('/').next().and_then(|v| v.parse::<u64>().ok());
            expected_size = blob.size.or(range_total);
            ensure!(expected_size.is_some(), "cannot verify completion of a resumed response with unknown total size");
        } else {
            ensure!(response.status() == 200, "unexpected download HTTP status {}", response.status());
            // A server that ignored Range sent the whole object; discard the old prefix.
            have = 0;
            expected_size = blob.size.or_else(|| response.headers().get("Content-Length")
                .and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<u64>().ok()));
        }
        let mut digest = Sha256::new();
        if have > 0 {
            let mut old = File::open(partial)?;
            let copied = std::io::copy(&mut old, &mut DigestWriter(&mut digest))?;
            ensure!(copied == have, "partial file changed while preparing resume");
        }
        let mut out = OpenOptions::new().create(true).write(true).append(have > 0).truncate(have == 0).open(partial)?;
        let mut reader = response.into_body().into_reader();
        let mut buffer = vec![0; 1024 * 1024];
        loop {
            cancelled(progress)?;
            let count = reader.read(&mut buffer).with_context(|| format!("read {} (partial file retained for resume)", blob.file))?;
            if count == 0 { break; }
            let next = have.checked_add(count as u64).context("download byte count overflow")?;
            ensure!(expected_size.is_none_or(|size| next <= size), "{}: response is larger than the declared file size", blob.file);
            out.write_all(&buffer[..count]).context("write partial GGUF")?;
            digest.update(&buffer[..count]);
            have = next;
            update(progress, |p| p.done = before.saturating_add(have))?;
        }
        out.sync_all()?;
        drop(out);
        ensure!(expected_size.is_none_or(|size| have == size), "{}: received {have} bytes, expected {}; partial file retained for resume", blob.file, expected_size.unwrap_or(0));
        let actual_hash = hex(&digest.finalize());
        if blob.sha256.as_ref().is_some_and(|expected| expected != &actual_hash) {
            std::fs::remove_file(partial)?;
            bail!("{}: SHA-256 mismatch; corrupt partial file removed, retry the download", blob.file);
        }
        verify_gguf_magic(partial)?;
        Ok(())
    }
}

/// Search only repositories containing an actually metadata-verified sparse MoE GGUF.
/// No content/"uncensored" keyword filtering is applied. Verification has bounded
/// candidate, transfer and time budgets; use an exact repository to inspect more choices.
pub fn search(query: &str, limit: usize) -> anyhow::Result<Vec<RepoSummary>> { Hub::official().search(query, limit) }

pub fn list_at(repo: &str, revision: &str) -> anyhow::Result<RepoFiles> { Hub::official().list_compatible(repo, revision) }

/// Inspect one exact GGUF choice (including all its shards) without downloading
/// weights. Uses the same bounded header checks and immutable revision as listing.
/// A metadata rejection is reported in `excluded`; unknown choices are errors.
pub fn list_file_at(repo: &str, revision: &str, file: &str) -> anyhow::Result<RepoFiles> {
    Hub::official().list_selected(repo, revision, Some(file))
}

/// Original browser API retained for compatibility.
pub fn list(repo: &str) -> anyhow::Result<Vec<(String, u64)>> {
    Ok(list_at(repo, "main")?.models.into_iter().map(|m| (m.file, m.size)).collect())
}

/// Download one GGUF or its complete shard group at an immutable resolved commit.
/// Returns the first shard's absolute path. `saved_path` is its model-directory ID.
pub fn download_at(repo: &str, file: &str, revision: &str, dir: &Path, progress: &Mutex<Progress>) -> anyhow::Result<PathBuf> {
    update(progress, |p| {
        p.repo = repo.to_owned(); p.file = file.to_owned(); p.running = true;
        p.error = None; p.saved_path = None; p.done = 0; p.total = 0;
    })?;
    let result = Hub::official().download_checked(repo, file, revision, dir, progress);
    update(progress, |p| {
        p.running = false;
        p.error = result.as_ref().err().map(|e| format!("{e:#}"));
        if result.is_err() { p.saved_path = None; }
    })?;
    result
}

pub fn download(repo: &str, file: &str, dir: &Path, progress: &Mutex<Progress>) -> anyhow::Result<()> {
    download_at(repo, file, "main", dir, progress).map(|_| ())
}

/// A repository ID, never an arbitrary URL or local path.
pub fn valid_repo(repo: &str) -> bool {
    let valid = |part: &str| !part.is_empty() && part != "." && part != ".." && part.len() <= 128
        && part.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b));
    repo.split_once('/').is_some_and(|(owner, name)| valid(owner) && valid(name))
}

/// Compatibility helper: return only the basename of a fully validated relative GGUF path.
pub fn local_name(file: &str) -> Option<&str> { safe_relative_file(file).then(|| file.rsplit('/').next()).flatten() }

fn safe_component(component: &str) -> bool {
    if component.is_empty() || component.starts_with('.') || component.ends_with('.') || component.ends_with(' ')
        || component.chars().any(|c| c.is_control() || "\\:<>\"|?*".contains(c)) { return false; }
    let stem = component.split('.').next().unwrap_or("").to_ascii_uppercase();
    !matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        && !(stem.len() == 4 && (stem.starts_with("COM") || stem.starts_with("LPT")) && matches!(stem.as_bytes()[3], b'1'..=b'9'))
}

fn safe_relative_file(file: &str) -> bool {
    file.len() <= 1024 && file.ends_with(".gguf") && file.split('/').all(safe_component)
}

fn valid_commit(commit: &str) -> bool {
    matches!(commit.len(), 40 | 64) && commit.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Protocol-specific proxy configuration without modifying process-global
/// environment variables. Invalid configured proxies fail rather than silently
/// falling back to an unproxied connection. Values never enter error messages.
fn proxy_for_url(url: &str, lookup: impl Fn(&str) -> Option<String>) -> anyhow::Result<Option<ureq::Proxy>> {
    let keys: &[&str] = if url.starts_with("https://") {
        &["HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"]
    } else {
        &["HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"]
    };
    let Some((key, value)) = keys.iter().find_map(|&key| lookup(key).filter(|value| !value.trim().is_empty()).map(|value| (key, value))) else {
        return Ok(None);
    };
    let parsed = ureq::Proxy::new(&value).map_err(|_| anyhow::anyhow!("invalid configured proxy URI in {key}"))?;
    let mut builder = ureq::Proxy::builder(parsed.protocol()).host(parsed.host()).port(parsed.port()).resolve_target(parsed.resolve_target());
    if let Some(username) = parsed.username() { builder = builder.username(username); }
    if let Some(password) = parsed.password() { builder = builder.password(password); }
    if let Some(exclusions) = lookup("NO_PROXY").or_else(|| lookup("no_proxy")) {
        for expression in exclusions.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            builder = builder.no_proxy(expression);
        }
    }
    Ok(Some(builder.build().map_err(|_| anyhow::anyhow!("invalid configured proxy in {key}"))?))
}

fn gated(value: &serde_json::Value) -> bool {
    value.as_bool().unwrap_or_else(|| value.as_str().is_some_and(|s| s != "false" && !s.is_empty()))
}

fn preferred_choice(file: &str) -> u8 {
    let name = file.to_ascii_lowercase();
    // Probe priority only, never a quality certificate or automatic download.
    // Actual tensor types are reported after the bounded metadata check.
    if name.contains("q4_k_m") { 0 } else if name.contains("q4_k_s") { 1 }
    else if name.contains("q5_k_m") { 2 } else if name.contains("q5_k_s") { 3 }
    else if name.contains("iq4_xs") || name.contains("iq4_nl") { 4 }
    else if name.contains("q6_k") || name.contains("q8_0") { 5 }
    else { 6 }
}

/// A rejection-only hint. A positive hint is never sufficient for acceptance.
fn advertised_incompatible(architecture: &str) -> bool {
    !matches!(architecture, "qwen3moe" | "qwen3_moe" | "qwen3next" | "qwen3_next"
        | "qwen35moe" | "qwen3_5_moe" | "qwen4exp" | "qwen4_exp" | "flint")
}

fn snapshot_from_json(repo: &str, revision: &str, value: &serde_json::Value) -> anyhow::Result<Snapshot> {
    let commit = value["sha"].as_str().context("Hugging Face did not resolve the revision to a commit")?;
    ensure!(valid_commit(commit), "Hugging Face returned an invalid immutable commit ID");
    let mut unique = BTreeMap::new();
    for entry in value["siblings"].as_array().context("Hugging Face metadata has no file list")? {
        let Some(file) = entry["rfilename"].as_str() else { continue };
        if !file.ends_with(".gguf") { continue; }
        ensure!(safe_relative_file(file), "repository contains a GGUF path unsafe on Windows or Linux: {file:?}");
        let raw_hash = entry["lfs"]["sha256"].as_str().or_else(|| entry["lfs"]["oid"].as_str());
        let sha256 = raw_hash.map(|hash| hash.strip_prefix("sha256:").unwrap_or(hash).to_ascii_lowercase());
        if let Some(hash) = &sha256 {
            ensure!(hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()), "invalid LFS SHA-256 for {file}");
        }
        let blob = Blob { file: file.to_owned(), size: entry["size"].as_u64().or_else(|| entry["lfs"]["size"].as_u64()), sha256 };
        ensure!(unique.insert(file.to_owned(), blob).is_none(), "duplicate repository filename: {file}");
    }
    let blobs: Vec<_> = unique.into_values().collect();
    let plain = blobs.iter().map(|b| (b.file.clone(), b.size.unwrap_or(0))).collect::<Vec<_>>();
    let listed = models(&plain).into_iter().map(|(file, size)| {
        let selected = select_blobs(&blobs, &file)?;
        Ok(ModelFile { file, size, shards: selected.len(), sha256: selected.first().and_then(|b| b.sha256.clone()), compatibility: None })
    }).collect::<anyhow::Result<Vec<_>>>()?;
    Ok(Snapshot {
        listing: RepoFiles {
            repo: repo.to_owned(), revision: revision.to_owned(), commit: commit.to_ascii_lowercase(), models: listed,
            license: value["cardData"]["license"].as_str().map(str::to_owned), gated: gated(&value["gated"]),
            policy: "unverified".into(), excluded: Vec::new(),
        }, blobs, advertised_architecture: value["gguf"]["architecture"].as_str()
            .or_else(|| value["config"]["model_type"].as_str()).map(str::to_owned),
    })
}

fn shard_parts(file: &str) -> Option<(&str, usize, usize)> {
    let (head, count) = file.strip_suffix(".gguf")?.rsplit_once("-of-")?;
    let (stem, index) = head.rsplit_once('-')?;
    let digits = |s: &str| s.len() == 5 && s.bytes().all(|b| b.is_ascii_digit());
    if !digits(count) || !digits(index) { return None; }
    Some((stem, index.parse().ok()?, count.parse().ok()?))
}

fn select_blobs<'a>(blobs: &'a [Blob], requested: &str) -> anyhow::Result<Vec<&'a Blob>> {
    ensure!(blobs.iter().any(|b| b.file == requested), "repository has no GGUF file {requested}");
    if let Some((stem, index, count)) = shard_parts(requested) {
        ensure!(count > 0 && index > 0 && index <= count && count <= blobs.len(), "invalid or incomplete GGUF shard group");
        (1..=count).map(|part| {
            let name = format!("{stem}-{part:05}-of-{count:05}.gguf");
            blobs.iter().find(|b| b.file == name).with_context(|| format!("missing shard {name}"))
        }).collect()
    } else {
        Ok(vec![blobs.iter().find(|b| b.file == requested).unwrap()])
    }
}

fn snapshot_relative(repo: &str, commit: &str) -> PathBuf {
    let readable: String = repo.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).take(64).collect();
    let hash = hex(&Sha256::digest(repo.as_bytes()));
    PathBuf::from("hf").join(format!("{readable}-{}", &hash[..32])).join(commit)
}

fn make_directory(base: &Path, relative: &Path) -> anyhow::Result<PathBuf> {
    let mut path = base.to_path_buf();
    for part in relative.components() {
        let std::path::Component::Normal(part) = part else { bail!("invalid cache directory component") };
        path.push(part);
        match std::fs::symlink_metadata(&path) {
            Ok(info) => ensure!(info.is_dir() && !info.file_type().is_symlink(), "cache directory is not a plain directory: {}", path.display()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => std::fs::create_dir(&path).with_context(|| format!("create cache directory {}", path.display()))?,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(path)
}

fn regular_or_absent(path: &Path) -> anyhow::Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(info) => ensure!(info.is_file() && !info.file_type().is_symlink(), "cache path is not a regular file: {}", path.display()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn verify_gguf_magic(path: &Path) -> anyhow::Result<()> {
    let mut header = [0; 8];
    File::open(path)?.read_exact(&mut header).context("download is too short to be a GGUF")?;
    ensure!(&header[..4] == b"GGUF" && matches!(u32::from_le_bytes(header[4..].try_into().unwrap()), 2 | 3), "downloaded file is not GGUF v2/v3: {}", path.display());
    Ok(())
}

fn verify_file(path: &Path, blob: &Blob) -> anyhow::Result<u64> {
    let size = std::fs::metadata(path)?.len();
    ensure!(blob.size.is_none_or(|expected| size == expected), "{}: cached size {size} differs from expected {}", blob.file, blob.size.unwrap_or(0));
    verify_gguf_magic(path)?;
    if let Some(expected) = &blob.sha256 {
        let mut digest = Sha256::new();
        std::io::copy(&mut File::open(path)?, &mut DigestWriter(&mut digest))?;
        ensure!(&hex(&digest.finalize()) == expected, "{}: cached SHA-256 mismatch", blob.file);
    }
    Ok(size)
}

fn validate_range(header: &str, start: u64, size: Option<u64>) -> anyhow::Result<()> {
    let (span, total) = header.strip_prefix("bytes ").and_then(|s| s.split_once('/')).context("invalid Content-Range")?;
    let (from, to) = span.split_once('-').context("invalid Content-Range span")?;
    let from: u64 = from.parse().context("invalid range start")?;
    let to: u64 = to.parse().context("invalid range end")?;
    ensure!(from == start && to >= from, "server returned an inconsistent resume range");
    if let Some(size) = size {
        ensure!(total.parse::<u64>().ok() == Some(size) && to < size, "server returned an inconsistent total size");
    } else if let Ok(size) = total.parse::<u64>() {
        ensure!(to < size, "server returned a range outside the total size");
    }
    Ok(())
}

struct DigestWriter<'a>(&'a mut Sha256);
impl Write for DigestWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> { self.0.update(bytes); Ok(bytes.len()) }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

fn update(progress: &Mutex<Progress>, f: impl FnOnce(&mut Progress)) -> anyhow::Result<()> {
    let mut guard = progress.lock().map_err(|_| anyhow::anyhow!("download progress lock poisoned"))?;
    f(&mut guard);
    Ok(())
}
fn cancelled(progress: &Mutex<Progress>) -> anyhow::Result<()> {
    ensure!(!progress.lock().map_err(|_| anyhow::anyhow!("download progress lock poisoned"))?.cancel, "download cancelled; partial files kept for resume");
    Ok(())
}
fn path_slashes(path: &Path) -> String { path.components().map(|c| c.as_os_str().to_string_lossy()).collect::<Vec<_>>().join("/") }
fn hex(bytes: &[u8]) -> String { bytes.iter().map(|b| format!("{b:02x}")).collect() }
fn encode(s: &str) -> String {
    let mut result = String::new();
    for byte in s.bytes() {
        if byte.is_ascii_alphanumeric() || b"-._~".contains(&byte) { result.push(byte as char); }
        else { result.push_str(&format!("%{byte:02X}")); }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::{AtomicU64, Ordering};

    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("forge-hf-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
            std::fs::create_dir_all(&path).unwrap(); Self(path)
        }
    }
    impl Drop for Temp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
    fn gguf(payload: &[u8]) -> Vec<u8> { [b"GGUF".as_slice(), &3u32.to_le_bytes(), payload].concat() }
    fn metadata(file: &str, bytes: &[u8]) -> serde_json::Value {
        serde_json::json!({"sha": COMMIT, "siblings": [{"rfilename":file,"size":bytes.len(),"lfs":{"sha256":hex(&Sha256::digest(bytes)),"size":bytes.len()}}]})
    }
    fn request(stream: &mut TcpStream) -> String {
        stream.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let mut result = Vec::new(); let mut byte = [0];
        while !result.ends_with(b"\r\n\r\n") { stream.read_exact(&mut byte).unwrap(); result.push(byte[0]); }
        String::from_utf8(result).unwrap()
    }
    fn respond(mut stream: TcpStream, code: &str, headers: &[(&str, String)], body: &[u8]) {
        write!(stream, "HTTP/1.1 {code}\r\nContent-Length: {}\r\nConnection: close\r\n", body.len()).unwrap();
        for (key, value) in headers { write!(stream, "{key}: {value}\r\n").unwrap(); }
        stream.write_all(b"\r\n").unwrap(); stream.write_all(body).unwrap();
    }
    fn hub() -> (Hub, TcpListener) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        (Hub { base: format!("http://{}", listener.local_addr().unwrap()), token: Some("test-token".into()) }, listener)
    }

    fn fixture(moe: bool) -> Vec<u8> {
        let name = if moe { "tiny-qwen3moe.gguf" } else { "tiny-qwen3.gguf" };
        std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures").join(name)).unwrap()
    }

    fn respond_header(stream: TcpStream, bytes: &[u8]) {
        respond(stream, "206 Partial Content", &[("Content-Range", format!("bytes 0-{}/{}",bytes.len()-1,bytes.len()))], bytes);
    }

    #[test]
    fn listing_verifies_actual_headers_and_excludes_dense_and_unknown_quantization() {
        let dense = fixture(false); let good = fixture(true); let mut bad = good.clone();
        let marker = b"token_embd.weight";
        let at = bad.windows(marker.len()).position(|b| b == marker).unwrap() + marker.len();
        let dimensions = u32::from_le_bytes(bad[at..at+4].try_into().unwrap()) as usize;
        bad[at+4+dimensions*8..at+8+dimensions*8].copy_from_slice(&999u32.to_le_bytes());
        let mut meta = metadata("a-misleading-MoE-name.gguf", &dense);
        meta["siblings"].as_array_mut().unwrap().push(metadata("b-uncensored.gguf", &good)["siblings"][0].clone());
        meta["siblings"].as_array_mut().unwrap().push(metadata("c-unknown.gguf", &bad)["siblings"][0].clone());
        let (hub, listener) = hub();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap(); request(&mut stream);
            respond(stream,"200 OK",&[],&serde_json::to_vec(&meta).unwrap());
            for body in [dense, good, bad] {
                let (mut stream, _) = listener.accept().unwrap();
                let req = request(&mut stream);
                assert!(req.contains(&format!("/resolve/{COMMIT}/")));
                assert!(req.to_ascii_lowercase().contains("range: bytes=0-67108863"));
                respond_header(stream, &body);
            }
        });
        let listed = hub.list_compatible("owner/models", "main").unwrap();
        server.join().unwrap();
        assert_eq!(listed.models.len(), 1);
        assert_eq!(listed.models[0].file,"b-uncensored.gguf");
        assert!(listed.models[0].compatibility.as_ref().unwrap().supported_metadata);
        assert_eq!(listed.excluded.len(),2);
        assert!(listed.excluded.iter().any(|x| x.reason.contains("sparse MoE")));
        assert!(listed.excluded.iter().any(|x| x.reason.contains("quantization type 999")));
    }

    #[test]
    fn selected_listing_probes_only_the_requested_choice_at_the_resolved_commit() {
        let bytes = fixture(true);
        let mut meta = metadata("other-Q4_K_M.gguf", &fixture(false));
        meta["siblings"].as_array_mut().unwrap().push(metadata("nested/selected-IQ4_XS.gguf", &bytes)["siblings"][0].clone());
        let (hub, listener) = hub();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            assert!(request(&mut stream).contains("/revision/moving-branch?blobs=true"));
            respond(stream, "200 OK", &[], &serde_json::to_vec(&meta).unwrap());
            let (mut stream, _) = listener.accept().unwrap();
            let req = request(&mut stream);
            assert!(req.contains(&format!("/resolve/{COMMIT}/nested/selected-IQ4_XS.gguf")));
            assert!(req.to_ascii_lowercase().contains("range: bytes=0-"));
            respond_header(stream, &bytes);
            // Closing the listener also ensures neither another choice nor a
            // subsequent full-weight download can succeed unnoticed.
        });
        let listed = hub.list_selected("owner/models", "moving-branch", Some("nested/selected-IQ4_XS.gguf")).unwrap();
        server.join().unwrap();
        assert_eq!(listed.commit, COMMIT);
        assert_eq!(listed.revision, "moving-branch");
        assert_eq!(listed.models.len(), 1);
        assert_eq!(listed.models[0].file, "nested/selected-IQ4_XS.gguf");
        assert!(listed.models[0].compatibility.as_ref().unwrap().supported_metadata);
        assert!(listed.excluded.is_empty());
    }

    #[test]
    fn selected_listing_reports_incompatibility_without_accepting_the_filename() {
        let bytes = fixture(false); let meta = metadata("uncensored-MoE.gguf", &bytes);
        let (hub, listener) = hub();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap(); request(&mut stream);
            respond(stream, "200 OK", &[], &serde_json::to_vec(&meta).unwrap());
            let (mut stream, _) = listener.accept().unwrap(); request(&mut stream); respond_header(stream, &bytes);
        });
        let listed = hub.list_selected("owner/models", "main", Some("uncensored-MoE.gguf")).unwrap();
        server.join().unwrap();
        assert!(listed.models.is_empty());
        assert_eq!(listed.excluded.len(), 1);
        assert_eq!(listed.excluded[0].file, "uncensored-MoE.gguf");
        assert!(listed.excluded[0].reason.contains("sparse MoE"));
    }

    #[test]
    fn selected_listing_rejects_unknown_and_nonselectable_files() {
        let (hub, listener) = hub(); let meta = metadata("available.gguf", &fixture(true));
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap(); request(&mut stream);
            respond(stream, "200 OK", &[], &serde_json::to_vec(&meta).unwrap());
        });
        let error = hub.list_selected("owner/models", "main", Some("missing.gguf")).unwrap_err();
        server.join().unwrap();
        assert!(error.to_string().contains("no selectable GGUF model missing.gguf"));
        for file in ["../escape.gguf", "m.gguf.part1of3", "m-00002-of-00003.gguf"] {
            let error = hub.list_selected("owner/models", "main", Some(file)).unwrap_err();
            assert!(error.to_string().contains("relative .gguf path") || error.to_string().contains("first GGUF shard"), "{file}: {error}");
        }
    }

    #[test]
    fn checked_download_rejects_dense_before_creating_files() {
        let tmp = Temp::new(); let bytes = fixture(false); let meta = metadata("looks-MoE.gguf", &bytes);
        let (hub, listener) = hub();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap(); request(&mut stream);
            respond(stream,"200 OK",&[],&serde_json::to_vec(&meta).unwrap());
            let (mut stream, _) = listener.accept().unwrap(); request(&mut stream); respond_header(stream,&bytes);
        });
        let result = hub.download_checked("owner/dense","looks-MoE.gguf","main",&tmp.0,&Mutex::new(Progress::default()));
        server.join().unwrap();
        assert!(result.unwrap_err().to_string().contains("sparse MoE"));
        assert_eq!(std::fs::read_dir(&tmp.0).unwrap().count(),0);
    }

    #[test]
    fn checked_download_preserves_verified_commit_and_requested_revision() {
        let tmp = Temp::new(); let bytes = fixture(true); let meta = metadata("uncensored.gguf", &bytes);
        let (hub, listener) = hub();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            assert!(request(&mut stream).contains("/revision/moving-branch?blobs=true"));
            respond(stream,"200 OK",&[],&serde_json::to_vec(&meta).unwrap());
            let (mut stream, _) = listener.accept().unwrap();
            assert!(request(&mut stream).contains(&format!("/resolve/{COMMIT}/uncensored.gguf")));
            respond_header(stream,&bytes);
            let (mut stream, _) = listener.accept().unwrap();
            let req = request(&mut stream);
            assert!(req.contains(&format!("/resolve/{COMMIT}/uncensored.gguf")));
            assert!(!req.to_ascii_lowercase().contains("range:"));
            respond(stream,"200 OK",&[],&bytes);
        });
        let path = hub.download_checked("owner/moe","uncensored.gguf","moving-branch",&tmp.0,&Mutex::new(Progress::default())).unwrap();
        server.join().unwrap();
        compatibility::inspect_path(&path,Policy::MoeOnly).unwrap().ensure_supported().unwrap();
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(path.with_extension("gguf.forge9070.json")).unwrap()).unwrap();
        assert_eq!(manifest["commit"],COMMIT);
        assert_eq!(manifest["requested_revision"],"moving-branch");
    }

    #[test]
    fn header_probe_rejects_ignored_range_without_reading_large_body() {
        let (hub, listener) = hub();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap(); request(&mut stream);
            write!(stream,"HTTP/1.1 200 OK\r\nContent-Length: 134217728\r\nConnection: close\r\n\r\n").unwrap();
            // No body is sent. A broken probe that tried to read it would fail differently.
        });
        let blob = Blob { file:"m.gguf".into(), size:Some(134217728), sha256:None };
        let error = hub.probe_header("a/b",COMMIT,&blob,&mut ProbeBudget::new(),None).err().unwrap();
        server.join().unwrap();
        assert!(error.to_string().contains("ignored Range"));
    }

    #[test]
    fn search_returns_only_repositories_with_a_verified_sparse_variant() {
        let dense = fixture(false); let good = fixture(true);
        let mut dense_meta = metadata("m.gguf", &dense); dense_meta["gguf"]=serde_json::json!({"architecture":"qwen3"});
        let mut good_meta = metadata("m.gguf", &good); good_meta["gguf"]=serde_json::json!({"architecture":"qwen3moe"});
        let (hub, listener) = hub();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap(); request(&mut stream);
            respond(stream,"200 OK",&[],br#"[{"id":"owner/dense"},{"id":"owner/uncensored"}]"#);
            let (mut stream, _) = listener.accept().unwrap(); assert!(request(&mut stream).contains("owner/dense"));
            respond(stream,"200 OK",&[],&serde_json::to_vec(&dense_meta).unwrap());
            let (mut stream, _) = listener.accept().unwrap(); assert!(request(&mut stream).contains("owner/uncensored"));
            respond(stream,"200 OK",&[],&serde_json::to_vec(&good_meta).unwrap());
            let (mut stream, _) = listener.accept().unwrap(); request(&mut stream); respond_header(stream,&good);
        });
        let results = hub.search("Qwen",1).unwrap(); server.join().unwrap();
        assert_eq!(results.len(),1); assert_eq!(results[0].id,"owner/uncensored");
        assert_eq!(results[0].verified_file,"m.gguf");
        assert!(results[0].compatibility.supported_metadata);
    }

    #[test]
    fn qwen4_exp_search_hint_reaches_header_checks_without_bypassing_them() {
        let dense = fixture(false); let good = fixture(true);
        let mut dense_meta = metadata("m.gguf", &dense);
        dense_meta["config"] = serde_json::json!({"model_type":"qwen4_exp"});
        let mut good_meta = metadata("m.gguf", &good);
        good_meta["config"] = serde_json::json!({"model_type":"qwen4_exp"});
        let (hub, listener) = hub();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap(); request(&mut stream);
            respond(stream, "200 OK", &[], br#"[{"id":"owner/dense"},{"id":"owner/valid"}]"#);
            for (repo, meta, bytes) in [("owner/dense", dense_meta, dense), ("owner/valid", good_meta, good)] {
                let (mut stream, _) = listener.accept().unwrap();
                assert!(request(&mut stream).contains(&format!("/api/models/{repo}/revision/")));
                respond(stream, "200 OK", &[], &serde_json::to_vec(&meta).unwrap());
                let (mut stream, _) = listener.accept().unwrap();
                assert!(request(&mut stream).contains(&format!("/{repo}/resolve/{COMMIT}/m.gguf")));
                respond_header(stream, &bytes);
            }
        });
        let results = hub.search("Qwen3.8", 1).unwrap(); server.join().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "owner/valid");
        // The actual GGUF architecture remains authoritative even when the
        // repository's Transformers hint differs from its advertised family.
        assert_eq!(results[0].compatibility.architecture, "qwen3moe");
        assert!(results[0].compatibility.supported_metadata);
    }

    #[test]
    fn successful_immutable_probe_is_cached_without_weight_data() {
        let bytes = fixture(true); let meta = metadata("m.gguf",&bytes);
        let snapshot = snapshot_from_json("a/b","main",&meta).unwrap(); let (hub, listener) = hub();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap(); request(&mut stream); respond_header(stream,&bytes);
        });
        let first = hub.probe_model("a/b",&snapshot,"m.gguf",&mut ProbeBudget::new(),None).unwrap();
        server.join().unwrap();
        let second = hub.probe_model("a/b",&snapshot,"m.gguf",&mut ProbeBudget::new(),None).unwrap();
        assert_eq!(first.tensor_weight_bytes,second.tensor_weight_bytes);
        assert!(second.supported_metadata);
    }

    #[test]
    fn paths_reject_traversal_windows_devices_and_ambiguous_separators() {
        for path in ["../a.gguf", "/a.gguf", "x/../a.gguf", "x\\a.gguf", "C:a.gguf", "CON.gguf", "x/LPT1.gguf", "x//a.gguf", ".hidden/a.gguf", "x/a?.gguf"] {
            assert!(local_name(path).is_none(), "{path}");
        }
        assert_eq!(local_name("Q4_K_M/my model.gguf"), Some("my model.gguf"));
        assert!(valid_repo("some-owner/community.Qwen-GGUF"));
        assert!(!valid_repo("owner/repo/extra"));
    }

    #[test]
    fn https_proxy_precedes_all_proxy_and_preserves_no_proxy_and_auth() {
        let vars = BTreeMap::from([
            ("ALL_PROXY", "socks5h://general.invalid:1080"),
            ("HTTP_PROXY", "http://plain.invalid:80"),
            ("HTTPS_PROXY", "http://test-user:test%40password@tls.invalid:8080"),
            ("NO_PROXY", "localhost, .internal.invalid"),
        ]);
        let proxy = proxy_for_url("https://huggingface.co", |key| vars.get(key).map(|value| value.to_string())).unwrap().unwrap();
        assert_eq!(proxy.host(), "tls.invalid");
        assert_eq!(proxy.port(), 8080);
        assert_eq!(proxy.username(), Some("test-user"));
        assert_eq!(proxy.password(), Some("test@password"));
        assert!(proxy.is_no_proxy(&"http://localhost/test".parse().unwrap()));
        assert!(proxy.is_no_proxy(&"https://a.internal.invalid/test".parse().unwrap()));
        assert!(!proxy.is_no_proxy(&"https://huggingface.co/test".parse().unwrap()));
    }

    #[test]
    fn proxy_selector_supports_lowercase_fallback_and_never_ignores_invalid_proxy() {
        let all = |key: &str| (key == "all_proxy").then(|| "socks5h://general.invalid:1080".to_owned());
        assert_eq!(proxy_for_url("https://huggingface.co", all).unwrap().unwrap().protocol(), ureq::ProxyProtocol::Socks5h);
        assert!(proxy_for_url("https://huggingface.co", |_| None).unwrap().is_none());
        let error = proxy_for_url("https://huggingface.co", |key| (key == "HTTPS_PROXY").then(|| "invalid secret value".to_owned())).unwrap_err();
        assert!(error.to_string().contains("HTTPS_PROXY"));
        assert!(!error.to_string().contains("secret"));
    }

    #[test]
    fn repository_and_revision_namespaces_do_not_alias() {
        assert_ne!(snapshot_relative("a/b--c", COMMIT), snapshot_relative("a--b/c", COMMIT));
        assert_ne!(snapshot_relative("Owner/Model", COMMIT), snapshot_relative("owner/model", COMMIT));
        assert_ne!(snapshot_relative("a/b", COMMIT), snapshot_relative("a/b", &"f".repeat(40)));
    }

    #[test]
    fn metadata_requires_immutable_commit_and_complete_shard_groups() {
        let bytes = gguf(b"data");
        let mut value = metadata("m-00001-of-00002.gguf", &bytes);
        let snapshot = snapshot_from_json("a/b", "main", &value).unwrap();
        assert!(snapshot.listing.models.is_empty());
        assert!(select_blobs(&snapshot.blobs, "m-00001-of-00002.gguf").is_err());
        value["sha"] = "main".into();
        assert!(snapshot_from_json("a/b", "main", &value).is_err());
    }

    #[test]
    fn resume_ranges_are_checked_before_appending() {
        assert!(validate_range("bytes 8-15/16", 8, Some(16)).is_ok());
        for bad in ["bytes 7-15/16", "bytes 8-15/17", "bytes 8-16/16", "nonsense"] {
            assert!(validate_range(bad, 8, Some(16)).is_err());
        }
    }

    #[test]
    fn download_pins_commit_resumes_verifies_and_records_provenance() {
        let tmp = Temp::new(); let bytes = gguf(b"fixture-data");
        let data = bytes.clone(); let metadata = metadata("nested/model.gguf", &bytes);
        let (hub, listener) = hub();
        let parent = tmp.0.join(snapshot_relative("owner/model", COMMIT)).join("nested");
        std::fs::create_dir_all(&parent).unwrap();
        std::fs::write(parent.join("model.gguf.part"), &bytes[..10]).unwrap();
        let server = std::thread::spawn(move || {
            let (mut first, _) = listener.accept().unwrap();
            let req = request(&mut first);
            assert!(req.contains("/revision/main?blobs=true")); assert!(req.to_ascii_lowercase().contains("authorization: bearer test-token"));
            respond(first, "200 OK", &[], &serde_json::to_vec(&metadata).unwrap());
            let (mut second, _) = listener.accept().unwrap();
            let req = request(&mut second);
            assert!(req.contains(&format!("/resolve/{COMMIT}/nested/model.gguf?download=true")));
            assert!(req.to_ascii_lowercase().contains("range: bytes=10-"));
            respond(second, "206 Partial Content", &[("Content-Range", format!("bytes 10-{}/{}", data.len()-1,data.len()))], &data[10..]);
        });
        let progress = Mutex::new(Progress::default());
        let result = hub.download("owner/model", "nested/model.gguf", "main", &tmp.0, &progress).unwrap();
        server.join().unwrap();
        assert_eq!(std::fs::read(&result).unwrap(), bytes);
        assert_eq!(progress.lock().unwrap().done, bytes.len() as u64);
        assert!(progress.lock().unwrap().saved_path.as_ref().unwrap().ends_with("nested/model.gguf"));
        let provenance: serde_json::Value = serde_json::from_slice(&std::fs::read(parent.join("model.gguf.forge9070.json")).unwrap()).unwrap();
        assert_eq!(provenance["commit"], COMMIT); assert_eq!(provenance["files"][0]["sha256_verified"], true);
        assert!(!provenance.to_string().contains("test-token"));
    }

    #[test]
    fn ignored_range_restarts_instead_of_duplicating_prefix() {
        let tmp = Temp::new(); let bytes = gguf(b"fresh-whole-file"); let data = bytes.clone();
        let meta = metadata("m.gguf", &bytes); let (hub, listener) = hub();
        let directory = tmp.0.join(snapshot_relative("a/b", COMMIT)); std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("m.gguf.part"), &bytes[..8]).unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap(); request(&mut stream); respond(stream,"200 OK",&[],&serde_json::to_vec(&meta).unwrap());
            let (mut stream, _) = listener.accept().unwrap(); assert!(request(&mut stream).to_ascii_lowercase().contains("range:")); respond(stream,"200 OK",&[],&data);
        });
        let result = hub.download("a/b","m.gguf","main",&tmp.0,&Mutex::new(Progress::default())).unwrap();
        server.join().unwrap(); assert_eq!(std::fs::read(result).unwrap(),bytes);
    }

    #[test]
    fn checksum_mismatch_never_publishes_a_model() {
        let tmp = Temp::new(); let good = gguf(b"correct"); let bad = gguf(b"changed");
        let meta=metadata("m.gguf",&good); let (hub,listener)=hub();
        let server=std::thread::spawn(move||{
            let (mut stream,_)=listener.accept().unwrap(); request(&mut stream); respond(stream,"200 OK",&[],&serde_json::to_vec(&meta).unwrap());
            let (mut stream,_)=listener.accept().unwrap(); request(&mut stream); respond(stream,"200 OK",&[],&bad);
        });
        let result=hub.download("a/b","m.gguf","main",&tmp.0,&Mutex::new(Progress::default()));
        server.join().unwrap(); assert!(result.unwrap_err().to_string().contains("SHA-256 mismatch"));
        let directory=tmp.0.join(snapshot_relative("a/b",COMMIT));
        assert!(!directory.join("m.gguf").exists()); assert!(!directory.join("m.gguf.part").exists());
    }

    #[test]
    fn matching_size_cached_file_still_needs_correct_digest() {
        let tmp=Temp::new(); let good=gguf(b"correct"); let bad=gguf(b"changed");
        let meta=metadata("m.gguf",&good); let snapshot=snapshot_from_json("a/b","main",&meta).unwrap();
        let path=tmp.0.join("m.gguf"); std::fs::write(&path,bad).unwrap();
        assert!(verify_file(&path,&snapshot.blobs[0]).is_err());
    }

    #[test]
    fn download_fetches_every_shard_and_returns_the_first() {
        let tmp=Temp::new();
        let first=gguf(b"first shard"); let second=gguf(b"second shard");
        let mut meta=metadata("split/m-00001-of-00002.gguf",&first);
        meta["siblings"].as_array_mut().unwrap().push(metadata("split/m-00002-of-00002.gguf",&second)["siblings"][0].clone());
        let (hub,listener)=hub();
        let server=std::thread::spawn(move||{
            let (mut stream,_)=listener.accept().unwrap(); request(&mut stream); respond(stream,"200 OK",&[],&serde_json::to_vec(&meta).unwrap());
            for (index,data) in [(1,first),(2,second)] {
                let (mut stream,_)=listener.accept().unwrap();
                assert!(request(&mut stream).contains(&format!("m-{index:05}-of-00002.gguf")));
                respond(stream,"200 OK",&[],&data);
            }
        });
        let path=hub.download("a/b","split/m-00002-of-00002.gguf","main",&tmp.0,&Mutex::new(Progress::default())).unwrap();
        server.join().unwrap();
        assert!(path.ends_with("m-00001-of-00002.gguf"));
        assert!(path.with_file_name("m-00002-of-00002.gguf").is_file());
    }

    #[test]
    fn bearer_token_is_not_forwarded_to_another_redirect_host() {
        let (hub,first)=hub();
        let second=TcpListener::bind("127.0.0.1:0").unwrap();
        let target=format!("http://localhost:{}/metadata",second.local_addr().unwrap().port());
        let server=std::thread::spawn(move||{
            let (mut stream,_)=first.accept().unwrap();
            assert!(request(&mut stream).to_ascii_lowercase().contains("authorization: bearer test-token"));
            respond(stream,"302 Found",&[("Location",target)],&[]);
            let (mut stream,_)=second.accept().unwrap();
            assert!(!request(&mut stream).to_ascii_lowercase().contains("authorization:"));
            respond(stream,"200 OK",&[],&serde_json::to_vec(&metadata("m.gguf",&gguf(b"fixture"))).unwrap());
        });
        let result=hub.snapshot("a/b","main").unwrap();
        server.join().unwrap(); assert_eq!(result.listing.commit,COMMIT);
    }

    #[cfg(unix)]
    #[test]
    fn cache_subdirectory_symlinks_cannot_escape_root() {
        let tmp=Temp::new(); let other=Temp::new();
        std::os::unix::fs::symlink(&other.0,tmp.0.join("hf")).unwrap();
        assert!(make_directory(&tmp.0,Path::new("hf/a")).is_err());
    }
}
