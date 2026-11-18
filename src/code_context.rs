//! Deterministic, bounded repository excerpts for a local model.
//!
//! Retrieval uses lexical terms, paths and declaration-shaped lines. It does
//! not provide embeddings, semantic indexing, dependency resolution or a
//! claim that a model has read the entire repository. Git supplies the file
//! inventory; repository programs, hooks, filters and build tools are not run.

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, Stdio};

const MAX_QUERY_BYTES: usize = 16 * 1024;
const MAX_PATH_BYTES: usize = 16 * 1024;
const MAX_INVENTORY_BYTES: usize = 64 * 1024 * 1024;
const MAX_LINES_PER_FILE: usize = 20_000;
const MAX_RETAINED_EXCERPT_BYTES: usize = 16 * 1024;
const RANKING: &str = "lexical terms, paths and declaration heuristics; no embeddings or semantic index";
const FOOTER: &str = "\nOnly the paths and numbered source lines above are included. Unlisted files and lines are outside this context.\n";

#[derive(Clone, Debug)]
pub struct Options {
    /// A Git worktree or a subdirectory. Inventory and includes are scoped here.
    pub root: PathBuf,
    pub query: String,
    /// Relative files or directory prefixes to prioritize, without overriding
    /// exclusions or the tracked-only default.
    pub include_paths: Vec<PathBuf>,
    pub include_untracked: bool,
    pub max_output_bytes: usize,
    pub max_tokens: Option<usize>,
    pub max_file_bytes: usize,
    pub max_scan_bytes: u64,
    pub max_inventory_files: usize,
    pub max_files: usize,
    pub max_excerpt_lines: usize,
    pub context_lines: usize,
    pub max_line_bytes: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            root: PathBuf::from("."), query: String::new(), include_paths: Vec::new(),
            include_untracked: false, max_output_bytes: 128 * 1024, max_tokens: None,
            max_file_bytes: 512 * 1024, max_scan_bytes: 128 * 1024 * 1024,
            max_inventory_files: 200_000, max_files: 24, max_excerpt_lines: 80,
            context_lines: 4, max_line_bytes: 4096,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ContextStats {
    /// NUL-delimited records seen from Git, before deduplication/exclusions.
    pub inventory_paths: usize,
    pub eligible_paths: usize,
    pub files_scanned: usize,
    pub bytes_scanned: u64,
    pub files_selected: usize,
    pub excerpts_selected: usize,
    pub lines_selected: usize,
    pub files_not_scanned: usize,
    pub files_omitted: usize,
    pub excluded_paths: usize,
    pub excluded_secrets: usize,
    pub excluded_generated: usize,
    pub excluded_binaries: usize,
    pub excluded_unsafe_paths: usize,
    pub unreadable_files: usize,
    pub changed_files: usize,
    pub truncated_files: usize,
    pub omitted_long_lines: usize,
    pub inventory_truncated: bool,
    pub output_limited: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct SelectedExcerpt {
    /// Relative to Options.root, with Git's forward-slash separators.
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub score: u64,
    pub file_truncated: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ContextReport {
    pub context: String,
    pub selected: Vec<SelectedExcerpt>,
    pub stats: ContextStats,
    pub warnings: Vec<String>,
    pub bytes: usize,
    /// Exact count of context, including query, headings, labels and omissions.
    /// None means no tokenizer was supplied; no byte-to-token estimate is used.
    pub tokens: Option<usize>,
    pub ranking: String,
}

type TokenCounter<'a> = Option<&'a dyn Fn(&str) -> Result<usize>>;

/// Build a pack without retaining whole-repository contents. An optional model
/// tokenizer counts each complete rendered trial, never individual snippets.
/// If a token cap is requested, an exact counter is mandatory.
pub fn build(options: &Options, count_tokens: TokenCounter<'_>) -> Result<ContextReport> {
    validate_options(options, count_tokens.is_some())?;
    let root = options.root.canonicalize().with_context(|| format!("resolve repository directory {}", options.root.display()))?;
    ensure!(root.is_dir(), "--repo must be a directory");
    let includes = normalize_includes(&root, &options.include_paths)?;
    let terms = query_terms(&options.query);
    ensure!(!terms.is_empty() || !includes.is_empty(), "query has no searchable terms; name a symbol, error, path or --include");
    let mut stats = ContextStats::default();
    let paths = inventory(&root, options, &mut stats)?;
    stats.eligible_paths = paths.len();
    let mut warnings = Vec::new();
    if includes.iter().any(|p| !paths.iter().any(|candidate| include_matches(p, candidate))) {
        warnings.push("Some include paths had no eligible inventoried files; includes do not override exclusions or include untracked files automatically.".into());
    }
    let mut ordered: Vec<_> = paths.into_iter().map(|path| {
        let score = path_score(&path, &terms, &includes);
        let priority = score.saturating_add(manifest_priority(&path));
        (Reverse(priority), path, score)
    }).collect();
    ordered.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let retain = options.max_files.saturating_mul(4).clamp(8, 256);
    let mut candidates: BTreeMap<(Reverse<u64>, String), Candidate> = BTreeMap::new();
    for (index, (_, path, path_score)) in ordered.iter().enumerate() {
        if stats.bytes_scanned >= options.max_scan_bytes {
            stats.files_not_scanned = ordered.len() - index;
            break;
        }
        if let Some(candidate) = inspect_file(&root, path, *path_score, &terms, options, &mut stats)? {
            let key = (Reverse(candidate.score), candidate.path.clone());
            candidates.insert(key, candidate);
            if candidates.len() > retain { candidates.pop_last(); }
        }
    }
    if stats.inventory_truncated { warnings.push("Git inventory reached its configured path or byte limit; additional files were not inventoried.".into()); }
    if stats.files_not_scanned > 0 { warnings.push("The scan byte budget stopped further file reads; lexical path priority determined which files were inspected first.".into()); }
    if stats.truncated_files > 0 { warnings.push("Some files were inspected only as bounded prefixes; their unread tails could contain additional matches.".into()); }
    if stats.changed_files > 0 { warnings.push("Files that changed while being read were omitted.".into()); }
    if candidates.is_empty() { warnings.push("No readable matching excerpts were found within the inventory, exclusions and scan limits.".into()); }

    let header = render_header(options, &stats)?;
    let mut body = String::new();
    let mut context = format!("{header}{FOOTER}");
    let tokens = exact_count(&context, count_tokens)?;
    ensure!(fits(&context, tokens, options), "the query and coverage header exceed the context budget; shorten the query or raise the byte/token budget");
    let mut selected = Vec::new();
    let matching_candidates = candidates.len();
    for candidate in candidates.values() {
        if stats.files_selected == options.max_files { stats.output_limited = true; break; }
        let full = candidate.render(candidate.lines.len());
        let trial = format!("{header}{body}{}{FOOTER}", full.0);
        let trial_tokens = exact_count_if_bytes_fit(&trial, options, count_tokens)?;
        let mut accepted = if fits(&trial, trial_tokens, options) { Some((full, trial, trial_tokens)) } else { None };
        if accepted.is_none() {
            stats.output_limited = true;
            // Try progressively sized sets of high-priority complete lines.
            // Every accepted trial is checked exactly; no monotonic-token
            // assumption can make the returned pack exceed its token cap.
            let (mut lo, mut hi) = (1, candidate.lines.len().saturating_sub(1));
            while lo <= hi {
                let middle = lo + (hi - lo) / 2;
                let rendered = candidate.render(middle);
                let trial = format!("{header}{body}{}{FOOTER}", rendered.0);
                let trial_tokens = exact_count_if_bytes_fit(&trial, options, count_tokens)?;
                if fits(&trial, trial_tokens, options) {
                    accepted = Some((rendered, trial, trial_tokens));
                    lo = middle + 1;
                } else { hi = middle.saturating_sub(1); }
            }
        }
        if let Some(((block, spans, lines), trial, _trial_tokens)) = accepted {
            body.push_str(&block);
            selected.extend(spans);
            stats.lines_selected += lines;
            stats.files_selected += 1;
            context = trial;
        }
    }
    stats.excerpts_selected = selected.len();
    stats.files_omitted = stats.eligible_paths.saturating_sub(stats.files_selected);
    if stats.files_selected < matching_candidates { stats.output_limited = true; }
    if stats.output_limited { warnings.push("Output limits omitted matching files or lines; selected excerpts remain complete source lines with explicit gaps.".into()); }
    // Count once more over the exact string returned to the caller. A callback
    // with side effects or a changing tokenizer must not invalidate the cap.
    let tokens = exact_count(&context, count_tokens)?;
    ensure!(fits(&context, tokens, options), "the exact tokenizer count changed during packing; no over-budget pack was returned");
    Ok(ContextReport { bytes: context.len(), context, selected, stats, warnings, tokens, ranking: RANKING.into() })
}

fn validate_options(o: &Options, has_tokenizer: bool) -> Result<()> {
    ensure!(!o.query.trim().is_empty() && o.query.len() <= MAX_QUERY_BYTES, "query must be nonempty and at most 16 KiB");
    ensure!((1024..=8 * 1024 * 1024).contains(&o.max_output_bytes), "context byte budget must be 1 KiB to 8 MiB");
    ensure!((1..=8 * 1024 * 1024).contains(&o.max_file_bytes), "per-file read limit must be 1 byte to 8 MiB");
    ensure!((1..=2 * 1024 * 1024 * 1024).contains(&o.max_scan_bytes), "scan byte budget must be 1 byte to 2 GiB");
    ensure!((1..=1_000_000).contains(&o.max_inventory_files), "inventory limit must be 1 to 1,000,000 eligible files");
    ensure!((1..=256).contains(&o.max_files), "output file limit must be 1 to 256");
    ensure!((1..=1000).contains(&o.max_excerpt_lines) && o.context_lines <= 100, "excerpt lines must be 1 to 1000, surrounding lines at most 100");
    ensure!((16..=65_536).contains(&o.max_line_bytes), "readable line limit must be 16 to 65536 bytes");
    if let Some(cap) = o.max_tokens {
        ensure!(cap > 0, "token budget must be positive");
        ensure!(has_tokenizer, "a token budget requires the selected model's exact tokenizer; bytes are not tokens");
    }
    Ok(())
}

fn exact_count(text: &str, count: TokenCounter<'_>) -> Result<Option<usize>> {
    count.map(|f| f(text)).transpose()
}

fn exact_count_if_bytes_fit(text: &str, options: &Options, count: TokenCounter<'_>) -> Result<Option<usize>> {
    if text.len() > options.max_output_bytes { Ok(None) } else { exact_count(text, count) }
}

fn fits(text: &str, tokens: Option<usize>, options: &Options) -> bool {
    text.len() <= options.max_output_bytes && options.max_tokens.is_none_or(|cap| tokens.is_some_and(|n| n <= cap))
}

fn normalize_includes(root: &Path, paths: &[PathBuf]) -> Result<Vec<String>> {
    let mut out = BTreeSet::new();
    for path in paths {
        let canonical;
        let relative = if path.is_absolute() {
            canonical = path.canonicalize().context("resolve absolute --include path")?;
            canonical.strip_prefix(root).context("--include must stay inside --repo")?
        } else { path.as_path() };
        let mut parts = Vec::new();
        for part in relative.components() {
            match part {
                Component::Normal(part) => parts.push(part.to_str().context("--include must be UTF-8")?),
                Component::CurDir => {}
                _ => bail!("--include cannot contain parent traversal, a drive prefix or an outside path"),
            }
        }
        let value = parts.join("/");
        ensure!(value.is_empty() || valid_relative_path(&value), "unsafe --include path");
        out.insert(value);
    }
    Ok(out.into_iter().collect())
}

fn include_matches(include: &str, path: &str) -> bool {
    include.is_empty() || path == include || path.strip_prefix(include).is_some_and(|s| s.starts_with('/'))
}

fn valid_relative_path(path: &str) -> bool {
    !path.is_empty() && path.len() <= MAX_PATH_BYTES && !path.starts_with('/')
        && !path.contains(['\\', ':']) && !path.chars().any(char::is_control)
        && path.split('/').all(|part| !part.is_empty() && part != "." && part != "..")
}

struct RunningGit(Child);
impl Drop for RunningGit {
    fn drop(&mut self) { let _ = self.0.kill(); let _ = self.0.wait(); }
}

fn inventory(root: &Path, options: &Options, stats: &mut ContextStats) -> Result<Vec<String>> {
    let mut command = Command::new("git");
    command.current_dir(root)
        .args(["-c", "core.fsmonitor=false", "-c", "core.untrackedCache=false", "ls-files", "-z", "--cached"])
        .env("GIT_OPTIONAL_LOCKS", "0").env("GIT_TERMINAL_PROMPT", "0")
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
    for key in ["GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES"] {
        command.env_remove(key);
    }
    if options.include_untracked { command.args(["--others", "--exclude-standard"]); }
    // An explicit current-directory pathspec scopes monorepo subdirectory use.
    command.args(["--", "."]);
    let mut child = RunningGit(command.spawn().context("start git ls-files; Git must be installed and --repo must be a Git worktree")?);
    let mut stdout = child.0.stdout.take().context("Git inventory stdout unavailable")?;
    let mut names = BTreeSet::new();
    let (mut bytes, mut pending, mut overlong) = (0usize, Vec::new(), false);
    let mut buffer = [0u8; 8192];
    'read: loop {
        let n = stdout.read(&mut buffer).context("read Git inventory")?;
        if n == 0 { break; }
        for &byte in &buffer[..n] {
            bytes += 1;
            if bytes > MAX_INVENTORY_BYTES { stats.inventory_truncated = true; break 'read; }
            if byte != 0 {
                if pending.len() < MAX_PATH_BYTES { pending.push(byte); } else { overlong = true; }
                continue;
            }
            stats.inventory_paths += 1;
            if overlong {
                stats.excluded_unsafe_paths += 1;
            } else if let Ok(name) = std::str::from_utf8(&pending) {
                if !valid_relative_path(name) { stats.excluded_unsafe_paths += 1; }
                else if let Some(reason) = exclusion(name) {
                    stats.excluded_paths += 1;
                    match reason {
                        Exclusion::Secret => stats.excluded_secrets += 1,
                        Exclusion::Generated => stats.excluded_generated += 1,
                        Exclusion::Binary => stats.excluded_binaries += 1,
                    }
                } else {
                    if !names.contains(name) && names.len() == options.max_inventory_files {
                        stats.inventory_truncated = true;
                        break 'read;
                    }
                    names.insert(name.to_owned());
                }
            } else { stats.excluded_unsafe_paths += 1; }
            pending.clear();
            overlong = false;
        }
    }
    drop(stdout);
    if stats.inventory_truncated {
        let _ = child.0.kill();
    } else {
        ensure!(pending.is_empty(), "Git returned an incomplete path record");
        let status = child.0.wait().context("wait for Git inventory")?;
        ensure!(status.success(), "git ls-files failed ({status}); --repo must be a readable Git worktree or subdirectory");
    }
    Ok(names.into_iter().collect())
}

#[derive(Clone, Copy)]
enum Exclusion { Secret, Generated, Binary }

fn exclusion(path: &str) -> Option<Exclusion> {
    let lower = path.to_lowercase();
    let parts: Vec<_> = lower.split('/').collect();
    let name = *parts.last()?;
    if parts.iter().any(|p| p.starts_with(".env") || matches!(*p, ".ssh" | ".aws" | ".azure" | ".gnupg" | ".kube" | ".secrets"))
        || matches!(name, ".netrc" | ".npmrc" | ".pypirc" | ".pgpass" | ".git-credentials" | "credentials" | "secret" | "secrets" | "kubeconfig" | "keys.json" | "api_keys.json" | "tokens.json" | "token.json" | "private_key")
        || ["credentials.", "credential.", "secrets.", "secret.", "private_key.", "private-key.", "client_secret", "service-account", "service_account", "serviceaccount", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"].iter().any(|p| name.starts_with(p))
        || [".pem", ".key", ".p12", ".pfx", ".jks", ".keystore", ".keytab", ".tfvars", ".tfstate"].iter().any(|s| name.ends_with(s))
        || name.contains(".tfstate.") {
        return Some(Exclusion::Secret);
    }
    if parts.iter().any(|p| matches!(*p, ".git" | ".hg" | ".svn" | "node_modules" | "vendor" | "vendors" | "third_party" | "third-party" | "thirdparty" | "target" | "dist" | "build" | "bin" | "obj" | "out" | "coverage" | "generated" | "__pycache__" | ".venv" | "venv" | ".next" | ".nuxt" | ".svelte-kit" | ".gradle" | ".idea" | ".cache" | ".mypy_cache" | ".pytest_cache" | ".ruff_cache" | ".tox"))
        || matches!(name, "cargo.lock" | "package-lock.json" | "pnpm-lock.yaml" | "yarn.lock" | "poetry.lock" | "uv.lock" | "pdm.lock" | "gemfile.lock" | "composer.lock" | "go.sum")
        || [".min.js", ".min.css", ".map", ".pb.go", ".pb.rs", ".pb.h", ".pb.cc", ".g.cs", ".snap"].iter().any(|s| name.ends_with(s))
        || name.contains(".generated.") {
        return Some(Exclusion::Generated);
    }
    if [".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz", ".xz", ".zst", ".7z", ".tar", ".mp3", ".mp4", ".mov", ".woff", ".woff2", ".ttf", ".otf", ".exe", ".dll", ".so", ".dylib", ".a", ".o", ".obj", ".class", ".jar", ".pyc", ".wasm", ".gguf", ".safetensors", ".pt", ".bin", ".db", ".sqlite", ".sqlite3"].iter().any(|s| name.ends_with(s)) {
        return Some(Exclusion::Binary);
    }
    None
}

fn query_terms(query: &str) -> Vec<String> {
    let stop = ["a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does", "for", "from", "how", "i", "in", "into", "is", "it", "me", "of", "on", "or", "please", "the", "this", "to", "use", "using", "want", "we", "what", "where", "which", "with", "would", "you"];
    let mut terms = BTreeSet::new();
    for raw in query.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if raw.is_empty() { continue; }
        let lower = raw.to_lowercase();
        if !stop.contains(&lower.as_str()) { terms.insert(lower); }
        let mut part = String::new();
        let mut previous_lower = false;
        for ch in raw.chars() {
            if ch == '_' || (ch.is_uppercase() && previous_lower) {
                if !part.is_empty() && !stop.contains(&part.as_str()) { terms.insert(std::mem::take(&mut part)); }
            }
            if ch != '_' { part.extend(ch.to_lowercase()); }
            previous_lower = ch.is_lowercase() || ch.is_numeric();
        }
        if !part.is_empty() && !stop.contains(&part.as_str()) { terms.insert(part); }
    }
    terms.into_iter().take(64).collect()
}

fn term_score(lower: &str, term: &str) -> u64 {
    lower.match_indices(term).take(8).map(|(at, _)| {
        let word = |c: char| c.is_alphanumeric() || c == '_';
        let before = lower[..at].chars().next_back().is_some_and(word);
        let after = lower[at + term.len()..].chars().next().is_some_and(word);
        if !before && !after { 8 } else if term.chars().count() >= 3 { 2 } else { 0 }
    }).sum()
}

fn path_score(path: &str, terms: &[String], includes: &[String]) -> u64 {
    let lower = path.to_lowercase();
    let lexical: u64 = terms.iter().map(|term| term_score(&lower, term) * 6).sum();
    lexical + if includes.iter().any(|include| include_matches(include, path)) { 1_000_000 } else { 0 }
}

fn manifest_priority(path: &str) -> u64 {
    let name = path.rsplit('/').next().unwrap_or(path).to_lowercase();
    if name.starts_with("readme") { 4 }
    else if matches!(name.as_str(), "cargo.toml" | "package.json" | "pyproject.toml" | "go.mod" | "cmakelists.txt") { 3 }
    else { 0 }
}

fn declaration(line: &str) -> bool {
    let s = line.trim_start();
    ["fn ", "pub fn ", "pub async fn ", "async fn ", "def ", "async def ", "class ", "struct ", "pub struct ", "enum ", "pub enum ", "trait ", "impl ", "interface ", "type ", "func ", "function ", "export function ", "export class ", "const "].iter().any(|p| s.starts_with(p))
}

#[derive(Clone)]
struct KeptLine { number: usize, text: String, priority: usize }
struct Candidate { path: String, score: u64, lines: Vec<KeptLine>, prefix_lines: usize, total_lines: Option<usize>, file_truncated: bool }

impl Candidate {
    fn render(&self, line_limit: usize) -> (String, Vec<SelectedExcerpt>, usize) {
        let mut chosen: Vec<_> = self.lines.iter().collect();
        chosen.sort_by_key(|line| line.priority);
        chosen.truncate(line_limit);
        chosen.sort_by_key(|line| line.number);
        let mut text = format!("\n--- FILE {} (lexical score {}) ---\n", serde_json::to_string(&self.path).unwrap(), self.score);
        let mut spans = Vec::new();
        let mut previous = 0usize;
        for line in &chosen {
            if line.number > previous + 1 {
                text.push_str(&format!("[source lines {}-{} omitted]\n", previous + 1, line.number - 1));
            }
            text.push_str(&format!("L{} | {}\n", line.number, line.text));
            if previous + 1 == line.number && !spans.is_empty() {
                spans.last_mut().map(|s: &mut SelectedExcerpt| s.end_line = line.number);
            } else {
                spans.push(SelectedExcerpt { path: self.path.clone(), start_line: line.number, end_line: line.number,
                    score: self.score, file_truncated: self.file_truncated });
            }
            previous = line.number;
        }
        match self.total_lines {
            Some(total) if total > previous => text.push_str(&format!("[source lines {}-{total} omitted]\n", previous + 1)),
            None => {
                if self.prefix_lines > previous { text.push_str(&format!("[source lines {}-{} omitted]\n", previous + 1, self.prefix_lines)); }
                text.push_str("[unread file tail omitted; file scan was capped]\n");
            }
            _ => {}
        }
        text.push_str("--- END FILE ---\n");
        (text, spans, chosen.len())
    }
}

fn inspect_file(root: &Path, relative: &str, path_score: u64, terms: &[String], options: &Options, stats: &mut ContextStats)
    -> Result<Option<Candidate>> {
    let file = match verified_file(root, relative) {
        Ok(Some(file)) => file,
        Ok(None) => { stats.excluded_unsafe_paths += 1; return Ok(None); }
        Err(_) => { stats.unreadable_files += 1; return Ok(None); }
    };
    let before = file.metadata()?;
    let limit = (options.max_scan_bytes - stats.bytes_scanned).min(options.max_file_bytes as u64);
    let mut bytes = Vec::new();
    if (&file).take(limit).read_to_end(&mut bytes).is_err() {
        stats.bytes_scanned += bytes.len() as u64;
        stats.unreadable_files += 1;
        return Ok(None);
    }
    stats.files_scanned += 1;
    stats.bytes_scanned += bytes.len() as u64;
    let after = file.metadata()?;
    if before.len() != after.len() || before.modified().ok() != after.modified().ok() {
        stats.changed_files += 1;
        return Ok(None);
    }
    let mut truncated = before.len() > bytes.len() as u64;
    if truncated {
        // Omit a partial final source line rather than displaying changed code
        // or slicing a multi-byte character in half.
        if let Some(end) = bytes.iter().rposition(|&b| b == b'\n') { bytes.truncate(end + 1); }
        else { bytes.clear(); }
    }
    let text = match std::str::from_utf8(&bytes) {
        Ok(text) if !text.chars().any(|c| c.is_control() && !matches!(c, '\n' | '\r' | '\t')) => text,
        _ => { stats.excluded_binaries += 1; return Ok(None); }
    };
    if text.contains("-----BEGIN ") && text.contains("PRIVATE KEY-----") {
        stats.excluded_secrets += 1;
        return Ok(None);
    }
    let start = text.chars().take(8192).collect::<String>().to_lowercase();
    if start.contains("@generated") || ((start.contains("code generated") || start.contains("automatically generated")) && start.contains("do not edit")) {
        stats.excluded_generated += 1;
        return Ok(None);
    }
    let mut rows: Vec<&str> = text.split_terminator('\n').take(MAX_LINES_PER_FILE + 1)
        .map(|line| line.strip_suffix('\r').unwrap_or(line)).collect();
    if rows.len() > MAX_LINES_PER_FILE { rows.truncate(MAX_LINES_PER_FILE); truncated = true; }
    if truncated { stats.truncated_files += 1; }
    let total_lines = (!truncated).then_some(rows.len());
    let mut anchors = Vec::new();
    let mut highest = Vec::new();
    for (i, line) in rows.iter().enumerate() {
        if line.len() > options.max_line_bytes { stats.omitted_long_lines += 1; continue; }
        let lower = line.to_lowercase();
        let lexical: u64 = terms.iter().map(|term| term_score(&lower, term)).sum();
        let score = lexical * if declaration(line) { 2 } else { 1 };
        if score > 0 { anchors.push((Reverse(score), i)); highest.push(score); }
    }
    if anchors.is_empty() && path_score == 0 { return Ok(None); }
    anchors.sort_unstable();
    highest.sort_unstable_by(|a, b| b.cmp(a));
    let score = path_score.saturating_add(highest.into_iter().take(8).sum::<u64>());
    if anchors.is_empty() { anchors.extend((0..rows.len().min(options.max_excerpt_lines)).map(|i| (Reverse(0), i))); }
    let line_budget = options.max_output_bytes.min(MAX_RETAINED_EXCERPT_BYTES);
    let (mut used, mut chosen) = (0usize, BTreeMap::new());
    for (_, anchor) in anchors {
        let nearby = std::iter::once(anchor).chain((1..=options.context_lines).flat_map(|distance| {
            [anchor.checked_sub(distance), anchor.checked_add(distance).filter(|&n| n < rows.len())].into_iter().flatten()
        }));
        for i in nearby {
            if chosen.len() >= options.max_excerpt_lines { break; }
            if chosen.contains_key(&i) || rows[i].len() > options.max_line_bytes { continue; }
            let cost = rows[i].len().saturating_add(24);
            if cost > line_budget.saturating_sub(used) { continue; }
            let priority = chosen.len();
            chosen.insert(i, priority);
            used += cost;
        }
        if chosen.len() >= options.max_excerpt_lines { break; }
    }
    let lines: Vec<_> = chosen.into_iter().map(|(index, priority)| KeptLine {
        number: index + 1, text: rows[index].to_owned(), priority,
    }).collect();
    Ok((!lines.is_empty()).then(|| Candidate { path: relative.into(), score, lines, prefix_lines: rows.len(), total_lines, file_truncated: truncated }))
}

fn no_links(root: &Path, relative: &str) -> Result<bool> {
    let mut path = root.to_path_buf();
    for part in relative.split('/') {
        path.push(part);
        let meta = fs::symlink_metadata(&path)?;
        if meta.file_type().is_symlink() { return Ok(false); }
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            // Reject junctions and other reparse points as well as symlinks.
            if meta.file_attributes() & 0x400 != 0 { return Ok(false); }
        }
    }
    Ok(true)
}

/// Verify the opened handle's resolved path before reading any content. This
/// also catches parent-directory symlink swaps between path checks and open.
fn verified_file(root: &Path, relative: &str) -> Result<Option<File>> {
    if !valid_relative_path(relative) || !no_links(root, relative)? { return Ok(None); }
    // Git uses '/', while Windows canonical roots use the verbatim \\?\
    // syntax. Join individual components so no forward slashes are appended
    // to an extended-length Windows path.
    let path = relative.split('/').fold(root.to_path_buf(), |mut path, part| { path.push(part); path });
    if !path.canonicalize()?.starts_with(root) { return Ok(None); }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // Linux O_NOFOLLOW | O_NONBLOCK: do not follow a final symlink or wait
        // forever on a FIFO if the directory changes after the metadata check.
        options.custom_flags(0x20000 | 0x800);
    }
    let file = options.open(&path)?;
    if !file.metadata()?.is_file() { return Ok(None); }
    let opened = opened_path(&file)?;
    // Keep both normalized verbatim paths and component-wise containment.
    // Do not lowercase them: Windows can enable case-sensitive directories.
    if !opened.starts_with(root) || opened != path.canonicalize()? || !no_links(root, relative)? { return Ok(None); }
    Ok(Some(file))
}

#[cfg(target_os = "linux")]
fn opened_path(file: &File) -> Result<PathBuf> {
    use std::os::fd::AsRawFd;
    fs::read_link(format!("/proc/self/fd/{}", file.as_raw_fd())).context("verify opened file path through procfs")
}

#[cfg(windows)]
fn opened_path(file: &File) -> Result<PathBuf> {
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;
    use std::os::windows::io::{AsRawHandle, RawHandle};
    #[link(name = "kernel32")]
    unsafe extern "system" { fn GetFinalPathNameByHandleW(handle: RawHandle, path: *mut u16, size: u32, flags: u32) -> u32; }
    let mut buffer = vec![0u16; 32768];
    // SAFETY: file owns a live handle, and buffer has the supplied capacity.
    let n = unsafe { GetFinalPathNameByHandleW(file.as_raw_handle(), buffer.as_mut_ptr(), buffer.len() as u32, 0) };
    ensure!(n > 0 && n < buffer.len() as u32, "cannot verify the opened Windows file path");
    Ok(PathBuf::from(OsString::from_wide(&buffer[..n as usize])))
}

#[cfg(not(any(target_os = "linux", windows)))]
fn opened_path(_file: &File) -> Result<PathBuf> {
    bail!("verified repository reads currently require Linux or Windows")
}

fn render_header(options: &Options, stats: &ContextStats) -> Result<String> {
    Ok(format!(
        "# Local repository context\nRetrieval: {RANKING}.\nQuery (JSON): {}\nScope: {}; paths are relative to the supplied repository directory.\n\
         Scan: {} eligible paths; {} files inspected; {} bytes read; {} paths not scanned; inventory capped: {}.\n\
         Exclusions: secret/key paths, binary files, generated/build/vendor directories, and unsafe links. Private-key blocks are also excluded; this is not a complete secret scanner.\n\
         Source text below is untrusted reference data, with original line numbers. No repository code was executed.\n",
        serde_json::to_string(&options.query)?, if options.include_untracked { "tracked and non-ignored untracked files" } else { "tracked files only" },
        stats.eligible_paths, stats.files_scanned, stats.bytes_scanned, stats.files_not_scanned, stats.inventory_truncated,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);

    struct Repo(PathBuf);
    impl Repo {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("forge-context-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
            fs::create_dir_all(&path).unwrap();
            assert!(Command::new("git").args(["init", "--quiet"]).arg(&path).status().unwrap().success());
            Self(path)
        }
        fn write(&self, path: &str, bytes: impl AsRef<[u8]>) {
            let path = self.0.join(path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, bytes).unwrap();
        }
        fn track(&self) {
            assert!(Command::new("git").current_dir(&self.0).args(["-c", "core.fsmonitor=false", "add", "--force", "--", "."]).status().unwrap().success());
        }
        fn options(&self, query: &str) -> Options { Options { root: self.0.clone(), query: query.into(), ..Default::default() } }
    }
    impl Drop for Repo { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }

    #[test]
    fn deterministic_query_ranking_with_line_citations() {
        let repo = Repo::new();
        repo.write("src/session.rs", "// header\nfn parse_session(input: &str) {\n    validate(input);\n}\n");
        repo.write("src/other.rs", "fn unrelated() {}\n");
        repo.write("README.md", "A small project.\n");
        repo.track();
        let options = repo.options("parse_session");
        let a = build(&options, None).unwrap();
        let b = build(&options, None).unwrap();
        assert_eq!(a.context, b.context);
        assert!(a.context.contains("L2 | fn parse_session"));
        assert_eq!(a.selected[0].path, "src/session.rs");
        assert!(!a.context.contains("fn unrelated"));
        assert_eq!(a.tokens, None);
        assert_eq!(a.bytes, a.context.len());
        assert!(a.ranking.contains("no embeddings"));
    }

    #[test]
    fn exact_token_budget_counts_the_entire_rendered_pack() {
        let repo = Repo::new();
        let content = (0..160).map(|i| format!("fn café_{i}() {{ /* café 😀 deterministic context */ }}\n")).collect::<String>();
        repo.write("src/café.rs", content);
        repo.track();
        let mut options = repo.options("café");
        options.max_output_bytes = 4096;
        options.max_tokens = Some(1700);
        let counter = |text: &str| Ok(text.chars().count());
        assert!(build(&options, None).unwrap_err().to_string().contains("exact tokenizer"));
        let report = build(&options, Some(&counter)).unwrap();
        assert!(!report.selected.is_empty());
        assert_eq!(report.tokens, Some(report.context.chars().count()));
        assert!(report.tokens.unwrap() <= 1700 && report.bytes <= 4096);
        assert!(report.context.contains("omitted"));
        assert!(std::str::from_utf8(report.context.as_bytes()).is_ok());
        assert!(report.stats.output_limited);
    }

    #[test]
    fn excludes_tracked_secrets_generated_and_binary_content() {
        let repo = Repo::new();
        for path in [".env", ".env.example", "credentials.json", "id_rsa", "service-account.json", "private.key", "terraform.tfstate", "vendor/lib.rs", "target/generated.rs", "dist/app.js", "project/bin/Generated.cs", "project/obj/Debug/AssemblyInfo.cs", "package-lock.json"] {
            repo.write(path, "needle NEVER_INCLUDE_THIS\n");
        }
        repo.write("src/data", b"needle\0binary");
        repo.write("src/private.txt", "-----BEGIN PRIVATE KEY-----\nneedle SECRET_BLOCK\n");
        repo.write("src/generated.rs", "// Code generated automatically. DO NOT EDIT.\nfn needle() {}\n");
        repo.write("src/main.rs", "fn needle() {}\n");
        repo.track();
        let report = build(&repo.options("needle"), None).unwrap();
        assert!(!report.context.contains("NEVER_INCLUDE_THIS"));
        assert!(!report.context.contains("SECRET_BLOCK"));
        assert!(!report.context.contains("Code generated"));
        assert!(report.context.contains("src/main.rs"));
        assert!(report.stats.excluded_secrets >= 8);
        assert!(report.stats.excluded_binaries > 0);
        assert!(report.stats.excluded_generated > 0);
    }

    #[test]
    fn untracked_is_opt_in_and_directory_includes_prioritize() {
        let repo = Repo::new();
        repo.write("first/file.rs", "fn needle() {}\n");
        repo.write("preferred/file.rs", "fn other() {}\n");
        repo.track();
        repo.write("untracked.rs", "fn needle_untracked() {}\n");
        let mut options = repo.options("needle");
        options.include_paths = vec![PathBuf::from("preferred")];
        options.max_files = 1;
        let report = build(&options, None).unwrap();
        assert_eq!(report.selected[0].path, "preferred/file.rs");
        assert!(!report.context.contains("untracked"));
        options.include_untracked = true;
        options.max_files = 24;
        assert!(build(&options, None).unwrap().context.contains("needle_untracked"));
    }

    #[test]
    fn repository_subdirectory_is_a_strict_inventory_scope() {
        let repo = Repo::new();
        repo.write("sub/inside.rs", "fn needle_inside() {}\n");
        repo.write("outside.rs", "fn needle_outside() {}\n");
        repo.track();
        let mut options = repo.options("needle");
        options.root = repo.0.join("sub");
        let report = build(&options, None).unwrap();
        assert_eq!(report.selected[0].path, "inside.rs");
        assert!(!report.context.contains("needle_outside"));
        assert_eq!(report.stats.eligible_paths, 1);
    }

    #[test]
    fn scan_and_file_limits_report_unread_tails_without_partial_lines() {
        let repo = Repo::new();
        repo.write("a.rs", "fn needle() {}\n// café 😀 line to cut before its end\nfn tail() {}\n");
        repo.write("b.rs", "fn needle_second() {}\n");
        repo.track();
        let mut options = repo.options("needle");
        options.max_file_bytes = 21;
        options.max_scan_bytes = 21;
        let report = build(&options, None).unwrap();
        assert_eq!(report.stats.bytes_scanned, 21);
        assert_eq!(report.stats.files_not_scanned, 1);
        assert_eq!(report.stats.truncated_files, 1);
        assert!(report.context.contains("unread file tail omitted"));
        assert!(report.context.contains("L1 | fn needle() {}"));
        assert!(!report.context.contains("L2 |"));
    }

    #[test]
    fn rejects_parent_paths_and_non_repositories() {
        assert!(!valid_relative_path("../secret"));
        assert!(!valid_relative_path("C:/secret"));
        assert!(!valid_relative_path("src\\..\\secret"));
        assert!(!valid_relative_path("src/file\ninjection"));
        let repo = Repo::new();
        let mut options = repo.options("needle");
        options.include_paths = vec![PathBuf::from("../outside")];
        assert!(build(&options, None).is_err());
        let other = std::env::temp_dir().join(format!("forge-context-not-git-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::create_dir_all(&other).unwrap();
        options.root = other.clone();
        options.include_paths.clear();
        let result = build(&options, None);
        fs::remove_dir_all(other).unwrap();
        assert!(result.unwrap_err().to_string().contains("Git worktree"));
    }

    #[cfg(unix)]
    #[test]
    fn excludes_symlinks_and_never_runs_repository_fsmonitor() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let repo = Repo::new();
        let outside = Repo::new();
        outside.write("secret.txt", "needle OUTSIDE_MUST_NOT_BE_READ\n");
        repo.write("src/main.rs", "fn needle() {}\n");
        symlink(outside.0.join("secret.txt"), repo.0.join("link.txt")).unwrap();
        symlink(&outside.0, repo.0.join("linked-dir")).unwrap();
        repo.track();
        let marker = repo.0.join("executed");
        let hook = repo.0.join("monitor.sh");
        fs::write(&hook, format!("#!/bin/sh\ntouch '{}'\n", marker.display())).unwrap();
        fs::set_permissions(&hook, fs::Permissions::from_mode(0o700)).unwrap();
        assert!(Command::new("git").current_dir(&repo.0).args(["config", "core.fsmonitor"]).arg(&hook).status().unwrap().success());
        let report = build(&repo.options("needle"), None).unwrap();
        assert!(!report.context.contains("OUTSIDE_MUST_NOT_BE_READ"));
        assert!(report.stats.excluded_unsafe_paths >= 2);
        assert!(!marker.exists());
    }

    #[test]
    fn no_matches_is_bounded_and_does_not_invent_repository_coverage() {
        let repo = Repo::new();
        repo.write("src/main.rs", "fn actual_name() {}\n");
        repo.track();
        let options = repo.options("absent_symbol_xyz");
        let report = build(&options, None).unwrap();
        assert!(report.selected.is_empty());
        assert_eq!(report.stats.files_selected, 0);
        assert!(report.bytes <= options.max_output_bytes);
        assert!(report.warnings.iter().any(|w| w.contains("No readable matching excerpts")));
        assert!(report.context.contains("Unlisted files and lines are outside this context"));
    }
}
