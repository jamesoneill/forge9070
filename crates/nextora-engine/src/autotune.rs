//! Separate-process, bounded performance measurements for one model and context.
use crate::{
    cli::{measure, write_report},
    gguf::Gguf,
    runtime::{Backend, BackendArgs, LoadStep, RuntimeModel},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
    sync::Mutex,
    time::{Duration, Instant},
};

const DECODE_STEPS: usize = 128;
const SCOPE: &str = "Single-request coding prompt; one warmup excluded, two repeated fixed 128-step decodes; selected context retained. The fastest measured candidate is selected. Full-window prompt processing and exhaustive hardware optimization are outside this benchmark.";
const PROMPT: &str = "fn square(x: i32) -> i32 {\n    // Return the square of x.\n";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkerRequest {
    pub model: PathBuf,
    pub context: usize,
    pub settings: Value,
    pub machine: String,
    #[serde(default)]
    pub prepare: Option<SetupGoal>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct SetupGoal {
    pub context: String,
    pub max_context: Option<usize>,
}
impl Default for SetupGoal {
    fn default() -> Self {
        Self {
            context: "maximum_fit".into(),
            max_context: None,
        }
    }
}
impl SetupGoal {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            matches!(
                self.context.as_str(),
                "maximum_fit" | "maximum_practical" | "fixed"
            ),
            "unknown context search goal"
        );
        ensure!(
            self.max_context
                .is_none_or(|n| (2..=1_048_576).contains(&n)),
            "context ceiling must be in 2..=1048576"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
struct Candidate {
    backend: String,
    threads: usize,
    settings: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_tps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_tps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    report: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn settings_number(settings: &Value, key: &str, fallback: usize) -> Result<usize> {
    match settings.get(key) {
        None => Ok(fallback),
        Some(value) => usize::try_from(value.as_u64().with_context(|| format!("invalid {key}"))?)
            .with_context(|| format!("{key} exceeds address range")),
    }
}

fn optional_number(settings: &Value, key: &str) -> Result<Option<usize>> {
    if settings.get(key).is_none_or(Value::is_null) {
        Ok(None)
    } else {
        settings_number(settings, key, 0).map(Some)
    }
}

fn backend_args(settings: &Value) -> Result<BackendArgs> {
    let backend = match settings["backend"].as_str() {
        Some("cpu") => Backend::Cpu,
        Some("hybrid") => Backend::Hybrid,
        Some("vulkan") => Backend::Vulkan,
        Some("auto") => Backend::Auto,
        _ => anyhow::bail!("unknown candidate backend"),
    };
    let args = BackendArgs {
        cpu: false,
        backend: Some(backend),
        device: optional_number(settings, "device")?,
        allow_software_vulkan: false,
        fast_math: settings["fast_math"].as_bool().unwrap_or(false),
        direct_experts: match settings.get("direct_experts") { None | Some(Value::Null) => None, Some(Value::Bool(value)) => Some(*value), _ => anyhow::bail!("direct_experts must be a boolean or null") },
        kv_placement: serde_json::from_value(
            settings
                .get("kv_placement")
                .cloned()
                .unwrap_or_else(|| json!("gpu")),
        )?,
        kv_ram_mib: optional_number(settings, "kv_ram_mib")?,
        kv_precision: serde_json::from_value(
            settings
                .get("kv_precision")
                .cloned()
                .unwrap_or_else(|| json!("f32")),
        )?,
        vram_mib: settings_number(settings, "vram_mib", 0)?,
        expert_ram_mib: optional_number(settings, "expert_ram_mib")?,
        ram_margin_mib: settings_number(settings, "ram_margin_mib", 4096)?,
        expert_warmup: settings["expert_warmup"].as_bool().unwrap_or(true),
        state_slots: 1,
    };
    args.selected()?;
    ensure!(
        args.device.is_none_or(|n| n <= 1023),
        "invalid adapter index"
    );
    for n in [
        args.vram_mib,
        args.expert_ram_mib.unwrap_or(0),
        args.kv_ram_mib.unwrap_or(0),
        args.ram_margin_mib,
    ] {
        ensure!(
            n <= 16 * 1024 * 1024,
            "candidate memory ceiling exceeds limits"
        );
    }
    Ok(args)
}

fn thread_counts(available: usize, baseline: usize) -> Vec<usize> {
    let mut counts = vec![
        baseline.max(1),
        (available / 4).max(1),
        (available / 2).max(1),
        available.max(1),
    ];
    let mut seen = BTreeSet::new();
    counts.retain(|count| seen.insert(*count));
    counts
}

fn make_candidates(
    request: &WorkerRequest,
    available: usize,
    hardware: &Value,
    weights: u64,
    architecture: &str,
) -> Result<Vec<Candidate>> {
    ensure!(
        request.settings.is_object(),
        "worker settings must be an object"
    );
    let baseline = optional_number(&request.settings, "threads")?.unwrap_or(available);
    ensure!((1..=1024).contains(&baseline), "invalid worker threads");
    let counts = thread_counts(available, baseline);
    let original = request.settings["backend"]
        .as_str()
        .context("backend missing")?;
    let device = optional_number(&request.settings, "device")?;
    let vram_mib = settings_number(&request.settings, "vram_mib", 0)?;
    let adapters = hardware["adapters"].as_array().cloned().unwrap_or_default();
    let fast_math = request.settings["fast_math"].as_bool().unwrap_or(false);
    let gpu = adapters.iter().find(|adapter| {
        adapter["compatible"].as_bool() == Some(true)
            && adapter["software"].as_bool() == Some(false)
            && (!fast_math || adapter["fast_math_available"].as_bool() == Some(true))
            && device.is_none_or(|index| adapter["index"].as_u64() == Some(index as u64))
    });
    let mut backends = vec![original.to_owned()];
    // Q8 activations intentionally have a different numerical reference.
    // Keep the requested mode stable rather than measuring F32 candidates
    // which would later fail the baseline-output check.
    let compact = request.settings["kv_precision"].as_str() == Some("q4_64");
    if !fast_math && !compact && request.settings["direct_experts"] != true {
        backends.push("cpu".into());
        if gpu.is_some() {
            backends.push("hybrid".into());
        }
    }
    if let Some(adapter) = gpu {
        let budget = if vram_mib > 0 {
            vram_mib as u64 * 1024 * 1024
        } else {
            adapter["allocation_budget_bytes"].as_u64().unwrap_or(0)
        };
        // A necessary ordinary-Qwen3 fit check avoids an expensive known failure.
        // Detailed staging/sequence allocations remain checked by the loader.
        if matches!(architecture, "qwen3moe" | "qwen4exp")
            || weights.saturating_add(256 * 1024 * 1024) < budget
        {
            backends.push("vulkan".into());
        }
    }
    if architecture == "qwen35" {
        backends.retain(|backend| matches!(backend.as_str(), "cpu" | "hybrid"));
    }
    let mut seen = BTreeSet::new();
    backends.retain(|backend| seen.insert(backend.clone()));
    let mut candidates = Vec::new();
    let mut unique = BTreeSet::new();
    for backend in backends {
        for &threads in &counts {
            let mut settings = request.settings.clone();
            settings
                .as_object_mut()
                .unwrap()
                .entry("kv_placement")
                .or_insert_with(|| json!("gpu"));
            settings
                .as_object_mut()
                .unwrap()
                .entry("kv_ram_mib")
                .or_insert(Value::Null);
            settings["backend"] = json!(backend);
            settings["threads"] = json!(threads);
            settings["ctx"] = json!(request.context);
            settings["parallel"] = json!(1);
            if matches!(backend.as_str(), "cpu" | "hybrid") {
                settings["fast_math"] = json!(false);
                settings.as_object_mut().unwrap().remove("direct_experts");
                settings["kv_placement"] = json!("gpu");
                settings["kv_ram_mib"] = Value::Null;
            }
            if backend == "cpu" {
                settings["device"] = Value::Null;
                settings["vram_mib"] = json!(0);
            } else if let Some(adapter) = gpu {
                // Keep adapter choice stable across all GPU trials.
                settings["device"] = adapter["index"].clone();
            }
            let encoded = serde_json::to_string(&settings)?;
            if unique.insert(encoded) {
                candidates.push(Candidate {
                    backend: backend.clone(),
                    threads,
                    settings,
                    decode_tps: None,
                    prefill_tps: None,
                    report: None,
                    error: None,
                });
            }
        }
    }
    Ok(candidates)
}

struct Progress<'a> {
    path: &'a Path,
    model: &'a Path,
    candidate: usize,
    candidate_total: usize,
    started: Instant,
    last_write: Mutex<Instant>,
}

impl Progress<'_> {
    fn write(
        &self,
        stage: &str,
        message: &str,
        counts: Option<(usize, usize, &str)>,
        force: bool,
    ) -> Result<()> {
        let mut last = self.last_write.lock().unwrap();
        if !force && last.elapsed() < Duration::from_millis(150) {
            return Ok(());
        }
        *last = Instant::now();
        write_report(
            self.path,
            &json!({"stage":stage,"message":message,"candidate":self.candidate,"candidate_total":self.candidate_total,
            "completed":counts.map(|(done,_,_)|done),"total":counts.map(|(_,total,_)|total),"unit":counts.map(|(_,_,unit)|unit),
            "elapsed_seconds":self.started.elapsed().as_secs_f64()}),
            Some(self.model),
        )
    }
    fn load(&self, step: LoadStep) {
        let (stage, message, counts, force) = match step {
            LoadStep::Stage(stage, message) => (stage, message, None, true),
            LoadStep::Upload {
                message,
                unit,
                completed,
                total,
                ..
            } => ("uploading", message, Some((completed, total, unit)), false),
            LoadStep::Warmup {
                considered, total, ..
            } => (
                "warming",
                "Warming the expert cache",
                Some((considered, total, "experts")),
                false,
            ),
        };
        // Report writes are mandatory at candidate boundaries. Frequent load
        // updates are best effort so a transient reader lock cannot abort load.
        let _ = self.write(stage, message, counts, force);
    }
}

fn bounded_measure(
    model: &RuntimeModel,
    prompt: &[u32],
    maximum: usize,
    minimum: usize,
    target: Duration,
    deadline: Instant,
    allow_shorter: bool,
) -> Result<Value> {
    let mut sequence = model.new_sequence()?;
    let prefill_start = Instant::now();
    let mut output = model.prefill(&mut sequence, prompt)?;
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();
    ensure!(
        Instant::now() < deadline,
        "candidate time budget exceeded during prefill"
    );
    let decode_start = Instant::now();
    let mut checksum = 0xcbf29ce484222325u64;
    let mut prefix = checksum;
    let mut count = 0;
    while count < maximum {
        ensure!(
            Instant::now() < deadline,
            "candidate time budget exceeded during autoregressive decoding"
        );
        let token = crate::runtime::greedy(&output.logits)?;
        checksum = (checksum ^ token as u64).wrapping_mul(0x100000001b3);
        model.step_into(&mut sequence, token, &mut output)?;
        count += 1;
        if count == 4 {
            prefix = checksum;
        }
        if allow_shorter && count >= minimum && decode_start.elapsed() >= target {
            break;
        }
    }
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    ensure!(
        decode_seconds.is_finite()
            && decode_seconds > 0.0
            && prefill_seconds.is_finite()
            && prefill_seconds > 0.0,
        "nonpositive or nonfinite benchmark time"
    );
    Ok(
        json!({"prefill_seconds":prefill_seconds,"decode_seconds":decode_seconds,
        "decode_steps":count,"prompt_tokens":prompt.len(),"greedy_checksum":[checksum],"validation_checksum":[prefix]}),
    )
}
fn measure_preparation_candidate(
    model: &RuntimeModel,
    candidate: &Candidate,
    progress: &Progress<'_>,
) -> Result<Value> {
    let mut prompt = model.tokenizer.encode(PROMPT)?;
    prompt.truncate(8);
    ensure!(
        !prompt.is_empty() && prompt.len() + 128 <= model.context(),
        "coding benchmark does not fit selected context"
    );
    let deadline = Instant::now() + Duration::from_secs(75);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(candidate.threads)
        .build()?;
    pool.install(||{
        progress.write("warmup","Running a short excluded warmup",Some((0,3,"runs")),true)?;
        bounded_measure(model,&prompt,8,2,Duration::from_secs(3),deadline,true)?;
        progress.write("measuring","Measuring autoregressive generation",Some((1,3,"runs")),true)?;
        let first=bounded_measure(model,&prompt,128,4,Duration::from_secs(3),deadline,true)?;
        let steps=first["decode_steps"].as_u64().unwrap() as usize;
        progress.write("measuring","Repeating the same decode length",Some((2,3,"runs")),true)?;
        let second=bounded_measure(model,&prompt,steps,steps,Duration::from_secs(3),deadline,false)?;
        ensure!(first["greedy_checksum"]==second["greedy_checksum"],"greedy output changed between repeated measurements");
        let decode=first["decode_seconds"].as_f64().unwrap()+second["decode_seconds"].as_f64().unwrap();
        let prefill=first["prefill_seconds"].as_f64().unwrap()+second["prefill_seconds"].as_f64().unwrap();
        Ok(json!({"decode_tps":(steps*2) as f64/decode,"prefill_tps":(prompt.len()*2) as f64/prefill,
            "prompt_tokens":prompt.len(),"decode_steps":steps,"validation_checksum":first["validation_checksum"],
            "warmup_excluded":1,"repetitions":2,"identical_greedy_checksums":true,
            "sampling_scope":"Adaptive whole-model autoregressive samples: 4–128 decode steps, about 3 seconds targeted per measured run, 75-second candidate deadline checked between engine steps. A single engine call cannot be interrupted in-process.",
            "samples":[first,second],"runtime":model.report()}))
    })
}

fn measure_candidate(
    model: &RuntimeModel,
    candidate: &Candidate,
    progress: &Progress<'_>,
) -> Result<Value> {
    let prompt = model.tokenizer.encode(PROMPT)?;
    ensure!(
        !prompt.is_empty()
            && prompt
                .len()
                .checked_add(DECODE_STEPS)
                .is_some_and(|n| n <= model.context()),
        "coding benchmark does not fit selected context"
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(candidate.threads)
        .build()?;
    pool.install(|| {
        progress.write("warmup","Running an excluded warmup",Some((0,3,"runs")),true)?;
        let warmup = measure(model, &prompt, DECODE_STEPS, 1)?;
        let mut samples = Vec::new();
        for run in 0..2 {
            progress.write("measuring","Measuring coding generation speed",Some((run+1,3,"runs")),true)?;
            let measured = measure(model, &prompt, DECODE_STEPS, 1)?;
            ensure!(measured.greedy_checksum == warmup.greedy_checksum, "greedy output changed between warmup and repeated measurements");
            samples.push(serde_json::to_value(measured)?);
        }
        let total_decode: f64 = samples.iter().map(|s|s["decode_seconds"].as_f64().unwrap()).sum();
        let total_prefill: f64 = samples.iter().map(|s|s["prefill_seconds"].as_f64().unwrap()).sum();
        ensure!(total_decode.is_finite() && total_decode>0.0 && total_prefill.is_finite() && total_prefill>0.0, "nonpositive or nonfinite benchmark time");
        Ok(json!({"decode_tps":(DECODE_STEPS*2) as f64/total_decode,"prefill_tps":(prompt.len()*2) as f64/total_prefill,
            "prompt_tokens":prompt.len(),"decode_steps":DECODE_STEPS,"warmup_excluded":1,"repetitions":2,"identical_greedy_checksums":true,
            "samples":samples,"runtime":model.report()}))
    })
}

fn study_fixed(request: &WorkerRequest, progress_path: &Path, adaptive: bool) -> Result<Value> {
    ensure!(
        (2..=1_048_576).contains(&request.context),
        "invalid tuning context"
    );
    ensure!(
        request.machine.len() <= 4096,
        "machine identity exceeds limit"
    );
    let available = std::thread::available_parallelism()?.get().min(128);
    let source = Gguf::open(&request.model)?;
    let architecture = source
        .meta
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .context("GGUF architecture missing")?;
    let mut hardware = crate::runtime::hardware_report();
    let options = backend_args(&request.settings)?.device_options()?;
    let selection = crate::vulkan::select_device(&options);
    let selected_device = if options.fast_math || options.kv_precision.is_compact() {
        selection.context("select GPU for requested fast-math tuning")?
    } else {
        selection.ok().flatten()
    };
    if let Some(adapters) = hardware["adapters"].as_array_mut() {
        adapters.retain(|adapter| adapter["index"].as_u64() == selected_device.map(|i| i as u64));
    }
    let mut candidates = make_candidates(
        request,
        available,
        &hardware,
        source.mapped_bytes() as u64,
        architecture,
    )?;
    if adaptive
        && request
            .prepare
            .as_ref()
            .is_some_and(|goal| goal.context != "fixed")
        && matches!(architecture, "qwen3" | "qwen3moe")
    {
        let alternatives: Vec<_> = candidates
            .iter()
            .filter(|c| c.backend == "vulkan")
            .map(|candidate| {
                let mut alternative = candidate.clone();
                if alternative.settings["kv_placement"] == "ram" {
                    alternative.settings["kv_placement"] = json!("gpu");
                    alternative.settings["kv_ram_mib"] = Value::Null;
                } else {
                    alternative.settings["kv_placement"] = json!("ram");
                }
                alternative
            })
            .collect();
        candidates.extend(alternatives);
    }
    drop(source);
    let started = Instant::now();
    let mut current: Option<(String, RuntimeModel)> = None;
    let mut failed_loads = BTreeMap::<String, String>::new();
    let mut expected_checksum: Option<Vec<Value>> = None;
    let mut slow_backends = BTreeSet::new();
    if adaptive {
        candidates.sort_by_key(|candidate| match candidate.backend.as_str() {
            "vulkan" => 0,
            "hybrid" => 1,
            "auto" => 2,
            _ => 3,
        });
    }
    for index in 0..candidates.len() {
        let progress = Progress {
            path: progress_path,
            model: &request.model,
            candidate: index + 1,
            candidate_total: candidates.len(),
            started,
            last_write: Mutex::new(Instant::now()),
        };
        let candidate = &mut candidates[index];
        let family_key = format!(
            "{}:{}:{}",
            candidate.backend, candidate.settings["kv_placement"], candidate.settings["direct_experts"]
        );
        if adaptive && slow_backends.contains(&family_key) {
            candidate.error=Some("Skipped further worker counts after this backend exceeded the candidate time budget".into());
            continue;
        }
        progress.write(
            "loading",
            &format!(
                "Testing {} with {} CPU workers",
                candidate.backend, candidate.threads
            ),
            None,
            true,
        )?;
        let attempt = (|| -> Result<Value> {
            let mut load_settings = candidate.settings.clone();
            load_settings.as_object_mut().unwrap().remove("threads");
            let key = serde_json::to_string(&load_settings)?;
            if let Some(error) = failed_loads.get(&key) {
                anyhow::bail!("This backend and memory configuration already failed: {error}");
            }
            if current.as_ref().is_none_or(|(loaded, _)| loaded != &key) {
                current = None;
                let options = backend_args(&candidate.settings)?;
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(candidate.threads)
                    .build()?;
                let loaded = pool.install(|| {
                    RuntimeModel::load_with_progress(
                        &request.model,
                        request.context,
                        &options,
                        |step| progress.load(step),
                    )
                });
                let model = match loaded {
                    Ok(model) => model,
                    Err(error) => {
                        failed_loads.insert(key, format!("{error:#}"));
                        return Err(error);
                    }
                };
                current = Some((key, model));
            }
            let report = if adaptive {
                measure_preparation_candidate(&current.as_ref().unwrap().1, candidate, &progress)?
            } else {
                measure_candidate(&current.as_ref().unwrap().1, candidate, &progress)?
            };
            let checksum_value = if adaptive {
                &report["validation_checksum"]
            } else {
                &report["samples"][0]["greedy_checksum"]
            };
            let checksum = checksum_value
                .as_array()
                .context("missing sample checksum")?;
            if let Some(expected) = &expected_checksum {
                ensure!(
                    checksum == expected,
                    "greedy output differs from the baseline; candidate rejected"
                );
            } else {
                expected_checksum = Some(checksum.clone());
            }
            Ok(report)
        })();
        match attempt {
            Ok(report) => {
                candidate.decode_tps = report["decode_tps"].as_f64();
                candidate.prefill_tps = report["prefill_tps"].as_f64();
                candidate.report = Some(report);
            }
            Err(error) => {
                let error = format!("{error:#}");
                if adaptive && error.contains("time budget") {
                    slow_backends.insert(family_key.clone());
                }
                candidate.error = Some(error);
                current = None;
            }
        }
    }
    let winner = candidates
        .iter()
        .filter(|c| c.error.is_none() && c.decode_tps.is_some())
        .max_by(|a, b| a.decode_tps.unwrap().total_cmp(&b.decode_tps.unwrap()))
        .cloned();
    Ok(
        json!({"schema_version":1,"machine":request.machine,"context":request.context,"candidates":candidates,"winner":winner,"scope":SCOPE,
        "elapsed_seconds":started.elapsed().as_secs_f64()}),
    )
}

// Search on a 256-token grid, retaining an exact declared/backend ceiling.
// All probes allocate a sequence and execute a small valid prefix. The full
// capacity estimate and the actually exercised prefix remain separate evidence.
fn search_context(
    maximum: usize,
    mut probe: impl FnMut(usize) -> Result<bool>,
) -> Result<Option<usize>> {
    ensure!(maximum >= 2, "context ceiling is too small");
    let minimum = maximum.min(256);
    let mut context = maximum.min(4096);
    loop {
        if probe(context)? {
            break;
        }
        if context == minimum {
            return Ok(None);
        }
        context = (context / 2).max(minimum);
    }
    let mut best = context;
    let failed = loop {
        if best == maximum {
            return Ok(Some(best));
        }
        let next = best.saturating_mul(2).min(maximum);
        if probe(next)? {
            best = next;
        } else {
            break next;
        }
    };
    let mut high = failed;
    while high > best.saturating_add(256) {
        let middle = ((best + (high - best) / 2) / 256) * 256;
        if middle <= best {
            break;
        }
        if probe(middle)? {
            best = middle;
        } else {
            high = middle;
        }
    }
    Ok(Some(best))
}
fn setup_progress(
    path: &Path,
    model: &Path,
    started: Instant,
    message: &str,
    completed: usize,
    total: usize,
) -> Result<()> {
    write_report(
        path,
        &json!({"stage":"fitting","message":message,"completed":completed,"total":total,
        "unit":"contexts","elapsed_seconds":started.elapsed().as_secs_f64()}),
        Some(model),
    )
}
fn backend_context_cap(
    source: &Gguf,
    architecture: &str,
    settings: &Value,
    declared: usize,
) -> usize {
    if settings["backend"] == "cpu" || settings["backend"] == "hybrid" {
        return declared;
    }
    if matches!(architecture, "qwen3" | "qwen3moe") {
        if settings["kv_precision"] == "q4_64" {
            declared
        } else {
            declared.min(65535)
        }
    } else if architecture == "qwen4exp" {
        let unindexed = source
            .tensors
            .keys()
            .filter_map(|name| name.strip_suffix("attn_q.weight"))
            .any(|prefix| {
                !source
                    .tensors
                    .contains_key(&format!("{prefix}indexer.k_proj.weight"))
            });
        declared.min(if unindexed { 65535 } else { 65535 * 4 })
    } else {
        declared
    }
}
fn prepare_study(request: &WorkerRequest, goal: &SetupGoal, progress_path: &Path) -> Result<Value> {
    goal.validate()?;
    let started = Instant::now();
    let source = Gguf::open(&request.model)?;
    let architecture = source
        .meta
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .context("model architecture missing")?
        .to_owned();
    ensure!(
        matches!(
            architecture.as_str(),
            "qwen3" | "qwen3moe" | "qwen4exp" | "qwen35"
        ),
        "native engine does not support architecture {architecture}"
    );
    let declared = source
        .meta
        .get(&format!("{architecture}.context_length"))
        .and_then(|v| v.as_u64())
        .and_then(|n| usize::try_from(n).ok())
        .context("declared context missing")?
        .min(1_048_576);
    let ceiling = if goal.context == "fixed" {
        ensure!(
            request.context <= declared,
            "fixed requested context exceeds the declared model limit"
        );
        request.context
    } else {
        goal.max_context.unwrap_or(declared).min(declared)
    };
    ensure!(ceiling >= 2, "model context is too small");
    let available = std::thread::available_parallelism()?.get().min(128);
    let hardware = crate::runtime::hardware_report();
    let mut baseline = request.clone();
    baseline.prepare = None;
    baseline.context = ceiling;
    baseline.settings["ctx"] = json!(ceiling);
    baseline.settings["parallel"] = json!(1);
    baseline.settings["max_tokens"] =
        json!(settings_number(&request.settings, "max_tokens", 128)?.min(ceiling - 1));
    let mut adjustments = Vec::new();
    if !matches!(architecture.as_str(), "qwen3" | "qwen3moe") {
        baseline.settings["fast_math"] = json!(false);
        baseline.settings.as_object_mut().unwrap().remove("direct_experts");
        baseline.settings["kv_precision"] = json!("f32");
        baseline.settings["kv_placement"] = json!("gpu");
        baseline.settings["kv_ram_mib"] = Value::Null;
        adjustments.push(
            "This model family uses standard activation math and F32 context storage.".to_string(),
        );
    }
    let mut families = make_candidates(
        &baseline,
        available,
        &hardware,
        source.mapped_bytes() as u64,
        &architecture,
    )?;
    if goal.context != "fixed" && matches!(architecture.as_str(), "qwen3" | "qwen3moe") {
        let alternatives: Vec<_> = families
            .iter()
            .filter(|c| c.backend == "vulkan")
            .map(|candidate| {
                let mut alternative = candidate.clone();
                if alternative.settings["kv_placement"] == "ram" {
                    alternative.settings["kv_placement"] = json!("gpu");
                    alternative.settings["kv_ram_mib"] = Value::Null;
                } else {
                    alternative.settings["kv_placement"] = json!("ram");
                }
                alternative
            })
            .collect();
        families.extend(alternatives);
    }
    let mut seen = BTreeSet::new();
    families.retain(|candidate| {
        let mut settings = candidate.settings.clone();
        settings.as_object_mut().unwrap().remove("threads");
        seen.insert(serde_json::to_string(&settings).unwrap())
    });
    let mut trials = Vec::new();
    let mut fitted = Vec::<(usize, Value, Value)>::new();
    for family in families {
        let cap = backend_context_cap(&source, &architecture, &family.settings, ceiling);
        if goal.context == "fixed" && cap < request.context {
            continue;
        }
        let mut last_runtime = Value::Null;
        let mut probe = |context: usize| -> Result<bool> {
            ensure!(
                started.elapsed() < Duration::from_secs(27 * 60),
                "context fitting exceeded its time budget"
            );
            setup_progress(
                progress_path,
                &request.model,
                started,
                &format!("Checking {} at {} context", family.backend, context),
                trials.len(),
                0,
            )?;
            let mut settings = family.settings.clone();
            settings["ctx"] = json!(context);
            settings["expert_warmup"] = json!(false);
            settings["max_tokens"] =
                json!(settings_number(&baseline.settings, "max_tokens", 128)?.min(context - 1));
            let trial_started = Instant::now();
            let attempt = (|| -> Result<Value> {
                let options = backend_args(&settings)?;
                let geometry = RuntimeModel::fit_plan(&request.model, context, &options)?;
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(family.threads)
                    .build()?;
                pool.install(||{
                    let model=RuntimeModel::load(&request.model,context,&options)?;
                    let capacity=model.probe_capacity()?;
                    let mut sequence=model.new_sequence()?;
                    let tokens=model.tokenizer.encode("fn f(")?;
                    ensure!(!tokens.is_empty(),"fit probe tokenization was empty");
                    let count=tokens.len().min(context.saturating_sub(1)).min(1).max(1);
                    let step=model.prefill(&mut sequence,&tokens[..count])?;
                    ensure!(step.logits.iter().all(|x|x.is_finite()),"fit probe produced invalid logits");
                    Ok(json!({"geometry_plan":geometry,"runtime":model.report(),"allocation_probe":capacity,
                        "actual_probe_tokens":count,"full_window_generation_tested":false}))
                })
            })();
            let success = attempt.is_ok();
            let mut trial = json!({"context":context,"settings":settings,"seconds":trial_started.elapsed().as_secs_f64(),"passed":success});
            match attempt {
                Ok(report) => {
                    last_runtime = report.clone();
                    trial["evidence"] = report;
                }
                Err(error) => trial["error"] = json!(format!("{error:#}")),
            }
            trials.push(trial);
            Ok(success)
        };
        let selected = if goal.context == "fixed" {
            if probe(cap)? { Some(cap) } else { None }
        } else {
            search_context(cap, &mut probe)?
        };
        if let Some(context) = selected {
            let mut settings = family.settings.clone();
            settings["ctx"] = json!(context);
            settings["max_tokens"] =
                json!(settings_number(&baseline.settings, "max_tokens", 128)?.min(context - 1));
            fitted.push((context, settings, last_runtime));
        }
    }
    drop(source);
    let (context, settings, evidence) =
        fitted
            .into_iter()
            .max_by_key(|item| item.0)
            .with_context(|| {
                format!(
                    "No supported configuration passed memory and execution checks: {}",
                    trials
                        .last()
                        .and_then(|x| x["error"].as_str())
                        .unwrap_or("no candidates")
                )
            })?;
    let mut selected = baseline;
    selected.context = context;
    selected.settings = settings;
    selected.prepare = Some(goal.clone());
    ensure!(
        goal.context != "fixed" || context == request.context,
        "fixed context search cannot change the requested window"
    );
    let mut report = study_fixed(&selected, progress_path, true)?;
    report["schema_version"] = json!(2);
    report["mode"] = json!("prepare");
    report["declared_context"] = json!(declared);
    report["requested_context_ceiling"] = json!(ceiling);
    report["memory_fit_context"] = json!(context);
    report["recommended_context"] = json!(context);
    report["context_search_resolution_tokens"] = json!(256);
    report["fit_trials"] = json!(trials);
    report["fit_evidence"] = evidence;
    report["capability_adjustments"] = json!(adjustments);
    report["goal"] = serde_json::to_value(goal)?;
    report["goal_interpretation"] = json!("maximum_memory_fit_without_a_latency_target");
    report["scope"] = json!(
        "Largest passing memory-fit context on a bounded 256-token search grid, within declared model and backend limits. Fitting disables expert warmup and executes a short prefix; benchmarks use up to eight coding-prompt tokens, excluded warmup and two equal-length 4–128-step autoregressive samples with per-candidate time limits. This is not a full-window speed or answer-quality validation. Saved settings persist; RAM/GPU caches warm again when the model is loaded."
    );
    Ok(report)
}
fn study(request: &WorkerRequest, progress_path: &Path) -> Result<Value> {
    if let Some(goal) = &request.prepare {
        prepare_study(request, goal, progress_path)
    } else {
        study_fixed(request, progress_path, false)
    }
}

pub(crate) fn run_worker(
    request_path: &Path,
    progress_path: &Path,
    result_path: &Path,
    watch_parent_stdin: bool,
) -> Result<()> {
    ensure!(
        request_path != progress_path
            && request_path != result_path
            && progress_path != result_path,
        "worker files must have distinct paths"
    );
    ensure!(
        fs::metadata(request_path)?.len() <= 1024 * 1024,
        "tuning request exceeds 1 MiB"
    );
    let request: WorkerRequest = serde_json::from_slice(&fs::read(request_path)?)?;
    if watch_parent_stdin {
        std::thread::Builder::new()
            .name("tuner-parent-watch".into())
            .spawn(|| {
                let mut byte = [0u8; 1];
                loop {
                    match std::io::stdin().read(&mut byte) {
                        Ok(0) | Err(_) => std::process::exit(0),
                        Ok(_) => {}
                    }
                }
            })?;
    }
    let result = study(&request, progress_path)?;
    write_report(result_path, &result, Some(&request.model))?;
    write_report(
        progress_path,
        &json!({"stage":"complete","message":"Measurements complete","completed":1,"total":1,"unit":"study"}),
        Some(&request.model),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request() -> WorkerRequest {
        WorkerRequest {
            model: PathBuf::from("test.gguf"),
            context: 40960,
            machine: "test".into(),
            prepare: None,
            settings: json!({"backend":"hybrid","device":0,"ctx":40960,"max_tokens":2048,"parallel":3,"vram_mib":1024,"expert_ram_mib":512,"ram_margin_mib":4096,"expert_warmup":false}),
        }
    }
    #[test]
    fn setup_context_search_obeys_limits_and_resolution() {
        let mut tested = Vec::new();
        let best = search_context(65535, |ctx| {
            tested.push(ctx);
            Ok(ctx <= 18000)
        })
        .unwrap()
        .unwrap();
        assert!((17745..=18000).contains(&best));
        assert!(tested.len() < 24 && tested.iter().all(|n| *n <= 65535));
        assert_eq!(search_context(512, |_| Ok(true)).unwrap(), Some(512));
        assert_eq!(search_context(512, |_| Ok(false)).unwrap(), None);
        assert!(
            SetupGoal {
                context: "other".into(),
                max_context: None
            }
            .validate()
            .is_err()
        );
    }
    #[test]
    fn qwen35_setup_uses_cpu_or_hybrid_without_claiming_resident_gpu_support() {
        let mut req = request();
        req.settings["backend"] = json!("vulkan");
        let gpu = json!({"adapters":[{"index":0,"compatible":true,"software":false,"allocation_budget_bytes":1024*1024*1024}]});
        let cases = make_candidates(&req, 16, &gpu, 20 * 1024 * 1024 * 1024, "qwen35").unwrap();
        assert!(!cases.is_empty());
        assert!(
            cases
                .iter()
                .all(|case| matches!(case.backend.as_str(), "cpu" | "hybrid"))
        );
        assert!(cases.iter().any(|case| case.backend == "hybrid"));
    }
    #[test]
    fn candidate_generation_preserves_context_caps_and_explicit_gpu() {
        let req = request();
        let gpu = json!({"adapters":[{"index":0,"compatible":true,"software":false,"allocation_budget_bytes":1024*1024*1024}]});
        let cases = make_candidates(&req, 16, &gpu, 2 * 1024 * 1024 * 1024, "qwen3moe").unwrap();
        assert!(
            cases.iter().all(
                |c| c.settings["expert_ram_mib"] == 512 && c.settings["expert_warmup"] == false
            )
        );
        assert!(cases.iter().all(|c| c.settings["ctx"] == 40960
            && c.settings["max_tokens"] == 2048
            && c.settings["ram_margin_mib"] == 4096
            && c.settings["parallel"] == 1));
        assert!(
            cases
                .iter()
                .filter(|c| c.backend == "cpu")
                .all(|c| c.settings["device"].is_null() && c.settings["vram_mib"] == 0)
        );
        assert!(
            cases
                .iter()
                .filter(|c| c.backend == "hybrid")
                .all(|c| c.settings["device"] == 0 && c.settings["vram_mib"] == 1024)
        );
        assert!(cases.iter().any(|c| c.backend == "vulkan"));
        assert!(
            cases.iter().any(|c| c.threads == 4)
                && cases.iter().any(|c| c.threads == 8)
                && cases.iter().any(|c| c.threads == 16)
        );
    }
    #[test]
    fn tuning_preserves_explicit_expert_access_and_never_enables_direct_automatically() {
        let mut req=request();req.settings["backend"]=json!("vulkan");
        let hardware=json!({"adapters":[{"index":0,"compatible":true,"software":false,"direct_expert_weights_supported":true,"allocation_budget_bytes":1024*1024*1024}]});
        for mode in [None,Some(false),Some(true)] {
            if let Some(mode)=mode {req.settings["direct_experts"]=json!(mode);} else {req.settings.as_object_mut().unwrap().remove("direct_experts");}
            let candidates=make_candidates(&req,8,&hardware,20*1024*1024*1024,"qwen3moe").unwrap();
            assert!(!candidates.is_empty());
            for candidate in &candidates {
                let selected=backend_args(&candidate.settings).unwrap().direct_experts;
                if candidate.backend=="vulkan" {assert_eq!(selected,mode);}
                else {assert_ne!(selected,Some(true));}
            }
            if mode==Some(true) {assert!(candidates.iter().all(|c|c.backend=="vulkan"));}
        }
    }
    #[test]
    fn fast_math_study_keeps_the_requested_numerical_mode() {
        let mut req = request();
        req.settings["backend"] = json!("vulkan");
        req.settings["fast_math"] = json!(true);
        let gpu = json!({"adapters":[{"index":0,"compatible":true,"software":false,"fast_math_available":true,"allocation_budget_bytes":1024*1024*1024}]});
        let cases = make_candidates(&req, 16, &gpu, 2 * 1024 * 1024 * 1024, "qwen3moe").unwrap();
        assert!(!cases.is_empty());
        assert!(
            cases
                .iter()
                .all(|c| c.backend == "vulkan" && c.settings["fast_math"] == true)
        );
        assert!(
            cases
                .iter()
                .all(|c| backend_args(&c.settings).unwrap().fast_math)
        );
    }

    #[test]
    fn compact_context_study_keeps_precision_and_gpu_execution() {
        let mut req = request();
        req.settings["backend"] = json!("vulkan");
        req.settings["kv_precision"] = json!("q4_64");
        req.settings["fast_math"] = json!(false);
        let gpu = json!({"adapters":[{"index":0,"compatible":true,"software":false,"allocation_budget_bytes":1024*1024*1024}]});
        let cases = make_candidates(&req, 16, &gpu, 2 * 1024 * 1024 * 1024, "qwen3moe").unwrap();
        assert!(!cases.is_empty());
        assert!(
            cases
                .iter()
                .all(|c| c.backend == "vulkan" && c.settings["kv_precision"] == "q4_64")
        );
        assert!(
            cases
                .iter()
                .all(|c| backend_args(&c.settings).unwrap().kv_precision.is_compact())
        );
    }

    #[test]
    fn cpu_only_study_never_adds_gpu_and_thread_counts_are_unique() {
        let mut req = request();
        req.settings["backend"] = json!("cpu");
        req.settings["device"] = Value::Null;
        req.settings["vram_mib"] = json!(0);
        let cases = make_candidates(&req, 1, &json!({"adapters":[]}), 1000, "qwen3moe").unwrap();
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].backend, "cpu");
        assert_eq!(thread_counts(2, 1), vec![1, 2]);
    }
}
