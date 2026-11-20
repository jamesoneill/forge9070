//! Standalone, explicitly selected execution and measurement for the new engine.
use crate::{
    gguf::Gguf,
    model::{Model, Step},
    runtime::{BackendArgs, RuntimeModel},
};
use anyhow::{Context, Result, ensure};
use clap::{Args, Subcommand};
use rayon::prelude::*;
use serde::Serialize;
use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

#[derive(Args)]
pub struct EngineArgs {
    /// CPU worker budget for this invocation; does not change process-global pools.
    #[arg(long, global = true)]
    pub threads: Option<usize>,
    #[command(subcommand)]
    pub command: EngineCommand,
}

#[derive(Subcommand)]
pub enum EngineCommand {
    /// Serve the independent runtime through the OpenAI chat API and bounded FIFO queue.
    Serve(crate::server::ServeArgs),
    /// Report CPU availability and eligible Vulkan adapters without claiming execution was verified.
    Doctor,
    /// Compare runtime-selected CPU dot kernels with their scalar numerical reference.
    KernelBench {
        #[arg(long, default_value_t = 2048)]
        width: usize,
        #[arg(long, default_value_t = 128)]
        rows: usize,
        #[arg(long, default_value_t = 8)]
        iterations: usize,
        #[arg(long, default_value_t = 5)]
        reps: usize,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Build bounded, tracked repository excerpts without running code.
    Context {
        /// Git worktree or subdirectory to inspect.
        #[arg(long)]
        repo: PathBuf,
        #[arg(long)]
        query: String,
        /// Optional local GGUF for exact tokenizer counts.
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long, default_value_t = 8192)]
        ctx: usize,
        #[arg(long, default_value_t = 2048)]
        reserve_output: usize,
        #[arg(long, default_value_t = 2048)]
        reserve_other: usize,
        #[arg(long)]
        budget_tokens: Option<usize>,
        #[arg(long, default_value_t = 131072)]
        max_bytes: usize,
        #[arg(long = "include")]
        include_paths: Vec<PathBuf>,
        #[arg(long)]
        include_untracked: bool,
        #[arg(long, default_value_t = 24)]
        max_files: usize,
        #[arg(long, default_value_t = 128)]
        scan_mib: usize,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Print the actual implementation and supported capabilities as JSON.
    Capabilities,
    /// Validate a supported checkpoint with the independent loader and model geometry.
    Inspect {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 4096)]
        ctx: usize,
    },
    /// Greedy raw-prompt generation using the explicitly selected independent backend.
    Generate {
        #[command(flatten)]
        input: Input,
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        /// Return generated text, token IDs and measured timings as one JSON object.
        #[arg(long)]
        json: bool,
    },
    /// Measure prefill and decode separately; warmup is excluded from reported samples.
    Bench {
        #[command(flatten)]
        input: Input,
        #[arg(long, default_value_t = 128)]
        decode_steps: usize,
        #[arg(long, default_value_t = 5)]
        reps: usize,
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        /// Independent sequences sharing model weights and the selected backend.
        #[arg(long, default_value_t = 1)]
        requests: usize,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
}

#[derive(Args)]
pub struct Input {
    #[arg(long)]
    model: PathBuf,
    #[command(flatten)]
    backend: BackendArgs,
    #[arg(long, default_value_t = 4096)]
    ctx: usize,
    /// Raw text, without an implicit chat template. Use - to read standard input.
    #[arg(long)]
    prompt: String,
}

pub fn run(args: EngineArgs) -> Result<()> {
    if let EngineCommand::Serve(server) = args.command {
        return crate::server::serve(server, args.threads);
    }
    if let Some(n) = args.threads {
        ensure!((1..=1024).contains(&n), "--threads must be in 1..=1024");
        let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build()?;
        pool.install(|| dispatch(args.command))
    } else {
        dispatch(args.command)
    }
}

fn dispatch(command: EngineCommand) -> Result<()> {
    match command {
        EngineCommand::Serve(_) => {
            unreachable!("serve dispatches with its own shared worker budget")
        }
        EngineCommand::Doctor => print_json(&crate::runtime::hardware_report()),
        EngineCommand::KernelBench {
            width,
            rows,
            iterations,
            reps,
            json_out,
        } => {
            let report = crate::performance::kernel_benchmark(width, rows, iterations, reps)?;
            if let Some(path) = json_out {
                write_report(&path, &report, None)?;
            }
            print_json(&report)
        }
        EngineCommand::Context { repo, query, model, ctx, reserve_output, reserve_other,
            budget_tokens, max_bytes, include_paths, include_untracked, max_files,
            scan_mib, output, json } => {
            ensure!((128..=131072).contains(&ctx), "ctx must be 128..=131072");
            ensure!(reserve_output > 0 && reserve_other > 0, "reserves must be positive");
            ensure!((1..=2048).contains(&scan_mib), "scan-mib must be 1..=2048");
            let available = ctx.checked_sub(reserve_output)
                .and_then(|n| n.checked_sub(reserve_other))
                .filter(|n| *n > 0)
                .context("context window cannot fit requested reserves")?;
            ensure!(budget_tokens.is_none() || model.is_some(),
                "--budget-tokens requires --model for exact tokenizer counts");
            let budget = model.as_ref().map(|_| budget_tokens.unwrap_or(available));
            ensure!(budget.is_none_or(|n| n > 0 && n <= available),
                "context-pack token budget exceeds available context");
            let tokenizer = if let Some(path) = &model {
                let source = Gguf::open(path)?;
                Some(crate::tokenizer::Tokenizer::from_gguf(&source)?)
            } else { None };
            let token_counter = |text: &str| -> Result<usize> {
                Ok(tokenizer.as_ref().context("tokenizer not selected")?.encode(text)?.len())
            };
            let options = crate::context::Options {
                root: repo, query, include_paths, include_untracked,
                max_output_bytes: max_bytes, max_tokens: budget,
                max_scan_bytes: (scan_mib as u64) << 20, max_files,
                ..Default::default()
            };
            let result = crate::context::build(&options,
                tokenizer.as_ref().map(|_| &token_counter as &dyn Fn(&str) -> Result<usize>))?;
            if let Some(path) = &output {
                std::fs::write(path, &result.context)?;
            }
            if json { print_json(&result) } else {
                if output.is_none() { print!("{}", result.context); }
                Ok(())
            }
        }
        EngineCommand::Capabilities => print_json(&crate::capabilities()),
        EngineCommand::Inspect { model, ctx } => {
            let started = Instant::now();
            let g = Arc::new(Gguf::open(&model)?);
            let m = Model::load(g.clone(), ctx)?;
            print_json(&serde_json::json!({
                "engine": crate::ENGINE_ID, "config": m.config(),
                "model": model, "weight_file_bytes": g.mapped_bytes(), "shards": g.shard_paths(),
                "qwen38": m.qwen38_config(),
                "load_seconds": started.elapsed().as_secs_f64(),
                "kv_precision": "F32", "compute_submitted": false,
                "gpu_execution_tested": false, "validation": "checkpoint_metadata_tensor_geometry_and_supported_quantization", "fallback": "none"
            }))
        }
        EngineCommand::Generate {
            input,
            max_tokens,
            json,
        } => generate(input, max_tokens, json),
        EngineCommand::Bench {
            input,
            decode_steps,
            reps,
            warmup,
            requests,
            json_out,
        } => bench(
            input,
            decode_steps,
            reps,
            warmup,
            requests,
            json_out.as_deref(),
        ),
    }
}

fn print_json(value: &impl Serialize) -> Result<()> {
    let mut out = std::io::stdout().lock();
    serde_json::to_writer_pretty(&mut out, value)?;
    writeln!(out)?;
    Ok(())
}

struct Loaded {
    model: RuntimeModel,
    prompt: Vec<u32>,
    load_seconds: f64,
    tokenize_seconds: f64,
}

fn load(input: &Input, reserve: usize) -> Result<Loaded> {
    ensure!(reserve > 0, "output/decode count must be positive");
    let start = Instant::now();
    let model = RuntimeModel::load(&input.model, input.ctx, &input.backend)?;
    let load_seconds = start.elapsed().as_secs_f64();
    let text = if input.prompt == "-" {
        let mut s = String::new();
        // Bound the transport too; callers can select exact context with --ctx.
        std::io::stdin()
            .take(64 * 1024 * 1024 + 1)
            .read_to_string(&mut s)?;
        ensure!(
            s.len() <= 64 * 1024 * 1024,
            "standard-input prompt exceeds 64 MiB"
        );
        s
    } else {
        input.prompt.clone()
    };
    let start = Instant::now();
    let prompt = model.tokenizer.encode(&text)?;
    ensure!(!prompt.is_empty(), "prompt must contain at least one token");
    ensure!(
        prompt
            .len()
            .checked_add(reserve)
            .is_some_and(|n| n <= model.context()),
        "prompt ({} tokens) plus output/decode reservation ({reserve}) exceeds context {}",
        prompt.len(),
        model.context()
    );
    Ok(Loaded {
        model,
        prompt,
        load_seconds,
        tokenize_seconds: start.elapsed().as_secs_f64(),
    })
}

fn greedy(logits: &[f32]) -> Result<u32> {
    ensure!(!logits.is_empty(), "empty logits");
    ensure!(
        logits.iter().all(|v| v.is_finite()),
        "non-finite logits; generation aborted"
    );
    let mut best = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    u32::try_from(best).context("token ID exceeds u32")
}

fn generate(input: Input, max_tokens: usize, json: bool) -> Result<()> {
    let loaded = load(&input, max_tokens)?;
    let model = &loaded.model;
    let mut sequence = model.new_sequence()?;
    let prefill_start = Instant::now();
    let mut output = model.prefill(&mut sequence, &loaded.prompt)?;
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();
    let decode_start = Instant::now();
    let mut ids = Vec::with_capacity(max_tokens);
    let mut finish = "length";
    let mut steps = 0usize;
    for i in 0..max_tokens {
        let token = greedy(&output.logits)?;
        if loaded.model.tokenizer.is_eos(token) {
            finish = "stop";
            break;
        }
        ids.push(token);
        if i + 1 < max_tokens {
            model.step_into(&mut sequence, token, &mut output)?;
            steps += 1;
        }
    }
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    let text = loaded.model.tokenizer.decode(&ids)?;
    if json {
        print_json(&serde_json::json!({
            "engine": crate::ENGINE_ID, "engine_version": crate::ENGINE_VERSION,
            "runtime": model.report(), "model": input.model,
            "text": text, "token_ids": ids, "finish_reason": finish,
            "prompt_tokens": loaded.prompt.len(), "generated_tokens": ids.len(),
            "load_seconds": loaded.load_seconds, "tokenize_seconds": loaded.tokenize_seconds,
            "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
            "decode_steps": steps,
            "decode_steps_per_second": rate(steps, decode_seconds),
            "kv_precision": "F32", "gpu_performance_verified": false
        }))
    } else {
        println!("{text}");
        eprintln!(
            "{} | {:?} | {} prompt tokens in {:.3}s | {} decode steps in {:.3}s | {} output tokens | {finish}",
            crate::ENGINE_ID,
            model.backend(),
            loaded.prompt.len(),
            prefill_seconds,
            steps,
            decode_seconds,
            ids.len()
        );
        Ok(())
    }
}

#[derive(Serialize)]
struct Measurement {
    prefill_seconds: f64,
    decode_seconds: f64,
    aggregate_prefill_tokens_per_second: f64,
    aggregate_decode_steps_per_second: f64,
    request_prefill_seconds: Vec<f64>,
    request_decode_completion_seconds: Vec<f64>,
    greedy_checksum: Vec<u64>,
}

fn measure(
    model: &RuntimeModel,
    prompt: &[u32],
    steps: usize,
    requests: usize,
) -> Result<Measurement> {
    let mut states: Vec<_> = (0..requests)
        .map(|_| Ok((model.new_sequence()?, Step::default())))
        .collect::<Result<_>>()?;
    let start = Instant::now();
    let prefill: Result<Vec<f64>> = states
        .par_iter_mut()
        .map(|(state, output)| {
            *output = model.prefill(state, prompt)?;
            Ok(start.elapsed().as_secs_f64())
        })
        .collect();
    let request_prefill_seconds = prefill?;
    let prefill_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let runs: Result<Vec<(f64, u64)>> = states
        .par_iter_mut()
        .map(|(state, output)| {
            let mut checksum = 0xcbf29ce484222325u64;
            for _ in 0..steps {
                let token = greedy(&output.logits)?;
                checksum = (checksum ^ token as u64).wrapping_mul(0x100000001b3);
                model.step_into(state, token, output)?;
            }
            Ok((start.elapsed().as_secs_f64(), checksum))
        })
        .collect();
    let runs = runs?;
    let decode_seconds = start.elapsed().as_secs_f64();
    Ok(Measurement {
        prefill_seconds,
        decode_seconds,
        aggregate_prefill_tokens_per_second: rate(prompt.len() * requests, prefill_seconds),
        aggregate_decode_steps_per_second: rate(steps * requests, decode_seconds),
        request_prefill_seconds,
        request_decode_completion_seconds: runs.iter().map(|r| r.0).collect(),
        greedy_checksum: runs.iter().map(|r| r.1).collect(),
    })
}

fn rate(n: usize, seconds: f64) -> f64 {
    if seconds > 0.0 {
        n as f64 / seconds
    } else {
        0.0
    }
}

fn percentile(mut values: Vec<f64>, p: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() as f64 * p).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1)]
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let i = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[i - 1] + values[i]) / 2.0
    } else {
        values[i]
    }
}

fn check_report_destination(model: &Path, report: &Path) -> Result<()> {
    if report.try_exists()? {
        ensure!(
            !same_file::is_same_file(model, report)?,
            "--json-out must not overwrite the loaded model (including links to that file)"
        );
        ensure!(report.is_file(), "--json-out must name a file");
    }
    Ok(())
}

fn write_report(path: &Path, report: &serde_json::Value, model: Option<&Path>) -> Result<()> {
    if let Some(model) = model {
        check_report_destination(model, path)?;
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let mut selected = None;
    for _ in 0..64 {
        let number = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let temp = parent.join(format!(
            ".nextora-engine-report-{}-{number}.tmp",
            std::process::id()
        ));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
        {
            Ok(file) => {
                selected = Some((temp, file));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    let (temp, mut file) = selected.context("unable to create unique report file")?;
    let result = (|| -> Result<()> {
        serde_json::to_writer_pretty(&mut file, report)?;
        writeln!(file)?;
        file.sync_all()?;
        drop(file);
        if let Some(model) = model {
            check_report_destination(model, path)?;
        }
        // Replace the directory entry, never truncate a potentially mapped inode.
        std::fs::rename(&temp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}

fn bench(
    mut input: Input,
    steps: usize,
    reps: usize,
    warmup: usize,
    requests: usize,
    path: Option<&Path>,
) -> Result<()> {
    ensure!((1..=100).contains(&reps), "--reps must be in 1..=100");
    ensure!((1..=20).contains(&warmup), "--warmup must be in 1..=20");
    ensure!((1..=4).contains(&requests), "--requests must be in 1..=4");
    if let Some(path) = path {
        check_report_destination(&input.model, path)?;
    }
    input.backend.state_slots = requests;
    let loaded = load(&input, steps)?;
    if let Some(report) = path {
        for shard in loaded.model.source.shard_paths() {
            check_report_destination(shard, report)?;
        }
    }
    for _ in 0..warmup {
        measure(&loaded.model, &loaded.prompt, steps, requests)?;
    }
    let samples: Vec<_> = (0..reps)
        .map(|_| measure(&loaded.model, &loaded.prompt, steps, requests))
        .collect::<Result<_>>()?;
    let base_checksum = samples[0].greedy_checksum[0];
    ensure!(
        samples
            .iter()
            .flat_map(|s| &s.greedy_checksum)
            .all(|v| *v == base_checksum),
        "greedy results changed across identical sequences/repetitions; refusing to report a successful benchmark"
    );
    let doc = serde_json::json!({
        "engine": crate::ENGINE_ID, "engine_version": crate::ENGINE_VERSION,
        "runtime": loaded.model.report(), "model": input.model,
        "model_file_bytes": loaded.model.source.mapped_bytes(),
        "model_config": loaded.model.config(),
        "worker_threads": rayon::current_num_threads(), "concurrent_requests": requests,
        "prompt_tokens_per_request": loaded.prompt.len(), "decode_steps_per_request": steps,
        "warmup_iterations_excluded": warmup, "repetitions": reps,
        "load_seconds_excluded": loaded.load_seconds, "tokenize_seconds_excluded": loaded.tokenize_seconds,
        "median_aggregate_decode_steps_per_second": median(samples.iter().map(|s| s.aggregate_decode_steps_per_second).collect()),
        "median_aggregate_prefill_tokens_per_second": median(samples.iter().map(|s| s.aggregate_prefill_tokens_per_second).collect()),
        "p95_request_decode_completion_seconds": percentile(samples.iter().flat_map(|s| s.request_decode_completion_seconds.iter().copied()).collect(),0.95),
        "identical_greedy_checksums": true,
        "samples": samples,
        "measurement_notes": [
            "Backend and actual adapter are recorded; software Vulkan is never a hardware performance result",
            "Fixed decode-step workload continues through EOS for repeatability",
            "Concurrent requests share immutable weights, a bounded worker pool, and independent state",
            "Request completion durations include scheduling within the measured phase",
            "Timing excludes weight loading, tokenization and warmup",
            "Tiny synthetic checkpoints measure harness behavior, not useful-model throughput"
        ]
    });
    if let Some(path) = path {
        for shard in loaded.model.source.shard_paths() {
            check_report_destination(shard, path)?;
        }
        write_report(path, &doc, Some(&input.model))?;
    }
    print_json(&doc)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn greedy_checks_finite_values_and_stable_ties() {
        assert_eq!(greedy(&[1.0, 2.0, 2.0]).unwrap(), 1);
        assert!(greedy(&[]).is_err());
        assert!(greedy(&[f32::NAN]).is_err());
        assert!(greedy(&[f32::INFINITY]).is_err());
    }
    #[test]
    fn rates_and_percentiles_are_well_defined() {
        assert_eq!(rate(10, 2.0), 5.0);
        assert_eq!(rate(10, 0.0), 0.0);
        assert_eq!(percentile(vec![3.0, 1.0, 2.0], 0.5), 2.0);
        assert_eq!(percentile(vec![3.0, 1.0, 2.0], 0.95), 3.0);
        assert_eq!(median(vec![1.0, 4.0, 3.0, 2.0]), 2.5);
    }
    #[test]
    fn report_rejects_the_model_and_hard_links_without_modifying_bytes() {
        let dir = std::env::temp_dir().join(format!("forge-report-alias-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("model.gguf");
        let link = dir.join("alias.json");
        std::fs::write(&model, b"unchanged source model").unwrap();
        assert!(check_report_destination(&model, &model).is_err());
        std::fs::hard_link(&model, &link).unwrap();
        assert!(check_report_destination(&model, &link).is_err());
        assert!(write_report(&link, &serde_json::json!({"bad":true}), Some(&model)).is_err());
        assert_eq!(std::fs::read(&model).unwrap(), b"unchanged source model");
        std::fs::remove_file(link).unwrap();
        std::fs::remove_file(model).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }
}
