//! Forge9070: independent Forge engine plus explicitly attributed compatibility tools.
//! Uses RustingStrata by Vasyl Trefilov. See LICENSE.md and MODIFICATIONS.md.

mod doctor;
mod inspect;
mod profile;
mod hub;
mod code_context;

use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand};
use std::{ffi::OsString, net::IpAddr, path::PathBuf};
use rusting_strata::serving_policy::{self, Parallelism};

#[derive(Parser)]
#[command(version, about = "Native Rust GGUF inference for Radeon RX 9070 XT through Vulkan", after_help = "Uses RustingStrata by Vasyl Trefilov. ROCm is not used. Run `forge9070 doctor` first.")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Independent Forge CPU/Vulkan engine, coding API and queue; no Strata fallback.
    Engine {
        #[command(flatten)]
        engine: forge_engine::cli::EngineArgs,
    },
    /// Read-only Vulkan and environment discovery; does not run a compute kernel.
    Doctor {
        #[arg(long)] json: bool,
        /// Return an error unless a suitable RX 9070 XT is discovered.
        #[arg(long)] require_gpu: bool,
    },
    /// Inspect GGUF architecture, quantization and total weight bytes without loading weights.
    Inspect {
        #[arg(long)] model: PathBuf,
        #[arg(long)] json: bool,
    },
    /// Preview the expert RAM budget from model headers and available host memory; no GPU or weight loading.
    Memory {
        #[arg(long)] model: PathBuf,
        #[arg(long, default_value = "auto")] expert_ram: String,
        #[arg(long, default_value_t = 4096)] ram_margin: usize,
        #[arg(long, default_value_t = 0)] max_ram: usize,
        #[arg(long, default_value_t = 512)] cache_ram: usize,
        #[arg(long)] json: bool,
    },
    /// Print chosen RX 9070 XT starting settings; not a measured hardware calibration.
    Profile {
        #[arg(long, value_enum, default_value_t = profile::Preset::General)] preset: profile::Preset,
    },
    /// Build bounded, query-focused repository excerpts locally; no model inference or repository code execution.
    Context {
        /// Git worktree or a directory inside it. Tracked files only unless explicitly requested.
        #[arg(long)] repo: PathBuf,
        #[arg(long)] query: String,
        /// GGUF tokenizer for exact pack token counts; reads metadata and maps shards, not model inference.
        #[arg(long)] model: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = profile::Preset::Code)] preset: profile::Preset,
        /// Override the preset's total context allowance for planning the pack.
        #[arg(long)] ctx: Option<usize>,
        /// Reserve tokens for generated reasoning and the final answer.
        #[arg(long)] reserve_output: Option<usize>,
        /// Reserve for system prompt, chat template, tools and history outside this pack.
        #[arg(long, default_value_t = 2048)] reserve_other: usize,
        /// Exact token ceiling for the whole pack, including headings/query; requires --model.
        #[arg(long)] budget_tokens: Option<usize>,
        /// Hard UTF-8 output byte ceiling, independent of tokens.
        #[arg(long, default_value_t = 131072)] max_bytes: usize,
        /// Prioritize an eligible repository-relative path; repeatable.
        #[arg(long = "include")] include_paths: Vec<PathBuf>,
        #[arg(long)] include_untracked: bool,
        #[arg(long, default_value_t = 24)] max_files: usize,
        #[arg(long, default_value_t = 128)] scan_mib: usize,
        /// Write the UTF-8 context pack to this file instead of stdout.
        #[arg(long)] output: Option<PathBuf>,
        /// Emit context plus selection/budget metadata as JSON.
        #[arg(long)] json: bool,
    },
    /// Search/download Hugging Face GGUFs and manage the running server's model.
    Models {
        #[command(subcommand)] command: hub::ModelsCommand,
    },
    /// Generate text from a raw prompt; use --prompt - to read stdin.
    Generate {
        #[command(flatten)] engine: profile::EngineOptions,
        #[arg(long)] prompt: String,
        #[arg(long, default_value_t = 128,
            default_value_if("preset", "code", "4096"), default_value_if("preset", "code-long", "8192"))] max_tokens: usize,
        #[command(flatten)] sampling: Sampling,
    },
    /// Interactive terminal chat.
    Chat {
        #[command(flatten)] engine: profile::EngineOptions,
        #[arg(long)] system: Option<String>,
        #[arg(long, default_value_t = 1024,
            default_value_if("preset", "code", "4096"), default_value_if("preset", "code-long", "8192"))] max_tokens: usize,
        #[command(flatten)] sampling: Sampling,
    },
    /// Serve a browser chat page and OpenAI-compatible chat API on localhost.
    Serve {
        #[command(flatten)] engine: profile::EngineOptions,
        /// Shared model catalog/download directory [default: user data directory].
        #[arg(long)] models_dir: Option<PathBuf>,
        #[arg(long, default_value = "127.0.0.1")] host: IpAddr,
        #[arg(long, default_value_t = 8080)] port: u16,
        /// Cap the completion length accepted by the server.
        #[arg(long, default_value_t = 1024,
            default_value_if("preset", "code", "4096"), default_value_if("preset", "code-long", "8192"))] max_tokens: usize,
        /// Maximum generation requests waiting for a free slot (0 = immediate admission only).
        #[arg(long, default_value_t = serving_policy::DEFAULT_QUEUE_CAPACITY)] queue_capacity: usize,
        /// Maximum seconds a generation request may wait for a slot (1..86400).
        #[arg(long, default_value_t = serving_policy::DEFAULT_QUEUE_TIMEOUT_SECS)] queue_timeout: u64,
        /// Limit all accepted connections, including readers, generation and health; at least queue capacity plus slots.
        #[arg(long, default_value_t = serving_policy::DEFAULT_MAX_WAITING)] max_waiting: usize,
        #[command(flatten)] sampling: Sampling,
    },
    /// Measure this machine and model; writes the engine's measured timings if --json-out is set.
    Bench {
        #[command(flatten)] engine: profile::EngineOptions,
        #[arg(long, default_value = "128,512", value_delimiter = ',')] pp: Vec<usize>,
        #[arg(long, default_value_t = 128)] tg: usize,
        #[arg(long, default_value_t = 3)] reps: usize,
        /// Untimed repetitions before measurement.
        #[arg(long, default_value_t = 1)] warmup: usize,
        #[arg(long)] json_out: Option<PathBuf>,
        #[arg(long)] prompt_file: Option<PathBuf>,
    },
    /// Select measured decode settings that match baseline greedy outputs on test prompts.
    Tune {
        #[arg(long)] model: PathBuf,
        #[arg(long, default_value_t = 4096)] ctx: usize,
        #[arg(long, default_value_t = 2048)] vram_margin: usize,
        #[arg(long, default_value = "auto")] expert_ram: String,
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)] expert_warmup: bool,
        #[arg(long, default_value_t = 4096)] ram_margin: usize,
        #[arg(long, default_value_t = 0)] max_ram: usize,
        #[arg(long, default_value_t = 512)] cache_ram: usize,
        #[arg(long, value_enum, default_value_t = profile::Kv::F16)] kv: profile::Kv,
        #[arg(long)] device: Option<usize>,
        /// UTF-8 workload prompt; defaults to three built-in chat prompts.
        #[arg(long)] prompt_file: Option<PathBuf>,
        #[arg(long, default_value_t = 128)] tg: usize,
        #[arg(long, default_value_t = 3)] reps: usize,
        #[arg(long, default_value_t = 1)] warmup: usize,
        #[arg(long)] json_out: Option<PathBuf>,
        /// Fixed CPU worker count. Compare thread counts in separate invocations.
        #[arg(long)] threads: Option<usize>,
    },
    /// Check model Vulkan pipeline shared-memory requirements; no inference is executed.
    Check {
        #[arg(long)] model: PathBuf,
        /// Backend device index shown by doctor.
        #[arg(long)] device: Option<usize>,
    },
    /// Show attribution and the engine's distribution license.
    License,
}

#[derive(Args)]
struct Sampling {
    #[arg(long, default_value_t = 0.0,
        default_value_if("preset", "code", "1.0"), default_value_if("preset", "code-long", "1.0"))] temperature: f32,
    #[arg(long, default_value_t = 1.0,
        default_value_if("preset", "code", "0.95"), default_value_if("preset", "code-long", "0.95"))] top_p: f32,
    #[arg(long, default_value_t = 0,
        default_value_if("preset", "code", "20"), default_value_if("preset", "code-long", "20"))] top_k: usize,
    #[arg(long)] seed: Option<u64>,
}

fn arg(args: &mut Vec<OsString>, flag: &str, value: impl Into<OsString>) {
    args.push(flag.into());
    args.push(value.into());
}

fn positive(name: &str, n: usize) -> Result<()> {
    ensure!(n > 0, "{name} must be greater than zero");
    Ok(())
}

impl Sampling {
    fn append(&self, args: &mut Vec<OsString>) -> Result<()> {
        ensure!(self.temperature.is_finite() && self.temperature >= 0.0,
            "--temperature must be a finite nonnegative number");
        ensure!(self.top_p.is_finite() && self.top_p > 0.0 && self.top_p <= 1.0,
            "--top-p must be greater than 0 and at most 1");
        arg(args, "--temp", self.temperature.to_string());
        arg(args, "--top-p", self.top_p.to_string());
        arg(args, "--top-k", self.top_k.to_string());
        if let Some(seed) = self.seed { arg(args, "--seed", seed.to_string()); }
        Ok(())
    }
}

fn engine_args(command: &str, engine: &profile::EngineOptions) -> Result<Vec<OsString>> {
    engine.validate_command(command)?;
    let capacity = if let Some(model) = &engine.model {
        inspect::inspect(model)?.ensure_supported()?;
        let mode = if command == "serve" { engine.parallel } else { Parallelism::Fixed(1) };
        let plan = rusting_strata::cli::resolve_model_parallel(model, mode, engine.max_parallel, !engine.cpu, engine.mtp)?;
        rusting_strata::cli::validate_model_runtime(model, plan.slots, engine.mtp, engine.mtp_file.as_deref())?;
        rusting_strata::cli::validate_model_context(model, engine.ctx)?;
        Some(plan)
    } else {
        ensure!(command == "serve", "--model is required for {command}");
        None
    };
    // Process startup, before Vulkan or server/worker threads are created.
    unsafe { std::env::set_var("STRATA_EXPERT_TAIL_MIN_RANK", "0"); }
    let device = if engine.cpu { None } else {
        Some(doctor::collect().select_9070_xt(engine.device).context(
            "GPU selection failed. Run `forge9070 doctor`. In WSL use scripts/serve-wsl.sh with --backend cpu or the Windows GPU launcher scripts/serve-wsl-windows.sh when native Vulkan is unavailable. --cpu explicitly selects CPU inference.")?)
    };
    eprintln!("Forge9070: {} | context {} per request | chunk {} | KV {} | MTP {} | slots {}",
        if engine.cpu { "CPU (explicit)" } else { "RX 9070 XT / Vulkan" }, engine.ctx, engine.chunk(), engine.kv.as_str(), engine.mtp,
        capacity.as_ref().map_or_else(|| "awaiting model".into(), |p| p.slots.to_string()));
    if let Some(plan) = capacity { eprintln!("Capacity policy {}: {}", plan.mode, plan.reason); }
    else { eprintln!("Capacity policy {} (auto ceiling {}): resolved when a model is loaded", engine.parallel, engine.max_parallel); }
    eprintln!("Starting preset: {} (explicit flags override it; hardware speed is unmeasured)", engine.preset.name());
    eprintln!("All selected experts retained. KV precision and checkpoint quantization remain explicit choices.");
    eprintln!("Uses RustingStrata by Vasyl Trefilov. ROCm is not used.");
    engine.arguments(command, device)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let args = match cli.command {
        Command::Engine { engine } => return forge_engine::cli::run(engine),
        Command::Doctor { json, require_gpu } => {
            let report = doctor::collect();
            if json { println!("{}", serde_json::to_string_pretty(&report)?); }
            else { report.print_human(); }
            if require_gpu { report.select_9070_xt(None)?; }
            return Ok(());
        }
        Command::Inspect { model, json } => {
            let report = inspect::inspect(&model)?;
            if json { println!("{}", serde_json::to_string_pretty(&report)?); }
            else { report.print_human(); }
            report.ensure_supported()?;
            return Ok(());
        }
        Command::Memory { model, expert_ram, ram_margin, max_ram, cache_ram, json } => {
            use rusting_strata::memory;
            let requested = memory::parse_expert_ram(&expert_ram)?;
            for (name, value) in [("--ram-margin", ram_margin), ("--max-ram", max_ram), ("--cache-ram", cache_ram)] {
                ensure!(value <= 1_048_576, "{name} exceeds the 1 TiB input limit (MiB)");
            }
            let report = inspect::inspect(&model)?;
            report.ensure_supported()?;
            let plan = memory::expert_plan(memory::snapshot(), report.weight_memory.routed_expert_bytes, 0,
                (ram_margin as u64) << 20, (max_ram as u64) << 20, (cache_ram as u64) << 20, requested);
            let note = "Preview before model/tokenizer allocation. Recomputed at load. Pageable owned payload ceiling, not an OS reservation; sparse embeddings stay lazy; no GPU fit or speed test.";
            if json {
                println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                    "schema":"forge9070.memory.v1", "model":report.path, "weight_memory":report.weight_memory,
                    "plan":plan, "note":note
                }))?);
            } else {
                let gib = |n: u64| n as f64 / 1073741824.0;
                println!("Model: {}", report.path);
                println!("Available host RAM: {} GiB", plan.memory.available_bytes.map(|n| format!("{:.2}", gib(n))).unwrap_or_else(|| "unknown".into()));
                println!("Expert payload: {:.2} GiB | owned ceiling: {:.2} GiB | all experts fit this budget: {}",
                    gib(plan.expert_weight_bytes), gib(plan.budget_bytes), plan.all_experts_fit_budget);
                println!("Lazy embeddings: {:.2} GiB | other weight payload: {:.2} GiB",
                    gib(report.weight_memory.lazy_embedding_bytes), gib(report.weight_memory.other_weight_bytes));
                println!("Reserves: {:.2} GiB OS + {:.2} GiB runtime + {:.2} GiB prompt cache",
                    gib(plan.os_margin_bytes), gib(plan.runtime_reserve_bytes), gib(plan.prompt_cache_reserve_bytes));
                println!("{note}");
            }
            return Ok(());
        }
        Command::Profile { preset } => {
            println!("{}", serde_json::to_string_pretty(&profile::selected_profile(preset))?);
            return Ok(());
        }
        Command::Context { repo, query, model, preset, ctx, reserve_output, reserve_other, budget_tokens,
                           max_bytes, include_paths, include_untracked, max_files, scan_mib, output, json } => {
            let context_tokens = ctx.unwrap_or_else(|| preset.context());
            let output_tokens = reserve_output.unwrap_or_else(|| preset.output());
            ensure!((128..=131072).contains(&context_tokens), "--ctx must be between 128 and 131072 tokens");
            ensure!(output_tokens > 0, "--reserve-output must be greater than zero");
            let allowed = context_tokens.checked_sub(output_tokens).and_then(|n| n.checked_sub(reserve_other))
                .filter(|&n| n > 0).context("Context must leave room after --reserve-output and --reserve-other")?;
            ensure!(budget_tokens.is_none() || model.is_some(), "--budget-tokens requires --model for exact tokenizer counts");
            let token_budget = model.as_ref().map(|_| budget_tokens.unwrap_or(allowed));
            ensure!(token_budget.is_none_or(|n| n > 0 && n <= allowed), "--budget-tokens must be positive and fit context after both reserves ({allowed} tokens available)");
            ensure!((1..=2048).contains(&scan_mib), "--scan-mib must be 1..=2048");
            let tokenizer = if let Some(path) = &model {
                rusting_strata::cli::validate_model_context(path, context_tokens)?;
                let gguf = rusting_strata::gguf::Gguf::open(path)?;
                Some(rusting_strata::tokenizer::Tokenizer::from_gguf(&gguf)?)
            } else { None };
            let counter = |text: &str| -> Result<usize> { Ok(tokenizer.as_ref().context("Tokenizer unavailable")?.encode(text)?.len()) };
            let opts = code_context::Options {
                root: repo, query, include_paths, include_untracked, max_output_bytes: max_bytes,
                max_tokens: token_budget, max_scan_bytes: (scan_mib as u64) << 20, max_files,
                ..Default::default()
            };
            let report = code_context::build(&opts, tokenizer.as_ref().map(|_| &counter as &dyn Fn(&str) -> Result<usize>))?;
            if let Some(path) = &output {
                std::fs::write(path, &report.context).with_context(|| format!("write {}", path.display()))?;
            }
            if json {
                println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                    "schema":"forge9070.code-context.v1", "preset":preset.name(), "context_tokens":context_tokens,
                    "reserved_output_tokens":output_tokens, "reserved_other_tokens":reserve_other,
                    "pack_token_budget":token_budget, "tokenizer_model":model, "output":output, "pack":report,
                    "note":"Exact tokens describe this pack only. Other history/tools/template must fit their reserve; /api/context checks the actual complete request. No generation was run."
                }))?);
            } else {
                if output.is_none() { print!("{}", report.context); }
                eprintln!("Context pack: {} files, {} UTF-8 bytes, {} tokens; {}",
                    report.stats.files_selected, report.bytes, report.tokens.map(|n| n.to_string()).unwrap_or_else(|| "unknown (no --model supplied)".into()), report.ranking);
                for warning in &report.warnings { eprintln!("Note: {warning}"); }
                if let Some(path) = output { eprintln!("Wrote {}", path.display()); }
            }
            return Ok(());
        }
        Command::Models { command } => return hub::run(command),
        Command::License => {
            println!("Forge9070 modifications, 2026. Uses RustingStrata by Vasyl Trefilov.\n\n{}", include_str!("../LICENSE.md"));
            return Ok(());
        }
        Command::Generate { engine, prompt, max_tokens, sampling } => {
            positive("--max-tokens", max_tokens)?;
            ensure!(max_tokens < engine.ctx, "--max-tokens must be smaller than --ctx");
            let mut args = engine_args("generate", &engine)?;
            arg(&mut args, "--prompt", prompt);
            arg(&mut args, "-n", max_tokens.to_string());
            sampling.append(&mut args)?;
            args
        }
        Command::Chat { engine, system, max_tokens, sampling } => {
            positive("--max-tokens", max_tokens)?;
            ensure!(max_tokens < engine.ctx, "--max-tokens must be smaller than --ctx");
            let mut args = engine_args("chat", &engine)?;
            if let Some(system) = system { arg(&mut args, "--system", system); }
            arg(&mut args, "-n", max_tokens.to_string());
            sampling.append(&mut args)?;
            args
        }
        Command::Serve { engine, models_dir, host, port, max_tokens, queue_capacity, queue_timeout, max_waiting, sampling } => {
            ensure!(port != 0, "--port must be from 1 to 65535");
            positive("--max-tokens", max_tokens)?;
            serving_policy::validate_queue_limits(queue_capacity, queue_timeout, max_waiting, engine.maximum_capacity()?.slots)?;
            ensure!(max_tokens < engine.ctx, "--max-tokens must be smaller than --ctx");
            if !host.is_loopback() {
                let key = std::env::var("STRATA_API_KEY").unwrap_or_default();
                ensure!(!key.trim().is_empty(), "Set STRATA_API_KEY before binding outside loopback; the local default needs no key");
            }
            let mut args = engine_args("serve", &engine)?;
            let models_dir = models_dir.unwrap_or_else(hub::default_models_dir);
            std::fs::create_dir_all(&models_dir).with_context(|| format!("create model directory {}", models_dir.display()))?;
            arg(&mut args, "--models-dir", models_dir);
            arg(&mut args, "--addr", std::net::SocketAddr::new(host, port).to_string());
            arg(&mut args, "--max-tokens", max_tokens.to_string());
            arg(&mut args, "--queue-capacity", queue_capacity.to_string());
            arg(&mut args, "--queue-timeout", queue_timeout.to_string());
            arg(&mut args, "--max-waiting", max_waiting.to_string());
            sampling.append(&mut args)?;
            args
        }
        Command::Bench { engine, pp, tg, reps, warmup, json_out, prompt_file } => {
            positive("--tg", tg)?;
            positive("--reps", reps)?;
            ensure!(reps <= 100 && warmup <= 10, "benchmark --reps must be at most 100 and --warmup at most 10");
            ensure!(!pp.is_empty(), "--pp needs at least one prompt length");
            for &n in &pp {
                positive("--pp", n)?;
                ensure!(n.checked_add(tg).is_some_and(|sum| sum <= engine.ctx),
                    "Each --pp value plus --tg must fit --ctx; increase --ctx explicitly");
            }
            let mut args = engine_args("bench", &engine)?;
            arg(&mut args, "--pp", pp.iter().map(usize::to_string).collect::<Vec<_>>().join(","));
            arg(&mut args, "--tg", tg.to_string());
            arg(&mut args, "--reps", reps.to_string());
            arg(&mut args, "--warmup", warmup.to_string());
            if let Some(path) = json_out { arg(&mut args, "--json-out", path); }
            if let Some(path) = prompt_file { arg(&mut args, "--prompt-file", path); }
            args
        }
        Command::Check { model, device } => {
            inspect::inspect(&model)?.ensure_supported()?;
            let index = doctor::collect().select_9070_xt(device)?;
            let mut args = vec!["forge9070".into(), "check".into()];
            arg(&mut args, "--model", model);
            arg(&mut args, "--backend", "vulkan");
            arg(&mut args, "--device", index.to_string());
            args
        }
        Command::Tune { model, ctx, vram_margin, expert_ram, expert_warmup, ram_margin, max_ram, cache_ram, kv, device, prompt_file, tg, reps, warmup, json_out, threads } => {
            rusting_strata::memory::parse_expert_ram(&expert_ram)?;
            for (name, value) in [("--ram-margin", ram_margin), ("--max-ram", max_ram), ("--cache-ram", cache_ram)] {
                ensure!(value <= 1_048_576, "{name} exceeds the 1 TiB input limit (MiB)");
            }
            ensure!((768..=131_072).contains(&ctx), "--ctx must be 768..=131072 for tuning");
            ensure!(vram_margin < 16_384, "--vram-margin must leave GPU memory for the model");
            ensure!((32..=1024).contains(&tg) && tg < ctx, "tuning --tg must be 32..=1024 and smaller than --ctx");
            ensure!((3..=15).contains(&reps), "tuning --reps must be 3..=15");
            ensure!((1..=5).contains(&warmup), "tuning --warmup must be 1..=5");
            ensure!(threads.is_none_or(|n| (1..=1024).contains(&n)), "tuning --threads must be 1..=1024");
            inspect::inspect(&model)?.ensure_supported()?;
            unsafe { std::env::set_var("STRATA_EXPERT_TAIL_MIN_RANK", "0"); }
            let index = doctor::collect().select_9070_xt(device)?;
            let mut args = vec!["forge9070".into(), "calibrate".into()];
            arg(&mut args, "--model", model);
            arg(&mut args, "--ctx", ctx.to_string());
            arg(&mut args, "--vram-margin", vram_margin.to_string());
            arg(&mut args, "--expert-ram", expert_ram);
            arg(&mut args, "--expert-warmup", expert_warmup.to_string());
            arg(&mut args, "--ram-margin", ram_margin.to_string());
            arg(&mut args, "--max-ram", max_ram.to_string());
            arg(&mut args, "--cache-ram", cache_ram.to_string());
            arg(&mut args, "--kv", kv.as_str());
            arg(&mut args, "--backend", "vulkan");
            arg(&mut args, "--device", index.to_string());
            arg(&mut args, "--tg", tg.to_string());
            arg(&mut args, "--reps", reps.to_string());
            arg(&mut args, "--warmup", warmup.to_string());
            if let Some(path) = prompt_file { arg(&mut args, "--prompt-file", path); }
            if let Some(path) = json_out { arg(&mut args, "--json-out", path); }
            if let Some(n) = threads { arg(&mut args, "--threads", n.to_string()); }
            args
        }
    };
    // In-process Rust library call: no C++ inference runtime or subprocess engine.
    rusting_strata::cli::run_from(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_rejects_nan_and_invalid_probability() {
        let mut args = Vec::new();
        assert!(Sampling { temperature: f32::NAN, top_p: 1.0, top_k: 0, seed: None }.append(&mut args).is_err());
        assert!(Sampling { temperature: 0.0, top_p: 0.0, top_k: 0, seed: None }.append(&mut args).is_err());
    }

    #[test]
    fn clap_interface_is_consistent() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
        let Cli { command: Command::Serve { engine, .. } } = Cli::parse_from(["forge9070", "serve", "--cpu"])
            else { panic!("serve arguments parsed as another command") };
        assert_eq!(engine.kv, profile::Kv::F16);
        assert_eq!((engine.parallel, engine.max_parallel, engine.mtp, engine.expert_tail_min_rank), (Parallelism::Auto, 4, 0, 0));
    }

    #[test]
    fn serve_admission_defaults_and_explicit_immediate_only_mode_are_valid() {
        let Cli { command: Command::Serve { engine, queue_capacity, queue_timeout, max_waiting, .. } } =
            Cli::parse_from(["forge9070", "serve", "--cpu"]) else { panic!("serve") };
        assert_eq!((queue_capacity, queue_timeout, max_waiting), (32, 120, 64));
        assert_eq!(engine.maximum_capacity().unwrap().slots, 1);
        serving_policy::validate_queue_limits(queue_capacity, queue_timeout, max_waiting, 1).unwrap();
        let Cli { command: Command::Serve { engine, queue_capacity, queue_timeout, max_waiting, .. } } =
            Cli::parse_from(["forge9070", "serve", "--cpu", "--parallel", "1", "--queue-capacity", "0",
                "--queue-timeout", "5", "--max-waiting", "1"]) else { panic!("serve") };
        assert_eq!(engine.parallel, Parallelism::Fixed(1));
        serving_policy::validate_queue_limits(queue_capacity, queue_timeout, max_waiting, 1).unwrap();
        assert!(serving_policy::validate_queue_limits(32, 120, 16, 1).is_err());
    }

    #[test]
    fn parallel_cli_parses_only_documented_modes_and_preserves_fixed_requests() {
        for value in ["auto", "1", "2", "3", "4"] {
            let Cli { command: Command::Serve { engine, .. } } =
                Cli::parse_from(["forge9070", "serve", "--parallel", value, "--max-parallel", "2"])
                else { panic!("serve") };
            assert_eq!(engine.parallel.to_string(), value);
            assert_eq!(engine.maximum_capacity().unwrap().slots, if value == "auto" { 2 } else { value.parse().unwrap() });
        }
        for value in ["0", "5", "-1", "all"] {
            assert!(Cli::try_parse_from(["forge9070", "serve", "--parallel", value]).is_err());
        }
    }

    #[test]
    fn code_presets_reserve_output_and_leave_explicit_overrides_in_control() {
        for (preset, ctx, output) in [("code",16384,4096),("code-long",32768,8192)] {
            let Cli { command: Command::Serve { engine, max_tokens, sampling, .. } } =
                Cli::parse_from(["forge9070","serve","--cpu","--preset",preset]) else { panic!("serve") };
            assert_eq!((engine.ctx,max_tokens),(ctx,output));
            assert_eq!((engine.ram_margin,engine.cache_ram,engine.parallel,engine.mtp),(6144,1024,Parallelism::Auto,0));
            assert_eq!((sampling.temperature,sampling.top_p,sampling.top_k),(1.0,0.95,20));
            assert_eq!(engine.kv,profile::Kv::F16);
            assert!(engine.think && engine.preserve_thinking);
            assert!(engine.prompt_cache.is_none());
        }
        let Cli { command: Command::Serve { engine, max_tokens, sampling, .. } } = Cli::parse_from([
            "forge9070","serve","--cpu","--preset","code-long","--ctx","2048","--max-tokens","256",
            "--cache-ram","0","--ram-margin","8192","--temperature","0.2","--top-k","40","--think","false"
        ]) else { panic!("serve") };
        assert_eq!((engine.ctx,max_tokens,engine.cache_ram,engine.ram_margin),(2048,256,0,8192));
        assert_eq!((sampling.temperature,sampling.top_k),(0.2,40));
        assert!(!engine.think);
    }

    #[test]
    fn profile_values_match_parsed_cli_defaults_for_each_preset() {
        for (preset, flag) in [(profile::Preset::General,"general"),(profile::Preset::Code,"code"),(profile::Preset::CodeLong,"code-long")] {
            let p=profile::selected_profile(preset);
            let Cli { command: Command::Serve { engine, max_tokens, .. } } = Cli::parse_from(["forge9070","serve","--cpu","--preset",flag]) else { panic!("serve") };
            assert_eq!(p["context_tokens"],engine.ctx);
            assert_eq!(p["maximum_output_tokens"],max_tokens);
            assert_eq!(p["prompt_cache_ram_mib"],engine.cache_ram);
            assert_eq!(p["ram_margin_mib"],engine.ram_margin);
        }
    }
}
