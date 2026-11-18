//! Native CLI bridge for applications that embed the RustingStrata engine.
//!
//! Forge9070 modification: the original executable implementation lives here so
//! callers can run the engine in the same process with OS-native arguments.
//! Original RustingStrata author: Vasyl Trefilov. See ../LICENSE.md.

use clap::{Args, Parser, Subcommand};
use anyhow::Context;
use rusting_strata::memory;
use rusting_strata::admission::{Admission, AdmissionError, Permit};
use rusting_strata::serving_policy::{Parallelism, CapacityPlan, ModelFeatures, resolve_capacity, validate_queue_limits};
use rusting_strata::model::{Model, State};
use rusting_strata::{backend, gpu_model::GpuModel};
use rusting_strata::{config::ModelConfig, gguf::Gguf, hf, ops, ops::Sampler, quant::GgmlType, serve, tokenizer::Tokenizer};
use rusting_strata::chat::{Hold, Msg, Template, Think};
use rusting_strata::vision::{self, VisionProjector};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::io::{BufRead, BufReader, Read, Write};
use std::{collections::BTreeMap, net::{TcpListener, TcpStream}, sync::atomic::Ordering::Relaxed, path::{Path, PathBuf}, time::{Duration, Instant, SystemTime}};

#[derive(Parser)]
#[command(name = "rustingstrata", version, about = "Rust inference engine for Qwen3.5/3.6-family GGUF models")]
struct Cli {
    /// Without a command, strata serves the chat page on 127.0.0.1:8080 on the GPU (else the CPU) and opens it in
    /// the browser.
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Args, Clone)]
struct Engine {
    /// Run on the GPU (experts that do not fit in VRAM run on the CPU) instead of the CPU reference.
    #[arg(long)]
    gpu: bool,
    /// Context tokens, including prompt and output (128..=131072 and no more than the model's training context).
    #[arg(long, default_value_t = 4096)]
    ctx: usize,
    /// Prefill chunk in tokens (GPU; 0 = token by token) [default: calibrated, else 4096].
    #[arg(long)]
    chunk: Option<usize>,
    /// VRAM left free after placing expert layers, in MB. One value applies to every device; `512,256` gives one
    /// per device in `--device` order, and `0:512,1:256` gives one per GPU index.
    #[arg(long, default_value = "512", value_delimiter = ',')]
    vram_margin: Vec<String>,
    /// Speculative decoding with the model's MTP draft head, N drafts per step (GPU; preserves greedy/sampling results when expert-tail approximation is off)
    /// [default: calibrated, else 0].
    #[arg(long, value_name = "N", default_missing_value = "1", num_args = 0..=1)]
    mtp: Option<usize>,
    /// Standalone MTP draft head GGUF (e.g. ~/models-local/mtp-Q8_0.gguf).
    #[arg(long)]
    mtp_model: Option<PathBuf>,
    /// External speculative draft model GGUF socket (e.g. future DFlash/EAGLE draft).
    #[arg(long)]
    draft_model: Option<PathBuf>,
    /// Speculative context-lookup drafter type(s), comma-separated: none, ngram-simple, ngram-map-k, ngram-map-k4v,
    /// ngram-mod, ngram-cache (combines with --mtp/--mtp-model in hybrid cascade).
    #[arg(long, default_value = "none")]
    spec_type: String,
    /// Maximum number of speculative draft tokens verified per step (1..=16).
    #[arg(long, default_value_t = 16)]
    spec_draft_n_max: usize,
    /// N-gram lookup key length in tokens (for ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod).
    #[arg(long, default_value_t = 12)]
    spec_ngram_size_n: usize,
    /// N-gram draft window length in tokens (for ngram-simple, ngram-map-k, ngram-map-k4v).
    #[arg(long, default_value_t = 48)]
    spec_ngram_size_m: usize,
    /// Minimum hit count before drafting (for ngram-map-k, ngram-map-k4v).
    #[arg(long, default_value_t = 1)]
    spec_ngram_min_hits: u16,
    /// Static lookup cache file (llama.cpp binary ngram cache or UTF-8 text scaffold corpus).
    #[arg(long)]
    lookup_cache_static: Option<PathBuf>,
    /// Dynamic lookup cache file loaded at startup and updated/saved across requests.
    #[arg(long)]
    lookup_cache_dynamic: Option<PathBuf>,
    /// Persistent disk cache file for KV and recurrent prefix states (Warm Start across requests and restarts).
    #[arg(long)]
    prompt_cache: Option<PathBuf>,
    /// Sampling temperature; 0 is greedy.
    #[arg(long, default_value_t = 0.0)]
    temp: f32,
    /// Sample only from the K most likely tokens (0 = all).
    #[arg(long, default_value_t = 0)]
    top_k: usize,
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    /// Random seed for sampling [default: from the clock].
    #[arg(long)]
    seed: Option<u64>,
    /// CPU threads for expert work [default: physical cores on the GPU, else all logical cores].
    #[arg(long)]
    threads: Option<usize>,
    /// Match llama.cpp's CPU numerics on the GPU (q8_K activations, f32 prefill attention, f16 decode attention
    /// accumulator) for parity runs; attention gets much slower.
    #[arg(long)]
    llama_numerics: bool,
    /// Place experts in layer order and never move them, ignoring and keeping the routing profile, so parity runs
    /// repeat exactly.
    #[arg(long)]
    fixed_experts: bool,
    /// Host RAM left available for the OS and other applications, in MiB.
    #[arg(long, default_value_t = 4096)]
    ram_margin: usize,
    /// Soft cap for estimated owned host RAM in MiB; the expert tier also respects available RAM.
    #[arg(long, default_value_t = 0)]
    max_ram: usize,
    /// Owned expert RAM ceiling: auto, 0 to disable, or MiB. Sparse embeddings remain lazy.
    #[arg(long, default_value = "auto")]
    expert_ram: String,
    /// Fill available expert RAM slots at startup; false loads entries only when needed.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    expert_warmup: bool,
    /// Host memory in MB for conversation states that a request with another history replaced (GPU): a later request
    /// that continues one of them takes it back instead of prefilling it again. 0 keeps none.
    #[arg(long, default_value_t = 512)]
    cache_ram: usize,
    /// GGUF LoRA adapter(s) to apply at runtime as "path:scale" (repeatable; CPU and GPU).
    #[arg(long, value_name = "PATH:SCALE", action = clap::ArgAction::Append)]
    lora: Vec<String>,
    /// Keep the think block of earlier replies in the prompt (Qwen3.5/3.6 templates' `preserve_thinking`; a request
    /// can set it in `chat_template_kwargs`). The template drops it from replies before the last user message, so
    /// each new user message changes the earlier text and the whole conversation is prefilled again; with this, a
    /// coding agent's long history stays cached across user messages. `--preserve-thinking false` follows the
    /// template's default.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    preserve_thinking: bool,
    /// Let the model think before answering (chat and `serve`); `--think false` turns it off. A request's `chat_template_kwargs.enable_thinking` overrides it in `serve`.
    #[arg(long = "think", default_value_t = true, action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    think_default: bool,
    /// KV cache format (GPU): f16, or q8 (q8_0 blocks, half the KV memory; numerics differ slightly from
    /// llama.cpp's default f16 cache).
    #[arg(long, default_value = "f16", value_parser = ["f16", "q8"])]
    kv: String,
    /// GPU backend: cuda or vulkan [default: cuda if it loads, else vulkan].
    #[arg(long)]
    backend: Option<String>,
    /// GPU index, or several separated by commas (such as 0,1) to split the layers over those GPUs in proportion to
    /// their free VRAM.
    #[arg(long, default_value = "0", value_delimiter = ',')]
    device: Vec<usize>,
    /// Prompts shorter than this skip the batched prefill (from `strata calibrate`, else 32).
    #[arg(skip)]
    short: usize,
    /// Run on the CPU when no GPU backend opens (the bare `strata` launch).
    #[arg(skip)]
    cpu_fallback: bool,
    /// Sequences in GPU memory at once (`serve --parallel`).
    #[arg(skip)]
    parallel: usize,
    /// Requested serving policy; the effective count is resolved per loaded model.
    #[arg(skip)]
    parallel_mode: Parallelism,
    #[arg(skip = 4usize)]
    max_parallel: usize,
    /// Model/device/configuration used to validate saved calibration settings.
    #[arg(skip)]
    calibration: Option<CalibrationContext>,
    /// Multimodal projector (mmproj) GGUF used by `--image`.
    #[arg(long)]
    mmproj: Option<PathBuf>,
}

/// `serve` options for a server that other people use (docs/HOSTING.md).
#[derive(clap::Args, Clone, Debug)]
struct Hosting {
    /// File of customer keys, one `name key` pair a line (`#` comments). These keys reach only the chat APIs and
    /// the model list, and each key's cached conversations are kept apart from the others'. `--api-key` stays
    /// the admin key for every route. A changed file is read again before the next request.
    #[arg(long)]
    api_keys: Option<PathBuf>,
    /// Append one JSON line per chat request (time, key name, token counts, timings, how it finished) to this file.
    #[arg(long)]
    usage_log: Option<PathBuf>,
    /// Open connections (being read, generated or queued, health checks included); more get `503` with `Retry-After`
    /// [default: 64; separate from --queue-capacity].
    #[arg(long)]
    max_waiting: Option<usize>,
    /// Requests one customer key may have read, generated or queued at once; more get `429` with `Retry-After`
    /// [default: no limit besides `--max-waiting`].
    #[arg(long)]
    max_per_key: Option<usize>,
    /// Extra `Host` header names accepted by the chat page, model list and stats, comma-separated. Needed when the
    /// server is reached through a LAN name while bound to `0.0.0.0`. [default: $STRATA_ALLOWED_HOSTS]
    #[arg(long, value_delimiter = ',')]
    allowed_hosts: Vec<String>,
    /// Most reply tokens a request may ask for (and the default when it asks for none) [default: what the context
    /// leaves].
    #[arg(long)]
    max_tokens: Option<usize>,
    /// Serving capacity: auto respects model/backend/MTP limits, or select an exact count from 1 to 4.
    /// Each GPU slot reserves its own --ctx state at model load; active slots follow demand.
    #[arg(long, default_value = "auto")]
    parallel: Parallelism,
    /// Upper slot limit for auto mode. Slots are allocated at model load, not grown while decoding.
    #[arg(long, default_value_t = 4)]
    max_parallel: usize,
    /// Generation requests allowed to wait beyond active slots; zero rejects work when all slots are busy.
    #[arg(long, default_value_t = 32)]
    queue_capacity: usize,
    /// Maximum seconds a generation request may wait for an execution slot.
    #[arg(long, default_value_t = 120)]
    queue_timeout: u64,
}

enum Runner<'a> { Cpu(&'a Model<'a>, State), Gpu(Box<GpuModel<'a>>), Slot(Client) }

impl Runner<'_> {
    fn expert_ram_stats(&self) -> Option<rusting_strata::expert_ram::ExpertRamStats> {
        match self {
            Runner::Cpu(m, _) => Some(m.w.expert_ram_stats()),
            Runner::Gpu(g) => Some(g.expert_ram_stats()),
            Runner::Slot(_) => None,
        }
    }
    fn step(&mut self, t: u32) -> anyhow::Result<Vec<f32>> {
        match self { Runner::Cpu(m, st) => Ok(m.step(st, t)), Runner::Gpu(g) => g.step(t), Runner::Slot(c) => Ok(c.step(t, true)?.1) }
    }
    /// Feed a prompt; returns the logits after its last token.
    fn prefill(&mut self, ts: &[u32]) -> anyhow::Result<Vec<f32>> {
        match self {
            Runner::Cpu(m, st) => Ok(ts.iter().map(|&t| m.step_prefill(st, t)).last().unwrap_or_default()),
            Runner::Gpu(g) => g.prefill(ts),
            Runner::Slot(_) => anyhow::bail!("prefill on a slot"),
        }
    }
    /// Prefill exactly the token ids already checked by the request budget, with complete-prefix state reuse.
    fn chat_tokens(&mut self, tokens: &serve::ChatTokens) -> anyhow::Result<(Vec<f32>, usize)> {
        match self {
            Runner::Gpu(g) => g.prefill_cached(&tokens.ids, tokens.sys, tokens.keep),
            Runner::Slot(c) => c.chat(&tokens.ids, tokens.sys, tokens.keep),
            _ => { self.reset()?; Ok((self.prefill(&tokens.ids)?, 0)) }
        }
    }
    /// Feed a prompt that contains image embeddings as a fresh sequence. Only the CPU runner carries the model
    /// state needed to inject embeddings; GPU image prefill is a later milestone.
    fn chat_image(&mut self, p: &vision::Prompt, embs: &[Vec<Vec<f32>>]) -> anyhow::Result<Vec<f32>> {
        match self {
            Runner::Cpu(m, st) => { *st = State::new(&m.cfg); vision::feed_prompt(m, st, p, embs) }
            Runner::Gpu(g) => g.prefill_image(p, embs),
            Runner::Slot(_) => anyhow::bail!("image input is not supported on a serve slot"),
        }
    }
    /// Forget the sequence so far.
    fn reset(&mut self) -> anyhow::Result<()> {
        match self { Runner::Cpu(m, st) => { *st = State::new(&m.cfg); Ok(()) } Runner::Gpu(g) => g.reset(), Runner::Slot(_) => Ok(()) }
    }
    /// Seconds spent waiting on CPU experts so far (GPU only).
    fn cpu_secs(&self) -> f64 { if let Runner::Gpu(g) = self { g.parts().map(|p| p.cpu_time.as_secs_f64()).sum() } else { 0.0 } }
    /// Bytes of routed-expert weights one decode token reads from the cache (0 without an expert cache).
    fn expert_bytes_per_token(&self) -> usize {
        if let Runner::Gpu(g) = self { g.parts().map(|p| p.expert_bytes_per_token()).sum() } else { 0 }
    }
    /// Split model: seconds per part but the last (forward until its GPU is done, copy to the next part).
    fn split_secs(&self) -> Vec<(f64, f64)> {
        let Runner::Gpu(g) = self else { return vec![] };
        let mut v: Vec<_> = g.parts().map(|p| (p.split_time.0.as_secs_f64(), p.split_time.1.as_secs_f64())).collect();
        v.pop();
        v
    }
    /// Backend name of the first part (`"cpu"` on the CPU, `"slot"` for a serve slot).
    fn backend_name(&self) -> &'static str {
        match self {
            Runner::Cpu(..) => "cpu",
            Runner::Gpu(g) => g.parts().next().map_or("cpu", |p| p.gpu.name()),
            Runner::Slot(_) => "slot",
        }
    }
    /// Expert-cache counters summed over the parts, for `finish()` and the benchmark JSON (`None` off the GPU).
    fn expert_stats(&self) -> Option<ExpertStats> {
        let Runner::Gpu(g) = self else { return None };
        let sum = |f: &dyn Fn(&GpuModel) -> usize| g.parts().map(f).sum::<usize>();
        let parts: Vec<_> = g.parts().collect();
        let cached: Vec<_> = parts.iter().filter(|p| p.cache.is_some()).collect();
        Some(ExpertStats {
            hits: sum(&|p| p.hits),
            misses: sum(&|p| p.misses),
            swaps: sum(&|p| p.cache.as_ref().map_or(0, |c| c.swaps)),
            resident: sum(&|p| p.cache.as_ref().map_or(0, |c| c.resident_count())),
            total: sum(&|p| p.cache.as_ref().map_or(0, |c| c.total_experts())),
            bytes_per_token: sum(&|p| p.expert_bytes_per_token()),
            cached: !cached.is_empty(),
            all_fast: cached.iter().all(|p| p.expert_fast_path()),
        })
    }
    /// The same counters, but only when an expert cache exists (the benchmark record's top-level `expert`).
    fn expert_summary(&self) -> Option<ExpertStats> {
        self.expert_stats().filter(|s| s.cached)
    }
    fn finish(&self) -> anyhow::Result<()> {
        if let Some(s) = self.expert_ram_stats().filter(|s| s.budget_bytes > 0) {
            eprintln!("expert RAM: {:.2}/{:.2} GiB, {} experts, {} hits / {} misses, {} evictions, {} mapped bypasses; matrix reads direct={} buffered={}",
                s.resident_bytes as f64 / 1073741824.0, s.budget_bytes as f64 / 1073741824.0,
                s.resident_experts, s.hits, s.misses, s.evictions, s.bypasses, s.direct_reads, s.buffered_reads);
        }
        if let Runner::Gpu(g) = self {
            let sum = |f: &dyn Fn(&GpuModel) -> usize| g.parts().map(f).sum::<usize>();
            let st = self.expert_stats().expect("Runner::Gpu has expert stats");
            if g.drafts.0 > 0 {
                let kind = if g.ngram.is_some() { "speculative ngram" } else { "MTP" };
                eprintln!("{kind} drafts accepted {}/{} ({:.1}%)", g.drafts.1, g.drafts.0, 100.0 * g.drafts.1 as f64 / g.drafts.0 as f64);
            }
            eprintln!("expert hit rate {:.1}% ({} misses), cache swaps {}", 100.0 * st.hits as f64 / (st.hits + st.misses).max(1) as f64, st.misses, st.swaps);
            // Issue #34 R1: printed on every run, so a fully resident run reads as a measured 0 in the log instead of
            // an absent line. `tokens` counts the token forwards the worker could have been asked about.
            let tokens = sum(&|p| p.tokens);
            let busy: f64 = g.parts().map(|p| p.cpu_time.as_secs_f64()).sum();
            let stalls = sum(&|p| p.resumed.get());
            eprintln!("worker busy {:.1} µs/token, worker stalls {stalls}", if tokens > 0 { 1e6 * busy / tokens as f64 } else { 0.0 });
            let mode = if !st.cached { "no expert cache".to_string() }
                else if st.all_fast { format!("fast path on, {}/{} experts resident", st.resident, st.total) }
                else { format!("fast path off, {}/{} experts resident", st.resident, st.total) };
            eprintln!("mailbox traffic: {} GPU writes, {} host routing reads ({mode})", sum(&|p| p.mailbox_writes.get()), sum(&|p| p.mailbox_reads.get()));
            for p in g.parts() { if let Some(c) = &p.cache { c.save_profile()?; } }
        }
        Ok(())
    }
}

impl Engine {
    /// Hand `--backend` and `--device` to the DeepSeek, GLM and Gemma 4 paths, which read them from the environment.
    fn ds_env(&self) {
        // SAFETY: called at the start of a command, before any thread reads these variables.
        if let Some(b) = &self.backend { unsafe { std::env::set_var("STRATA_DS_BACKEND", b) } }
        if self.device != [0] { unsafe { std::env::set_var("STRATA_DS_DEVICES", self.device.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",")) } }
    }

    /// Fills settings not given on the command line from the model's calibration file, else built-in defaults.
    fn ram_cap(&self) -> Option<u64> { memory::ram_cap((self.ram_margin as u64) << 20, (self.max_ram as u64) << 20) }
    /// A copy with the load options of an `/api/load` body applied (`ctx`, `kv`, `cache_ram`, `vram_margin`, `mtp`,
    /// `chunk`); a field that is absent or null keeps its value.
    fn with(&self, v: &serde_json::Value) -> anyhow::Result<Engine> {
        let num = |k: &str, lo: u64, hi: u64| -> anyhow::Result<Option<usize>> {
            if v[k].is_null() { return Ok(None); }
            let n = v[k].as_u64().filter(|n| (lo..=hi).contains(n)).with_context(|| format!("{k} must be a whole number from {lo} to {hi}"))?;
            Ok(Some(n as usize))
        };
        let mut e = self.clone();
        if let Some(n) = num("ctx", CONTEXT_MIN as u64, CONTEXT_MAX as u64)? { e.ctx = n; }
        if let Some(n) = num("cache_ram", 0, 1 << 20)? { e.cache_ram = n; }
        if let Some(n) = num("ram_margin", 0, 1 << 20)? { e.ram_margin = n; }
        if let Some(n) = num("max_ram", 0, 1 << 20)? { e.max_ram = n; }
        if !v["expert_ram"].is_null() {
            e.expert_ram = if let Some(s) = v["expert_ram"].as_str() { s.to_owned() }
                else { num("expert_ram", 0, 1 << 20)?.context("expert_ram must be auto or MiB")?.to_string() };
        }
        memory::parse_expert_ram(&e.expert_ram)?;
        if !v["expert_warmup"].is_null() {
            e.expert_warmup = v["expert_warmup"].as_bool().context("expert_warmup must be true or false")?;
        }
        if let Some(n) = num("vram_margin", 0, 1 << 20)? { e.vram_margin = vec![n.to_string()]; }
        if let Some(n) = num("mtp", 0, 2)? { e.mtp = Some(n); }
        if !v["parallel"].is_null() {
            e.parallel_mode = if let Some(mode) = v["parallel"].as_str() { mode.parse().map_err(anyhow::Error::msg)? }
                else { Parallelism::Fixed(num("parallel", 1, 4)?.context("parallel must be auto or 1..4")?) };
        }
        if let Some(n) = num("max_parallel", 1, 4)? { e.max_parallel = n; }
        e.parallel = resolve_capacity(e.parallel_mode, e.max_parallel, e.gpu, e.mtp(), ModelFeatures::default())?.slots;
        if let Some(n) = num("chunk", 0, 1 << 20)? { e.chunk = Some(n); }
        if !v["kv"].is_null() { e.kv = v["kv"].as_str().filter(|k| ["f16", "q8"].contains(k)).context("kv must be f16 or q8")?.into(); }
        anyhow::ensure!(e.gpu || (e.parallel <= 1 && e.mtp() == 0), "CPU mode serves one request at a time and cannot use MTP");
        anyhow::ensure!(e.parallel <= 1 || e.mtp() == 0, "parallel requests and MTP cannot run together; set MTP to 0 for throughput mode");
        validate_context_length(e.ctx, None)?;
        anyhow::ensure!(e.chunk.is_none_or(|n| n <= e.ctx), "prefill chunk must not exceed context length");
        Ok(e)
    }

    fn runner<'a>(&mut self, m: &'a Model<'a>, model: &Path) -> anyhow::Result<Runner<'a>> {
        validate_context_length(self.ctx, model_context_length(&m.w.g))?;
        anyhow::ensure!(self.chunk.is_none_or(|n| n <= self.ctx), "prefill chunk must not exceed context length");
        memory::parse_expert_ram(&self.expert_ram)?;
        // A previous runner has stopped. Its residency must not count as unavoidable model storage in this
        // cap check, nor reduce the available-memory snapshot used by the new budget.
        m.w.configure_expert_ram(0)?;
        if self.prompt_cache.is_some() && self.cache_ram == 0 { self.cache_ram = 2048; }
        if self.mtp() > 0 {
            rusting_strata::compatibility::validate_mtp_geometry(m, m.w.aux_g)?;
        }
        self.cache_ram = prompt_cache_cap(m.ram_estimate() as u64, self.cache_ram, self.ram_cap())?;
        setup_expert_ram(m, self.ram_margin, self.max_ram, self.cache_ram, &self.expert_ram)?;
        if self.llama_numerics { unsafe { std::env::set_var("STRATA_LLAMA_NUMERICS", "1") } }  // read by Gpu::new
        if self.fixed_experts { unsafe { std::env::set_var("STRATA_FIXED_EXPERTS", "1") } }
        // Set both directions before constructing the backend, including a UI
        // switch from q8 to f16. The tuning label must describe actual kernels.
        unsafe {
            if self.kv == "q8" { std::env::set_var("STRATA_KV_Q8", "1"); }
            else { std::env::remove_var("STRATA_KV_Q8"); }
        }
        let open = || self.device.iter().map(|&d| backend::open(self.backend.as_deref(), d)).collect::<anyhow::Result<Vec<_>>>();
        let bk = match self.gpu.then(open).transpose() {
            Err(e) if self.cpu_fallback => { eprintln!("no GPU backend ({e:#}), running on the CPU"); None }
            r => r?,
        };
        // A global Rayon pool can be configured only once. Record the actual
        // worker count rather than treating a later request as an applied change.
        let gpu_threads = bk.is_some().then(physical_cores).flatten();
        if let Some(n) = self.threads.or(gpu_threads) {
            let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
        }
        self.calibration = bk.as_ref().map(|b| {
            calibration_context(m, self.ctx, &self.kv,
                calibration_devices(&self.device, b.iter().map(|g| &**g))?,
                vram_margins(&self.vram_margin, &self.device)?.into_iter().map(|mb| mb << 20).collect(),
                self.parallel.max(1), rayon::current_num_threads(), self.cache_ram, self.expert_warmup)
        }).transpose()?;
        let v = self.calibration.as_ref().and_then(read_calibration).unwrap_or_default();
        if !v.is_null() {
            eprintln!("calibration: {}", calibration_path(self.calibration.as_ref().unwrap()).display());
        } else if bk.is_some() {
            eprintln!("no matching calibration for this model revision, device and settings: `forge9070 tune --model {}` measures prefill and supported MTP options", model.display());
        }
        let get = |k: &str| v[k].as_u64().map(|x| x as usize);
        if let Some(dp) = &self.draft_model {
            anyhow::ensure!(dp.exists(), "draft model path does not exist: {}", dp.display());
            eprintln!("draft model socket: registered candidate {}", dp.display());
        }
        let mut ngram_cfg = self.ngram_config()?;
        ngram_cfg.n_vocab = m.cfg.n_vocab;
        // Keep the resolved default valid for subsequent UI reloads too. A
        // short context must not acquire an implicit 4096-token chunk that
        // Engine::with would reject on the next model-management request.
        self.chunk.get_or_insert(get("chunk").unwrap_or(4096).min(self.ctx));
        if self.mtp_model.is_some() && self.mtp.is_none() {
            self.mtp = Some(1);
        } else if ngram_cfg.is_enabled() && self.mtp.is_none() {
            self.mtp = Some(0);
        } else {
            self.mtp.get_or_insert(get("mtp").unwrap_or(0));
        }
        self.short = get("short").unwrap_or(32);
        let Some(bk) = bk else {
            let (notes, _, mtp) = resolve_flags(false, self.parallel, self.mtp(), m.cfg.n_nextn, m.cfg.hc_n, false)?;
            for n in notes { eprintln!("{n}"); }
            self.mtp = Some(mtp);
            let state = State::new(&m.cfg);
            warm_expert_ram(m, self.expert_warmup, false)?;
            return Ok(Runner::Cpu(m, state));
        };
        let split = bk.len() > 1;
        // Explicit feature requests must not silently become different execution modes.
        anyhow::ensure!(self.parallel <= 1 || self.mtp() == 0, "--parallel > 1 cannot be combined with MTP");
        anyhow::ensure!(self.parallel <= 1 || (m.cfg.hc_n == 0 && m.cfg.n_embd_ple == 0 && m.cfg.idx_top_k == 0),
            "this model's hyper-connections, per-layer embeddings or sparse attention do not support parallel slots in this engine");
        anyhow::ensure!(self.mtp() == 0 || (!split && m.cfg.hc_n == 0 && m.cfg.n_nextn > 0),
            "requested MTP is unavailable: use a supported matching MTP head without hyper-connections and a single GPU, or set --mtp 0");
        if self.mtp() > 0 {
            rusting_strata::compatibility::validate_mtp_geometry(m, m.w.aux_g)?;
        }
        let (notes, parallel, mtp) = resolve_flags(true, self.parallel, self.mtp(), m.cfg.n_nextn, m.cfg.hc_n, split)?;
        for n in notes { eprintln!("{n}"); }
        (self.parallel, self.mtp) = (parallel, Some(mtp));
        // Before the first weight is uploaded: fail if a pipeline these weight types need does not fit the device's
        // shared memory (Vulkan; a no-op elsewhere). No pipeline or model buffer exists yet.
        let types = rusting_strata::gpu_model::gpu_weight_types(&m.w.g);
        for b in &bk { b.preflight(&types, false)?; }
        let margins = vram_margins(&self.vram_margin, &self.device)?.into_iter().map(|mb| mb << 20).collect::<Vec<_>>();
        let spec_k = if mtp > 0 { mtp } else if ngram_cfg.is_enabled() && self.parallel <= 1 { ngram_cfg.draft_max } else { 0 };
        let mut g = if self.parallel > 1 {
            GpuModel::load_split_margins(m, bk, self.ctx, self.chunk(), &margins, 0, self.parallel).context(LOAD_HINT)?
        } else {
            GpuModel::load_split_margins(m, bk, self.ctx, self.chunk(), &margins, spec_k, 1).context(LOAD_HINT)?
        };
        if ngram_cfg.is_enabled() && self.parallel <= 1 {
            let tok = Tokenizer::from_gguf(&m.w.g).ok();
            g.ngram = Some(rusting_strata::ngram::NgramDrafter::new(ngram_cfg, tok.as_ref())?);
        }
        g.short = self.short;
        g.cache_ram = self.cache_ram << 20;
        if let Some(p) = &self.prompt_cache {
            if let Err(e) = g.load_prompt_cache(p) {
                eprintln!("warning: could not load prompt cache {}: {e:#}", p.display());
            }
        }
        // Use the same complete-expert placement policy for generation,
        // benchmarking and serving. Calibration measures this policy too.
        g.set_decode_only(true);
        warm_expert_ram(m, self.expert_warmup, mtp > 0)?;
        Ok(Runner::Gpu(Box::new(g)))
    }
    fn ngram_config(&self) -> anyhow::Result<rusting_strata::ngram::NgramConfig> {
        let types = rusting_strata::ngram::SpecType::parse_list(&self.spec_type)?;
        anyhow::ensure!((1..=rusting_strata::cpu_worker::TMAX - 1).contains(&self.spec_draft_n_max),
            "--spec-draft-n-max {}: must be between 1 and {}", self.spec_draft_n_max, rusting_strata::cpu_worker::TMAX - 1);
        Ok(rusting_strata::ngram::NgramConfig {
            types,
            draft_max: self.spec_draft_n_max,
            size_n: self.spec_ngram_size_n.max(1),
            size_m: self.spec_ngram_size_m.max(1),
            min_hits: self.spec_ngram_min_hits.max(1),
            mod_n_match: self.spec_ngram_size_n.max(1),
            mod_n_min: 2,
            lookup_cache_static: self.lookup_cache_static.clone(),
            lookup_cache_dynamic: self.lookup_cache_dynamic.clone(),
            n_vocab: 0,
        })
    }
    fn sampler(&self) -> Sampler {
        Sampler { temp: self.temp, top_k: self.top_k, top_p: self.top_p, rng: seed(self.seed) }
    }
    fn chunk(&self) -> usize { self.chunk.unwrap_or(4096) }
    fn mtp(&self) -> usize { self.mtp.unwrap_or(0) }
}

/// Called after the model's small owned tensors and tokenizer have loaded,
/// before GPU expert uploads or worker creation. The GGUF/PLE stays mapped.
fn setup_expert_ram(m: &Model, margin_mib: usize, max_mib: usize, prompt_mib: usize,
    requested: &str) -> anyhow::Result<memory::ExpertRamPlan> {
    // Runners are not live at this point. Reset a previous configuration before
    // taking a new available-memory snapshot, so residency is not counted twice.
    m.w.configure_expert_ram(0)?;
    let mut inventory = rusting_strata::compatibility::weight_memory(&m.w.g.tensors)?;
    if let Some(aux) = m.w.aux_g {
        let extra = rusting_strata::compatibility::weight_memory(&aux.tensors)?;
        inventory.routed_expert_bytes = inventory.routed_expert_bytes.checked_add(extra.routed_expert_bytes)
            .context("expert memory inventory overflow")?;
    }
    let plan = memory::expert_plan(memory::snapshot(), inventory.routed_expert_bytes, m.ram_estimate() as u64,
        (margin_mib as u64) << 20, (max_mib as u64) << 20, (prompt_mib as u64) << 20,
        memory::parse_expert_ram(requested)?);
    m.w.configure_expert_ram(usize::try_from(plan.budget_bytes).context("expert RAM budget exceeds address space")?)?;
    let gib = |n: u64| n as f64 / 1073741824.0;
    eprintln!("host available: {} GiB; expert payload {:.2} GiB; owned expert ceiling {:.2} GiB ({})",
        plan.memory.available_bytes.map(|n| format!("{:.2}", gib(n))).unwrap_or_else(|| "unknown".into()),
        gib(inventory.routed_expert_bytes), gib(plan.budget_bytes),
        if plan.budget_bytes == 0 { "mapped path" } else if plan.all_experts_fit_budget { "all expert payloads fit the ceiling" } else { "bounded LRU; cold experts backed by disk" });
    eprintln!("host reserve: {:.2} GiB OS + {:.2} GiB runtime + {:.2} GiB prompt cache; {:.2} GiB sparse embeddings stay lazy",
        gib(plan.os_margin_bytes), gib(plan.runtime_reserve_bytes), gib(plan.prompt_cache_reserve_bytes), gib(inventory.lazy_embedding_bytes));
    Ok(plan)
}

/// Warm only routed experts after runtime buffers have been allocated. PLE stays lazy and this pass never evicts
/// an existing RAM entry. It is startup I/O; its duration is not an inference throughput measurement.
fn warm_expert_ram(m: &Model, enabled: bool, include_draft: bool) -> anyhow::Result<()> {
    if !enabled || !m.w.expert_ram_enabled() { return Ok(()); }
    let warmed = rusting_strata::expert_cache::warm_expert_ram(m, include_draft)?;
    eprintln!("expert RAM warmup: {} experts, {:.2} GiB loaded in {:.2} s; {:.2} GiB resident ({})",
        warmed.experts_loaded, warmed.bytes_loaded as f64 / 1073741824.0, warmed.elapsed.as_secs_f64(),
        warmed.resident_bytes as f64 / 1073741824.0, if warmed.ranking_used { "saved routing profile" } else { "layer order" });
    Ok(())
}

fn prompt_cache_cap(owned_bytes: u64, requested_mib: usize, cap: Option<u64>) -> anyhow::Result<usize> {
    let Some(cap) = cap else { return Ok(requested_mib); };
    anyhow::ensure!(owned_bytes <= cap,
        "model needs ~{} MiB of owned host RAM, over the {} MiB cap; lower --ram-margin or raise --max-ram",
        owned_bytes >> 20, cap >> 20);
    let fit = usize::try_from((cap - owned_bytes) >> 20).unwrap_or(usize::MAX);
    if requested_mib > fit {
        eprintln!("RAM cap: clamping --cache-ram from {requested_mib} MiB to {fit} MiB");
    }
    Ok(requested_mib.min(fit))
}

/// Per-device VRAM margins in MB from `--vram-margin`: one value for every device, one per device like `512,256`,
/// or one per GPU index like `0:512,1:256`. The result is in the order of `devices`.
fn vram_margins(spec: &[String], devices: &[usize]) -> anyhow::Result<Vec<usize>> {
    let parse = |s: &str| -> anyhow::Result<(Option<usize>, usize)> {
        let (idx, mb) = match s.split_once(':') {
            Some((i, m)) => (Some(i.trim().parse().with_context(|| format!("--vram-margin {s}: `{}` is not a GPU index", i.trim()))?), m),
            None => (None, s),
        };
        let mb = mb.trim().parse().with_context(|| format!("--vram-margin {s}: `{}` is not a value in MB", mb.trim()))?;
        Ok((idx, mb))
    };
    let vals = spec.iter().map(|s| parse(s)).collect::<anyhow::Result<Vec<_>>>()?;
    anyhow::ensure!(!vals.is_empty(), "--vram-margin needs at least one value");
    if vals.len() == 1 && vals[0].0.is_none() {
        return Ok(vec![vals[0].1; devices.len()]);
    }
    if vals.iter().all(|v| v.0.is_none()) {
        anyhow::ensure!(vals.len() == devices.len(), "--vram-margin {}: {} value(s) for {} device(s) (give one value, \
            or one per device in --device order)", spec.join(","), vals.len(), devices.len());
        return Ok(vals.into_iter().map(|v| v.1).collect());
    }
    anyhow::ensure!(vals.iter().all(|v| v.0.is_some()), "--vram-margin {}: mixing `IDX:MB` and plain MB values", spec.join(","));
    let mut out = vec![0; devices.len()];
    let mut seen = vec![false; devices.len()];
    for (idx, mb) in vals {
        let idx = idx.unwrap();
        let pos = devices.iter().position(|&d| d == idx)
            .with_context(|| format!("--vram-margin: GPU {idx} is not in --device"))?;
        anyhow::ensure!(!seen[pos], "--vram-margin: GPU {idx} is given twice");
        seen[pos] = true;
        out[pos] = mb;
    }
    if let Some(pos) = seen.iter().position(|&s| !s) {
        anyhow::bail!("--vram-margin: no value for GPU {} (give every device one)", devices[pos]);
    }
    Ok(out)
}

/// A projector opened once and the images encoded with it.
struct ImageInput {
    grids: Vec<(usize, usize)>,
    embeddings: Vec<Vec<Vec<f32>>>,
}

/// Physical cores, not SMT siblings: the CPU expert workers are memory-bound, and oversubscribing them with the
/// hyperthreads slows the CPU/GPU token loop. Linux reads the CPU topology; other platforms keep every logical CPU.
fn physical_cores() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let groups = cpu_topology_groups(std::path::Path::new("/sys/devices/system/cpu"));
        let avail = std::thread::available_parallelism().map_or(0, |n| n.get());
        if groups > 0 {
            return Some(if avail > 0 { groups.min(avail) } else { groups });
        }
    }
    None
}

/// Distinct `thread_siblings_list` values under `dir` (one per physical core).
#[cfg(target_os = "linux")]
fn cpu_topology_groups(dir: &std::path::Path) -> usize {
    let groups: std::collections::BTreeSet<String> = std::fs::read_dir(dir).into_iter().flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name();
            let n = name.to_str()?.strip_prefix("cpu")?;
            if n.is_empty() || !n.bytes().all(|b| b.is_ascii_digit()) { return None; }
            std::fs::read_to_string(e.path().join("topology/thread_siblings_list")).ok().map(|s| s.trim().to_string())
        })
        .collect();
    groups.len()
}

/// Decode and encode every `--image` with the projector at `mmproj`.
fn load_images(mmproj: &Path, images: &[String]) -> anyhow::Result<ImageInput> {
    let projector = VisionProjector::open(mmproj)?;
    let mut grids = Vec::with_capacity(images.len());
    let mut embeddings = Vec::with_capacity(images.len());
    for src in images {
        let img = vision::load_image(src)?;
        let (feats, nx, ny) = vision::encode_image(&projector, &img)?;
        grids.push((nx, ny));
        embeddings.push(feats);
    }
    Ok(ImageInput { grids, embeddings })
}

/// Expand `<image>` placeholders into the projector's vision tokens and assign M-RoPE positions.
fn build_image_prompt(tok: &Tokenizer, input: &ImageInput, images: &[String], prompt: &str) -> anyhow::Result<vision::Prompt> {
    let (start, end, pad) = (
        tok.token_id("<|vision_start|>").context("tokenizer has no <vision_start|>")?,
        tok.token_id("<|vision_end|>").context("tokenizer has no <vision_end|>")?,
        tok.token_id("<|image_pad|>").context("tokenizer has no <image_pad|>")?,
    );
    let tokens = "<|vision_start|><|image_pad|><|vision_end|>";
    let n_mark = prompt.matches("<image>").count();
    let text = if n_mark == 0 {
        format!("{}{}", tokens.repeat(images.len()), prompt)
    } else {
        anyhow::ensure!(n_mark == images.len(), "prompt has {n_mark} <image> placeholder(s) for {} --image(s)", images.len());
        prompt.replace("<image>", tokens)
    };
    let ids = tok.encode(&text)?;
    vision::expand_images(&ids, start, end, pad, &input.grids)
}

/// The negative log-likelihood of `tok` under `logits`, and in `row` the log-probabilities as llama-perplexity stores
/// them: scale and minimum as two f32, then each logit as a u16 step over the 16 nats below the largest (0 below
/// that), padded to an even count. Same arithmetic as its `log_softmax`.
fn log_probs(logits: &[f32], tok: u32, row: &mut Vec<u16>) -> f64 {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min = logits.iter().fold(f32::INFINITY, |a, &b| a.min(b)).max(max - 16.0);
    let lse = logits.iter().map(|&x| (x - max).exp() as f64).sum::<f64>().ln() as f32;
    let scale = (max - min) / 65535.0;
    row.clear();
    for f in [scale, min - max - lse] { let b = f.to_bits(); row.extend([b as u16, (b >> 16) as u16]); }
    row.extend(logits.iter().map(|&x| if scale != 0.0 && x > min { ((x - min) * (1.0 / scale)).round_ties_even() as u16 } else { 0 }));
    if logits.len() % 2 == 1 { row.push(0); }
    (max + lse - logits[tok as usize]) as f64
}

/// A nonzero xorshift state from `s`, or from the clock.
fn seed(s: Option<u64>) -> u64 {
    s.unwrap_or_else(|| SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64))
        .wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1
}

const LOAD_HINT: &str = "loading the model onto the GPU failed (if VRAM ran out: lower --ctx, add --kv q8, or close programs using VRAM)";

/// Settle the flag combinations that cannot all take effect: an invalid request (`--parallel` outside 1..=255) is an
/// error; a feature that cannot run is turned off with one message naming the flag and the reason. `parallel` 0 means
/// "not set" (commands other than `serve`), treated as one slot. Returns the notices, the slots in effect and the MTP
/// drafts in effect.
fn resolve_flags(gpu: bool, parallel: usize, mtp: usize, n_nextn: usize, hc_n: usize, split: bool) -> anyhow::Result<(Vec<String>, usize, usize)> {
    let p = parallel.max(1);
    anyhow::ensure!((1..=255).contains(&p), "--parallel {parallel}: 1 to 255 slots");
    let mut notes = vec![];
    if !gpu {
        if p > 1 { notes.push(format!("--parallel {p}: requires --gpu (the CPU answers one request at a time)")); }
        if mtp > 0 { notes.push(format!("--mtp {mtp}: requires --gpu: decoding without drafts")); }
        return Ok((notes, p, 0));
    }
    if p > 1 && mtp > 0 {
        notes.push(format!("--parallel {p} and --mtp {mtp}: not supported together, decoding without MTP drafts"));
        return Ok((notes, p, 0));
    }
    let why = if mtp == 0 { None }
        else if split { Some("not available with a multi-GPU layer split") }
        else if hc_n > 0 { Some("not supported for this model (hyper-connections)") }
        else if n_nextn == 0 { Some("this model has no MTP head") }
        else { None };
    let mut k = mtp;
    if let Some(why) = why { notes.push(format!("--mtp {mtp}: {why}: decoding without drafts")); k = 0; }
    Ok((notes, p, k))
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct CalibrationDevice {
    index: usize,
    backend: String,
    name: String,
    total_vram_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct CalibrationContext {
    model_identity: String,
    companion_identity: Option<String>,
    has_lora: bool,
    devices: Vec<CalibrationDevice>,
    ctx: usize,
    kv: String,
    vram_margin_bytes: Vec<usize>,
    expert_ram_budget_bytes: usize,
    expert_ram_policy: String,
    expert_warmup: bool,
    prompt_cache_budget_bytes: usize,
    expert_tail_min_rank: usize,
    parallel: usize,
    threads: usize,
    max_mtp_drafts: usize,
    llama_numerics: bool,
    decode_cache_policy: String,
    inference_overrides: BTreeMap<String, String>,
}

const CALIBRATION_SCHEMA: &str = "forge9070.calibration.v5";
const DECODE_CACHE_POLICY: &str = "server_decode_following";
const EXPERT_RAM_POLICY: &str = "owned_lru_prefill_transient_v1";

/// Only inference controls belong here; never serialize API keys or arbitrary
/// environment values into benchmark/calibration records.
fn inference_overrides() -> BTreeMap<String, String> {
    ["STRATA_LLAMA_NUMERICS", "STRATA_FIXED_EXPERTS", "STRATA_ADAPT_EVERY",
     "STRATA_ADAPT_SWAPS", "STRATA_PF_ADAPT", "STRATA_EXPERT_MB",
     "STRATA_MOE_PREFETCH", "STRATA_MTP_ADAPT", "STRATA_NO_GRAPH",
     "STRATA_VK_NO_COOP", "STRATA_VK_DP_OLD", "STRATA_VK_NO_COND",
     "STRATA_VK_NO_TQ", "STRATA_VK_ATTN_BUDGET", "STRATA_VK_SMEM_LIMIT",
     "STRATA_VK_PROF", "STRATA_VK_STATS"]
        .into_iter().filter_map(|key| std::env::var(key).ok().map(|value| (key.into(), value))).collect()
}

fn calibration_devices<'a>(indices: &[usize], backends: impl IntoIterator<Item = &'a dyn backend::Backend>)
    -> anyhow::Result<Vec<CalibrationDevice>> {
    let backends: Vec<_> = backends.into_iter().collect();
    anyhow::ensure!(indices.len() == backends.len() && !indices.is_empty(), "calibration device list mismatch");
    indices.iter().zip(backends).map(|(&index, gpu)| Ok(CalibrationDevice {
        index, backend: gpu.name().into(), name: gpu.device(), total_vram_bytes: gpu.mem_info()?.1,
    })).collect()
}

fn calibration_mtp_budget(n_nextn: usize, hc_n: usize, devices: usize) -> usize {
    if n_nextn > 0 && hc_n == 0 && devices == 1 { 2 } else { 0 }
}

fn calibration_context(m: &Model, ctx: usize, kv: &str, devices: Vec<CalibrationDevice>,
    margin: Vec<usize>, parallel: usize, threads: usize, cache_ram: usize, expert_warmup: bool) -> anyhow::Result<CalibrationContext> {
    anyhow::ensure!(devices.len() == margin.len(), "calibration VRAM-margin list mismatch");
    Ok(CalibrationContext {
        model_identity: m.w.g.identity.clone(), companion_identity: m.w.aux_g.map(|g| g.identity.clone()),
        has_lora: !m.loras.is_empty(), max_mtp_drafts: if parallel == 1
            && rusting_strata::compatibility::validate_mtp_geometry(m, m.w.aux_g).is_ok() {
            calibration_mtp_budget(m.cfg.n_nextn, m.cfg.hc_n, devices.len())
        } else { 0 },
        devices, ctx, kv: kv.into(), vram_margin_bytes: margin, parallel, threads,
        expert_ram_budget_bytes: m.w.expert_ram_stats().budget_bytes,
        expert_ram_policy: EXPERT_RAM_POLICY.into(), expert_warmup,
        prompt_cache_budget_bytes: cache_ram << 20,
        expert_tail_min_rank: rusting_strata::expert_tail::Policy::from_env()?.min_rank,
        llama_numerics: std::env::var_os("STRATA_LLAMA_NUMERICS").is_some(),
        decode_cache_policy: DECODE_CACHE_POLICY.into(), inference_overrides: inference_overrides(),
    })
}

/// Both the model revision and the measured configuration are in the key. The
/// complete configuration is also checked inside the record before any setting
/// is used. Old basename-only files are deliberately not migrated or trusted.
fn calibration_path(context: &CalibrationContext) -> PathBuf {
    let encoded = serde_json::to_string(context).expect("calibration context is serializable");
    let settings = rusting_strata::cache_identity::combine_identities(CALIBRATION_SCHEMA, [&encoded]);
    rusting_strata::expert_cache::cache_dir().join(format!("calibration-v5-{}-{}.json",
        context.model_identity, &settings[..24]))
}

fn calibration_matches(value: &serde_json::Value, context: &CalibrationContext) -> bool {
    value["schema"] == CALIBRATION_SCHEMA
        && value["status"] == "measured"
        && value["objective"] == "median_decode_tokens_per_second"
        && value["fidelity"]["selected_candidate_all_measured_outputs_match_baseline"] == true
        && serde_json::from_value::<CalibrationContext>(value["configuration"].clone()).is_ok_and(|c| c == *context)
        && value["chunk"].as_u64().is_some_and(|n| n > 0 && n <= context.ctx as u64)
        && value["mtp"].as_u64().is_some_and(|n| n <= context.max_mtp_drafts as u64)
        && value["short"].as_u64().is_some_and(|n| n > 0 && n <= context.ctx as u64)
        && value["threads"].as_u64() == Some(context.threads as u64)
}

fn read_calibration(context: &CalibrationContext) -> Option<serde_json::Value> {
    let value = serde_json::from_slice(&std::fs::read(calibration_path(context)).ok()?).ok()?;
    calibration_matches(&value, context).then_some(value)
}

fn runner_tail_stats(runner: &Runner) -> rusting_strata::expert_tail::Stats {
    let Runner::Gpu(g) = runner else { return Default::default(); };
    g.parts().map(|part| part.expert_tail_stats()).fold(Default::default(), |mut total, next| {
        total.min_rank = total.min_rank.max(next.min_rank);
        total.skipped_assignments = total.skipped_assignments.saturating_add(next.skipped_assignments);
        total.skipped_expert_groups = total.skipped_expert_groups.saturating_add(next.skipped_expert_groups);
        total
    })
}

/// Routing-byte estimates do not measure GPU/PCIe traffic. Once assignments are
/// omitted, their unadjusted byte counts are only an upper bound.
fn annotate_runtime_bench(record: &mut serde_json::Value, tail: rusting_strata::expert_tail::Stats, identity: &str) {
    record["expert_tail"] = json!(tail);
    record["approximate_expert_tail_enabled"] = json!(tail.min_rank > 0);
    record["model"]["cache_identity"] = json!(identity);
    let estimate = if tail.skipped_assignments > 0 { "upper_bound_before_expert_omissions" }
        else { "inferred_routed_weight_bytes" };
    record["expert_traffic_estimate"] = json!({"kind": estimate,
        "measured_gpu_or_pcie_bandwidth": false,
        "note": "weight bytes inferred from routing multiplied by decode rate; not a hardware traffic measurement"});
    if let Some(expert) = record.get_mut("expert") { expert["byte_estimate"] = json!(estimate); }
    if let Some(results) = record["results"].as_array_mut() {
        for result in results {
            if let Some(expert) = result.get_mut("expert") { expert["byte_estimate"] = json!(estimate); }
        }
    }
}

#[cfg(test)]
mod calibration_tests {
    use super::*;

    fn context() -> CalibrationContext {
        CalibrationContext {
            model_identity: "model-revision-a".into(), companion_identity: None, has_lora: false,
            devices: vec![CalibrationDevice { index: 0, backend: "vulkan".into(),
                name: "test device".into(), total_vram_bytes: 16 << 30 }],
            ctx: 4096, kv: "q8".into(), vram_margin_bytes: vec![512 << 20],
            expert_ram_budget_bytes: 40 << 30, expert_ram_policy: EXPERT_RAM_POLICY.into(), expert_warmup: true,
            prompt_cache_budget_bytes: 512 << 20,
            expert_tail_min_rank: 0, parallel: 1, threads: 8, max_mtp_drafts: 2,
            llama_numerics: false, decode_cache_policy: DECODE_CACHE_POLICY.into(), inference_overrides: BTreeMap::new(),
        }
    }

    fn record(context: &CalibrationContext) -> serde_json::Value {
        json!({"schema":CALIBRATION_SCHEMA, "status":"measured", "configuration": context,
            "objective":"median_decode_tokens_per_second","fidelity":{"selected_candidate_all_measured_outputs_match_baseline":true},
            "chunk": 1024, "mtp": 2, "short": 32, "threads": context.threads})
    }

    #[test]
    fn calibration_does_not_cross_model_or_execution_configurations() {
        let original = context();
        let measured = record(&original);
        assert!(calibration_matches(&measured, &original));
        let changes: &[fn(&mut CalibrationContext)] = &[
            |c| c.model_identity = "model-revision-b".into(),
            |c| c.companion_identity = Some("different-mtp-head".into()),
            |c| c.has_lora = true,
            |c| c.ctx = 8192,
            |c| c.kv = "f16".into(),
            |c| c.devices[0].backend = "cuda".into(),
            |c| c.devices[0].name = "another device".into(),
            |c| c.devices[0].index = 1,
            |c| c.devices[0].total_vram_bytes = 8 << 30,
            |c| c.vram_margin_bytes[0] += 256 << 20,
            |c| c.expert_ram_budget_bytes += 1 << 20,
            |c| c.expert_ram_policy = "different-ram-policy".into(),
            |c| c.expert_warmup = false,
            |c| c.prompt_cache_budget_bytes = 0,
            |c| c.expert_tail_min_rank = 8,
            |c| c.parallel = 4,
            |c| c.threads = 4,
            |c| c.max_mtp_drafts = 0,
            |c| c.llama_numerics = true,
            |c| c.decode_cache_policy = "other-policy".into(),
            |c| { c.inference_overrides.insert("STRATA_MOE_PREFETCH".into(), "1".into()); },
        ];
        for change in changes {
            let mut other = original.clone();
            change(&mut other);
            assert!(!calibration_matches(&measured, &other), "accepted stale calibration: {other:?}");
            assert_ne!(calibration_path(&original), calibration_path(&other));
        }
    }

    #[test]
    fn calibration_rejects_legacy_partial_and_invalid_records() {
        let context = context();
        for (field, bad) in [
            ("schema", json!("basename-v1")), ("configuration", json!(null)),
            ("chunk", json!(0)), ("chunk", json!(8192)), ("chunk", json!(-1)),
            ("mtp", json!(3)), ("short", json!(0)), ("short", json!(8192)),
            ("threads", json!(16)),
            ("schema", json!("forge9070.calibration.v3")),
            ("schema", json!("forge9070.calibration.v4")),
            ("objective", json!("request_output_tokens_per_second")),
            ("fidelity", json!({"selected_candidate_all_measured_outputs_match_baseline":false})),
            ("status", json!("unstable_or_insufficient_reference")),
        ] {
            let mut value = record(&context);
            value[field] = bad;
            assert!(!calibration_matches(&value, &context), "accepted invalid {field}: {value}");
        }
        let mut missing = record(&context);
        missing["configuration"].as_object_mut().unwrap().remove("kv");
        assert!(!calibration_matches(&missing, &context));
        let mut old_ram_policy = record(&context);
        old_ram_policy["configuration"].as_object_mut().unwrap().remove("expert_ram_budget_bytes");
        assert!(!calibration_matches(&old_ram_policy, &context));
        assert!(!calibration_matches(&json!({"chunk":1024,"mtp":1,"short":32,"threads":8}), &context));
    }

    #[test]
    fn calibration_never_allocates_unsupported_mtp_heads() {
        assert_eq!(calibration_mtp_budget(1, 0, 1), 2);
        assert_eq!(calibration_mtp_budget(0, 0, 1), 0);
        assert_eq!(calibration_mtp_budget(1, 4, 1), 0);
        assert_eq!(calibration_mtp_budget(1, 0, 2), 0);
        assert_eq!(calibration_mtp_budget(1, 0, 0), 0);
    }

    #[test]
    fn benchmark_labels_omitted_experts_without_claiming_hardware_bandwidth() {
        let base = json!({"model":{"name":"fixture"}, "expert":{"bytes_per_token":1024},
            "results":[{"expert":{"bytes_per_token":1024,"gb_per_s":0.25}}]});
        let mut exact = base.clone();
        annotate_runtime_bench(&mut exact, Default::default(), "identity");
        assert_eq!(exact["expert_traffic_estimate"]["kind"], "inferred_routed_weight_bytes");
        assert_eq!(exact["approximate_expert_tail_enabled"], false);
        let mut skipped = base;
        annotate_runtime_bench(&mut skipped, rusting_strata::expert_tail::Stats {
            min_rank: 8, skipped_assignments: 12, skipped_expert_groups: 4,
        }, "identity");
        assert_eq!(skipped["expert_traffic_estimate"]["kind"], "upper_bound_before_expert_omissions");
        assert_eq!(skipped["expert_traffic_estimate"]["measured_gpu_or_pcie_bandwidth"], false);
        assert_eq!(skipped["model"]["cache_identity"], "identity");
        assert_eq!(skipped["expert_tail"]["skipped_assignments"], 12);
        assert_eq!(skipped["results"][0]["expert"]["byte_estimate"], "upper_bound_before_expert_omissions");
        assert_eq!(skipped["results"][0]["expert"]["gb_per_s"], 0.25);
        assert_eq!(skipped["expert"]["bytes_per_token"], 1024);
    }
}

impl Engine {
    pub fn lora_specs(&self) -> Vec<(String, f32)> {
        self.lora.iter().map(|s| {
            if let Some(idx) = s.rfind(':') {
                if let Ok(scale) = s[idx+1..].parse::<f32>() {
                    return (s[..idx].to_string(), scale);
                }
            }
            (s.clone(), 1.0)
        }).collect()
    }
}

/// Expert-cache counters summed over the GPU parts (`Runner::finish` and the benchmark JSON share them).
#[derive(Debug, Clone, Copy, PartialEq)]
struct ExpertStats {
    hits: usize,
    misses: usize,
    swaps: usize,
    resident: usize,
    total: usize,
    bytes_per_token: usize,
    /// At least one part has an expert cache.
    cached: bool,
    /// Every part with a cache takes the fully-resident fast path.
    all_fast: bool,
}

/// Round to the one decimal the text block prints, so the JSON carries exactly those digits (no second, different
/// rounding of the raw float).
fn round1(x: f64) -> f64 {
    format!("{x:.1}").parse().unwrap_or(x)
}

/// Mean/min/max of a benchmark series. The text block and the JSON record both read these fields, so the numbers
/// cannot drift apart.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Stat {
    mean: f64,
    min: f64,
    max: f64,
}

impl Stat {
    fn of(v: &[f64]) -> Self {
        if v.is_empty() { return Self { mean: 0.0, min: 0.0, max: 0.0 }; }
        // The same fold as the text block always used, so formatting cannot change any printed digit.
        let (lo, hi) = v.iter().fold((f64::INFINITY, 0f64), |(a, b), &x| (a.min(x), b.max(x)));
        Self { mean: round1(v.iter().sum::<f64>() / v.len() as f64), min: round1(lo), max: round1(hi) }
    }
    /// The `MEAN (MIN-MAX)` field of the `pp …` line.
    fn text(&self) -> String { format!("{:.1} ({:.1}-{:.1})", self.mean, self.min, self.max) }
}

/// The run-wide values of a `strata bench` invocation (`strata.bench.v1`).
struct BenchMeta<'a> {
    backend: &'a str,
    backend_arg: Option<&'a str>,
    device: &'a [usize],
    model: &'a Path,
    ctx: usize,
    chunk: usize,
    kv: &'a str,
    mtp: usize,
    tg: usize,
    reps: usize,
}

/// One `--pp` length: the two `Stat`s the text prints and what the expert record needs.
struct BenchResult {
    pp: usize,
    prefill: Stat,
    decode: Stat,
    /// Bytes of expert weights one decode token reads (0 without an expert cache: no `expert` object).
    bytes_per_token: usize,
    /// The same median decode rate the `expert traffic:` line prints.
    decode_median: f64,
}

/// The `strata.bench.v1` record. Pure so it can be tested without a model or a GPU.
fn bench_record(meta: &BenchMeta, results: &[BenchResult], expert: Option<&ExpertStats>, unix_time: u64) -> serde_json::Value {
    let mut v = json!({
        "schema": "strata.bench.v1",
        "unix_time": unix_time,
        "backend": meta.backend,
        "device": meta.device,
        "model": {
            "path": meta.model.display().to_string(),
            "name": meta.model.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        },
        "ctx": meta.ctx,
        "chunk": meta.chunk,
        "kv": meta.kv,
        "mtp": meta.mtp,
        "tg": meta.tg,
        "reps": meta.reps,
    });
    if let Some(b) = meta.backend_arg { v["backend_arg"] = json!(b); }
    let results: Vec<_> = results.iter().map(|r| {
        let mut o = json!({
            "pp": r.pp,
            "prefill_tok_s": {"mean": r.prefill.mean, "min": r.prefill.min, "max": r.prefill.max},
            "decode_tok_s": {"mean": r.decode.mean, "min": r.decode.min, "max": r.decode.max},
        });
        if r.bytes_per_token > 0 {
            o["expert"] = json!({
                "bytes_per_token": r.bytes_per_token,
                "mb_per_token": round1(r.bytes_per_token as f64 / 1048576.0),
                "gb_per_s": round1(r.bytes_per_token as f64 * r.decode_median / 1e9),
                "at_decode_tok_s": round1(r.decode_median),
            });
        }
        o
    }).collect();
    v["results"] = json!(results);
    if let Some(e) = expert {
        v["expert"] = json!({
            "hit_rate_pct": round1(100.0 * e.hits as f64 / (e.hits + e.misses).max(1) as f64),
            "misses": e.misses,
            "swaps": e.swaps,
            "resident": e.resident,
            "total": e.total,
            "bytes_per_token": e.bytes_per_token,
            "mb_per_token": round1(e.bytes_per_token as f64 / 1048576.0),
        });
    }
    v
}

/// `--json-out` values: a file path, never `-` (stdout keeps the text block).
fn json_out_path(s: &str) -> Result<PathBuf, String> {
    if s == "-" {
        Err("`-` is not a file: stdout carries the text block, give a path".into())
    } else {
        Ok(PathBuf::from(s))
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Print model config and tensor type histogram.
    Info {
        #[arg(long)]
        model: PathBuf,
    },
    /// Teacher-force token ids; print argmax and top-5 logprobs for each continuation position.
    /// Input JSON: {"prompt": [ids], "continuation": [ids]} ("-" reads stdin).
    Score {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: String,
        #[command(flatten)]
        engine: Engine,
    },
    /// Interactive chat on stdin (greedy unless --temp; on the GPU each turn prefills only what follows the previous turn's messages).
    Chat {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        system: Option<String>,
        /// Max tokens per reply.
        #[arg(short, default_value_t = 2048)]
        n: usize,
        /// Image path, data URL or http(s) URL (repeatable), attached to the conversation. Needs --mmproj (or a
        /// sibling mmproj next to the model).
        #[arg(long, action = clap::ArgAction::Append)]
        image: Vec<String>,
        #[command(flatten)]
        engine: Engine,
    },
    /// OpenAI-compatible HTTP server: POST /v1/chat/completions (streaming or not), GET /v1/models.
    /// One request at a time; temperature, top_k, top_p and seed are honored, greedy when left out. A request that
    /// extends the previous one's messages prefills only the new part (usage.prompt_tokens_details.cached_tokens);
    /// one that keeps only the system prompt and tools (a compacted conversation) starts after them.
    /// GET / is a chat page that can also switch between the models in `--models-dir` (POST /api/load), so the
    /// server can start without `--model`.
    Serve {
        #[arg(long)]
        model: Option<PathBuf>,
        /// Folder of .gguf models the chat page offers [default: $XDG_DATA_HOME/rusting_strata/models,
        /// %LOCALAPPDATA%\rusting_strata\models on Windows, else ~/.local/share/rusting_strata/models].
        #[arg(long)]
        models_dir: Option<PathBuf>,
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
        /// Open the chat page in the browser (if the address is taken, open it and exit: strata already runs there).
        #[arg(long)]
        open: bool,
        /// Require `Authorization: Bearer <key>` on every request [default: $STRATA_API_KEY, which keeps the key out
        /// of the process list].
        #[arg(long)]
        api_key: Option<String>,
        #[command(flatten)]
        host: Hosting,
        #[command(flatten)]
        engine: Engine,
    },
    /// Measure decode speed at fixed model/KV/context settings. Only candidates
    /// matching repeated greedy reference outputs are eligible for selection.
    Calibrate {
        #[arg(long)]
        model: PathBuf,
        /// GPU context length used while measuring and assumed for the results.
        #[arg(long, default_value_t = 4096)]
        ctx: usize,
        /// VRAM left free per device, as `--vram-margin`.
        #[arg(long, default_value = "512", value_delimiter = ',')]
        vram_margin: Vec<String>,
        /// Owned expert RAM ceiling: auto, 0 to disable, or MiB, as for run/serve.
        #[arg(long, default_value = "auto")]
        expert_ram: String,
        /// Fill free expert RAM capacity before measuring each candidate.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        expert_warmup: bool,
        /// Host RAM left available to the OS and other applications, in MiB.
        #[arg(long, default_value_t = 4096)]
        ram_margin: usize,
        /// Soft owned-host-memory cap in MiB; 0 uses available RAM.
        #[arg(long, default_value_t = 0)]
        max_ram: usize,
        /// Reserve for conversation-state caching, in MiB, as for run/serve.
        #[arg(long, default_value_t = 512)]
        cache_ram: usize,
        #[arg(long)]
        backend: Option<String>,
        /// GPU index, or several separated by commas (such as 0,1) to calibrate the layer split over those GPUs.
        #[arg(long, default_value = "0", value_delimiter = ',')]
        device: Vec<usize>,
        /// Accepted for consistency with the other commands; calibration always uses the base model.
        #[arg(long, value_name = "PATH:SCALE", action = clap::ArgAction::Append)]
        lora: Vec<String>,
        /// KV cache format to calibrate with (q8 leaves more VRAM for experts), as `--kv` of the runs that use it.
        #[arg(long, default_value = "f16", value_parser = ["f16", "q8"])]
        kv: String,
        /// One UTF-8 user prompt in place of the three built-in chat workloads.
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        /// Maximum generated tokens per workload (32..=1024).
        #[arg(long, default_value_t = 128)]
        tg: usize,
        /// Measured repetitions per workload (3..=15), after excluded warmup.
        #[arg(long, default_value_t = 3)]
        reps: usize,
        /// Excluded warmup repetitions per workload and candidate (1..=5).
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        /// Write the full calibration report, including rejected candidates.
        #[arg(long, value_parser = json_out_path)]
        json_out: Option<PathBuf>,
        /// Fixed CPU worker count; compare counts using separate processes.
        #[arg(long)]
        threads: Option<usize>,
        /// Accepted for symmetry with the other commands: calibration always runs on the GPU.
        #[arg(long, hide = true)]
        gpu: bool,
    },
    /// Check that every Vulkan pipeline this model needs fits the device's `maxComputeSharedMemorySize`, naming the
    /// kernel, specialization, bytes needed and limit when one does not; exits non-zero on overflow. No model is
    /// loaded, no weight uploaded and no pipeline created: it is SPIR-V plus spec-constant arithmetic only.
    Check {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        backend: Option<String>,
        #[arg(long, default_value_t = 0)]
        device: usize,
    },
    /// Speed test: for each prompt length, prefill that many tokens of prose, then greedily decode `tg` more at that
    /// depth. Prints prefill and decode tok/s as mean (min-max) over `reps` runs. `--ctx` grows to fit.
    /// `--json-out PATH` also writes the same numbers as one `strata.bench.v1` JSON object plus a newline (`-` is
    /// rejected: stdout keeps the text block).
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "512,4096")]
        pp: Vec<usize>,
        #[arg(long, default_value_t = 128)]
        tg: usize,
        #[arg(long, default_value_t = 3)]
        reps: usize,
        /// Excluded warmup repetitions per prompt length before measurement.
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        #[arg(long, value_parser = json_out_path)]
        json_out: Option<PathBuf>,
        /// Prefill this text file instead of the built-in paragraph (repeated when shorter than `--pp`). The built-in
        /// prose routes few experts; a real prompt shows what a first request costs.
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        #[command(flatten)]
        engine: Engine,
    },
    /// Perplexity over a text file as llama-perplexity computes it: chunks of `ctx-len` tokens, each from an empty
    /// context, scored over their second half (the first half prefilled, the rest decoded token by token).
    /// `--kld-base` writes the log-probabilities in llama-perplexity's format, so
    /// `llama-perplexity -m MODEL --kl-divergence --kl-divergence-base FILE` compares llama.cpp's logits with these
    /// (KLD, top-token agreement and both perplexities) on the same tokens.
    Perplexity {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        file: PathBuf,
        #[arg(long, default_value_t = 512)]
        ctx_len: usize,
        /// Score only the first N chunks.
        #[arg(long)]
        chunks: Option<usize>,
        #[arg(long)]
        kld_base: Option<PathBuf>,
        #[command(flatten)]
        engine: Engine,
    },
    /// Text generation (greedy unless --temp).
    Generate {
        #[arg(long)]
        model: PathBuf,
        /// Prompt text ("-" reads stdin; long prompts exceed the 128 KiB argument limit).
        #[arg(long)]
        prompt: String,
        #[arg(short, default_value_t = 32)]
        n: usize,
        /// Image path, data URL or http(s) URL (repeatable). Each is inserted where the prompt has `<image>`;
        /// without placeholders all images are prepended. Needs --mmproj (or a sibling mmproj next to the model).
        #[arg(long, action = clap::ArgAction::Append)]
        image: Vec<String>,
        #[command(flatten)]
        engine: Engine,
    },
}

/// Run the native engine CLI in this process.
///
/// `args` includes the executable name, as with [`std::env::args_os`]. Paths
/// remain OS-native; no shell or child inference process is involved. Argument
/// errors are returned to the caller. Help and version output return success.
/// With no subcommand, the original interactive server-launch behavior remains.
pub fn run_from<I, T>(args: I) -> anyhow::Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let cli = match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(error) if matches!(
            error.kind(),
            clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
        ) => {
            error.print()?;
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    let Some(cmd) = cli.cmd else {
        // Started without arguments, as by a double click: keep the window open on an error so it can be read.
        let Some(Cmd::Serve { model, models_dir, addr, api_key, open, host, mut engine }) = Cli::parse_from(["rustingstrata", "serve", "--gpu", "--open"]).cmd else { unreachable!() };
        engine.cpu_fallback = true;
        let r = run(Cmd::Serve { model, models_dir, addr, api_key, open, host, engine });
        if let Err(e) = &r {
            eprintln!("error: {e:#}\npress Enter to close");
            let _ = std::io::stdin().read_line(&mut String::new());
        }
        return r;
    };
    run(cmd)
}

fn run(cmd: Cmd) -> anyhow::Result<()> {
    // The CPU kernels are chosen at compile time. A build without AVX2 (RUSTFLAGS in the environment replaces
    // .cargo/config.toml's target-cpu=native) runs MoE experts that miss the VRAM cache about 10x slower (#44).
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    if std::arch::is_x86_feature_detected!("avx2") {
        eprintln!("warning: this binary was built without AVX2 but this CPU has it: CPU work is about 10x slower. Rebuild with \
                   RUSTFLAGS=\"-C target-cpu=native\" (or x86-64-v3), or use a release binary.");
    }
    match cmd {
        Cmd::Info { model } => {
            let g = Gguf::open(&model)?;
            println!("{:#?}", ModelConfig::from_gguf(&g)?);
            let mut hist = BTreeMap::<String, usize>::new();
            for t in g.tensors.values() {
                let name = GgmlType::from_u32(t.ty).map(|ty| ty.name().to_string()).unwrap_or(format!("type{}", t.ty));
                *hist.entry(name).or_default() += 1;
            }
            for (ty, n) in hist { println!("{n:5} {ty}"); }
        }
        Cmd::Score { model, input, mut engine } => {
            let text = if input == "-" {
                let mut s = String::new();
                std::io::stdin().read_to_string(&mut s)?;
                s
            } else {
                std::fs::read_to_string(&input)?
            };
            let v: serde_json::Value = serde_json::from_str(&text)?;
            let ids = |k: &str| -> anyhow::Result<Vec<u32>> {
                v[k].as_array().with_context(|| format!("missing {k}"))?
                    .iter().map(|x| x.as_u64().and_then(|x| u32::try_from(x).ok()).context("bad id")).collect()
            };
            let (prompt, cont) = (ids("prompt")?, ids("continuation")?);
            anyhow::ensure!(!prompt.is_empty() && !cont.is_empty(), "empty prompt or continuation");
            let g = Gguf::open(&model)?;
            let aux_g = open_mtp_gguf(engine.mtp_model.as_deref())?;
            let m = Model::load_with_aux(&g, aux_g.as_ref(), &engine.lora_specs())?;
            if let Some(&t) = prompt.iter().chain(&cont).find(|&&t| t as usize >= m.cfg.n_vocab) {
                anyhow::bail!("token id {t} out of range (vocabulary has {} tokens)", m.cfg.n_vocab);
            }
            let feed: Vec<u32> = prompt.iter().chain(&cont[..cont.len() - 1]).copied().collect();
            anyhow::ensure!(feed.len() <= engine.ctx,
                "scoring needs {} prompt/continuation tokens, exceeding --ctx {}; shorten the input or choose a supported larger context", feed.len(), engine.ctx);
            let mut r = engine.runner(&m, &model)?;
            let (mut argmax, mut top5) = (Vec::new(), Vec::new());
            let mut logits = r.prefill(&prompt)?;
            for i in prompt.len() - 1..feed.len() {
                if i >= prompt.len() { logits = r.step(feed[i])?; }
                let lse = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let lse = lse + logits.iter().map(|x| (x - lse).exp()).sum::<f32>().ln();
                argmax.push(ops::argmax(&logits));
                top5.push(ops::top_k(&logits, 5).into_iter().map(|j| (j, logits[j] - lse)).collect::<Vec<_>>());
            }
            r.finish()?;
            println!("{}", serde_json::json!({"argmax": argmax, "top5": top5}));
        }
        Cmd::Generate { model, prompt, n, image, mut engine } => {
            engine.ds_env();
            let g = Gguf::open(&model)?;
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some("gemma4") {
                let prompt = if prompt == "-" { let mut s = String::new(); std::io::stdin().read_to_string(&mut s)?; s } else { prompt };
                if image.is_empty() { return rusting_strata::gemma4::generate(&g, &prompt, n); }
                let mmproj = engine.mmproj.clone().or_else(|| vision::find_sibling(&model));
                let mmproj = mmproj.as_deref().context("--image needs --mmproj")?;
                let input = load_images(mmproj, &image)?;
                return rusting_strata::gemma4::generate_image(&g, &prompt, &input.grids, &input.embeddings, n);
            }
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some("glm5-next") {
                anyhow::ensure!(image.is_empty(), "--image is not supported for glm5-next");
                let prompt = if prompt == "-" { let mut s = String::new(); std::io::stdin().read_to_string(&mut s)?; s } else { prompt };
                return rusting_strata::glm::generate(&g, &prompt, n);
            }
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some("deepseek41") {
                anyhow::ensure!(image.is_empty(), "--image is not supported for deepseek41");
                let prompt = if prompt == "-" { let mut s = String::new(); std::io::stdin().read_to_string(&mut s)?; s } else { prompt };
                return rusting_strata::deepseek::chat::generate(&g, &prompt, n);
            }
            let aux_g = open_mtp_gguf(engine.mtp_model.as_deref())?;
            let m = Model::load_with_aux(&g, aux_g.as_ref(), &engine.lora_specs())?;
            let tok = Tokenizer::from_gguf(&g)?;
            let prompt = if prompt == "-" {
                let mut s = String::new();
                std::io::stdin().read_to_string(&mut s)?;
                s
            } else { prompt };
            let vision_prompt = if image.is_empty() {
                None
            } else {
                let mmproj = engine.mmproj.clone().or_else(|| vision::find_sibling(&model));
                let mmproj = mmproj.as_deref().context("--image needs --mmproj")?;
                let input = load_images(mmproj, &image)?;
                Some((build_image_prompt(&tok, &input, &image, &prompt)?, input))
            };
            let ids = tok.encode(&prompt)?;
            anyhow::ensure!(!ids.is_empty() || vision_prompt.is_some(), "empty prompt");
            let n_prompt = vision_prompt.as_ref().map_or(ids.len(), |(p, _)| p.ids.len());
            let cap = serve::ContextBudget::new(engine.ctx, n_prompt, Some(n), usize::MAX).ensure_fits()?;
            let mut r = engine.runner(&m, &model)?;
            let t0 = Instant::now();
            let (logits, n_prompt) = match vision_prompt {
                None => (r.prefill(&ids)?, ids.len()),
                Some((p, input)) => (r.chat_image(&p, &input.embeddings)?, p.ids.len()),
            };
            let t1 = Instant::now();
            let (cpu0, split0) = (r.cpu_secs(), r.split_secs());
            let (out, _) = decode(&mut r, &tok, engine.mtp() > 0, &mut engine.sampler(), logits, cap, &mut |t| { print!("{t}"); std::io::stdout().flush()?; Ok(false) })?;
            let (pp, tg) = ((t1 - t0).as_secs_f64(), t1.elapsed().as_secs_f64());
            eprintln!("\nprompt {} tok {:.2} tok/s | gen {} tok {:.2} tok/s", n_prompt, n_prompt as f64 / pp, out.len(), out.len() as f64 / tg);
            if engine.gpu { eprintln!("CPU expert compute, share of gen time: {:.1}%", 100.0 * (r.cpu_secs() - cpu0) / tg); }
            let ms = |s: f64| 1e3 * s / out.len().max(1) as f64;
            for (i, ((a, b), (a0, b0))) in r.split_secs().into_iter().zip(split0).enumerate() {
                eprintln!("device {i}: {:.2} ms/token forward, {:.2} ms/token copy to the next device", ms(a - a0), ms(b - b0));
            }
            if !r.split_secs().is_empty() { eprintln!("gen total {:.2} ms/token", ms(tg)); }
            r.finish()?;
        }
        Cmd::Perplexity { model, file, ctx_len: n, chunks, kld_base, mut engine } => {
            let g = Gguf::open(&model)?;
            let aux_g = open_mtp_gguf(engine.mtp_model.as_deref())?;
            let m = Model::load_with_aux(&g, aux_g.as_ref(), &engine.lora_specs())?;
            let tok = Tokenizer::from_gguf(&g)?;
            // ponytail: no BOS at chunk starts (llama-perplexity adds one where the vocabulary asks for it; Qwen does not)
            let ids = tok.encode(&std::fs::read_to_string(&file)?)?;
            let n_chunk = (ids.len() / n).min(chunks.unwrap_or(usize::MAX));
            anyhow::ensure!(n >= 4 && n_chunk > 0, "{} tokens: need at least one chunk of --ctx-len {n} (>= 4)", ids.len());
            engine.ctx = engine.ctx.max(n);
            let mut r = engine.runner(&m, &model)?;
            let nv = m.cfg.n_vocab;
            let mut out = kld_base.map(|p| std::fs::File::create(p).map(std::io::BufWriter::new)).transpose()?;
            if let Some(o) = &mut out {
                o.write_all(b"_logits_")?;
                for v in [n, nv, n_chunk] { o.write_all(&(v as u32).to_le_bytes())?; }
                for &t in &ids[..n_chunk * n] { o.write_all(&t.to_le_bytes())?; }
            }
            let (mut nll, mut count, mut row) = (0f64, 0usize, Vec::new());
            let first = n / 2;
            for c in ids.chunks_exact(n).take(n_chunk) {
                r.reset()?;
                let mut logits = r.prefill(&c[..=first])?;
                for i in first..n - 1 {
                    if i > first { logits = r.step(c[i])?; }
                    anyhow::ensure!(logits.len() == nv, "{} logits, vocabulary {nv}", logits.len());
                    nll += log_probs(&logits, c[i + 1], &mut row);
                    if let Some(o) = &mut out { for v in &row { o.write_all(&v.to_le_bytes())?; } }
                    count += 1;
                }
                eprint!("[{}]{:.4},", count / (n - 1 - first), (nll / count as f64).exp());
            }
            if let Some(o) = &mut out { o.flush()?; }
            println!("\nPPL = {:.4} over {n_chunk} chunks of {n} tokens ({count} scored)", (nll / count as f64).exp());
            r.finish()?;
        }
        Cmd::Bench { model, pp, tg, reps, warmup, json_out, prompt_file, mut engine } => {
            anyhow::ensure!(tg > 0 && reps > 0 && reps <= 100 && warmup <= 10, "bench needs positive tg/reps, reps <= 100 and warmup <= 10");
            if let Some(f) = &prompt_file {
                let text = std::fs::read_to_string(f).with_context(|| format!("--prompt-file {}", f.display()))?;
                anyhow::ensure!(!text.trim().is_empty(), "--prompt-file {} is empty", f.display());
                let _ = BENCH_TEXT.set(text);
            }
            let g = Gguf::open(&model)?;
            engine.ctx = engine.ctx.max(pp.iter().max().unwrap_or(&0) + tg);
            if let Some(arch) = g.get("general.architecture").ok().and_then(|v| v.as_str()) {
                if arch == "deepseek41" || arch == "glm5-next" {
                    engine.ds_env();
                    return if arch == "deepseek41" {
                        bench_deepseek(&g, &model, &pp, tg, reps, warmup, json_out, &mut engine)
                    } else {
                        bench_glm(&g, &model, &pp, tg, reps, warmup, json_out, &mut engine)
                    };
                }
            }
            let aux_g = open_mtp_gguf(engine.mtp_model.as_deref())?;
            let m = Model::load_with_aux(&g, aux_g.as_ref(), &engine.lora_specs())?;
            let tok = Tokenizer::from_gguf(&g)?;
            let mut r = engine.runner(&m, &model)?;
            let mut results = Vec::with_capacity(pp.len());
            let mut raw_samples = Vec::new();
            let mut prompts = Vec::new();
            for n in pp {
                let ids = prose(&tok, n)?;
                prompts.push(json!({"tokens":n,"token_ids_sha256":token_digest(&ids)}));
                let (mut p, mut d) = (Vec::new(), Vec::new());
                for iteration in 0..warmup + reps {
                    r.reset()?;
                    let t = Instant::now();
                    let l = r.prefill(&ids)?;
                    let prefill_seconds = t.elapsed().as_secs_f64();
                    let t = Instant::now();
                    let k = decode(&mut r, &tok, engine.mtp() > 0, &mut Sampler::greedy(), l, tg, &mut |_| Ok(false))?.0.len();
                    let decode_seconds = t.elapsed().as_secs_f64();
                    if iteration >= warmup {
                        p.push(n as f64 / prefill_seconds);
                        d.push(k as f64 / decode_seconds);
                        raw_samples.push(json!({"prompt_tokens":n,"repetition":iteration-warmup,
                            "completion_tokens":k,"prefill_seconds":prefill_seconds,"decode_seconds":decode_seconds,
                            "decode_tokens_per_second":k as f64/decode_seconds}));
                    }
                }
                let (prefill, decode_stat) = (Stat::of(&p), Stat::of(&d));
                println!("pp {n:6}: prefill {} tok/s | tg {tg}: decode {} tok/s", prefill.text(), decode_stat.text());
                // R3: the attained expert-read bandwidth (bytes per token x tokens per second), against the ~900 GB/s
                // of the V100, for the prefetch decision the window records.
                let eb = r.expert_bytes_per_token();
                let mid = if eb > 0 {
                    let mid = { let mut v = d.clone(); v.sort_by(f64::total_cmp); v[v.len() / 2] };
                    let estimate = if runner_tail_stats(&r).skipped_assignments > 0 { "expert traffic upper bound before omissions" }
                        else { "inferred routed expert traffic" };
                    eprintln!("{estimate}: {:.1} MB/token -> {:.1} GB/s at {mid:.1} tok/s (not measured GPU/PCIe bandwidth)", eb as f64 / 1048576.0, eb as f64 * mid / 1e9);
                    mid
                } else { 0.0 };
                results.push(BenchResult { pp: n, prefill, decode: decode_stat, bytes_per_token: eb, decode_median: mid });
            }
            r.finish()?;
            // After `finish()`: the expert counters are final, and the record cannot reorder `save_profile()`.
            if let Some(path) = json_out {
                let device: &[usize] = if r.backend_name() == "cpu" { &[] } else { &engine.device };
                let meta = BenchMeta {
                    backend: r.backend_name(),
                    backend_arg: engine.backend.as_deref(),
                    device,
                    model: &model,
                    ctx: engine.ctx,
                    chunk: engine.chunk(),
                    kv: &engine.kv,
                    mtp: engine.mtp(),
                    tg,
                    reps,
                };
                let unix_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_secs());
                let mut record = bench_record(&meta, &results, r.expert_summary().as_ref(), unix_time);
                record["expert_ram"] = json!(r.expert_ram_stats());
                record["expert_ram_requested"] = json!(engine.expert_ram);
                record["expert_warmup"] = json!(engine.expert_warmup);
                annotate_runtime_bench(&mut record, runner_tail_stats(&r), &g.identity);
                record["warmup_repetitions_per_prompt_excluded"] = json!(warmup);
                record["samples"] = json!(raw_samples);
                record["prompts"] = json!(prompts);
                record["decode_cache_policy"] = json!(DECODE_CACHE_POLICY);
                record["threads"] = json!(rayon::current_num_threads());
                record["llama_numerics"] = json!(std::env::var_os("STRATA_LLAMA_NUMERICS").is_some());
                record["inference_overrides"] = json!(inference_overrides());
                let mut line = serde_json::to_string(&record)?;
                line.push('\n');
                std::fs::write(&path, line).with_context(|| format!("writing the JSON record to {}", path.display()))?;
            }
        }
        Cmd::Calibrate { model, ctx, vram_margin, expert_ram, expert_warmup, ram_margin, max_ram, cache_ram,
            backend, device, kv, lora, gpu: _, prompt_file, tg, reps, warmup, json_out, threads } => {
            if kv == "q8" { unsafe { std::env::set_var("STRATA_KV_Q8", "1") } }  // read by Gpu::new and Vk::new
            anyhow::ensure!(lora.is_empty(), "calibration measures the base model; runtime LoRA adapters are not supported by this tuner");
            let margins = vram_margins(&vram_margin, &device)?.into_iter().map(|mb| mb << 20).collect::<Vec<_>>();
            calibrate(&model, ctx, &margins, backend.as_deref(), &device, &kv,
                &TuneOptions { prompt_file, tg, reps, warmup, json_out, threads,
                    expert_ram, expert_warmup, ram_margin, max_ram, cache_ram })?
        }
        Cmd::Check { model, backend, device } => {
            let g = Gguf::open(&model)?;
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some("deepseek41") {
                let c = rusting_strata::deepseek::DsConfig::from_gguf(&g)?;
                c.validate(&g)?;
                println!("check: deepseek41 ok: {} layers, {} experts ({} used), {} tensors, tensor table valid (CPU reference only; no GPU path yet)", c.n_layer, c.n_expert, c.n_used, g.tensors.len());
                return Ok(());
            }
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some("glm5-next") {
                let c = rusting_strata::glm::GlmConfig::from_gguf(&g)?;
                c.validate(&g)?;
                println!("check: glm5-next ok: {} layers ({} KDA, {} DSA) + {} NextN, {} experts ({} used), {} tensors, tensor table valid", c.n_layer, c.n_kda(), c.n_dsa(), c.n_nextn, c.n_expert, c.n_used, g.tensors.len());
                return Ok(());
            }
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some("flint") {
                let c = ModelConfig::from_gguf(&g)?;
                let n_moe = c.layers.iter().enumerate().filter(|&(il, _)| c.is_moe_layer(il)).count();
                println!("check: flint ok: {} layers ({} dense, {} MoE), {} experts ({} used), {}/{} sliding-window / no-RoPE layers, {} tensors", c.n_layer, c.n_dense, n_moe, c.n_expert, c.n_expert_used, c.swa.iter().filter(|&&s| s).count(), c.rope_layer.iter().filter(|&&r| !r).count(), g.tensors.len());
            }
            let types = rusting_strata::gpu_model::gpu_weight_types(&g);
            let b = backend::open(backend.as_deref(), device)?;
            if b.name() != "vulkan" {
                eprintln!("check: the shared-memory preflight is Vulkan-only; nothing to check on {}", b.name());
                return Ok(());
            }
            b.preflight(&types, true)?;
        }
        Cmd::Chat { model, system, n, image, mut engine } => {
            let think = engine.think_default;
            engine.ds_env();
            let g = Gguf::open(&model)?;
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some("gemma4") {
                let input = if image.is_empty() { None } else {
                    let mmproj = engine.mmproj.clone().or_else(|| vision::find_sibling(&model));
                    let mmproj = mmproj.as_deref().context("--image needs --mmproj")?;
                    Some(load_images(mmproj, &image)?)
                };
                return match &input {
                    Some(i) => rusting_strata::gemma4::chat(&g, system, n, think, Some((&i.grids, &i.embeddings))),
                    None => rusting_strata::gemma4::chat(&g, system, n, think, None),
                };
            }
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some("glm5-next") {
                anyhow::ensure!(image.is_empty(), "--image is not supported for glm5-next");
                return rusting_strata::glm::chat(&g, system, n);
            }
            if g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some("deepseek41") {
                anyhow::ensure!(image.is_empty(), "--image is not supported for deepseek41");
                return rusting_strata::deepseek::chat::chat(&g, system, n, think);
            }
            let aux_g = open_mtp_gguf(engine.mtp_model.as_deref())?;
            let m = Model::load_with_aux(&g, aux_g.as_ref(), &engine.lora_specs())?;
            let tok = Tokenizer::from_gguf(&g)?;
            let image_input = if image.is_empty() { None } else {
                let mmproj = engine.mmproj.clone().or_else(|| vision::find_sibling(&model));
                let mmproj = mmproj.as_deref().context("--image needs --mmproj")?;
                Some(load_images(mmproj, &image)?)
            };
            let mut r = engine.runner(&m, &model)?;
            let mut smp = engine.sampler();
            let tpl = template(&g);
            let mut msgs: Vec<Msg> = system.iter().map(|s| Msg::new("system", s)).collect();
            let stdin = std::io::stdin();
            loop {
                print!("> ");
                std::io::stdout().flush()?;
                let mut line = String::new();
                if stdin.lock().read_line(&mut line)? == 0 { break; }
                if line.trim().is_empty() { continue; }
                msgs.push(Msg::new("user", line.trim_end()));
                let prompt = tpl.render(&msgs, &[], think, engine.preserve_thinking, true);
                let vision_prompt = image_input.as_ref().map(|input| build_image_prompt(&tok, input, &image, &prompt)).transpose()?;
                let tokens = if vision_prompt.is_none() {
                    let n_sys = msgs.iter().take_while(|m| m.role == "system").count();
                    Some(serve::ChatTokens::new(&tok, &prompt,
                        &tpl.render(&msgs, &[], think, engine.preserve_thinking, false),
                        &tpl.render(&msgs[..n_sys], &[], think, engine.preserve_thinking, false))?)
                } else { None };
                let used = vision_prompt.as_ref().map_or_else(|| tokens.as_ref().unwrap().ids.len(), |p| p.ids.len());
                let cap = match serve::ContextBudget::new(engine.ctx, used, Some(n), usize::MAX).ensure_fits() {
                    Ok(cap) => cap,
                    Err(err) => { msgs.pop(); eprintln!("{err:#}\nThis turn was not added; the earlier conversation is unchanged."); continue; }
                };
                let logits = match &vision_prompt {
                    Some(p) => r.chat_image(p, &image_input.as_ref().unwrap().embeddings)?,
                    None => r.chat_tokens(tokens.as_ref().unwrap())?.0,
                };
                let (out, _) = decode(&mut r, &tok, engine.mtp() > 0, &mut smp, logits, cap, &mut |t| { print!("{t}"); std::io::stdout().flush()?; Ok(false) })?;
                println!();
                msgs.push(Msg::new("assistant", &answer(&tok.decode(&out)?)));
            }
            r.finish()?;
        }
        Cmd::Serve { model, models_dir, addr, api_key, open, host, mut engine } => {
            engine.ds_env();
            if let Some(mp) = &model && let Ok(g) = Gguf::open(mp) && g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some("gemma4") {
                let name = mp.file_name().map_or("gemma4".into(), |n| n.to_string_lossy().into_owned());
                let key = api_key.or_else(|| std::env::var("STRATA_API_KEY").ok());
                anyhow::ensure!(key.as_ref().is_none_or(|k| !k.is_empty()), "empty API key");
                let mmproj = engine.mmproj.clone().or_else(|| vision::find_sibling(mp));
                return rusting_strata::gemma4::serve(&g, &name, &addr, key.as_deref(), mmproj.as_deref());
            }
            if let Some(mp) = &model && let Ok(g) = Gguf::open(mp) && g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some("glm5-next") {
                let name = mp.file_name().map_or("glm".into(), |n| n.to_string_lossy().into_owned());
                let key = api_key.or_else(|| std::env::var("STRATA_API_KEY").ok());
                anyhow::ensure!(key.as_ref().is_none_or(|k| !k.is_empty()), "empty API key");
                return rusting_strata::glm::serve(&g, &name, &addr, key.as_deref());
            }
            if let Some(mp) = &model && let Ok(g) = Gguf::open(mp) && g.get("general.architecture").ok().and_then(|v| v.as_str()) == Some("deepseek41") {
                let name = mp.file_name().map_or("deepseek".into(), |n| n.to_string_lossy().into_owned());
                let key = api_key.or_else(|| std::env::var("STRATA_API_KEY").ok());
                anyhow::ensure!(key.as_ref().is_none_or(|k| !k.is_empty()), "empty API key");
                return rusting_strata::deepseek::chat::serve(&g, &name, &addr, key.as_deref());
            }
            engine.parallel_mode = host.parallel;
            engine.max_parallel = host.max_parallel;
            engine.parallel = resolve_capacity(host.parallel, host.max_parallel, engine.gpu, engine.mtp(), ModelFeatures::default())?.slots;
            validate_context_length(engine.ctx, None)?;
            anyhow::ensure!(host.max_tokens != Some(0), "--max-tokens must be at least 1");
            // This is a conservative connection budget before model-specific limits are known.
            let max_waiting = host.max_waiting.unwrap_or(64);
            validate_queue_limits(host.queue_capacity, host.queue_timeout, max_waiting, engine.parallel)?;
            let api_key = api_key.or_else(|| std::env::var("STRATA_API_KEY").ok());
            anyhow::ensure!(api_key.as_ref().is_none_or(|k| !k.is_empty()), "empty API key");
            let mut allowed_hosts = host.allowed_hosts.clone();
            if allowed_hosts.is_empty() && let Ok(v) = std::env::var("STRATA_ALLOWED_HOSTS") {
                allowed_hosts.extend(v.split(',').map(str::trim).filter(|h| !h.is_empty()).map(str::to_owned));
            }
            let mut keys_time = host.api_keys.as_ref().and_then(|f| std::fs::metadata(f).and_then(|m| m.modified()).ok());
            let users = match &host.api_keys {
                Some(f) => serve::Keys::parse(&std::fs::read_to_string(f).with_context(|| format!("reading {}", f.display()))?)
                    .with_context(|| format!("in {}", f.display()))?,
                None => vec![],
            };
            anyhow::ensure!(!users.iter().any(|u| api_key.as_ref() == Some(&u.1)), "the admin key is also a customer key");
            // An empty key file would leave the server open to anyone.
            anyhow::ensure!(host.api_keys.is_none() || !users.is_empty() || api_key.is_some(), "the key file has no keys");
            let keys = std::sync::Arc::new(std::sync::RwLock::new(serve::Keys { admin: api_key, users }));
            let keys_file = host.api_keys.clone();
            let usage = match &host.usage_log {
                Some(f) => Some(std::sync::Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(f)
                    .with_context(|| format!("opening {}", f.display()))?)),
                None => None,
            };
            anyhow::ensure!(host.max_per_key != Some(0), "--max-per-key must be at least 1");
            let url = format!("http://{addr}/");
            let listener = match TcpListener::bind(&addr) {
                Err(e) if open && e.kind() == std::io::ErrorKind::AddrInUse => { open_browser(&url); return Ok(()); }
                l => l?,
            };
            // Connections are read on their own threads, so a client that is slow to send its request does not hold
            // up the others; complete requests run one at a time, in arrival order.
            let (tx, rx) = std::sync::mpsc::channel();
            let open_conns = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let waiting = open_conns.clone();
            let stats = std::sync::Arc::new(std::sync::Mutex::new(serve::Stats::default()));
            let admission = Admission::new(0, host.queue_capacity, Duration::from_secs(host.queue_timeout));
            let policy = std::sync::Arc::new(std::sync::Mutex::new(json!({
                "mode": host.parallel.to_string(), "ceiling": host.max_parallel,
                "slots": 0, "reason": "no model loaded"
            })));
            let (admit, capacity_policy) = (admission.clone(), policy.clone());
            let output_cap = host.max_tokens;
            let (st, a, k, allowed, t0) = (stats.clone(), addr.clone(), keys.clone(), allowed_hosts.clone(), Instant::now());
            let (per_key, held) = (host.max_per_key, serve::Open::default());
            let stop = on_sigterm();
            // SIGTERM (systemctl stop or restart): new requests get 503, the requests accepted before it finish, then
            // the server exits. The model thread does that itself, after saving the expert profile (`Serving::stopped`);
            // this thread covers a server with no model or one loading. Two looks 0.2 s apart cover a connection counted
            // just before the signal, and the model thread gets 3 s to save first.
            let oc = open_conns.clone();
            std::thread::spawn(move || loop {
                std::thread::sleep(Duration::from_millis(200));
                if !stop.load(Relaxed) { continue; }
                let n = oc.load(Relaxed);
                if n == 0 { std::thread::sleep(Duration::from_millis(200)); if oc.load(Relaxed) == 0 { std::thread::sleep(Duration::from_secs(3)); eprintln!("stopped"); std::process::exit(0); } }
                else { eprintln!("stopping: {n} requests to finish"); std::thread::sleep(Duration::from_secs(5)); }
            });
            std::thread::spawn(move || for (i, conn) in listener.incoming().enumerate() {
                let mut c = match conn { Ok(c) => c, Err(e) => { eprintln!("request {i}: {e:#}"); continue } };
                if stop.load(Relaxed) {
                    let b = json!({"error": {"message": "server is stopping, retry later", "type": "overloaded_error"}}).to_string();
                    let _ = c.set_write_timeout(Some(Duration::from_secs(5)));
                    let _ = write!(c, "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 5\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
                    let _ = c.shutdown(std::net::Shutdown::Write);
                    continue;
                }
                // ponytail: one cap on connections read or queued (each may hold an 8 MB body), not one per key.
                if open_conns.fetch_add(1, Relaxed) >= max_waiting {
                    open_conns.fetch_sub(1, Relaxed);
                    eprintln!("request {i}: {max_waiting} requests already waiting, answered 503");
                    let b = json!({"error": {"message": "server busy, retry later", "type": "overloaded_error"}}).to_string();
                    let _ = c.set_write_timeout(Some(Duration::from_secs(5)));
                    let _ = write!(c, "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 2\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
                    let _ = c.shutdown(std::net::Shutdown::Write);
                    continue;
                }
                // A changed key file takes effect from the next request; one that does not parse keeps the old keys.
                if let Some(f) = &keys_file && let Ok(t) = std::fs::metadata(f).and_then(|m| m.modified()) && Some(t) != keys_time {
                    keys_time = Some(t);
                    let mut kw = k.write().unwrap();
                    match std::fs::read_to_string(f).map_err(Into::into).and_then(|t| serve::Keys::parse(&t)) {
                        Ok(u) if u.iter().any(|u| kw.admin.as_ref() == Some(&u.1)) => eprintln!("{}: the admin key is also a customer key, keeping the old keys", f.display()),
                        // An empty file (or one caught mid-write) would leave the server open.
                        Ok(u) if u.is_empty() && kw.admin.is_none() => eprintln!("{}: no keys, keeping the old keys", f.display()),
                        Ok(u) => { eprintln!("{}: {} keys", f.display(), u.len()); kw.users = u; }
                        Err(e) => eprintln!("{}: {e:#}, keeping the old keys", f.display()),
                    }
                }
                let (tx, oc, st, a, k, allowed, held, admit, policy) = (tx.clone(), open_conns.clone(), st.clone(), a.clone(), k.clone(), allowed.clone(), held.clone(), admit.clone(), capacity_policy.clone());
                std::thread::spawn(move || {
                    let q = c.set_read_timeout(Some(Duration::from_secs(30))).and_then(|_| c.set_write_timeout(Some(Duration::from_secs(30))))
                        .map_err(Into::into).and_then(|_| serve::read_request(&mut BufReader::new(&c)));
                    let reply = |v: serde_json::Value, status: &str| {
                        let b = v.to_string();
                        let _ = write!(&c, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nCache-Control: no-store\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
                        oc.fetch_sub(1, Relaxed);
                    };
                    // For load balancers: no key, no Host check, and an answer while a reply is generated.
                    if let Ok(r) = &q && r.method == "GET" && r.path.split('?').next() == Some("/health") {
                        let (model, loading, slots) = { let g = st.lock().unwrap(); (g.model.is_some() && !g.loading, g.loading, g.slots) };
                        let waiting = oc.load(Relaxed) - 1;
                        return if model { reply(json!({"status": "ok", "waiting": waiting, "slots": slots, "admission": admit.snapshot()}), "200 OK") }
                            else { reply(json!({"status": if loading { "loading" } else { "no model" }, "admission": admit.snapshot()}), "503 Service Unavailable") };
                    }
                    // The page's side panel is answered here, so it stays current while a reply is generated or a
                    // model loads. A bad Host or key falls through to `handle`, which refuses it.
                    if let Ok(r) = &q && r.method == "GET" && matches!(r.path.as_str(), "/api/stats" | "/metrics") && let k = k.read().unwrap().clone()
                        && (k.any() || serve::host_ok(&r.host, &a, &allowed)) && k.who(&r.auth).is_some_and(|w| w.admin) {
                        if r.path == "/metrics" {
                            let mut b = st.lock().unwrap().prometheus(oc.load(Relaxed) - 1, t0.elapsed().as_secs());
                            b.push_str(&admission_metrics(&admit));
                            let _ = write!(&c, "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nCache-Control: no-store\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
                            oc.fetch_sub(1, Relaxed);
                            return;
                        }
                        let mut v = json!(*st.lock().unwrap());
                        (v["system"], v["uptime_s"]) = (serve::system(), t0.elapsed().as_secs().into());
                        v["admission"] = json!(admit.snapshot());
                        v["concurrency"] = policy.lock().unwrap().clone();
                        return reply(v, "200 OK");
                    }
                    // A malformed request, a wrong key, the page and the model list are answered here, not in the queue,
                    // where without `--parallel` they would wait for the reply being generated (`handle` checks the key
                    // again: the keys may change meanwhile). The page holds no secret, so it loads without the key (it
                    // asks for it); without keys the Host check stops DNS rebinding.
                    let who = q.as_ref().ok().and_then(|r| k.read().unwrap().who(&r.auth));
                    match &q {
                        Err(e) => {
                            eprintln!("request {i}: {e:#}");
                            return reply(json!({"error": {"message": format!("{e:#}"), "type": "invalid_request_error"}}), "400 Bad Request");
                        }
                        Ok(r) if r.method == "GET" && r.path == "/" && (k.read().unwrap().any() || serve::host_ok(&r.host, &a, &allowed)) => {
                            let _ = write!(&c, "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nX-Frame-Options: DENY\r\nContent-Security-Policy: frame-ancestors 'none'\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{PAGE}", PAGE.len());
                            oc.fetch_sub(1, Relaxed);
                            return;
                        }
                        Ok(_) if who.is_none() =>
                            return reply(json!({"error": {"message": "missing or wrong API key", "type": "invalid_api_key"}}), "401 Unauthorized"),
                        // Clients list the models often; a model still loading is listed, as its chats wait for the load.
                        Ok(r) if r.method == "GET" && r.path.split('?').next() == Some("/v1/models") && (k.read().unwrap().any() || serve::host_ok(&r.host, &a, &allowed)) => {
                            let created = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_secs());
                            let (m, n_ctx, n_ctx_train) = {
                                let state = st.lock().unwrap();
                                (state.model.clone(), state.ctx, state.ctx_train)
                            };
                            return reply(json!({"object": "list", "data": m.map(|id| json!([{
                                "id": id, "object": "model", "created": created, "owned_by": "strata",
                                "meta": {"n_ctx": n_ctx, "n_ctx_train": n_ctx_train}
                            }])).unwrap_or(json!([]))}), "200 OK");
                        }
                        _ => {}
                    }
                    // Coding clients can inspect capacity without entering the generation queue.
                    if let Ok(r) = &q {
                        let route = r.path.split('?').next().unwrap_or_default();
                        if !k.read().unwrap().any() && !serve::host_ok(&r.host, &a, &allowed) {
                            return reply(json!({"error":{"message":"bad Host header","type":"forbidden"}}), "403 Forbidden");
                        }
                        if r.method == "GET" && route == "/v1/status" {
                            return reply(serving_status(&st, &admit, &policy, output_cap), "200 OK");
                        }
                        if r.method == "POST" && matches!(route, "/v1/responses" | "/v1/completions") {
                            return reply(json!({"error": {"message": "This server supports /v1/chat/completions and /v1/messages. Configure the client's Chat Completions provider with base URL /v1.", "type": "unsupported_endpoint", "code": "unsupported_endpoint"}}), "501 Not Implemented");
                        }
                    }
                    // A customer key over `--max-per-key` is refused here, before it takes a place in the queue.
                    let hold = match (who, per_key) {
                        (Some(w), Some(cap)) if !w.admin => match serve::Hold::take(&held, w.id, cap) {
                            Some(h) => Some(h),
                            None => {
                                eprintln!("request {i}: {} already has {cap} requests open, answered 429", w.name);
                                let b = json!({"error": {"message": format!("this key already has {cap} requests open, retry later"), "type": "rate_limit_error"}}).to_string();
                                let _ = write!(&c, "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
                                oc.fetch_sub(1, Relaxed);
                                return;
                            }
                        },
                        _ => None,
                    };
                    // Queue only validated generation requests. Control routes never consume generation permits.
                    let permit = if let Ok(r) = &q && r.method == "POST"
                        && matches!(r.path.split('?').next(), Some("/v1/chat/completions" | "/v1/messages")) {
                        if !r.json {
                            return reply(json!({"error":{"message":"Content-Type must be application/json","type":"invalid_request_error"}}), "415 Unsupported Media Type");
                        }
                        let parsed = if r.path.split('?').next() == Some("/v1/messages") { serve::messages_body(&r.body) }
                            else { Ok(r.body.clone()) }.and_then(|b| serve::chat_request(&b));
                        if let Err(error) = parsed {
                            return reply(json!({"error":{"message":format!("{error:#}"),"type":"invalid_request_error"}}), "400 Bad Request");
                        }
                        let (loaded, loading) = { let state = st.lock().unwrap(); (state.model.clone(), state.loading) };
                        if loaded.is_none() && !loading {
                            return reply(json!({"error":{"message":"no model loaded","type":"server_error","code":"model_not_ready"}}), "503 Service Unavailable");
                        }
                        if r.path.split('?').next() == Some("/v1/chat/completions") && !loading
                            && let Ok(body) = serde_json::from_slice::<serde_json::Value>(&r.body)
                            && let Some(requested) = body["model"].as_str().filter(|m| !m.trim().is_empty())
                            && loaded.as_deref() != Some(requested) {
                            return reply(json!({"error":{"message":format!("requested model {requested:?} is not loaded"),"type":"model_not_loaded","loaded_model":loaded}}), "409 Conflict");
                        }
                        let mut ticket = match admit.enqueue() {
                            Ok(ticket) => ticket,
                            Err(error) => {
                                let _ = admission_error(&c, i, error);
                                oc.fetch_sub(1, Relaxed);
                                return;
                            }
                        };
                        loop {
                            if serve::gone(&c) {
                                ticket.cancel();
                                oc.fetch_sub(1, Relaxed);
                                return;
                            }
                            match ticket.poll_wait(Duration::from_millis(75)) {
                                Ok(Some(permit)) => break Some(permit),
                                Ok(None) => {},
                                Err(error) => {
                                    let _ = admission_error(&c, i, error);
                                    oc.fetch_sub(1, Relaxed);
                                    return;
                                }
                            }
                        }
                    } else { None };
                    if tx.send((c, q, i, hold, permit)).is_err() { oc.fetch_sub(1, Relaxed); }
                });
            });
            let s = Serving { rx: rx.into(), waiting, stop, addr, allowed_hosts, keys, usage, max_tokens: host.max_tokens.unwrap_or(usize::MAX), dir: models_dir.unwrap_or_else(models_dir_default), model: model.clone(),
                dl: Default::default(), hf_busy: Default::default(), stats, admission, policy, max_waiting, opts: Default::default() };
            // The model to load next and the /api/load request waiting for the outcome (none for --model).
            let mut next = model.map(|p| (p, None));
            eprintln!("listening on {url} (models folder: {})", s.dir.display());
            // Chats wait while a model loads; the page shows at once.
            if open { open_browser(&url); }
            loop {
                let (p, mut w) = match next.take() {
                    Some(x) => x,
                    None => match s.serve(None, &engine) {
                        Some(ModelChange::Load(p, w)) => (p, Some(w)),
                        Some(ModelChange::Unload(w)) => { let _ = acknowledge_unload(w); continue; }
                        None => return Ok(()),
                    },
                };
                // The page's load options stay for later loads; a load that fails takes the old ones back.
                let old = engine.clone();
                let o = s.opts.lock().unwrap().take();
                if !o.is_null() {
                    engine = engine.with(&o)?;
                    // `runner` only sets it.
                    if engine.kv != "q8" { unsafe { std::env::remove_var("STRATA_KV_Q8") } }
                }
                // The page's calibrate button: measure with no model loaded, then load with the measured chunk and MTP
                // (they replace `--chunk`/`--mtp` and the page's earlier values).
                let cal = if o["calibrate"] == true && engine.gpu {
                    // Resolve the actual checkpoint before recording a calibration identity. Auto may have
                    // an ordinary ceiling of four while this Qwen checkpoint supports only one slot.
                    let capacity = resolve_model_parallel(&p, engine.parallel_mode, engine.max_parallel, engine.gpu, 0);
                    if let Ok(plan) = &capacity { engine.parallel = plan.slots; }
                    (engine.chunk, engine.mtp) = (None, None);
                    if engine.kv == "q8" { unsafe { std::env::set_var("STRATA_KV_Q8", "1") } }
                    s.stats.lock().unwrap().unload(p.file_stem().and_then(|n| n.to_str()).map(String::from));
                    capacity.and_then(|_| vram_margins(&engine.vram_margin, &engine.device))
                        .and_then(|mb| calibrate(&p, engine.ctx, &mb.into_iter().map(|mb| mb << 20).collect::<Vec<_>>(), engine.backend.as_deref(), &engine.device, &engine.kv,
                            &TuneOptions::for_engine(&engine)))
                } else { Ok(()) };
                match cal.and_then(|()| serve_model(&p, engine.clone(), &s, &mut w)) {
                    Ok(Some(ModelChange::Load(p, w))) => next = Some((p, Some(w))),
                    Ok(Some(ModelChange::Unload(w))) => { let _ = acknowledge_unload(w); },
                    Ok(None) => return Ok(()),
                    Err(e) if w.is_none() => return Err(e),
                    Err(e) => {
                        engine = old;
                        s.stats.lock().unwrap().unload(None);
                        eprintln!("loading {}: {e:#}", p.display());
                        let b = json!({"error": {"message": format!("{e:#}"), "type": "server_error"}}).to_string();
                        let _ = write!(w.unwrap(), "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
                    }
                }
            }
        }
    }
    Ok(())
}

/// `n` tokens of varied prose, repeated: the prefill prompt for `bench` and `calibrate`.
/// `bench --prompt-file`: the text `prose` repeats instead of its paragraph.
static BENCH_TEXT: std::sync::OnceLock<String> = std::sync::OnceLock::new();

fn prose(tok: &Tokenizer, n: usize) -> anyhow::Result<Vec<u32>> {
    if let Some(text) = BENCH_TEXT.get() {
        let mut ids = tok.encode(text)?;
        anyhow::ensure!(!ids.is_empty(), "--prompt-file gives no tokens");
        while ids.len() < n { ids.extend_from_within(..); }
        ids.truncate(n);
        return Ok(ids);
    }
    let text = "The river town woke before dawn. Fishermen checked their nets, bakers lit ovens, and a clockmaker \
        argued with his apprentice about gears, tides and the price of brass. In 1874 the bridge washed away; \
        the council voted 7 to 5 to rebuild it in stone. ";
    let ids: Vec<u32> = tok.encode(&text.repeat(n / 20 + 1))?.into_iter().take(n).collect();
    anyhow::ensure!(ids.len() == n, "prose prompt too short");
    Ok(ids)
}

/// Expert-cache counters for a DeepSeek/GLM bench run. `tensors_per_expert` is 3 (gate, up, down).
fn big_expert_stats(layers: usize, n_expert: usize) -> Option<ExpertStats> {
    let (_backend, resident, _bytes) = rusting_strata::deepseek::mat::dev_info()?;
    let l = &rusting_strata::deepseek::mat::EX_LOOKUPS;
    Some(ExpertStats {
        hits: l[0].swap(0, Relaxed) as usize,
        misses: l[1].swap(0, Relaxed) as usize,
        swaps: 0,
        resident,
        total: 3 * layers * n_expert,
        bytes_per_token: 0,
        cached: true,
        all_fast: false,
    })
}

fn write_bench_json(path: Option<PathBuf>, meta: &BenchMeta, results: &[BenchResult], expert: Option<&ExpertStats>) -> anyhow::Result<()> {
    let Some(path) = path else { return Ok(()) };
    let unix_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let record = bench_record(meta, results, expert, unix_time);
    let mut line = serde_json::to_string(&record)?;
    line.push('\n');
    std::fs::write(&path, line).with_context(|| format!("writing the JSON record to {}", path.display()))?;
    Ok(())
}

fn decode_greedy_big(m: &mut dyn FnMut(u32) -> Vec<f32>, logits: Vec<f32>, eos: &[u32], tg: usize) -> usize {
    let mut logits = logits;
    let mut n = 0;
    for i in 0..tg {
        let next = ops::argmax(&logits) as u32;
        if eos.contains(&next) { break; }
        n += 1;
        if i + 1 < tg { logits = m(next); }
    }
    n
}

/// `strata bench` for DeepSeek-V4.1 (token-at-a-time reference forward, GPU matvecs when enabled).
fn bench_deepseek(g: &Gguf, model: &Path, pp: &[usize], tg: usize, reps: usize, warmup: usize, json_out: Option<PathBuf>, engine: &mut Engine) -> anyhow::Result<()> {
    rusting_strata::deepseek::chat::gpu_defaults();
    let tok = Tokenizer::from_gguf(g)?;
    let mut m = rusting_strata::deepseek::forward::DsModel::new(g)?;
    let eos = tok.eos_ids();
    let mut results = Vec::with_capacity(pp.len());
    for &n in pp {
        let ids = prose(&tok, n)?;
        let (mut p, mut d) = (Vec::new(), Vec::new());
        for iteration in 0..warmup + reps {
            m.reset();
            let t = Instant::now();
            let logits = m.prefill(&ids);
            let prefill_seconds = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let k = decode_greedy_big(&mut |t| m.step(t), logits, &eos, tg);
            let decode_seconds = t.elapsed().as_secs_f64();
            if iteration >= warmup {
                p.push(n as f64 / prefill_seconds);
                d.push(k as f64 / decode_seconds);
            }
        }
        let (prefill, decode_stat) = (Stat::of(&p), Stat::of(&d));
        println!("pp {n:6}: prefill {} tok/s | tg {tg}: decode {} tok/s", prefill.text(), decode_stat.text());
        results.push(BenchResult { pp: n, prefill, decode: decode_stat, bytes_per_token: 0, decode_median: 0.0 });
    }
    let expert = big_expert_stats(m.cfg.n_layer, m.cfg.n_expert);
    let backend = rusting_strata::deepseek::mat::dev_info().map_or("cpu", |i| i.0);
    let meta = BenchMeta {
        backend,
        backend_arg: engine.backend.as_deref(),
        device: if backend == "cpu" { &[] } else { &engine.device },
        model,
        ctx: engine.ctx,
        chunk: 1,
        kv: &engine.kv,
        mtp: 0,
        tg,
        reps,
    };
    write_bench_json(json_out, &meta, &results, expert.as_ref())
}

/// `strata bench` for GLM-5.3-Flash (chunked reference forward, GPU matvecs when enabled).
fn bench_glm(g: &Gguf, model: &Path, pp: &[usize], tg: usize, reps: usize, warmup: usize, json_out: Option<PathBuf>, engine: &mut Engine) -> anyhow::Result<()> {
    rusting_strata::glm::gpu_defaults();
    let tok = Tokenizer::from_gguf(g)?;
    let mut m = rusting_strata::glm::forward::GlmModel::new(g)?;
    if engine.ctx > rusting_strata::glm::forward::max_ctx(&m.cfg) {
        unsafe { std::env::set_var("STRATA_GLM_CTX", engine.ctx.to_string()) }
    }
    let eos = tok.eos_ids();
    let chunk = std::env::var("STRATA_GLM_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    let mut results = Vec::with_capacity(pp.len());
    for &n in pp {
        let ids = prose(&tok, n)?;
        let (mut p, mut d) = (Vec::new(), Vec::new());
        for iteration in 0..warmup + reps {
            m.reset();
            let t = Instant::now();
            let logits = m.prefill(&ids);
            let prefill_seconds = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let k = decode_greedy_big(&mut |t| m.step(t), logits, &eos, tg);
            let decode_seconds = t.elapsed().as_secs_f64();
            if iteration >= warmup {
                p.push(n as f64 / prefill_seconds);
                d.push(k as f64 / decode_seconds);
            }
        }
        let (prefill, decode_stat) = (Stat::of(&p), Stat::of(&d));
        println!("pp {n:6}: prefill {} tok/s | tg {tg}: decode {} tok/s", prefill.text(), decode_stat.text());
        results.push(BenchResult { pp: n, prefill, decode: decode_stat, bytes_per_token: 0, decode_median: 0.0 });
    }
    let expert = big_expert_stats(m.cfg.n_layer.saturating_sub(m.cfg.n_dense), m.cfg.n_expert);
    let backend = rusting_strata::deepseek::mat::dev_info().map_or("cpu", |i| i.0);
    let meta = BenchMeta {
        backend,
        backend_arg: engine.backend.as_deref(),
        device: if backend == "cpu" { &[] } else { &engine.device },
        model,
        ctx: engine.ctx,
        chunk,
        kv: &engine.kv,
        mtp: 0,
        tg,
        reps,
    };
    write_bench_json(json_out, &meta, &results, expert.as_ref())
}

#[derive(Clone, Debug)]
struct TuneOptions {
    prompt_file: Option<PathBuf>,
    tg: usize,
    reps: usize,
    warmup: usize,
    json_out: Option<PathBuf>,
    threads: Option<usize>,
    expert_ram: String,
    expert_warmup: bool,
    ram_margin: usize,
    max_ram: usize,
    cache_ram: usize,
}

impl Default for TuneOptions {
    fn default() -> Self {
        Self { prompt_file: None, tg: 128, reps: 3, warmup: 1, json_out: None, threads: None,
            expert_ram: "auto".into(), expert_warmup: true, ram_margin: 4096, max_ram: 0, cache_ram: 512 }
    }
}

impl TuneOptions {
    fn for_engine(engine: &Engine) -> Self {
        Self { threads: engine.threads, expert_ram: engine.expert_ram.clone(), expert_warmup: engine.expert_warmup,
            ram_margin: engine.ram_margin, max_ram: engine.max_ram, cache_ram: engine.cache_ram, ..Default::default() }
    }

    fn validate(&self, ctx: usize) -> anyhow::Result<()> {
        memory::parse_expert_ram(&self.expert_ram)?;
        anyhow::ensure!([self.ram_margin, self.max_ram, self.cache_ram].into_iter().all(|n| n <= 1 << 20),
            "tuning RAM settings use MiB and must not exceed 1 TiB each");
        anyhow::ensure!((768..=131_072).contains(&ctx), "tuning context must be between 768 and 131072 tokens");
        anyhow::ensure!((32..=1024).contains(&self.tg), "tuning --tg must be between 32 and 1024 tokens");
        anyhow::ensure!(self.tg < ctx, "tuning output must fit the context beside a prompt");
        anyhow::ensure!((3..=15).contains(&self.reps), "tuning --reps must be between 3 and 15");
        anyhow::ensure!((1..=5).contains(&self.warmup), "tuning --warmup must be between 1 and 5");
        anyhow::ensure!(self.threads.is_none_or(|n| n > 0 && n <= 1024), "tuning --threads must be between 1 and 1024");
        Ok(())
    }
}

#[derive(Clone, Debug, serde::Serialize)]
struct TunePrompt {
    name: String,
    rendered_sha256: String,
    token_ids_sha256: String,
    prompt_tokens: usize,
    #[serde(skip)]
    ids: Vec<u32>,
}

fn token_digest(ids: &[u32]) -> String {
    let mut hash = Sha256::new();
    for id in ids { hash.update(id.to_le_bytes()); }
    format!("{:x}", hash.finalize())
}

fn tuning_prompts(g: &Gguf, tok: &Tokenizer, options: &TuneOptions, ctx: usize) -> anyhow::Result<Vec<TunePrompt>> {
    let texts = if let Some(path) = &options.prompt_file {
        let file = std::fs::File::open(path).with_context(|| format!("open tuning prompt {}", path.display()))?;
        let mut text = String::new();
        file.take(1024 * 1024 + 1).read_to_string(&mut text)?;
        anyhow::ensure!(!text.trim().is_empty() && text.len() <= 1024 * 1024, "tuning prompt must contain 1 byte to 1 MiB of UTF-8 text");
        vec![("custom".to_string(), text)]
    } else {
        vec![
            ("short_explanation".into(), "Explain how a CPU and GPU cooperate to run a mixture-of-experts language model. Give a detailed answer with examples covering expert selection, memory bandwidth, caching and the difference between latency and throughput. Aim for at least 250 words.".into()),
            ("medium_analysis".into(), "A software team runs a service with these observations: median response time is 180 ms, p95 is 2.4 seconds and p99 is 8 seconds. CPU use averages 45 percent, memory use is stable, and database query duration rises sharply during bursts. Requests pass through an API gateway, a worker queue and a connection pool of 24 database connections. A recent change added two independent reads to each request. Retries use a fixed one-second delay. Traffic doubled but the database instance stayed the same. The service must preserve transaction consistency and cannot drop customer work. The team proposes doubling worker concurrency, increasing the connection pool to 200, and caching every response for five minutes. Analyse the proposal carefully. Distinguish what the measurements establish from hypotheses, explain which additional measurements to collect, and give a staged plan with rollback criteria. Discuss tail latency, queueing, retry amplification and cache correctness. Write a detailed technical answer of at least 300 words.".into()),
            ("rust_code".into(), "Write a robust Rust implementation of a bounded in-memory work queue for a single producer and multiple worker threads using only the standard library. Jobs are boxed functions. The producer must receive a clear result when the queue is full or shut down. Workers must sleep while idle, and shutdown must wake them and let them drain accepted work. Explain ownership, panic handling, lock scope and why no accepted job is executed twice. Include the complete implementation followed by a short usage example and an explanation of the concurrency invariants.".into()),
        ]
    };
    let tpl = template(g);
    texts.into_iter().map(|(name, text)| {
        // Use the same chat-template route as a normal request. Thinking remains
        // enabled; both reasoning and answer tokens count as generated output.
        let rendered = tpl.render(&[Msg::new("user", &text)], &[], true, true, true);
        let ids = tok.encode(&rendered)?;
        anyhow::ensure!(!ids.is_empty(), "tuning prompt {name} tokenized to an empty sequence");
        anyhow::ensure!(ids.len().checked_add(options.tg).is_some_and(|n| n <= ctx),
            "tuning prompt {name} has {} tokens plus {} requested output tokens, exceeding context {ctx}; increase context or shorten the prompt/output", ids.len(), options.tg);
        Ok(TunePrompt { name, rendered_sha256: format!("{:x}", Sha256::digest(rendered.as_bytes())),
            token_ids_sha256: token_digest(&ids), prompt_tokens: ids.len(), ids })
    }).collect()
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
struct ReferenceOutput {
    token_ids: Vec<u32>,
    stopped: bool,
}

#[derive(Clone, Debug, serde::Serialize)]
struct DecodeSample {
    prompt: usize,
    repetition: usize,
    phase: &'static str,
    prompt_tokens: usize,
    completion_tokens: usize,
    prefill_seconds: f64,
    decode_seconds: f64,
    decode_tokens_per_second: f64,
    token_ids: Vec<u32>,
    stopped: bool,
    matches_reference: Option<bool>,
}

#[derive(Clone, Debug, serde::Serialize)]
struct TuneCandidate {
    chunk: usize,
    mtp: usize,
    eligible: bool,
    reason: Option<String>,
    median_decode_tokens_per_second: Option<f64>,
    median_prefill_tokens_per_second: Option<f64>,
    measured_round_decode_tokens_per_second: Vec<f64>,
    samples: Vec<DecodeSample>,
}

impl TuneCandidate {
    fn rejected(chunk: usize, mtp: usize, reason: String) -> Self {
        Self { chunk, mtp, eligible: false, reason: Some(reason), median_decode_tokens_per_second: None,
            median_prefill_tokens_per_second: None, measured_round_decode_tokens_per_second: vec![], samples: vec![] }
    }
}

fn median(values: &[f64]) -> anyhow::Result<f64> {
    anyhow::ensure!(!values.is_empty() && values.iter().all(|n| n.is_finite() && *n >= 0.0), "timing series is empty or non-finite");
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let i = values.len() / 2;
    Ok(if values.len() % 2 == 0 { values[i - 1] / 2.0 + values[i] / 2.0 } else { values[i] })
}

/// Pair each round's actual token count with exactly that round's elapsed
/// time. Warmup samples never contribute, and missing/duplicate cases fail.
fn tuning_round_rates(samples: &[DecodeSample], prompts: usize, reps: usize) -> anyhow::Result<(Vec<f64>, Vec<f64>)> {
    let mut decode = Vec::with_capacity(reps);
    let mut prefill = Vec::with_capacity(reps);
    for repetition in 0..reps {
        let rows: Vec<_> = samples.iter().filter(|s| s.phase == "measured" && s.repetition == repetition).collect();
        anyhow::ensure!(rows.len() == prompts, "incomplete tuning repetition {repetition}");
        for prompt in 0..prompts {
            anyhow::ensure!(rows.iter().filter(|s| s.prompt == prompt).count() == 1, "duplicate or missing tuning prompt {prompt}");
        }
        anyhow::ensure!(rows.iter().all(|s| s.prefill_seconds.is_finite() && s.prefill_seconds > 0.0
            && s.decode_seconds.is_finite() && s.decode_seconds > 0.0), "invalid tuning duration");
        let generated: usize = rows.iter().map(|s| s.completion_tokens).sum();
        let input: usize = rows.iter().map(|s| s.prompt_tokens).sum();
        let decode_seconds: f64 = rows.iter().map(|s| s.decode_seconds).sum();
        let prefill_seconds: f64 = rows.iter().map(|s| s.prefill_seconds).sum();
        decode.push(generated as f64 / decode_seconds);
        prefill.push(input as f64 / prefill_seconds);
    }
    Ok((decode, prefill))
}

fn decode_sample(r: &mut Runner, tok: &Tokenizer, prompt: &TunePrompt, prompt_index: usize,
    repetition: usize, phase: &'static str, mtp: usize, tg: usize) -> anyhow::Result<DecodeSample> {
    r.reset()?;
    let started = Instant::now();
    let logits = r.prefill(&prompt.ids)?;
    let prefill_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let (token_ids, stopped) = decode(r, tok, mtp > 0, &mut Sampler::greedy(), logits, tg, &mut |_| Ok(false))?;
    let decode_seconds = started.elapsed().as_secs_f64();
    anyhow::ensure!(decode_seconds.is_finite() && decode_seconds > 0.0 && prefill_seconds > 0.0, "non-positive tuning duration");
    Ok(DecodeSample { prompt: prompt_index, repetition, phase, prompt_tokens: prompt.ids.len(),
        completion_tokens: token_ids.len(), prefill_seconds, decode_seconds,
        decode_tokens_per_second: token_ids.len() as f64 / decode_seconds,
        token_ids, stopped, matches_reference: None })
}

fn qualify_tune_outputs(samples: &mut [DecodeSample], references: &[ReferenceOutput]) -> anyhow::Result<()> {
    let mut first_failure = None;
    let mut measured = 0;
    for sample in samples.iter_mut().filter(|s| s.phase == "measured") {
        measured += 1;
        let matches = references.get(sample.prompt).is_some_and(|reference|
            sample.token_ids == reference.token_ids && sample.stopped == reference.stopped);
        sample.matches_reference = Some(matches);
        if !matches {
            first_failure.get_or_insert_with(|| format!("greedy output differs from the reference on prompt {} repetition {}", sample.prompt, sample.repetition));
        }
        if sample.completion_tokens != sample.token_ids.len() || sample.completion_tokens < 32 {
            first_failure.get_or_insert_with(|| format!("prompt {} returned only {} counted tokens or inconsistent token accounting; at least 32 are required", sample.prompt, sample.completion_tokens));
        }
    }
    anyhow::ensure!(measured > 0, "no measured outputs to compare");
    if let Some(reason) = first_failure { anyhow::bail!(reason); }
    Ok(())
}

fn measure_tune_candidate(mut r: Runner, tok: &Tokenizer, prompts: &[TunePrompt], options: &TuneOptions,
    chunk: usize, mtp: usize, reference: Option<&[ReferenceOutput]>) -> anyhow::Result<(TuneCandidate, Vec<ReferenceOutput>)> {
    anyhow::ensure!(reference.is_none_or(|r| r.len() == prompts.len()), "tuning reference prompt count differs");
    let mut expected: Vec<Option<ReferenceOutput>> = reference.map_or_else(|| vec![None; prompts.len()], |r| r.iter().cloned().map(Some).collect());
    let mut result = TuneCandidate { chunk, mtp, eligible: true, reason: None,
        median_decode_tokens_per_second: None, median_prefill_tokens_per_second: None,
        measured_round_decode_tokens_per_second: vec![], samples: vec![] };
    // Every fresh allocation starts from the same saved routing profile. Let its
    // normal adaptation run during both warmup and measurement; do not write a
    // routing profile here, which would change later candidates' starting seed.
    for repetition in 0..options.warmup {
        for (index, prompt) in prompts.iter().enumerate() {
            result.samples.push(decode_sample(&mut r, tok, prompt, index, repetition, "warmup", mtp, options.tg)?);
        }
    }
    for repetition in 0..options.reps {
        for (index, prompt) in prompts.iter().enumerate() {
            let sample = decode_sample(&mut r, tok, prompt, index, repetition, "measured", mtp, options.tg)?;
            let actual = ReferenceOutput { token_ids: sample.token_ids.clone(), stopped: sample.stopped };
            expected[index].get_or_insert(actual);
            result.samples.push(sample);
        }
    }
    let expected = expected.into_iter().collect::<Option<Vec<_>>>().context("tuning reference is incomplete")?;
    if let Err(error) = qualify_tune_outputs(&mut result.samples, &expected) {
        result.eligible = false;
        result.reason = Some(error.to_string());
    }
    let (decode_rates, prefill_rates) = tuning_round_rates(&result.samples, prompts.len(), options.reps)?;
    result.median_decode_tokens_per_second = Some(median(&decode_rates)?);
    result.median_prefill_tokens_per_second = Some(median(&prefill_rates)?);
    result.measured_round_decode_tokens_per_second = decode_rates;
    Ok((result, expected))
}

/// The baseline wins close results. Fidelity failures can never win even when
/// their timing is faster, and the threshold is against the common baseline.
fn select_tune_candidate(candidates: &[TuneCandidate]) -> Option<usize> {
    let baseline = candidates.first()?;
    if !baseline.eligible { return None; }
    let base = baseline.median_decode_tokens_per_second.filter(|v| v.is_finite() && *v > 0.0)?;
    let mut selected = 0;
    let mut best = base;
    for (i, candidate) in candidates.iter().enumerate().skip(1) {
        let Some(rate) = candidate.median_decode_tokens_per_second.filter(|v| v.is_finite() && *v > 0.0) else { continue };
        if candidate.eligible && rate >= base * 1.03 && rate > best { selected = i; best = rate; }
    }
    Some(selected)
}

fn routing_seed(model: &Model) -> anyhow::Result<Option<String>> {
    let identities = std::iter::once(model.w.g.identity.as_str()).chain(model.w.aux_g.map(|g| g.identity.as_str()));
    let identity = rusting_strata::cache_identity::combine_identities("expert-profile-base-and-companion-v1", identities);
    let path = rusting_strata::expert_cache::cache_dir().join(format!("profile-v2-{identity}.f32"));
    let file = match std::fs::File::open(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).with_context(|| format!("read routing seed {}", path.display())),
    };
    anyhow::ensure!(file.metadata()?.len() <= 64 * 1024 * 1024, "routing profile is unexpectedly large");
    let mut bytes = Vec::new();
    file.take(64 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
    anyhow::ensure!(bytes.len() <= 64 * 1024 * 1024, "routing profile grew beyond the size limit");
    Ok(Some(format!("{:x}", Sha256::digest(&bytes))))
}

fn write_tune_report(path: &Path, report: &serde_json::Value) -> anyhow::Result<()> {
    let temporary = path.with_extension(format!("json.{}.tmp", std::process::id()));
    std::fs::write(&temporary, serde_json::to_string_pretty(report)?)?;
    std::fs::rename(&temporary, path).with_context(|| format!("write tuning report {}", path.display()))
}

fn calibrate(model: &Path, ctx: usize, margin: &[usize], kind: Option<&str>, device: &[usize], kv: &str,
    options: &TuneOptions) -> anyhow::Result<()> {
    options.validate(ctx)?;
    anyhow::ensure!(matches!(kv, "q8" | "f16"), "calibration KV format must be q8 or f16");
    anyhow::ensure!(rusting_strata::expert_tail::Policy::from_env()?.min_rank == 0,
        "accuracy-preserving tuning requires expert-tail omission disabled");
    unsafe {
        if kv == "q8" { std::env::set_var("STRATA_KV_Q8", "1"); }
        else { std::env::remove_var("STRATA_KV_Q8"); }
    }
    let desired = options.threads.or_else(physical_cores)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    let _ = rayon::ThreadPoolBuilder::new().num_threads(desired).build_global();
    let threads = rayon::current_num_threads();
    anyhow::ensure!(options.threads.is_none_or(|n| n == threads),
        "Rayon already has {threads} workers; compare a different --threads setting in a fresh process");
    let g = Gguf::open(model)?;
    let m = Model::load(&g)?;
    validate_context_length(ctx, model_context_length(&g))?;
    let tok = Tokenizer::from_gguf(&g)?;
    let prompts = tuning_prompts(&g, &tok, options, ctx)?;
    // Freeze the same startup budget for all candidates. Each allocation below clears prior residency, so a
    // later candidate cannot inherit the baseline's RAM working set or silently measure the old mmap policy.
    let cache_ram = prompt_cache_cap(m.ram_estimate() as u64, options.cache_ram,
        memory::ram_cap((options.ram_margin as u64) << 20, (options.max_ram as u64) << 20))?;
    let ram_plan = setup_expert_ram(&m, options.ram_margin, options.max_ram, cache_ram, &options.expert_ram)?;
    let ram_budget = usize::try_from(ram_plan.budget_bytes).context("expert RAM budget exceeds address space")?;
    let max_mtp = if rusting_strata::compatibility::validate_mtp_geometry(&m, None).is_ok() {
        calibration_mtp_budget(m.cfg.n_nextn, m.cfg.hc_n, device.len())
    } else { 0 };
    let seed = routing_seed(&m)?;
    let load = |chunk: usize, drafts: usize| -> anyhow::Result<GpuModel> {
        anyhow::ensure!(routing_seed(&m)? == seed, "routing profile changed while tuning; stop other processes using this checkpoint and repeat");
        m.w.configure_expert_ram(0)?;
        m.w.configure_expert_ram(ram_budget)?;
        let bk = device.iter().map(|&d| backend::open(kind, d)).collect::<anyhow::Result<Vec<_>>>()?;
        let mut gm = GpuModel::load_split_margins(&m, bk, ctx, chunk, margin, drafts, 1).context(LOAD_HINT)?;
        gm.set_decode_only(true);
        gm.short = 32;
        gm.cache_ram = cache_ram << 20;
        warm_expert_ram(&m, options.expert_warmup, drafts > 0)?;
        anyhow::ensure!(routing_seed(&m)? == seed, "routing profile changed during candidate allocation");
        Ok(gm)
    };
    let baseline_chunk = 256.min(ctx);
    let gm = load(baseline_chunk, 0)?;
    let context = calibration_context(&m, ctx, kv,
        calibration_devices(device, gm.parts().map(|part| &*part.gpu))?, margin.to_vec(), 1, threads,
        cache_ram, options.expert_warmup)?;
    eprintln!("tuning decode with {} chat workloads, {} excluded warmup(s) and {} measured repetitions per workload; fixed {kv} KV, one slot, full expert computation", prompts.len(), options.warmup, options.reps);
    let (baseline, references) = measure_tune_candidate(Runner::Gpu(Box::new(gm)), &tok, &prompts, options, baseline_chunk, 0, None)?;
    let mut candidates = vec![baseline];
    eprintln!("baseline chunk {baseline_chunk}, MTP 0: {}", if candidates[0].eligible {
        format!("{:.2} median decode tok/s; repeated outputs stable", candidates[0].median_decode_tokens_per_second.unwrap())
    } else { candidates[0].reason.clone().unwrap_or_else(|| "invalid reference".into()) });
    if candidates[0].eligible {
        for chunk in [64, 128, 256, 512, 1024].into_iter().filter(|chunk| *chunk <= ctx) {
            for drafts in 0..=max_mtp {
                if chunk == baseline_chunk && drafts == 0 { continue; }
                let measured = load(chunk, drafts).and_then(|gm| measure_tune_candidate(Runner::Gpu(Box::new(gm)), &tok, &prompts, options, chunk, drafts, Some(&references)));
                let candidate = match measured {
                    Ok((candidate, _)) => candidate,
                    Err(error) => TuneCandidate::rejected(chunk, drafts, format!("candidate could not complete: {error:#}")),
                };
                eprintln!("chunk {chunk:4}, MTP {drafts}: {}", if candidate.eligible {
                    format!("{:.2} median decode tok/s; outputs match reference", candidate.median_decode_tokens_per_second.unwrap())
                } else { format!("excluded: {}", candidate.reason.as_deref().unwrap_or("invalid measurement")) });
                candidates.push(candidate);
            }
        }
    }
    anyhow::ensure!(routing_seed(&m)? == seed, "routing profile changed while tuning; no recommendation saved");
    let selected = select_tune_candidate(&candidates);
    let mut report = json!({"schema":CALIBRATION_SCHEMA,"configuration":context,
        "objective":"median_decode_tokens_per_second","minimum_improvement_fraction":0.03,
        "status":if selected.is_some(){"measured"}else{"unstable_or_insufficient_reference"},
        "workloads":prompts,"baseline_outputs":references,"candidates":candidates,
        "warmup_repetitions_per_workload_excluded":options.warmup,"measured_repetitions_per_workload":options.reps,
        "requested_generation_tokens":options.tg,"chat_template_enable_thinking":true,
        "expert_ram":{"requested":options.expert_ram,"startup_plan":ram_plan,
            "warmup_enabled":options.expert_warmup,"fresh_residency_per_candidate":true,
            "policy":EXPERT_RAM_POLICY},
        "sampling":{"temperature":0.0,"top_k":0,"top_p":1.0},
        "routing":{"initial_profile_sha256":seed,"fresh_allocation_per_candidate":true,
            "adaptive_placement_enabled":std::env::var_os("STRATA_FIXED_EXPERTS").is_none()
                && std::env::var("STRATA_ADAPT_EVERY").ok().and_then(|n|n.parse::<usize>().ok()).unwrap_or(16)>0,
            "adapt_every":std::env::var("STRATA_ADAPT_EVERY").ok().and_then(|n|n.parse::<usize>().ok()).unwrap_or(16),
            "adapt_swaps":std::env::var("STRATA_ADAPT_SWAPS").ok().and_then(|n|n.parse::<usize>().ok()).unwrap_or(16),
            "profile_written_during_tuning":false,"policy":DECODE_CACHE_POLICY},
        "fidelity":{"selected_candidate_all_measured_outputs_match_baseline":selected.is_some(),
            "scope":"selected candidate: exact greedy token IDs and stopping behaviour on every measured workload/repetition only; not a factual accuracy benchmark or universal numerical equivalence guarantee"},
        "rate_definition":"median across measured rounds of actual generated tokens divided by summed decode wall time; prefill and warmup excluded; reasoning tokens included",
        "ctx":ctx,"kv":kv,"threads":threads,"short":32,
        "notes":["Model weights, KV format, context and memory margins were held fixed.",
            "Only chunk and supported embedded MTP choices were compared; this is not a global hardware optimum.",
            "The baseline is retained unless an eligible candidate improves median decode speed by at least 3 percent."]});
    let Some(selected) = selected else {
        // Retain the failed measurement and invalidate any earlier v5 result
        // for this configuration rather than leaving a stale success active.
        let path = calibration_path(&context);
        std::fs::create_dir_all(path.parent().unwrap())?;
        write_tune_report(&path, &report)?;
        if let Some(path) = &options.json_out { write_tune_report(path, &report)?; }
        anyhow::bail!("baseline outputs were unstable or too short to measure; no usable calibration was saved. Keep --chunk {baseline_chunk} --mtp 0 and inspect {} or use a longer representative prompt", path.display());
    };
    let chosen = &candidates[selected];
    report["selected_candidate"] = json!(selected);
    report["chunk"] = json!(chosen.chunk);
    report["mtp"] = json!(chosen.mtp);
    report["measured"] = json!({"median_decode_tokens_per_second":chosen.median_decode_tokens_per_second,
        "median_prefill_tokens_per_second":chosen.median_prefill_tokens_per_second,
        "baseline_median_decode_tokens_per_second":candidates[0].median_decode_tokens_per_second});
    let path = calibration_path(&context);
    std::fs::create_dir_all(path.parent().unwrap())?;
    write_tune_report(&path, &report)?;
    if let Some(output) = &options.json_out { if *output != path { write_tune_report(output, &report)?; } }
    println!("Selected chunk {}, MTP {}: {:.2} median decode tok/s; measured outputs match the baseline. Full report: {}",
        chosen.chunk, chosen.mtp, chosen.median_decode_tokens_per_second.unwrap(), path.display());
    println!("Recommended flags: --ctx {ctx} --kv {kv} --chunk {} --mtp {} --threads {threads}", chosen.chunk, chosen.mtp);
    println!("Keep --expert-ram {} --expert-warmup {} --ram-margin {} --max-ram {} --cache-ram {}; measured expert ceiling {:.2} GiB.",
        options.expert_ram, options.expert_warmup, options.ram_margin, options.max_ram, cache_ram,
        ram_budget as f64 / 1073741824.0);
    if margin.len() == 1 { println!("Keep --vram-margin {} --parallel 1, the same model/device and expert omission disabled.", margin[0] >> 20); }
    Ok(())

}

#[cfg(test)]
mod decode_tuning_tests {
    use super::*;

    fn sample(prompt: usize, repetition: usize, tokens: usize, seconds: f64) -> DecodeSample {
        DecodeSample { prompt, repetition, phase: "measured", prompt_tokens: 128,
            completion_tokens: tokens, prefill_seconds: 0.5, decode_seconds: seconds,
            decode_tokens_per_second: tokens as f64 / seconds, token_ids: vec![65; tokens],
            stopped: false, matches_reference: Some(true) }
    }

    #[test]
    fn timing_pairs_actual_counts_with_their_round_and_excludes_warmup() {
        let mut samples = vec![];
        for (repetition, scale) in [1.0, 2.0, 0.5].into_iter().enumerate() {
            // A mean of the two per-prompt rates would be wrong: use the
            // complete round's 96 actual tokens divided by its paired 4s.
            samples.push(sample(0, repetition, 32, scale));
            samples.push(sample(1, repetition, 64, 3.0 * scale));
        }
        let mut warmup = sample(0, 0, 100_000, 0.001);
        warmup.phase = "warmup";
        samples.push(warmup);
        let (decode, prefill) = tuning_round_rates(&samples, 2, 3).unwrap();
        assert_eq!(decode, vec![24.0, 12.0, 48.0]);
        assert_eq!(median(&decode).unwrap(), 24.0);
        assert_eq!(prefill, vec![256.0; 3]);
    }

    #[test]
    fn incomplete_duplicate_and_nonfinite_measurements_are_rejected() {
        assert!(tuning_round_rates(&[sample(0, 0, 32, 1.0)], 2, 1).is_err());
        assert!(tuning_round_rates(&[sample(0, 0, 32, 1.0), sample(0, 0, 32, 1.0)], 2, 1).is_err());
        assert!(tuning_round_rates(&[sample(0, 0, 32, f64::NAN)], 1, 1).is_err());
        assert!(tuning_round_rates(&[sample(0, 0, 32, 0.0)], 1, 1).is_err());
        assert!(median(&[]).is_err());
        assert!(median(&[1.0, f64::INFINITY]).is_err());
    }

    fn candidate(rate: f64, eligible: bool) -> TuneCandidate {
        TuneCandidate { chunk: 256, mtp: 0, eligible, reason: (!eligible).then(|| "output mismatch".into()),
            median_decode_tokens_per_second: Some(rate), median_prefill_tokens_per_second: Some(1.0),
            measured_round_decode_tokens_per_second: vec![rate; 3], samples: vec![] }
    }

    #[test]
    fn fidelity_failures_never_win_and_small_gains_keep_the_baseline() {
        assert_eq!(select_tune_candidate(&[candidate(100.0, true), candidate(1000.0, false), candidate(102.9, true)]), Some(0));
        assert_eq!(select_tune_candidate(&[candidate(100.0, true), candidate(103.5, true), candidate(108.0, true)]), Some(2));
        assert_eq!(select_tune_candidate(&[candidate(100.0, false), candidate(1000.0, true)]), None);
        assert_eq!(select_tune_candidate(&[candidate(0.0, true)]), None);
        assert_eq!(select_tune_candidate(&[candidate(100.0, true), candidate(f64::NAN, true)]), Some(0));
    }

    #[test]
    fn tuning_requires_warmup_repeated_measurements_and_bounded_output() {
        assert!(TuneOptions::default().validate(4096).is_ok());
        for options in [TuneOptions { warmup: 0, ..Default::default() },
            TuneOptions { reps: 2, ..Default::default() }, TuneOptions { tg: 0, ..Default::default() },
            TuneOptions { tg: 1025, ..Default::default() }, TuneOptions { threads: Some(0), ..Default::default() },
            TuneOptions { expert_ram: "all".into(), ..Default::default() },
            TuneOptions { ram_margin: (1 << 20) + 1, ..Default::default() }] {
            assert!(options.validate(4096).is_err());
        }
        assert!(TuneOptions::default().validate(128).is_err());
    }

    #[test]
    fn output_gate_checks_every_measured_row_complete_ids_and_stopping() {
        let references = vec![ReferenceOutput { token_ids: vec![65; 32], stopped: false },
            ReferenceOutput { token_ids: vec![66; 32], stopped: false }];
        let mut samples = vec![];
        for repetition in 0..2 {
            samples.push(sample(0, repetition, 32, 1.0));
            let mut second = sample(1, repetition, 32, 1.0);
            second.token_ids.fill(66);
            samples.push(second);
        }
        let mut warmup = sample(0, 0, 32, 1.0);
        warmup.phase = "warmup";
        warmup.token_ids.fill(99);
        samples.push(warmup);
        assert!(qualify_tune_outputs(&mut samples, &references).is_ok());
        samples[2].token_ids[31] = 66;
        assert!(qualify_tune_outputs(&mut samples, &references).is_err());
        assert_eq!(samples[2].matches_reference, Some(false));
        assert_eq!(samples[3].matches_reference, Some(true));
        samples[2].token_ids[31] = 65;
        samples[2].stopped = true;
        assert!(qualify_tune_outputs(&mut samples, &references).is_err());
        samples[2].stopped = false;
        samples[2].completion_tokens = 31;
        assert!(qualify_tune_outputs(&mut samples, &references).is_err());
    }
}

/// Decoding of up to `n` tokens after `logits`; `emit` gets each new complete piece of text and returns true to stop.
/// Returns the tokens and whether an end-of-turn token or `emit` stopped it.
fn decode(r: &mut Runner, tok: &Tokenizer, mtp: bool, smp: &mut Sampler, mut logits: Vec<f32>, n: usize,
          emit: &mut dyn FnMut(&str) -> anyhow::Result<bool>) -> anyhow::Result<(Vec<u32>, bool)> {
    let eos = tok.eos_ids();
    // `ahead`: the next token when `spec_step` or `step_greedy` already picked it.
    // `start`: the first token whose text is not emitted yet (the text before it ends on a whole character).
    let (mut out, mut start, mut row, mut ahead, mut stopped) = (Vec::new(), 0, 0, None, false);
    'outer: while out.len() < n {
        let next = match ahead.take() { Some(t) => t, None => smp.pick(&logits)? as u32 };
        if eos.contains(&next) { stopped = true; break; }
        let toks = match r {
            Runner::Gpu(g) if mtp && g.ngram.is_some() => {
                let (toks, nx, rw) = g.hybrid_spec_step(next, row, smp)?;
                (ahead, row) = (Some(nx), rw);
                toks
            }
            Runner::Gpu(g) if mtp => {
                let (toks, nx, rw) = g.spec_step(next, row, smp)?;
                (ahead, row) = (Some(nx), rw);
                toks
            }
            Runner::Gpu(g) if g.ngram.is_some() => {
                let (toks, nx) = g.ngram_spec_step(next, smp)?;
                ahead = Some(nx);
                toks
            }
            // Greedy without MTP: the argmax on the GPU, so only the token id is copied back, not the logits.
            Runner::Gpu(g) if smp.temp <= 0.0 => { ahead = Some(g.step_greedy(next)?); vec![next] }
            Runner::Slot(c) if smp.temp <= 0.0 => { ahead = Some(c.step(next, false)?.0); vec![next] }
            _ => { logits = r.step(next)?; vec![next] }
        };
        for t in toks {
            if eos.contains(&t) { stopped = true; break 'outer; }
            if out.len() == n { break; }
            out.push(t);
            let s = tok.decode(&out[start..])?;
            if !s.ends_with('\u{fffd}') {
                if emit(&s)? { stopped = true; break 'outer; }
                start = out.len();
            }
        }
    }
    if let Runner::Gpu(g) = r && let Some(ng) = g.ngram.as_mut() {
        let _ = ng.save_dynamic();
    }
    Ok((out, stopped))
}

/// Forge's public context ceiling is shared by CLI execution and model-management requests. Metadata is an
/// additional ceiling, never a promise that the model and this context will fit the selected GPU.
pub const CONTEXT_MIN: usize = 128;
pub const CONTEXT_MAX: usize = 131072;

fn model_context_length(g: &Gguf) -> Option<u64> {
    let arch = g.meta.get("general.architecture")?.as_str()?;
    g.meta.get(&format!("{arch}.context_length"))?.as_u64().filter(|n| *n > 0)
}

fn validate_context_length(ctx: usize, trained: Option<u64>) -> anyhow::Result<()> {
    anyhow::ensure!((CONTEXT_MIN..=CONTEXT_MAX).contains(&ctx),
        "--ctx must be {CONTEXT_MIN}..={CONTEXT_MAX} tokens, including prompt and output; got {ctx}");
    if let Some(trained) = trained {
        anyhow::ensure!(ctx as u64 <= trained,
            "--ctx {ctx} exceeds this model's training context {trained}; select a context no larger than {}. Context extension is not enabled by this engine.", trained.min(CONTEXT_MAX as u64));
    }
    Ok(())
}

/// Validate context against both Forge's range and the selected checkpoint before allocating GPU state.
pub fn validate_model_context(path: &Path, ctx: usize) -> anyhow::Result<()> {
    validate_context_length(ctx, None)?;
    validate_context_length(ctx, model_context_length(&Gguf::open(path)?))
}

/// The chat format from the GGUF's `tokenizer.chat_template`.
fn template(g: &Gguf) -> Template {
    let t = Template::detect(g.meta.get("tokenizer.chat_template").and_then(|v| v.as_str()));
    eprintln!("chat template: {t:?}");
    t
}

/// Reply text without the thinking part (the template drops it from history too).
/// Streams one (reasoning, answer) pair as `reasoning_content` and `content` deltas.
fn send_split(w: &mut impl Write, chunk: &impl Fn(serde_json::Value, Option<&str>) -> serde_json::Value, (r, a): (String, String)) -> anyhow::Result<()> {
    if !r.is_empty() { write!(w, "data: {}\n\n", chunk(json!({"reasoning_content": r}), None))?; }
    if !a.is_empty() { write!(w, "data: {}\n\n", chunk(json!({"content": a}), None))?; }
    Ok(())
}

fn answer(s: &str) -> String { s.rsplit_once("</think>").map_or(s, |x| x.1).trim().to_string() }

/// A model `serve` has loaded.
struct Loaded<'a> { r: Runner<'a>, tok: &'a Tokenizer, tpl: Template, name: String, path: PathBuf, identity: String }

/// A model management request. The outer loop acknowledges unload only after
/// the current `Loaded`, model weights and GPU buffers have been dropped.
enum ModelChange { Load(PathBuf, TcpStream), Unload(TcpStream) }

fn acknowledge_unload(mut stream: TcpStream) -> std::io::Result<()> {
    let body = r#"{"unloaded":true,"loaded":null}"#;
    write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
}

/// Capacity metadata is available while inference or loading is busy. No prompts or credentials are exposed.
fn serving_status(stats: &std::sync::Mutex<serve::Stats>, admission: &Admission,
    policy: &std::sync::Mutex<serde_json::Value>, max_tokens: Option<usize>) -> serde_json::Value {
    let state = stats.lock().unwrap().clone();
    json!({
        "object": "forge.server.status",
        "status": if state.loading { "loading" } else if state.model.is_some() { "ready" } else { "no_model" },
        "model": state.model, "backend": state.backend,
        "context_tokens_per_request": state.ctx,
        "max_output_tokens": max_tokens,
        "generating_requests": state.busy,
        "concurrency": policy.lock().unwrap().clone(),
        "admission": admission.snapshot(),
        "api": {
            "models": "/v1/models", "chat_completions": "/v1/chat/completions",
            "messages": "/v1/messages", "status": "/v1/status", "context_preview": "/api/context",
            "streaming": true, "function_tools": true,
            "tool_execution": "client", "responses_api": false,
            "queue_persistence": "connection_bound_memory"
        }
    })
}

fn admission_error(mut stream: &TcpStream, id: usize, error: AdmissionError) -> std::io::Result<()> {
    let (status, code, message) = match error {
        AdmissionError::Full => ("429 Too Many Requests", "queue_full", "The generation queue is full. Retry after capacity is available."),
        AdmissionError::Timeout => ("503 Service Unavailable", "queue_timeout", "The request exceeded the queue wait limit and was not started. Retry later or increase --queue-timeout."),
        AdmissionError::Cancelled => ("503 Service Unavailable", "request_cancelled", "The queued request was cancelled before execution."),
    };
    let body = json!({"error":{"message":message,"type":"overloaded_error","code":code},"request_id":format!("forge-{id}")}).to_string();
    write!(stream, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nCache-Control: no-store\r\nRetry-After: 2\r\nX-Request-ID: forge-{id}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
}

fn admission_metrics(admission: &Admission) -> String {
    let value = json!(admission.snapshot());
    let mut out = String::new();
    for (key, name, kind, help) in [
        ("active", "active_requests", "gauge", "Requests admitted for execution, including prompt preparation."),
        ("reserved", "reserved_slots", "gauge", "Slots reserved for waiting threads but not yet claimed."),
        ("queued", "queued_requests", "gauge", "Generation requests waiting for an execution slot."),
        ("capacity", "concurrency_capacity", "gauge", "Current execution slot capacity; zero while unavailable."),
        ("queue_capacity", "queue_capacity", "gauge", "Configured maximum additional waiting generation requests."),
        ("rejected_full_total", "queue_rejected_total", "counter", "Generation requests rejected because the queue was full."),
        ("timed_out_total", "queue_timed_out_total", "counter", "Requests expired before execution."),
        ("cancelled_total", "queue_cancelled_total", "counter", "Queued or reserved requests cancelled before execution."),
    ] {
        out.push_str(&format!("# HELP forge_{name} {help}\n# TYPE forge_{name} {kind}\nforge_{name} {}\n", value[key]));
    }
    out
}

/// What `serve` keeps across model loads.
struct Serving {
    /// Complete requests, with their key's place when `--max-per-key` counts it (freed once answered).
    rx: std::sync::Mutex<std::sync::mpsc::Receiver<(TcpStream, anyhow::Result<serve::Request>, usize, Option<serve::Hold>, Option<Permit>)>>,
    waiting: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// Set by SIGTERM.
    stop: &'static std::sync::atomic::AtomicBool,
    addr: String,
    allowed_hosts: Vec<String>,
    keys: std::sync::Arc<std::sync::RwLock<serve::Keys>>,
    usage: Option<std::sync::Mutex<std::fs::File>>,
    max_tokens: usize,
    dir: PathBuf,
    /// The --model, offered beside the models folder.
    model: Option<PathBuf>,
    /// The Hugging Face download running on its own thread, or the last one.
    dl: std::sync::Arc<std::sync::Mutex<hf::Progress>>,
    /// At most one remote metadata inspection; it never runs on the decode loop.
    hf_busy: std::sync::Arc<std::sync::atomic::AtomicBool>,
    stats: std::sync::Arc<std::sync::Mutex<serve::Stats>>,
    admission: Admission,
    policy: std::sync::Arc<std::sync::Mutex<serde_json::Value>>,
    max_waiting: usize,
    /// Load options of the last /api/load (`Engine::with`), applied before the next model loads.
    opts: std::sync::Mutex<serde_json::Value>,
}

/// Metadata Range requests can be slow; keep them outside every decode scheduler.
/// One outstanding job bounds resources independently of the chat queue.
fn spawn_hf_metadata(s: &Serving, mut stream: TcpStream,
    job: impl FnOnce() -> anyhow::Result<serde_json::Value> + Send + 'static) -> anyhow::Result<()> {
    fn reply(stream: &mut TcpStream, status: &str, value: serde_json::Value) -> std::io::Result<()> {
        let body=value.to_string();
        let retry=if status.starts_with("429") { "Retry-After: 1\r\n" } else { "" };
        write!(stream,"HTTP/1.1 {status}\r\n{retry}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len())
    }
    if s.hf_busy.compare_exchange(false,true,std::sync::atomic::Ordering::AcqRel,Relaxed).is_err() {
        reply(&mut stream,"429 Too Many Requests",json!({"error":{"message":"a Hugging Face metadata inspection is already running; retry when it finishes","type":"rate_limit_error"}}))?;
        return Ok(());
    }
    struct Busy(std::sync::Arc<std::sync::atomic::AtomicBool>);
    impl Drop for Busy { fn drop(&mut self) { self.0.store(false,std::sync::atomic::Ordering::Release); } }
    let busy=Busy(s.hf_busy.clone());
    std::thread::Builder::new().name("forge9070-hf-metadata".into()).spawn(move || {
        let _busy=busy;
        let (status,value)=match job() {
            Ok(value)=>("200 OK",value),
            Err(error)=>("502 Bad Gateway",json!({"error":{"message":format!("{error:#}"),"type":"server_error"}})),
        };
        let _=reply(&mut stream,status,value);
    })?;
    Ok(())
}

/// A flag that SIGTERM sets (instead of ending the process). Never set on Windows.
fn on_sigterm() -> &'static std::sync::atomic::AtomicBool {
    static STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    #[cfg(unix)]
    {
        extern "C" fn h(_: i32) { STOP.store(true, Relaxed); }
        unsafe extern "C" { fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize; }
        // SAFETY: the handler only stores to an atomic, which is async-signal-safe. 15 is SIGTERM on Linux and macOS.
        unsafe { signal(15, h); }
    }
    &STOP
}

impl Serving {
    /// Answers requests until one asks for a model to be loaded; returns its path and connection (None once the
    /// listener is gone).
    fn serve(&self, mut l: Option<&mut Loaded>, e: &Engine) -> Option<ModelChange> {
        if let Some(l) = l.as_deref_mut() && matches!(&l.r, Runner::Gpu(g) if g.slots() > 1) { return self.serve_slots(l, e); }
        let rx = self.rx.lock().unwrap();
        loop {
            let (c, q, i, _hold, permit) = match rx.recv_timeout(Duration::from_millis(200)) {
                Ok(x) => x,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) if self.stopped() => { eprintln!("stopped"); return None }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return None,
            };
            let mut next = None;
            let queue_wait = permit.as_ref().map_or(Duration::ZERO, Permit::wait_duration);
            let res = handle(c, q, l.as_deref_mut(), e, self, i, queue_wait, &mut next);
            drop(permit);
            self.waiting.fetch_sub(1, Relaxed);
            if let Err(e) = res { eprintln!("request {i}: {e:#}"); serve::exit_if_gpu_dead(&e); }
            if next.is_some() { return next; }
        }
    }
    /// SIGTERM came and the requests accepted before it are answered: `serve` returns None, so the expert profile
    /// is saved and the server exits.
    fn stopped(&self) -> bool { self.stop.load(Relaxed) && self.waiting.load(Relaxed) == 0 }
    /// The models the page offers: the models folder, then --model if it is elsewhere.
    fn models(&self) -> Vec<(String, PathBuf, u64)> {
        let mut v = serve::list_models(&self.dir);
        if let Some(p) = &self.model {
            let real = p.canonicalize().unwrap_or_else(|_| p.clone());
            if !v.iter().any(|x| x.1 == real) {
                v.push((p.display().to_string(), real, std::fs::metadata(p).map_or(0, |m| m.len())));
            }
        }
        v
    }

    fn model_id(&self, path: &Path) -> String {
        let real = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        self.models().into_iter().find(|(_, p, _)| *p == real)
            .map(|(id, _, _)| id).unwrap_or_else(|| path.display().to_string())
    }
}

/// What a request thread asks of the model's thread under `--parallel` (`Serving::serve_slots`).
enum Job {
    /// `prefill_cached` in a free slot; answers the logits, the tokens reused, the slot and when the prefill began.
    /// `gone` is set when the client leaves while the chat waits for a slot; the chat is then dropped.
    Chat { ids: Vec<u32>, sys: usize, keep: usize, owner: u64, gone: std::sync::Arc<std::sync::atomic::AtomicBool>, back: std::sync::mpsc::Sender<anyhow::Result<(Vec<f32>, usize, usize, Instant)>> },
    /// One decode step in a slot the request holds; answers the greedy next token and the logits if `logits`.
    Step { slot: usize, token: u32, logits: bool, back: std::sync::mpsc::Sender<anyhow::Result<(u32, Vec<f32>)>> },
    /// The request is done with its slot.
    Done(usize),
}

/// A request's handle on the model under `--parallel`; dropping it frees the slot.
/// `conn` is the request's connection, watched while the chat waits for a slot; `started` is when the model took
/// the chat, after that wait.
struct Client { tx: std::sync::mpsc::Sender<Job>, slot: Option<usize>, owner: u64, conn: Option<TcpStream>, started: Option<Instant> }

impl Client {
    fn chat(&mut self, ids: &[u32], sys: usize, keep: usize) -> anyhow::Result<(Vec<f32>, usize)> {
        if let Some(s) = self.slot.take() { let _ = self.tx.send(Job::Done(s)); }
        let (back, r) = std::sync::mpsc::channel();
        let gone = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        self.tx.send(Job::Chat { ids: ids.to_vec(), sys, keep, owner: self.owner, gone: gone.clone(), back }).map_err(|_| anyhow::anyhow!("model gone"))?;
        let (logits, reused, s, started) = loop {
            match r.recv_timeout(Duration::from_millis(250)) {
                Ok(x) => break x?,
                // A client that left is flagged, and the wait goes on: the model drops the chat if it is still queued
                // (the channel closes), or sends its slot if the prefill already started, which `handle` then frees.
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => if !gone.load(Relaxed) && self.conn.as_ref().is_some_and(serve::gone) {
                    gone.store(true, Relaxed);
                },
                Err(_) if gone.load(Relaxed) => anyhow::bail!("client closed the connection"),
                Err(_) => anyhow::bail!("model gone"),
            }
        };
        (self.slot, self.started) = (Some(s), Some(started));
        Ok((logits, reused))
    }
    fn step(&mut self, token: u32, logits: bool) -> anyhow::Result<(u32, Vec<f32>)> {
        let slot = self.slot.context("step before chat")?;
        let (back, r) = std::sync::mpsc::channel();
        self.tx.send(Job::Step { slot, token, logits, back }).map_err(|_| anyhow::anyhow!("model gone"))?;
        r.recv()?
    }
}

impl Drop for Client {
    fn drop(&mut self) { if let Some(s) = self.slot { let _ = self.tx.send(Job::Done(s)); } }
}

/// The model thread's view of the slots under `--parallel` (`Serving::serve_slots`).
#[derive(Default)]
struct Sched {
    /// Slots held by a request, and when each was last freed (the least recently used goes first).
    held: Vec<bool>,
    freed: Vec<u64>,
    tick: u64,
    /// The key that holds each slot (`Client::owner`).
    who: Vec<u64>,
    /// Chat jobs waiting for a free slot, and decode steps waiting for a forward (since the first came).
    chats: std::collections::VecDeque<Job>,
    steps: Vec<(usize, u32, bool, std::sync::mpsc::Sender<anyhow::Result<(u32, Vec<f32>)>>)>,
    since: Option<Instant>,
}

impl Sched {
    /// How long a decode step waits for the other decoding slots' steps to share its forward.
    const GATHER: Duration = Duration::from_millis(2);

    fn decoding(&self) -> usize { self.held.iter().filter(|&&h| h).count() }

    /// The queued chat to start next: the one whose key holds the fewest slots, the oldest among those, so that one
    /// customer's burst of requests does not hold back the others'. With one key (or none) it is the oldest.
    fn next_chat(&self) -> Option<usize> {
        let holds = |o: u64| self.held.iter().zip(&self.who).filter(|&(&h, &w)| h && w == o).count();
        (0..self.chats.len()).min_by_key(|&j| (if let Job::Chat { owner, .. } = &self.chats[j] { holds(*owner) } else { usize::MAX }, j))
    }

    /// How long to wait for jobs: until the steps of all decoding slots are in, at most `GATHER` after the first.
    fn wait(&self) -> Duration {
        match self.since {
            Some(t) if self.steps.len() < self.decoding() => Self::GATHER.saturating_sub(t.elapsed()),
            Some(_) => Duration::ZERO,
            None => Duration::from_millis(5),
        }
    }

    /// Take the jobs that come within `wait`.
    fn take(&mut self, jobs: &std::sync::mpsc::Receiver<Job>, wait: Duration) {
        let mut got: Vec<Job> = jobs.recv_timeout(wait).into_iter().collect();
        got.extend(jobs.try_iter());
        for j in got {
            match j {
                Job::Done(s) => { self.held[s] = false; self.tick += 1; self.freed[s] = self.tick; }
                Job::Step { slot, token, logits, back } => { self.since.get_or_insert_with(Instant::now); self.steps.push((slot, token, logits, back)); }
                c => self.chats.push_back(c),
            }
        }
    }

    fn ready(&self) -> bool { self.since.is_some_and(|t| self.steps.len() >= self.decoding() || t.elapsed() >= Self::GATHER) }

    /// One forward over the oldest steps (with more than `STEP_ROWS` slots decoding, each waits its turn), in slot order
    /// (fewer graphs to capture).
    fn run(&mut self, g: &mut GpuModel) {
        // ponytail: fixed row count from which the GEMM forward (`step_batch`) replaces the matvec one; `calibrate`
        // should measure it per model and machine. STRATA_BATCH_MIN overrides.
        let min = std::env::var("STRATA_BATCH_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        let big = g.batch_cap() > 0 && self.steps.len() >= min;
        let cap = if big { g.batch_cap() } else { rusting_strata::gpu_model::STEP_ROWS };
        let mut batch: Vec<_> = self.steps.drain(..self.steps.len().min(cap)).collect();
        batch.sort_by_key(|s| s.0);
        self.since = if self.steps.is_empty() { None } else { Some(Instant::now() - Self::GATHER) };
        let rows: Vec<_> = batch.iter().map(|b| (b.0, b.1, b.2)).collect();
        match if big { g.step_batch(&rows) } else { g.step_rows(&rows) } {
            Ok(rows) => for (b, l) in batch.into_iter().zip(rows) { let _ = b.3.send(Ok(l)); },
            Err(err) => for b in batch { let _ = b.3.send(Err(serve::copy_err(&err))); },
        }
    }
}

impl Serving {
    /// `serve` with several slots (`--parallel`): each request runs `handle` on its own thread and sends its model
    /// work here, where prefills run one at a time (the decoding slots run between its chunks) and the decode steps of
    /// up to `TMAX` slots share a forward.
    fn serve_slots(&self, l: &mut Loaded, e: &Engine) -> Option<ModelChange> {
        use std::sync::mpsc::{TryRecvError, channel};
        let Loaded { r, tok, tpl, name, path, identity } = l;
        let Runner::Gpu(g) = r else { unreachable!() };
        let (tx, jobs) = channel::<Job>();
        let (next, live) = (std::sync::Mutex::new(None), std::sync::atomic::AtomicUsize::new(0));
        let rx = self.rx.lock().unwrap();
        let n = g.slots();
        std::thread::scope(|sc| {
            // Owned here, so that a panic drops the queued jobs' reply channels and the request threads end (the
            // scope then joins them and the panic goes on).
            let (jobs, mut s) = (jobs, Sched { held: vec![false; n], freed: vec![0; n], who: vec![0; n], ..Default::default() });
            let mut open = true;
            loop {
                while open && next.lock().unwrap().is_none() {
                    match rx.try_recv() {
                        Ok((c, q, i, hold, permit)) => {
                            live.fetch_add(1, Relaxed);
                            let (tx, tok, tpl, name, path, identity, next, live) = (tx.clone(), &**tok, *tpl, name.clone(), path.clone(), identity.clone(), &next, &live);
                            sc.spawn(move || {
                                let mut l = Loaded { r: Runner::Slot(Client { tx, slot: None, owner: 0, conn: None, started: None }), tok, tpl, name, path, identity };
                                let mut nx = None;
                                // A panic still gives back the counts below, so the model thread can switch models.
                                let queue_wait = permit.as_ref().map_or(Duration::ZERO, Permit::wait_duration);
                                let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handle(c, q, Some(&mut l), e, self, i, queue_wait, &mut nx)))
                                    .unwrap_or_else(|_| Err(anyhow::anyhow!("panicked")));
                                drop((l, hold, permit));
                                self.waiting.fetch_sub(1, Relaxed);
                                if let Err(e) = res { eprintln!("request {i}: {e:#}"); serve::exit_if_gpu_dead(&e); }
                                if nx.is_some() { *next.lock().unwrap() = nx; }
                                live.fetch_sub(1, Relaxed);
                            });
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => open = false,
                    }
                }
                if open && self.stopped() { eprintln!("stopped"); open = false; }
                if (!open || next.lock().unwrap().is_some()) && live.load(Relaxed) == 0 { break; }
                s.take(&jobs, s.wait());
                // Prefill between decode steps, in the free slot whose saved state the prompt extends the furthest, else
                // the least recently used one.
                s.chats.retain(|j| !matches!(j, Job::Chat { gone, .. } if gone.load(Relaxed)));
                // Models with a batched forward prefill the queued prompts together, one free slot each.
                let lim = g.batch_cap().max(1);
                let fits = |c: &Job| matches!(c, Job::Chat { ids, .. } if ids.len() <= g.ctx());
                let mut adm: Vec<(usize, Job)> = vec![];
                while adm.len() < lim {
                    let Some(j) = s.next_chat() else { break };
                    if !adm.is_empty() && !(fits(&s.chats[j]) && fits(&adm[0].1)) { break; }
                    let Job::Chat { ids, keep, owner, .. } = &s.chats[j] else { break };
                    let Some(sl) = (0..n).filter(|&i| !s.held[i]).max_by_key(|&i| {
                        let (sv, o) = g.slot_saved(i);
                        (if o == *owner && ids[..*keep].starts_with(sv) { sv.len() } else { 0 }, std::cmp::Reverse(s.freed[i]))
                    }) else { break };
                    // Held while the rest are chosen, so that `next_chat` counts it against its key.
                    (s.held[sl], s.who[sl]) = (true, *owner);
                    adm.push((sl, s.chats.remove(j).unwrap()));
                }
                for (sl, _) in &adm { s.held[*sl] = false; }
                if !adm.is_empty() {
                    // Between two prefill forwards the decoding slots run for as long as the forward before took, so
                    // long prompts share the GPU with them evenly instead of stalling them.
                    // ponytail: an even share; weigh it (or cap the prompt's chunk) if first-token time matters more.
                    let (started, mut t0) = (Instant::now(), Instant::now());
                    let res: Vec<anyhow::Result<(Vec<f32>, usize)>> = {
                        let mut between = |g: &mut GpuModel| {
                            let end = Instant::now() + t0.elapsed();
                            while s.decoding() > 0 && Instant::now() < end {
                                s.take(&jobs, s.wait().min(end.saturating_duration_since(Instant::now())));
                                if s.ready() { s.run(g); }
                            }
                            t0 = Instant::now();
                            Ok(())
                        };
                        if adm.len() > 1 {
                            let list: Vec<_> = adm.iter().map(|(sl, c)| {
                                let Job::Chat { ids, sys, keep, owner, .. } = c else { unreachable!() };
                                (*sl, *owner, &ids[..], *sys, *keep)
                            }).collect();
                            match g.prefill_slots(&list, &mut between) {
                                Ok(v) => v.into_iter().map(Ok).collect(),
                                Err(e) => adm.iter().map(|_| Err(serve::copy_err(&e))).collect(),
                            }
                        } else {
                            let (sl, Job::Chat { ids, sys, keep, owner, .. }) = &adm[0] else { unreachable!() };
                            g.use_slot(*sl);
                            g.owner = *owner;
                            vec![g.prefill_cached_with(ids, *sys, *keep, &mut between)]
                        }
                    };
                    for ((sl, c), r) in adm.into_iter().zip(res) {
                        let Job::Chat { owner, back, .. } = c else { unreachable!() };
                        (s.held[sl], s.who[sl]) = (r.is_ok(), owner);
                        // A client that left during the prefill takes no slot.
                        if back.send(r.map(|(l, u)| (l, u, sl, started))).is_err() && s.held[sl] {
                            s.held[sl] = false;
                            s.tick += 1;
                            s.freed[sl] = s.tick;
                        }
                    }
                    gpu_model_stats(g, &mut self.stats.lock().unwrap());
                }
                if s.ready() { s.run(g); }
            }
        });
        next.into_inner().unwrap()
    }
}

/// Resolve `--mtp-model` (`.gguf` file, HF `.safetensors` file, or directory containing either `model.gguf` or
/// `model.safetensors` + `config.json`) and open it as a companion GGUF. If `.gguf` is missing (or
/// `STRATA_EAGLE3_RECONVERT` is set) and `.safetensors` + `config.json` are present, converts natively in Rust.
fn open_mtp_gguf(path: Option<&Path>) -> anyhow::Result<Option<Gguf>> {
    let Some(p) = path else { return Ok(None) };
    let reconvert = std::env::var_os("STRATA_EAGLE3_RECONVERT").is_some();
    let (gguf, st_cfg) = if p.is_dir() {
        (p.join("model.gguf"), Some((p.join("model.safetensors"), p.join("config.json"))))
    } else if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
        (p.with_extension("gguf"), Some((p.to_path_buf(), p.with_file_name("config.json"))))
    } else if p.extension().and_then(|e| e.to_str()) == Some("bin") {
        (p.with_file_name("model.gguf"), None)
    } else {
        (p.to_path_buf(), None)
    };
    if let Some((st, cfg)) = st_cfg
        && (!gguf.exists() || reconvert)
        && st.exists()
        && cfg.exists()
    {
        eprintln!("note: native Rust EAGLE3 conversion {} -> {}", st.display(), gguf.display());
        rusting_strata::gguf::convert_safetensors_eagle3_to_gguf(&st, &cfg, &gguf, 48)?;
    }
    anyhow::ensure!(
        gguf.exists(),
        "{}: not found (for PyTorch .bin EAGLE3 checkpoints, convert first via tools/convert_eagle3_to_gguf.py)",
        gguf.display()
    );
    Ok(Some(Gguf::open(&gguf)?))
}

/// A lightweight model-switch check. It maps GGUF headers and builds a tokenizer,
/// but neither materializes all weights nor creates another GPU model. A failed
/// check leaves the current model active.
fn preflight_model(path: &Path, e: &Engine) -> anyhow::Result<()> {
    rusting_strata::compatibility::inspect_path(path, rusting_strata::compatibility::Policy::MoeOnly)?.ensure_supported()?;
    let plan = resolve_model_parallel(path, e.parallel_mode, e.max_parallel, e.gpu, e.mtp())?;
    validate_model_runtime(path, plan.slots, e.mtp(), e.mtp_model.as_deref())?;
    let g=Gguf::open(path)?;
    validate_context_length(e.ctx, model_context_length(&g))?;
    Tokenizer::from_gguf(&g)?;
    Ok(())
}

/// Resolve the serving ceiling from checkpoint features without opening a driver or loading weights.
/// Auto chooses a supported slot count; it does not certify VRAM fit or measured speed.
pub fn resolve_model_parallel(path: &Path, mode: Parallelism, ceiling: usize, gpu: bool, mtp: usize) -> anyhow::Result<CapacityPlan> {
    let g = Gguf::open(path)?;
    let cfg = ModelConfig::from_gguf(&g)?;
    resolve_capacity(mode, ceiling, gpu, mtp, ModelFeatures {
        hyper_connections: cfg.hc_n > 0,
        per_layer_embeddings: cfg.n_embd_ple > 0,
        sparse_attention: cfg.idx_top_k > 0,
    })
}

/// Validate requested acceleration before a driver opens or the active model is released.
/// This is a feature/shape preflight, not a GPU execution or performance test.
pub fn validate_model_runtime(path: &Path, parallel: usize, mtp: usize, mtp_model: Option<&Path>) -> anyhow::Result<()> {
    anyhow::ensure!((1..=4).contains(&parallel), "Forge9070 supports 1 to 4 parallel slots");
    anyhow::ensure!(mtp <= 2, "Forge9070 supports 0 to 2 MTP drafts");
    anyhow::ensure!(parallel == 1 || mtp == 0, "MTP and concurrent GPU slots cannot be enabled together");
    let g=Gguf::open(path)?;
    let cfg=ModelConfig::from_gguf(&g)?;
    anyhow::ensure!(parallel == 1 || (cfg.hc_n == 0 && cfg.n_embd_ple == 0 && cfg.idx_top_k == 0),
        "this checkpoint's hyper-connections, per-layer embeddings or sparse attention require --parallel 1 in this engine");
    if mtp > 0 {
        anyhow::ensure!(cfg.hc_n == 0, "MTP for hyper-connection models is not implemented in this Rust engine");
        let aux=mtp_model.map(Gguf::open).transpose()?;
        // The model loader checks the actual head and may reject its quantization;
        // do that before dropping a currently working model.
        let loaded=Model::load_with_aux(&g,aux.as_ref(),&[])?;
        anyhow::ensure!(loaded.cfg.n_nextn > 0,
            "this checkpoint has no usable MTP head; select a matching supported --mtp-file or set --mtp 0");
        rusting_strata::compatibility::validate_mtp_geometry(&loaded, aux.as_ref())?;
    }
    Ok(())
}

/// Loads `model`, answers `w` (the /api/load request that asked for it) and serves until another model is asked
/// for. The model and its GPU memory are freed on return.
fn serve_model(model: &Path, mut engine: Engine, s: &Serving, w: &mut Option<TcpStream>) -> anyhow::Result<Option<ModelChange>> {
    eprintln!("loading {}", model.display());
    if w.is_none() { preflight_model(model, &engine)?; }
    s.admission.set_capacity(0);
    let plan = resolve_model_parallel(model, engine.parallel_mode, engine.max_parallel, engine.gpu, engine.mtp())?;
    engine.parallel = plan.slots;
    *s.policy.lock().unwrap() = json!(plan);
    eprintln!("serving capacity: {} slots ({}) | queue {} | timeout {}s", plan.slots, plan.reason,
        s.admission.snapshot().queue_capacity, s.admission.snapshot().queue_timeout_seconds);
    let name = s.model_id(model);
    {
        let mut st = s.stats.lock().unwrap();
        st.unload(Some(name.clone()));
        st.ctx = engine.ctx;
        st.ram_cap = engine.ram_cap();
    }
    let g = Gguf::open(model)?;
    s.stats.lock().unwrap().ctx_train = model_context_length(&g);
    let aux_g = open_mtp_gguf(engine.mtp_model.as_deref())?;
    let mut m = Model::load_with_aux(&g, aux_g.as_ref(), &engine.lora_specs())?;
    if !engine.lora.is_empty() { m.apply_lora(&engine.lora_specs())?; }
    let tok = Tokenizer::from_gguf(&g)?;
    let mut r = engine.runner(&m, model)?;
    // Serving decodes one reply after a (usually cached) prompt: follow the reply's routing, not the prompt's.
    if let Runner::Gpu(g) = &mut r { g.set_decode_only(true); }
    {
        let mut st = s.stats.lock().unwrap();
        st.loading = false;
        st.mtp_drafts = engine.mtp();
        (st.backend, st.device, st.ctx, st.slots) = match &r { Runner::Gpu(g) => (g.gpu.name().into(), g.gpu.device(), engine.ctx, g.slots()), _ => ("cpu".into(), String::new(), engine.ctx, 1) };
        gpu_stats(&r, &mut st);
    }
    let mut l = Loaded { r, tok: &tok, tpl: template(&g), name, path: model.canonicalize().unwrap_or_else(|_| model.to_path_buf()), identity: g.identity.clone() };
    s.admission.set_capacity(s.stats.lock().unwrap().slots);
    eprintln!("loaded {}", l.name);
    if let Some(mut w) = w.take() {
        let b = json!({"loaded": l.name}).to_string();
        let _ = write!(w, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
    }
    let next = s.serve(Some(&mut l), &engine);
    s.admission.set_capacity(0);
    s.stats.lock().unwrap().unload(None);
    // A failed profile save must not stop the server between two models.
    if let Err(e) = l.r.finish() { eprintln!("{e:#}"); }
    Ok(next)
}

/// The side panel's device memory, expert cache and MTP counts.
fn gpu_stats(r: &Runner, s: &mut serve::Stats) {
    s.expert_ram = r.expert_ram_stats();
    if let Runner::Gpu(g) = r { gpu_model_stats(g, s) }
}

fn gpu_model_stats(g: &GpuModel, s: &mut serve::Stats) {
    s.expert_ram = Some(g.expert_ram_stats());
    s.vram = g.gpu.mem_info().ok().map(|(f, t)| ((t - f) as u64, t as u64));
    s.experts = g.cache.as_ref().map(|_| (g.hits as u64, g.misses as u64));
    s.drafts = (g.drafts.0 > 0).then_some((g.drafts.0 as u64, g.drafts.1 as u64));
    s.expert_tail = g.parts().map(|part| part.expert_tail_stats()).fold(rusting_strata::expert_tail::Stats::default(), |mut total, next| {
        total.min_rank = total.min_rank.max(next.min_rank);
        total.skipped_assignments = total.skipped_assignments.saturating_add(next.skipped_assignments);
        total.skipped_expert_groups = total.skipped_expert_groups.saturating_add(next.skipped_expert_groups);
        total
    });
}

/// Default models folder: `$XDG_DATA_HOME`, `%LOCALAPPDATA%` on Windows, else `~/.local/share`.
fn models_dir_default() -> PathBuf {
    std::env::var_os("XDG_DATA_HOME").or_else(|| std::env::var_os("LOCALAPPDATA")).map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".local/share"))
        .join("rusting_strata").join("models")
}

fn open_browser(url: &str) {
    #[cfg(windows)]
    let r = std::process::Command::new("cmd").args(["/c", "start", "", url]).spawn();
    #[cfg(target_os = "macos")]
    let r = std::process::Command::new("open").arg(url).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let r = std::process::Command::new("xdg-open").arg(url).spawn();
    if let Err(e) = r { eprintln!("could not open a browser ({e}); open {url}"); }
}

/// One Messages API stream event.
fn sse(w: &mut impl Write, v: serde_json::Value) -> anyhow::Result<()> { Ok(write!(w, "event: {}\ndata: {v}\n\n", v["type"].as_str().unwrap_or_default())?) }

/// Streamed reply text as a Messages API text block (index 0), started by the first piece.
fn text_delta(w: &mut impl Write, opened: &mut bool, t: &str) -> anyhow::Result<()> {
    if !std::mem::replace(opened, true) { sse(w, json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}))?; }
    sse(w, json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": t}}))
}

/// The `timings` block llama-server puts in its responses. llama-swap and other
/// OpenAI-compatible clients read it for prefill/decode rates. `prompt_n` is the
/// uncached part of the prompt and `cache_n` the part served from cache, as in
/// llama-server.
fn llama_timings(l: &serve::Last) -> serde_json::Value {
    let prompt_n = l.prompt_tokens.saturating_sub(l.cached_tokens);
    let per_token_ms = |ms: f64, n: usize| if n == 0 { 0.0 } else { ms / n as f64 };
    let per_second = |ms: f64, n: usize| if ms > 0.0 { n as f64 * 1e3 / ms } else { 0.0 };
    json!({
        "prompt_n": prompt_n,
        "prompt_ms": l.prefill_ms,
        "prompt_per_token_ms": per_token_ms(l.prefill_ms, prompt_n),
        "prompt_per_second": per_second(l.prefill_ms, prompt_n),
        "predicted_n": l.completion_tokens,
        "predicted_ms": l.decode_ms,
        "predicted_per_token_ms": per_token_ms(l.decode_ms, l.completion_tokens),
        "predicted_per_second": per_second(l.decode_ms, l.completion_tokens),
        "cache_n": l.cached_tokens,
    })
}

/// The chat page, served by the connection threads of `strata serve`.
const PAGE: &str = include_str!("web/index.html");

fn handle(mut w: TcpStream, q: anyhow::Result<serve::Request>, l: Option<&mut Loaded>, e: &Engine, s: &Serving, i: usize, queue_wait: Duration,
    next: &mut Option<ModelChange>) -> anyhow::Result<()> {
    let send = |w: &mut TcpStream, status: &str, v: serde_json::Value| -> anyhow::Result<()> {
        let b = v.to_string();
        Ok(write!(w, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nX-Request-ID: forge-{i}\r\nX-Forge-Queue-Ms: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", queue_wait.as_millis(), b.len())?)
    };
    // A malformed request gets a 400 (best effort: the client may be gone) and is still logged.
    let serve::Request { method, path, host, auth, json, body } = match q {
        Ok(q) => q,
        Err(err) => {
            let _ = send(&mut w, "400 Bad Request", json!({"error": {"message": format!("{err:#}"), "type": "invalid_request_error"}}));
            return Err(err);
        }
    };
    let created = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?.as_secs();
    // Browsers can reach a loopback server: the Host check stops DNS rebinding, and requiring a JSON content type
    // makes a cross-site POST need a CORS preflight, which is never answered.
    // With a key the check is not needed (a rebinding page has no key) and would refuse a reverse proxy's Host.
    let keys = s.keys.read().unwrap().clone();
    if !keys.any() && !serve::host_ok(&host, &s.addr, &s.allowed_hosts) { return send(&mut w, "403 Forbidden", json!({"error": {"message": "bad Host header", "type": "forbidden"}})); }
    let Some(who) = keys.who(&auth) else { return send(&mut w, "401 Unauthorized", json!({"error": {"message": "missing or wrong API key", "type": "invalid_api_key"}})) };
    if method == "POST" && !json { return send(&mut w, "415 Unsupported Media Type", json!({"error": {"message": "Content-Type must be application/json", "type": "invalid_request_error"}})); }
    // Claude Code adds `?beta=true`.
    let route = path.split('?').next().unwrap_or_default();
    // Customer keys may not load models, download files or read the server's stats.
    if !who.admin && !serve::customer_route(route) {
        return send(&mut w, "403 Forbidden", json!({"error": {"message": "this key may use only the chat APIs, /v1/messages/count_tokens, /api/context and /v1/models", "type": "permission_error"}}));
    }
    match (method.as_str(), route) {
        ("GET", "/api/models") => {
            let all = s.models();
            let all_count = all.len();
            let models: Vec<_> = all.into_iter().filter_map(|(id, p, n)| {
                let report = rusting_strata::compatibility::inspect_path(&p, rusting_strata::compatibility::Policy::MoeOnly).ok()?;
                report.ensure_supported().ok()?;
                Some(json!({"id":id,"bytes":n,"compatibility":report,"image":false,"loaded":l.as_ref().is_some_and(|l|l.path==p)}))
            }).collect();
            let excluded_count = all_count - models.len();
            // The values in effect: with a model loaded, `chunk` and `mtp` include its calibration.
            let options = json!({"ctx": e.ctx, "kv": e.kv, "cache_ram": e.cache_ram, "expert_ram": e.expert_ram,
                "context_min": CONTEXT_MIN, "context_max": CONTEXT_MAX,
                "max_tokens": (s.max_tokens != usize::MAX).then_some(s.max_tokens),
                "default_max_tokens": s.max_tokens.min(e.ctx.saturating_sub(1)),
                "think": e.think_default, "preserve_thinking": e.preserve_thinking,
                "temperature": e.temp, "top_p": e.top_p, "top_k": e.top_k,
                "expert_warmup": e.expert_warmup,"ram_margin":e.ram_margin,"max_ram":e.max_ram,
                "vram_margin": e.vram_margin.join(","), "mtp": e.mtp, "chunk": e.chunk,
                "threads": l.as_ref().map(|_| rayon::current_num_threads()).or(e.threads),
                "llama_numerics": std::env::var_os("STRATA_LLAMA_NUMERICS").is_some(),
                "model_identity": l.as_ref().map(|loaded| &loaded.identity),
                "decode_cache_policy": if e.gpu { DECODE_CACHE_POLICY } else { "not_applicable_cpu" },
                "inference_overrides": inference_overrides(),
                "gpu": e.gpu, "parallel":e.parallel.max(1), "parallel_mode": e.parallel_mode.to_string(),
                "max_parallel": e.max_parallel, "concurrency_reason": s.policy.lock().unwrap()["reason"],
                "expert_tail_min_rank":rusting_strata::expert_tail::Policy::from_env().map_or(0,|p|p.min_rank), "ctx_train": s.stats.lock().unwrap().ctx_train,
                // A filename alone is not proof that settings apply to this revision/configuration.
                "calibrated": l.as_ref().filter(|_| e.gpu).map(|_| e.calibration.as_ref().and_then(read_calibration).is_some())});
            send(&mut w, "200 OK", json!({"dir": s.dir, "models": models, "options": options,"policy":"moe_only","excluded_count":excluded_count}))
        }
        // Answered once the model has loaded (or failed to).
        ("POST", "/api/load") => {
            let v = serde_json::from_slice::<serde_json::Value>(&body).unwrap_or_default();
            let id = v["model"].as_str().unwrap_or_default();
            let mut requested = match e.with(&v["options"]) {
                Ok(value)=>value,
                Err(err)=>return send(&mut w,"400 Bad Request",json!({"error":{"message":format!("{err:#}"),"type":"invalid_request_error"}})),
            };
            let admission = s.admission.snapshot();
            // Model reloads must obey the same connection budget as startup, including a larger new ceiling.
            if let Err(error) = validate_queue_limits(admission.queue_capacity, admission.queue_timeout_seconds as u64,
                s.max_waiting, requested.parallel.max(1)) {
                return send(&mut w, "400 Bad Request", json!({"error":{"message":format!("{error:#}"),"type":"invalid_request_error"}}));
            }
            match s.models().into_iter().find(|m| m.0 == id) {
                Some((_, p, _)) => {
                    if v["calibrate"] == true {
                        requested.mtp = Some(0); // calibration chooses only supported embedded drafts
                        let capacity = resolve_model_parallel(&p, requested.parallel_mode, requested.max_parallel, requested.gpu, 0);
                        if !requested.gpu || requested.ctx < 768 || !capacity.as_ref().is_ok_and(|plan| plan.slots == 1) {
                            return send(&mut w,"400 Bad Request",json!({"error":{"message":"calibration needs GPU mode, one supported parallel slot and context >= 768; use --parallel 1 or --max-parallel 1 when needed","type":"invalid_request_error"}}));
                        }
                    }
                    // Validate metadata and tokenizer before releasing the working model.
                    // Allocation or numerical failures can still occur during actual loading.
                    if let Err(err) = preflight_model(&p, &requested) {
                        return send(&mut w, "400 Bad Request", json!({"error": {"message": format!("{err:#}"), "type": "incompatible_model"}}));
                    }
                    // `calibrate: true` measures the model first; it travels with the options to the load loop.
                    let mut o = v["options"].clone();
                    if v["calibrate"] == true { if !o.is_object() { o = json!({}); } o["calibrate"] = true.into(); }
                    *s.opts.lock().unwrap() = o; s.admission.set_capacity(0); *next = Some(ModelChange::Load(p, w)); Ok(())
                }
                None => send(&mut w, "404 Not Found", json!({"error": {"message": format!("no model {id:?} in {}", s.dir.display()), "type": "not_found"}})),
            }
        }
        ("POST", "/api/unload") => {
            if l.is_some() { s.admission.set_capacity(0); *next = Some(ModelChange::Unload(w)); Ok(()) }
            else { acknowledge_unload(w).map_err(Into::into) }
        }
        ("POST", "/api/hf/search") => {
            let v = serde_json::from_slice::<serde_json::Value>(&body).unwrap_or_default();
            let query = v["query"].as_str().unwrap_or_default().trim().to_string();
            let limit = v["limit"].as_u64().unwrap_or(6);
            if query.is_empty() || query.len() > 200 || !(1..=10).contains(&limit) {
                return send(&mut w, "400 Bad Request", json!({"error": {"message": "query needs 1–200 bytes and limit must be 1–10", "type": "invalid_request_error"}}));
            }
            spawn_hf_metadata(s,w,move || {
                Ok(json!({"models":hf::search(&query,limit as usize)?,"policy":"moe_only"}))
            })
        }
        ("POST", "/api/hf/files") => {
            let v = serde_json::from_slice::<serde_json::Value>(&body).unwrap_or_default();
            let repo = v["repo"].as_str().unwrap_or_default().to_string();
            let revision = v["revision"].as_str().unwrap_or("main").to_string();
            spawn_hf_metadata(s,w,move || {
                let f=hf::list_at(&repo,&revision)?;
                Ok(json!({"repo":f.repo,"revision":f.revision,"commit":f.commit,
                    "license":f.license,"gated":f.gated,"policy":f.policy,"excluded":f.excluded,
                    "files":f.models.into_iter().map(|m|json!({"file":m.file,"bytes":m.size,"shards":m.shards,"sha256":m.sha256,"compatibility":m.compatibility})).collect::<Vec<_>>()}))
            })
        }
        ("GET", "/api/download") => send(&mut w, "200 OK", json!(*s.dl.lock().unwrap())),
        ("POST", "/api/download/cancel") => { s.dl.lock().unwrap().cancel = true; send(&mut w, "200 OK", json!(*s.dl.lock().unwrap())) }
        // Runs on its own thread so chats go on meanwhile; the page polls GET /api/download.
        ("POST", "/api/download") => {
            let v: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
            let (repo, file) = (v["repo"].as_str().unwrap_or_default().to_string(), v["file"].as_str().unwrap_or_default().to_string());
            let revision = v["revision"].as_str().unwrap_or("main").to_string();
            let mut g = s.dl.lock().unwrap();
            if g.running { drop(g); return send(&mut w, "409 Conflict", json!({"error": {"message": "a download is already running", "type": "conflict"}})); }
            *g = hf::Progress { repo: repo.clone(), file: file.clone(), running: true, ..Default::default() };
            drop(g);
            let (dl, dir) = (s.dl.clone(), s.dir.clone());
            std::thread::spawn(move || {
                let r = hf::download_at(&repo, &file, &revision, &dir, &dl);
                if let Err(e) = &r { eprintln!("download {repo} {file}: {e:#}"); } else { eprintln!("downloaded {repo} {file}"); }
                let mut g = dl.lock().unwrap();
                (g.running, g.error) = (false, r.err().map(|e| format!("{e:#}")));
            });
            send(&mut w, "202 Accepted", json!(*s.dl.lock().unwrap()))
        }
        // The prompt's tokens as the model would see them, template and tools included.
        ("POST", "/v1/messages/count_tokens") => {
            let Some(l) = l else { return send(&mut w, "503 Service Unavailable", json!({"error": {"message": "no model loaded", "type": "server_error"}})) };
            match serve::messages_body(&body).and_then(|b| serve::chat_request(&b)) {
                Ok(req) => {
                    let rendered = serve::render_chat(l.tpl, &req, e.think_default, e.preserve_thinking);
                    let (prompt, images) = (rendered.prompt, rendered.images);
                    let prepared = (|| -> anyhow::Result<usize> {
                        if images.is_empty() { Ok(l.tok.encode(&prompt)?.len()) } else {
                            let mmproj = e.mmproj.clone().or_else(|| vision::find_sibling(&l.path));
                            let mmproj = mmproj.as_deref().context("image input needs --mmproj")?;
                            let input = load_images(mmproj, &images)?;
                            Ok(build_image_prompt(l.tok, &input, &images, &prompt)?.ids.len())
                        }
                    })();
                    let n = match prepared {
                        Ok(n) => n,
                        Err(error) => return send(&mut w, "400 Bad Request", json!({"error":{"message":format!("{error:#}"),"type":"invalid_request_error","code":"prompt_preparation_failed"}})),
                    };
                    send(&mut w, "200 OK", json!({"input_tokens": n}))
                }
                Err(err) => send(&mut w, "400 Bad Request", json!({"error": {"message": format!("{err:#}"), "type": "invalid_request_error"}})),
            }
        }
        // The Anthropic Messages API (Claude Code) is translated to a chat request, and the reply back.
        ("POST", p @ ("/v1/chat/completions" | "/v1/messages" | "/api/context")) => {
            let anth = p == "/v1/messages";
            let preview = p == "/api/context";
            let Some(l) = l else { return send(&mut w, "503 Service Unavailable", json!({"error": {"message": "no model loaded", "type": "server_error"}})) };
            let (r, tok, tpl, name) = (&mut l.r, &l.tok, l.tpl, l.name.as_str());
            // A queued client may still name the previous model after a switch.
            // Never silently answer an explicit OpenAI model ID with another
            // model. Omitted/empty IDs use the current model, as the chat page does.
            // Anthropic aliases retain upstream semantics for Claude Code.
            if !anth && let Ok(v) = serde_json::from_slice::<serde_json::Value>(&body)
                && let Some(requested) = v["model"].as_str().filter(|id| !id.trim().is_empty())
                && requested != name {
                return send(&mut w, "409 Conflict", json!({"error": {
                    "message": format!("requested model {requested:?} is not loaded; current model is {name:?}. Load the requested catalog ID with /api/load first."),
                    "type": "model_not_loaded", "loaded_model": name
                }}));
            }
            let req = match if anth { serve::messages_body(&body) } else { Ok(body.clone()) }.and_then(|b| serve::chat_request(&b)) {
                Ok(q) => q,
                Err(err) => return send(&mut w, "400 Bad Request", json!({"error": {"message": format!("{err:#}"), "type": "invalid_request_error"}})),
            };
            // Room is checked on the full prompt before any state changes.
            let rendered = serve::render_chat(tpl, &req, e.think_default, e.preserve_thinking);
            let (prompt, history, system, images) = (rendered.prompt, rendered.history, rendered.system, rendered.images);
            if preview && !images.is_empty() {
                return send(&mut w, "400 Bad Request", json!({"error": {"message": "the read-only context preview accepts text and tools; image token counts require the vision encoder and are not previewed", "type": "invalid_request_error", "code": "image_preview_unsupported"}}));
            }
            // Preparation may fail before inference starts, including a missing vision projector. Always
            // return an HTTP error here, including for clients that requested an SSE response.
            let prepared = (|| {
                let image_input = if images.is_empty() { None } else {
                    let mmproj = e.mmproj.clone().or_else(|| vision::find_sibling(&l.path));
                    let mmproj = mmproj.as_deref().context("image input needs --mmproj")?;
                    Some(load_images(mmproj, &images)?)
                };
                let vision_prompt = match &image_input {
                    Some(input) => Some(build_image_prompt(tok, input, &images, &prompt)?),
                    None => None,
                };
                let tokens = if vision_prompt.is_none() { Some(serve::ChatTokens::new(tok, &prompt, &history, &system)?) } else { None };
                let n_system = match &tokens { Some(t) => t.system_tokens, None => tok.encode(&system)?.len() };
                Ok::<_, anyhow::Error>((image_input, vision_prompt, tokens, n_system))
            })();
            let (image_input, vision_prompt, tokens, n_system) = match prepared {
                Ok(value) => value,
                Err(error) => return send(&mut w, "400 Bad Request", json!({"error":{"message":format!("{error:#}"),"type":"invalid_request_error","code":"prompt_preparation_failed"}})),
            };
            let ids = match &vision_prompt { Some(p) => &p.ids, None => &tokens.as_ref().unwrap().ids };
            let budget = serve::ContextBudget::new(e.ctx, ids.len(), req.max_tokens, s.max_tokens);
            if preview {
                let mut result = serde_json::to_value(&budget)?;
                result["model"] = name.into();
                result["model_context_tokens"] = json!(s.stats.lock().unwrap().ctx_train);
                result["system_tokens"] = n_system.into();
                result["context_max"] = CONTEXT_MAX.into();
                return send(&mut w, "200 OK", result);
            }
            if !budget.fits {
                return send(&mut w, "400 Bad Request", json!({"error": &budget.error, "budget": &budget}));
            }
            let n = budget.output_tokens;
            match r { Runner::Gpu(g) => g.owner = who.id, Runner::Slot(c) => (c.owner, c.conn) = (who.id, w.try_clone().ok()), _ => {} }
            // One line per chat request, and one when it finishes (see `log` below).
            let key_name = if who.name.is_empty() { "open" } else { who.name.as_str() };
            eprintln!("request: {method} {p} key={key_name} model={name} prompt={} tok stream={} tools={}",
                ids.len(), req.stream, req.tools.len());
            let mut smp = e.sampler();
            smp.temp = req.temperature.unwrap_or(smp.temp);
            smp.top_k = req.top_k.unwrap_or(smp.top_k);
            smp.top_p = req.top_p.unwrap_or(smp.top_p);
            if req.seed.is_some() { smp.rng = seed(req.seed); }
            let id = format!("chatcmpl-{created}-{i}");
            let include_usage = req.include_usage;
            let chunk = |delta: serde_json::Value, finish: Option<&str>| {
                let mut value = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
                if include_usage { value["usage"] = serde_json::Value::Null; }
                value
            };
            // With tools, a stream holds back the text from a `<tool_call>` tag on until the reply is parsed.
            let tools = req.tool_calls && !req.tools.is_empty();
            let mut cached = 0;
            let mut k = 0;  // tokens decoded
            let mut opened = false;  // Messages API stream: the text block has started
            let mut stream_started = false;
            let mut hold = Hold::new(tools, req.stop);
            // A prompt that ends in an open think block makes the reply start with reasoning, up to `</think>`.
            let think_open = prompt.trim_end().ends_with("<think>");
            let mut think = Think::new(think_open);
            let (mut t0, mut t1) = (Instant::now(), Instant::now());
            // Under `--parallel` a request is busy (and its prefill timed) from when the model takes it, not while it
            // waits for a slot.
            let mut busy = false;
            let start = |busy: &mut bool| {
                let mut st = s.stats.lock().unwrap();
                if st.busy == 0 { st.live_tokens = 0; }
                st.busy += 1;
                *busy = true;
            };
            if !matches!(r, Runner::Slot(_)) { start(&mut busy); }
            // Whether the prompt reached the model: a client that left while it waited in the queue is not logged.
            let mut prefilled = false;
            let res = (|| -> anyhow::Result<(Vec<u32>, bool)> {
                let logits;
                if serve::gone(&w) { anyhow::bail!("client closed the connection"); }
                let (chat_logits, chat_cached) = match &vision_prompt {
                    Some(p) => (r.chat_image(p, &image_input.as_ref().unwrap().embeddings)?, 0),
                    None => r.chat_tokens(tokens.as_ref().unwrap())?,
                };
                (logits, cached) = (chat_logits, chat_cached);
                if let Runner::Slot(c) = &*r && let Some(t) = c.started { t0 = t; start(&mut busy); }
                prefilled = true;
                t1 = Instant::now();
                if serve::gone(&w) { anyhow::bail!("client closed the connection"); }
                if req.stream {
                    // A failed write may have sent a partial header. Never append a second HTTP response.
                    stream_started = true;
                    write!(w, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nX-Request-ID: forge-{i}\r\nX-Forge-Queue-Ms: {}\r\nConnection: close\r\n\r\n", queue_wait.as_millis())?;
                    if anth {
                        sse(&mut w, json!({"type": "message_start", "message": {"id": format!("msg_{i}"), "type": "message", "role": "assistant", "model": name,
                            "content": [], "stop_reason": null, "stop_sequence": null,
                            "usage": {"input_tokens": ids.len() - cached, "cache_read_input_tokens": cached, "output_tokens": 0}}}))?;
                    } else {
                        write!(w, "data: {}\n\n", chunk(json!({"role": "assistant", "content": ""}), None))?;
                    }
                }
                let mut last_stream_activity = Instant::now();
                decode(r, tok, e.mtp() > 0, &mut smp, logits, n, &mut |t| {
                    s.stats.lock().unwrap().live_tokens += 1;
                    // A client that left stops the reply (a stream notices on the next write; held text writes nothing).
                    k += 1;
                    if k % 16 == 0 && serve::gone(&w) { anyhow::bail!("client closed the connection"); }
                    let t = hold.push(t);
                    if req.stream {
                        if !t.is_empty() {
                            if !anth { send_split(&mut w, &chunk, think.feed(&t))?; } else { text_delta(&mut w, &mut opened, &t)?; }
                            last_stream_activity = Instant::now();
                        } else if last_stream_activity.elapsed() >= Duration::from_secs(15) {
                            w.write_all(b": keep-alive\n\n")?;
                            last_stream_activity = Instant::now();
                        }
                    }
                    Ok(hold.stopped)
                })
            })();
            {
                let mut st = s.stats.lock().unwrap();
                if busy { st.busy -= 1; }
                gpu_stats(r, &mut st);
            }
            let decoded = t1.elapsed();
            let ms = |d: Duration| d.as_secs_f64() * 1e3;
            let mut last = serve::Last { at: created, model: name.into(), system_tokens: n_system, cached_tokens: cached,
                prompt_tokens: ids.len(), completion_tokens: k, prefill_ms: ms(t1 - t0), decode_ms: ms(decoded), finish: String::new() };
            let log = |last: &serve::Last| {
                let decode_s = last.decode_ms / 1e3;
                let prefill_s = last.prefill_ms / 1e3;
                // Cached prompt tokens were not prefilled again, so they do not count toward the prefill speed.
                let new_prompt = last.prompt_tokens.saturating_sub(last.cached_tokens);
                let decode_tps = if decode_s > 0.0 { last.completion_tokens as f64 / decode_s } else { 0.0 };
                let prefill_tps = if prefill_s > 0.0 { new_prompt as f64 / prefill_s } else { 0.0 };
                eprintln!("done: key={key_name} model={} prompt={} tok ({} cached) -> {} tok in {decode_s:.2} s, \
                    decode {decode_tps:.1} tok/s, prefill {prefill_tps:.1} tok/s, finish={}",
                    last.model, last.prompt_tokens, last.cached_tokens, last.completion_tokens, last.finish);
                if let Some(f) = &s.usage {
                    let mut v = json!(last);
                    v["key"] = who.name.clone().into();
                    v["queue_ms"] = (queue_wait.as_millis() as u64).into();
                    // A failed write must not fail the reply; it is reported.
                    if let Err(err) = writeln!(f.lock().unwrap(), "{v}") { eprintln!("usage log: {err}"); }
                }
            };
            // The next request starts from the saved history state or from scratch. A stream that already started is just cut off.
            let (out, stop) = match res {
                Ok(x) => x,
                Err(err) => {
                    // Logged with the tokens decoded so far, which a client that left may have received.
                    let gone = serve::gone(&w);
                    last.finish = if gone { "cut" } else { "error" }.into();
                    // A request that failed before its prefill ended measured no prefill speed (t1 is still t0).
                    if !prefilled { last.prefill_ms = 0.0; }
                    if prefilled || !gone { log(&last); }
                    if (!req.stream || !stream_started) && !gone {
                        let sent = send(&mut w, "500 Internal Server Error", json!({"error": {"message": format!("{err:#}"), "type": "server_error"}}));
                        serve::exit_if_gpu_dead(&err);
                        return sent;
                    }
                    return Err(err);
                }
            };
            // A malformed call goes out as plain text.
            let calls = if tools { tpl.parse_calls(&hold.text, &req.tools) } else { None };
            last.completion_tokens = out.len();
            last.finish = if calls.is_some() { "tool_calls" } else if stop { "stop" } else { "length" }.into();
            let timings = llama_timings(&last);
            log(&last);
            s.stats.lock().unwrap().record(last);
            let calls_json = |(content, calls): &(String, Vec<(String, String)>), stream: bool| {
                let c: Vec<_> = calls.iter().enumerate().map(|(k, (f, a))| {
                    let mut c = json!({"id": format!("call_{i}_{k}"), "type": "function", "function": {"name": f, "arguments": a}});
                    if stream { c["index"] = k.into(); }
                    c
                }).collect();
                if stream { json!({"tool_calls": c}) }
                else { json!({"role": "assistant", "content": if content.is_empty() { None } else { Some(content) }, "tool_calls": c}) }
            };
            if anth {
                let reason = if calls.is_some() { "tool_use" } else if stop { "end_turn" } else { "max_tokens" };
                let usage = json!({"input_tokens": ids.len() - cached, "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0, "output_tokens": out.len()});
                let uses: Vec<_> = calls.iter().flat_map(|c| &c.1).enumerate().map(|(k, (f, a))| json!({"type": "tool_use", "id": format!("toolu_{i}_{k}"),
                    "name": f, "input": serde_json::from_str::<serde_json::Value>(a).unwrap_or(json!({}))})).collect();
                if !req.stream {
                    let text = calls.as_ref().map_or(&hold.text, |c| &c.0);
                    let content: Vec<_> = (!text.is_empty()).then(|| json!({"type": "text", "text": text})).into_iter().chain(uses).collect();
                    return send(&mut w, "200 OK", json!({"id": format!("msg_{i}"), "type": "message", "role": "assistant", "model": name, "content": content,
                        "stop_reason": reason, "stop_sequence": null, "usage": usage, "timings": timings}));
                }
                if calls.is_none() && !hold.rest().is_empty() { text_delta(&mut w, &mut opened, hold.rest())?; }
                if opened { sse(&mut w, json!({"type": "content_block_stop", "index": 0}))?; }
                for (k, u) in uses.iter().enumerate() {
                    let (index, input) = (k + opened as usize, u["input"].to_string());
                    let mut start = u.clone();
                    start["input"] = json!({});
                    sse(&mut w, json!({"type": "content_block_start", "index": index, "content_block": start}))?;
                    sse(&mut w, json!({"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": input}}))?;
                    sse(&mut w, json!({"type": "content_block_stop", "index": index}))?;
                }
                sse(&mut w, json!({"type": "message_delta", "delta": {"stop_reason": reason, "stop_sequence": null}, "usage": {"output_tokens": out.len()}, "timings": timings}))?;
                return sse(&mut w, json!({"type": "message_stop"}));
            }
            let finish = if calls.is_some() { "tool_calls" } else if stop { "stop" } else { "length" };
            let usage = json!({"prompt_tokens": ids.len(), "completion_tokens": out.len(), "total_tokens": ids.len() + out.len(),
                "prompt_tokens_details": {"cached_tokens": cached}});
            if req.stream {
                match &calls {
                    Some(c) => write!(w, "data: {}\n\n", chunk(calls_json(c, true), None))?,
                    None => {
                        let (r, a) = think.feed(hold.rest());
                        send_split(&mut w, &chunk, (r, a))?;
                        send_split(&mut w, &chunk, think.finish())?;
                    }
                }
                let mut last = chunk(json!({}), Some(finish));
                last["timings"] = timings;
                write!(w, "data: {last}\n\n")?;
                if include_usage {
                    let usage_chunk = json!({"id": id, "object": "chat.completion.chunk", "created": created,
                        "model": name, "choices": [], "usage": usage});
                    write!(w, "data: {usage_chunk}\n\n")?;
                }
                Ok(write!(w, "data: [DONE]\n\n")?)
            } else {
                let message = match &calls { Some(c) => calls_json(c, false), None => {
                    let mut th = Think::new(think_open);
                    let (mut r, mut a) = th.feed(&hold.text);
                    let (r2, a2) = th.finish();
                    r += &r2; a += &a2;
                    if r.is_empty() { json!({"role": "assistant", "content": a}) } else { json!({"role": "assistant", "content": a, "reasoning_content": r}) }
                } };
                send(&mut w, "200 OK", json!({"id": id, "object": "chat.completion", "created": created, "model": name,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage, "timings": timings}))
            }
        }
        _ => send(&mut w, "404 Not Found", json!({"error": {"message": format!("no route {method} {path}"), "type": "not_found"}})),
    }
}

#[cfg(test)]
mod flag_tests {
    use super::{resolve_flags, vram_margins};
    #[cfg(target_os = "linux")]
    use super::cpu_topology_groups;

    #[test]
    fn load_options_apply_and_validate() {
        #[derive(clap::Parser)]
        struct T { #[command(flatten)] e: super::Engine }
        let mut e = <T as clap::Parser>::parse_from(["t"]).e;
        e.gpu=true;
        let n = e.with(&serde_json::json!({"ctx": 8192, "kv": "q8", "mtp": 2, "vram_margin": 256})).unwrap();
        assert_eq!((n.ctx, n.kv.as_str(), n.mtp, n.vram_margin, n.cache_ram), (8192, "q8", Some(2), vec!["256".to_string()], e.cache_ram));
        assert_eq!(e.with(&serde_json::Value::Null).unwrap().ctx, e.ctx);
        for bad in [serde_json::json!({"ctx": 8}), serde_json::json!({"ctx": "big"}), serde_json::json!({"kv": "q4"}), serde_json::json!({"mtp": -1}),
            serde_json::json!({"expert_warmup": "false"}), serde_json::json!({"expert_ram": "all"}), serde_json::json!({"ram_margin": -1})] {
            assert!(e.with(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn tuning_and_engine_memory_flags_agree_and_ui_preserves_them() {
        use clap::Parser;
        #[derive(Parser)]
        struct T { #[command(flatten)] e: super::Engine }
        let defaults = T::parse_from(["t"]).e;
        assert_eq!((defaults.expert_ram.as_str(), defaults.expert_warmup, defaults.ram_margin,
            defaults.max_ram, defaults.cache_ram), ("auto", true, 4096, 0, 512));
        let flags = ["--expert-ram", "40960", "--expert-warmup", "false", "--ram-margin", "8192",
            "--max-ram", "50000", "--cache-ram", "256"];
        let engine = T::try_parse_from(std::iter::once("t").chain(flags)).unwrap().e;
        let cli = super::Cli::try_parse_from(["t", "calibrate", "--model", "model.gguf"].into_iter().chain(flags)).unwrap();
        let Some(super::Cmd::Calibrate { expert_ram, expert_warmup, ram_margin, max_ram, cache_ram, .. }) = cli.cmd
            else { panic!("expected calibration command") };
        assert_eq!((expert_ram.as_str(), expert_warmup, ram_margin, max_ram, cache_ram),
            (engine.expert_ram.as_str(), engine.expert_warmup, engine.ram_margin, engine.max_ram, engine.cache_ram));
        let changed = defaults.with(&serde_json::json!({"expert_ram": "40960", "expert_warmup": false,
            "ram_margin":8192,"max_ram":50000,"cache_ram":256})).unwrap();
        let tune = super::TuneOptions::for_engine(&changed);
        assert_eq!((tune.expert_ram.as_str(), tune.expert_warmup, tune.ram_margin, tune.max_ram, tune.cache_ram),
            ("40960", false, 8192, 50000, 256));
        assert!(T::try_parse_from(["t", "--expert-warmup", "sometimes"]).is_err());
    }

    #[test]
    fn context_limits_apply_to_cli_and_model_switch_preflight() {
        use clap::Parser;
        #[derive(Parser)]
        struct T { #[command(flatten)] e: super::Engine }
        let e = T::parse_from(["t"]).e;
        for ctx in [128, 4096, 131072] {
            assert_eq!(e.with(&serde_json::json!({"ctx":ctx})).unwrap().ctx, ctx);
        }
        for ctx in [0, 127, 131073, 1048576] {
            assert!(e.with(&serde_json::json!({"ctx":ctx})).is_err());
        }
        assert!(super::validate_context_length(32768, Some(32768)).is_ok());
        assert!(super::validate_context_length(32769, Some(32768)).is_err());
        assert!(super::validate_context_length(131072, Some(262144)).is_ok());
        assert!(super::validate_context_length(262144, Some(262144)).is_err());
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe.gguf");
        assert!(super::validate_model_context(&path, 512).is_ok());
        assert!(super::validate_model_context(&path, 513).is_err());
        let mut request = e.clone();
        request.ctx = 512;
        assert!(super::preflight_model(&path, &request).is_ok());
        request.ctx = 1024;
        assert!(super::preflight_model(&path, &request).unwrap_err().to_string().contains("training context 512"));
    }

    #[test]
    fn prompt_cache_reserve_obeys_the_same_owned_ram_cap() {
        let mib = 1u64 << 20;
        assert_eq!(super::prompt_cache_cap(2 * mib, 512, Some(10 * mib)).unwrap(), 8);
        assert_eq!(super::prompt_cache_cap(2 * mib, 3, Some(10 * mib)).unwrap(), 3);
        assert_eq!(super::prompt_cache_cap(2 * mib, 512, None).unwrap(), 512);
        assert!(super::prompt_cache_cap(11 * mib, 0, Some(10 * mib)).is_err());
    }

    #[test]
    fn vram_margins_are_per_device() {
        let spec = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(vram_margins(&spec(&["512"]), &[0, 1]).unwrap(), [512, 512]);
        assert_eq!(vram_margins(&spec(&["512", "256"]), &[1, 0]).unwrap(), [512, 256]);
        assert_eq!(vram_margins(&spec(&["0:512", "1:256"]), &[1, 0]).unwrap(), [256, 512]);
        assert_eq!(vram_margins(&spec(&["0:512", "1:256"]), &[0, 1]).unwrap(), [512, 256]);
    }

    #[test]
    fn vram_margins_reject_bad_specs() {
        let spec = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        for bad in [&["512", "256"][..], &["0:512", "0:256"][..], &["0:512", "256"][..]] {
            let e = vram_margins(&spec(bad), &[0]).unwrap_err().to_string();
            assert!(e.starts_with("--vram-margin"), "{bad:?}: {e}");
        }
        assert!(vram_margins(&spec(&["2:512"]), &[0, 1]).unwrap_err().to_string().contains("not in --device"));
        assert!(vram_margins(&spec(&["x:512"]), &[0]).unwrap_err().to_string().contains("not a GPU index"));
    }

    /// SMT siblings share one `thread_siblings_list`: 4 logical CPUs, 2 physical cores.
    #[cfg(target_os = "linux")]
    #[test]
    fn physical_cores_count_distinct_sibling_groups() {
        let d = std::env::temp_dir().join(format!("strata_cpu_topology_{}", std::process::id()));
        for (cpu, siblings) in [(0, "0,2"), (2, "0,2"), (1, "1,3"), (3, "1,3")] {
            let p = d.join(format!("cpu{cpu}/topology"));
            std::fs::create_dir_all(&p).unwrap();
            std::fs::write(p.join("thread_siblings_list"), siblings).unwrap();
        }
        std::fs::create_dir_all(d.join("cpufreq")).unwrap();
        assert_eq!(cpu_topology_groups(&d), 2);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn parallel_mtp_are_explained() {
        // `--parallel` with MTP: MTP off, one message naming both flags.
        let (n, p, k) = resolve_flags(true, 4, 1, 1, 0, false).unwrap();
        assert_eq!((p, k), (4, 0));
        assert_eq!(n, ["--parallel 4 and --mtp 1: not supported together, decoding without MTP drafts"]);
    }

    #[test]
    fn mtp_without_a_head_runs_without_drafts() {
        let (n, _, k) = resolve_flags(true, 1, 3, 0, 0, false).unwrap();
        assert_eq!(k, 0);
        assert_eq!(n, ["--mtp 3: this model has no MTP head: decoding without drafts"]);
        // A model with a head keeps its drafts.
        let (n, _, k) = resolve_flags(true, 1, 3, 1, 0, false).unwrap();
        assert!(n.is_empty() && k == 3);
        // A layer split or hyper-connections explain themselves instead.
        for (split, hc, why) in [(true, 0, "not available with a multi-GPU layer split"),
                                 (false, 4, "not supported for this model (hyper-connections)")] {
            let (n, _, k) = resolve_flags(true, 1, 3, 1, hc, split).unwrap();
            assert_eq!(k, 0);
            assert_eq!(n, [format!("--mtp 3: {why}: decoding without drafts")]);
        }
    }

    #[test]
    fn cpu_requires_the_gpu_for_both() {
        let (n, _, k) = resolve_flags(false, 2, 1, 1, 0, false).unwrap();
        assert_eq!(k, 0);
        assert_eq!(n, ["--parallel 2: requires --gpu (the CPU answers one request at a time)",
                       "--mtp 1: requires --gpu: decoding without drafts"]);
        // No flags set (0 means unset) prints nothing.
        assert_eq!(resolve_flags(false, 0, 0, 0, 0, false).unwrap(), (vec![], 1, 0));
    }

    #[test]
    fn parallel_range_is_rejected() {
        for n in [256, 1000] {
            let e = resolve_flags(true, n, 0, 0, 0, false).unwrap_err().to_string();
            assert_eq!(e, format!("--parallel {n}: 1 to 255 slots"));
        }
        assert!(resolve_flags(true, 255, 0, 0, 0, false).is_ok());
        assert_eq!(resolve_flags(true, 1, 0, 0, 0, false).unwrap(), (vec![], 1, 0));
    }
}

#[cfg(test)]
mod timing_tests {
    use super::{llama_timings, serve::Last};

    #[test]
    fn timings_match_llama_server() {
        let last = Last { prompt_tokens: 100, cached_tokens: 40, completion_tokens: 20,
            prefill_ms: 60.0, decode_ms: 400.0, ..Default::default() };
        let t = llama_timings(&last);
        // llama-server counts the uncached prompt in prompt_n and the reused part in cache_n.
        assert_eq!(t["prompt_n"], 60);
        assert_eq!(t["cache_n"], 40);
        assert_eq!(t["prompt_ms"], 60.0);
        assert_eq!(t["prompt_per_second"], 1000.0);
        assert_eq!(t["prompt_per_token_ms"], 1.0);
        assert_eq!(t["predicted_n"], 20);
        assert_eq!(t["predicted_ms"], 400.0);
        assert_eq!(t["predicted_per_second"], 50.0);
        assert_eq!(t["predicted_per_token_ms"], 20.0);
    }
}

#[cfg(test)]
mod bench_tests {
    use super::{bench_record, round1, BenchMeta, BenchResult, Cli, ExpertStats, Stat};
    use clap::Parser;
    use serde_json::Value;
    use std::path::Path;

    fn meta() -> BenchMeta<'static> {
        BenchMeta {
            backend: "cpu",
            backend_arg: None,
            device: &[],
            model: Path::new("/mnt/models/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"),
            ctx: 32896,
            chunk: 4096,
            kv: "f16",
            mtp: 0,
            tg: 128,
            reps: 3,
        }
    }

    fn result(pp: usize, prefill: Stat, decode: Stat) -> BenchResult {
        BenchResult { pp, prefill, decode, bytes_per_token: 0, decode_median: decode.mean }
    }

    fn parsed(record: &Value) -> Value {
        // One compact object plus a newline is what reaches the file: it must parse back unchanged.
        let mut line = serde_json::to_string(record).unwrap();
        line.push('\n');
        assert!(line.ends_with("}\n") && line.matches('\n').count() == 1);
        serde_json::from_slice::<Value>(line.as_bytes()).unwrap()
    }

    #[test]
    fn bench_record_schema_and_one_result_per_pp_in_order() {
        let st = |x: f64| Stat { mean: round1(x), min: round1(x), max: round1(x) };
        let results = [result(512, st(2135.7), st(129.5)), result(4096, st(1900.0), st(120.0)), result(32768, st(1500.0), st(80.0))];
        let v = parsed(&bench_record(&meta(), &results, None, 1_762_617_600));
        assert_eq!(v["schema"], "strata.bench.v1");
        assert_eq!(v["unix_time"], 1_762_617_600u64);
        assert_eq!(v["backend"], "cpu");
        assert!(v.get("backend_arg").is_none());
        assert_eq!(v["device"], serde_json::json!([]));
        assert_eq!(v["model"]["name"], "Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf");
        assert_eq!(v["model"]["path"], "/mnt/models/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf");
        assert_eq!((v["ctx"].as_u64(), v["chunk"].as_u64(), v["mtp"].as_u64()), (Some(32896), Some(4096), Some(0)));
        assert_eq!((v["tg"].as_u64(), v["reps"].as_u64()), (Some(128), Some(3)));
        let rs = v["results"].as_array().unwrap();
        assert_eq!(rs.iter().map(|r| r["pp"].as_u64().unwrap()).collect::<Vec<_>>(), [512, 4096, 32768]);
        assert_eq!(rs[0]["prefill_tok_s"]["mean"], 2135.7);
    }

    #[test]
    fn bench_record_rates_match_the_text_rendering() {
        let raw_p = [2144.9, 2129.6, 2132.6];
        let raw_d = [128.9, 129.8, 129.8];
        let (p, d) = (Stat::of(&raw_p), Stat::of(&raw_d));
        let text = format!("pp {:6}: prefill {} tok/s | tg {}: decode {} tok/s", 512, p.text(), 32, d.text());
        let v = parsed(&bench_record(&meta(), &[result(512, p, d)], None, 0));
        for (key, stat) in [("prefill_tok_s", p), ("decode_tok_s", d)] {
            let o = &v["results"][0][key];
            assert_eq!(o["mean"], stat.mean);
            assert_eq!(o["min"], stat.min);
            assert_eq!(o["max"], stat.max);
            // The same decimal strings the text line carries.
            assert!(text.contains(&format!("{:.1}", o["mean"].as_f64().unwrap())));
        }
        assert_eq!(format!("{:.1}", p.mean), "2135.7");
        assert_eq!(format!("{:.1}", p.min), "2129.6");
        assert_eq!(format!("{:.1}", p.max), "2144.9");
        assert!(text.starts_with("pp    512: prefill 2135.7 (2129.6-2144.9) tok/s | tg 32: decode 129.5 (128.9-129.8) tok/s"));
    }

    #[test]
    fn bench_record_omits_expert_without_a_cache() {
        let v = parsed(&bench_record(&meta(), &[result(512, Stat { mean: 1.0, min: 1.0, max: 1.0 }, Stat { mean: 2.0, min: 2.0, max: 2.0 })], None, 0));
        assert!(v.get("expert").is_none());
        assert!(v["results"][0].get("expert").is_none());
    }

    #[test]
    fn bench_record_expert_object_uses_the_printed_numbers() {
        let stats = ExpertStats {
            hits: 1024, misses: 0, swaps: 3, resident: 10240, total: 10240,
            bytes_per_token: 329_006_694, cached: true, all_fast: true,
        };
        let mut r = result(512, Stat { mean: 2135.7, min: 2129.6, max: 2144.9 }, Stat { mean: 129.5, min: 128.9, max: 129.8 });
        r.bytes_per_token = 329_006_694;
        r.decode_median = 129.7;
        let mut m = meta();
        m.backend = "vulkan";
        m.backend_arg = Some("vulkan");
        m.device = &[0];
        let v = parsed(&bench_record(&m, &[r], Some(&stats), 0));
        assert_eq!(v["backend"], "vulkan");
        assert_eq!(v["backend_arg"], "vulkan");
        assert_eq!(v["device"], serde_json::json!([0]));
        let e = &v["expert"];
        assert_eq!(e["hit_rate_pct"], 100.0);
        assert_eq!((e["misses"].as_u64(), e["swaps"].as_u64()), (Some(0), Some(3)));
        assert_eq!((e["resident"].as_u64(), e["total"].as_u64()), (Some(10240), Some(10240)));
        assert_eq!(e["bytes_per_token"], 329_006_694u64);
        assert_eq!(e["mb_per_token"], 313.8);
        let pe = &v["results"][0]["expert"];
        assert_eq!(pe["bytes_per_token"], 329_006_694u64);
        assert_eq!(pe["mb_per_token"], 313.8);
        assert_eq!(pe["gb_per_s"], round1(329_006_694.0 * 129.7 / 1e9));
        assert_eq!(pe["at_decode_tok_s"], 129.7);
    }

    #[test]
    fn bench_record_rejects_json_out_dash() {
        let e = Cli::try_parse_from(["rustingstrata", "bench", "--model", "m.gguf", "--json-out", "-"]).err().unwrap();
        assert!(e.to_string().contains("stdout carries the text block"), "{e}");
        assert!(Cli::try_parse_from(["rustingstrata", "bench", "--model", "m.gguf", "--json-out", "/tmp/x.json"]).is_ok());
    }
}
