> **Historical compatibility guide:** The previous Strata-based application
> and port 8080 have been retired from ForgeAI. The original documentation
> below is retained solely as migration reference and may describe commands,
> cache settings or API behaviour that the native Rust service does not offer.
> For the active runtime, use [the native guide](NATIVE-ENGINE.md) and
> [migration details](COMPATIBILITY.md).

# Large code projects with Forge9070 0.6

**Scope:** this guide documents the existing compatibility application. For the
independent Forge 0.9 runtime, its Vulkan/CPU settings, coding API and queue,
use [NATIVE-ENGINE.md](NATIVE-ENGINE.md). Flags and endpoints differ.

This release adds explicit coding presets, complete-request token checks,
query-focused local repository excerpts and corrected conversation-cache
handling. It is a starting configuration for the target RX 9070 XT / approximately
62 GB host, not a measured hardware optimum or an autonomous coding agent.

## Choose a context and keep room for the answer

| Setting | `general` | `code` | `code-long` |
| --- | ---: | ---: | ---: |
| Total context tokens | 4,096 | 16,384 | 32,768 |
| Default server output cap | 1,024 | 4,096 | 8,192 |
| Prefill chunk | 256 | 256 | 256 |
| KV representation | F16 | F16 | F16 |
| Concurrent generation slots | 1 | 1 | 1 |
| MTP drafts | 0 | 0 | 0 |
| Parked conversation RAM | 512 MiB | 1,024 MiB | 1,024 MiB |
| OS/application-growth RAM allowance | 4,096 MiB | 6,144 MiB | 6,144 MiB |
| Owned expert RAM | auto | auto | auto |

Explicit flags override a preset. The preset does not select or download a
model, alter checkpoint precision, invent GPU support or measure throughput.
For Qwen3.8, one generation slot and MTP disabled remain required.

The coding presets also select Qwen3.8's published thinking-mode sampling
values: temperature 1.0, top_p 0.95 and top_k 20. Other sampling penalties remain
at their implemented neutral values. API request parameters override these
server defaults. A browser's existing explicitly saved sampling settings are
retained. These defaults are intended for Qwen3.8; compare appropriate settings
when switching to a different family or a community derivative.

Start with `code` for a first hardware check. For larger edits and analysis,
use `code-long` once loading and ordinary generation work:

```powershell
$Forge = 'C:\code\forge9070-0.6.0\target-windows\x86_64-pc-windows-msvc\release\forge9070.exe'
& $Forge profile --preset code-long
& $Forge serve --models-dir 'C:\Models' --preset code-long
```

Open `http://localhost:8080` and select the downloaded model. The browser's
engine settings also offer explicit coding presets. A context change needs a
model reload because it changes GPU state allocation. The server's output cap
is established at startup; a browser control cannot raise that cap beyond the
running server's value.

Context includes system instructions, tools, code excerpts, conversation
history, reasoning and the final answer. The 32,768-token preset reserves up to
8,192 output tokens when a request asks for that amount; its rendered prompt
must then fit within 24,576 tokens. Reasoning uses the same output allowance as
visible code, so a long reasoning trace can consume a substantial part of it.

An explicit request that exceeds the server output cap or whose complete
rendered prompt plus requested output does not fit receives an actionable
HTTP 400 error before inference. Forge does not silently remove source or
shorten an explicit output budget. A request omitting its output limit uses a
default bounded by the server cap and remaining space. A normally completed
generation can still reach its requested limit; inspect `finish_reason`.

The model advertises 262,144 native context tokens, but this Forge build caps
configured context at 131,072 and also checks the checkpoint's declared limit.
The API and browser use the same checks. The advertised model maximum does not
certify GPU allocation, numerical behavior or useful speed on this machine.
No RoPE or YaRN override is added by these presets.

## What the extra context costs

For the inspected Qwen3.8 geometry (12 QSA layers, two KV heads and head size
256), F16 K/V payload is 24,576 bytes per configured token per sequence:

| Context | F16 K/V payload | Additional pooled index keys | One full parked prefix payload |
| --- | ---: | ---: | ---: |
| 16,384 | 384 MiB | 24 MiB | about 521 MiB |
| 32,768 | 768 MiB | 48 MiB | about 929 MiB |
| 65,536 | 1,536 MiB | 96 MiB | about 1,745 MiB |

These are source-derived allocation/payload estimates, not measured total
VRAM or process memory. Dense weights, expert VRAM slots, recurrent state,
prefill scratch, system-prefix snapshots and driver reserves add to them.
The complete calculation is in
[context-memory-estimates-0.6.0.json](validation/context-memory-estimates-0.6.0.json).

The old 512 MiB parked-cache allowance could not hold one completely filled
16K Qwen prefix. The new 1 GiB coding allowance can hold one full 32K snapshot
by payload size, not many independent long conversations. Live contiguous
conversation reuse is distinct from these parked snapshots.

Longer context reduces space for GPU experts. To try 64K deliberately:

```powershell
& $Forge serve --models-dir 'C:\Models' --preset code-long --ctx 65536 --cache-ram 2048
```

This is an optional experiment requiring a real load and workload test. It
does not establish that 64K is faster or that all allocations will fit. Keep
F16 as the accuracy baseline; Q8 KV is an explicit precision change and needs
its own answer-quality comparison.

Start the IDE, containers and normal development services before loading the
model so the available-memory calculation sees them. The 6 GiB coding margin
leaves additional growth room but is not an OS reservation. It reduces the
expert budget compared with the earlier 4 GiB setting. If the development
workload is stable, `--ram-margin 4096` is an explicit way to trade some of that
headroom back to experts. Watch paging and measured latency during builds.

## Build useful context from a large repository

`context` creates a local UTF-8 pack with the task, relevant code excerpts,
repository-relative paths and line ranges. It uses lexical query, identifier
and path matching. It does not create embeddings, parse a complete symbol
graph, run repository code or send files to an external service.

Use the actual first-shard path of the selected GGUF as `$Model`:

```powershell
$Model = 'C:\Models\REPLACE-WITH-THE-ACTUAL-FIRST-SHARD-PATH.gguf'

& $Forge context --repo 'C:\code\MyLargeProject' --query 'Trace order cancellation, transaction boundaries and refund retries' --model $Model --preset code-long --include 'src/Orders' --include 'tests/Orders' --output '.\repo-context.txt'
```

The placeholder must be replaced with the downloaded local path. A model
tokenizer is sufficient; this command does not initialize Vulkan or run
inference. It counts the whole rendered pack, including labels and the query,
using that tokenizer. The default `code-long` pack allowance is 22,528 tokens:
32,768 total minus 8,192 output minus 2,048 reserved for other prompt material.
Its independent UTF-8 byte ceiling can make the result smaller.

That 2,048-token allowance is a planning value, not an exact count of a coding
client's system prompt, tool definitions or existing history. Increase
`--reserve-other` or reduce `--budget-tokens` for a client with larger overhead.
The server's request preview is the exact check for the complete chat request.

Useful controls:

- `--include PATH` prioritizes eligible files or a directory; it does not
  override path containment, exclusions or the final budget.
- `--include-untracked` includes eligible untracked files as well as tracked
  working-tree files. Tracked files include their current edits.
- `--max-files 24` and `--scan-mib 128` bound default selection and scan work.
  Scan limits and omitted material are reported; absence from a pack is not
  proof that a repository has no matching implementation.
- `--budget-tokens N` requires `--model`; without a model, only the byte ceiling
  is enforced and the token count is explicitly unknown.
- `--json` includes selection and budget metadata. `--output` writes the text
  pack; otherwise it is printed to stdout.

Common credential filenames, binary files, generated/build/dependency
directories and symlinks are excluded. This is not a comprehensive secret
scanner. Review the resulting pack before giving it to a model or a remote
coding client. Everything produced by the command stays local until you use it.

For a large monorepo, supply the relevant project directory as `--repo` to
scope work, and use concrete subsystem/symbol names in the query. Refresh the
pack after significant code changes. Reuse stable project instructions and
architecture notes across requests, and add the current files, callers,
interfaces, tests, errors and diffs needed for the task. Pack selection is
bounded retrieval, not proof of complete repository understanding.

## Check the complete request without inference

`POST /api/context` accepts the same text chat body as
`POST /v1/chat/completions`. It applies the same template and tokenizer and
reports the actual prompt count, output allowance, server cap, remaining
space and whether the request fits. It does not call the model runner.

```powershell
$body = @{
  messages = @(
    @{ role = 'system'; content = 'Use the supplied repository evidence. Preserve public APIs unless the task requires a change. Identify missing context and propose tests.' },
    @{ role = 'user'; content = (Get-Content '.\repo-context.txt' -Raw -Encoding UTF8) }
  )
  max_tokens = 8192
  temperature = 1.0
  top_p = 0.95
  top_k = 20
  chat_template_kwargs = @{ enable_thinking = $true; preserve_thinking = $true }
} | ConvertTo-Json -Depth 10

Invoke-RestMethod 'http://localhost:8080/api/context' -Method Post -ContentType 'application/json' -Body $body
```

Use the configured bearer key if the server requires one. A valid preview can
return HTTP 200 with `fits: false` and a budget error. Only submit the complete
chat request when it fits. The preview covers text; image-bearing previews are
rejected rather than reporting an approximate token count as exact.

The browser performs the same budget check before sending. It preserves
earlier reasoning content when requested and no longer trims the assistant's
trailing code whitespace from continuation history.

## Reuse prefixes and measure the right workload

An unchanged system prompt and stable tool definitions give the cache a
reusable prefix. Keep those before changing task-specific excerpts. Do not
expect a cache hit when earlier messages, template settings or source bytes
change. Preserving earlier reasoning can improve continuity and prefix reuse,
but it also occupies context; start a new task with explicit project notes
when the history is no longer relevant. Forge does not silently summarize or
discard conversation history.

Optional persistent GPU conversation-state storage is exposed with
`--prompt-cache PATH`. Version-2 cache files check model/shard metadata identity,
adapter contents/scales, runtime layout and payload integrity; previous cache
formats are rejected. Model identity is not a complete multi-gigabyte weight
checksum: changing weights while preserving path, length and timestamp requires
clearing the cache. State may contain recoverable prompt/token information, so
keep the file in an appropriate local directory. Persistence is disabled unless
requested; ordinary in-process GPU prefix reuse does not require a cache file.

Measure representative code prompts before deciding on larger prefill chunks
or a different CPU worker count. Stop the GPU server before a separate tuning
run so the two processes do not compete:

```powershell
& $Forge tune --model $Model --ctx 32768 --kv f16 --ram-margin 6144 --cache-ram 1024 --prompt-file '.\repo-context.txt' --tg 128 --reps 3 --warmup 1 --json-out '.\code-tuning.json'
```

The tuner reads the actual workload, rejects a prompt that does not fit, holds
precision fixed and compares candidates against the greedy baseline outputs.
It records both prefill and decode timings; it does not certify answer quality
or choose every possible context/memory/thread setting. Use tests and builds
from the target project to evaluate proposed code changes.

Forge is the local inference server and context packer. A coding client still
owns file editing, tool execution, builds and tests. Its configured context and
output budgets should match the server. Both original and uncensored models
use these controls; their answer quality needs separate evaluation.

## Sources and validation

- [Qwen3.8 official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next):
  native context, thinking-mode sampling and preserved thinking.
- `vendor/rusting-strata/src/gpu_model.rs`: actual cache/state allocation.
- [context-memory-estimates-0.6.0.json](validation/context-memory-estimates-0.6.0.json): source-derived sizing.
- [VALIDATION.md](../VALIDATION.md): observed tests and hardware limits.

Uses RustingStrata by Vasyl Trefilov.
