# Native engine architecture

Forge9070 contains an independent native engine and a retained compatibility
application. This page describes `forge-engine` 0.2.1. The native service is
also reachable through `forge9070 engine`; ordinary `forge9070 serve` selects
the RX 9070 XT compatibility service.

## Execution policy and hardware

The native runtime keeps CPU, resident Vulkan and diagnostic hybrid execution
behind one model interface. Vulkan adapter eligibility comes from the API
version, compute queues, kernel limits and memory capabilities. Selection does
not require an AMD vendor ID or an RX 9070 XT name. The same native kernels are
the path for eligible AMD, NVIDIA and Intel hardware on Windows and Linux.

`--backend auto` resolves to Vulkan when eligible hardware is available, or CPU
when it is unavailable. This decision happens before host-memory planning, so
CPU state and expert caches receive the correct budget. Reports retain both the
requested policy and the actual backend, with a selection reason. Supplying an
explicit device, nonzero VRAM cap or software-Vulkan test permission requires
strict device selection. Explicit CPU/Vulkan/hybrid choices also remain strict.
Model, device-creation, shader and allocation failures are returned to the caller.

Automatic selection is a hardware policy, not a new kernel implementation.
There are no native CUDA, HIP or Metal backends. Multi-vendor capability checks
do not substitute for physical GPU execution, model acceptance or benchmarking.
See [hardware support](HARDWARE-SUPPORT.md) for requirements and validation scope.

## Model storage and expert execution

```mermaid
flowchart TD
    Files["Validated GGUF shard set"] --> Host["Bounded owned expert RAM"]
    Files --> PLE["Lazy PLE embedding rows"]
    Host --> GPU["Bounded GPU expert cache"]
    Host --> CPU["CPU expert computation"]
    GPU --> Combine["Combine all selected experts"]
    CPU --> Combine
    PLE --> State["Sequence state"]
    Combine --> State
```

This is a data-flow diagram, not a promise that transfer and compute overlap.
Expert leases protect entries while in use. Prefill misses do not evict the
protected decode working set. Cache budgets account for owned payload; other
runtime and OS allocations still need headroom. The engine retains every
selected expert. When its packed working set cannot fit the GPU expert budget,
the native path can compute that expert on the CPU and counts the fallback.

GGUF metadata, shards, tensor dimensions and quantization formats are checked
before loading. The selected checkpoint still has to allocate and execute
successfully. Sparse embedding tables remain separately mapped and accessed
as required. See [the native guide](NATIVE-ENGINE.md) and
[the source comparison](AVALON-COMPARISON.md).

## Service lifecycle

1. The service can start without a model, exposing its configuration panel;
   a fresh unloaded configuration defaults to the `auto` policy.
2. The catalog scans a bounded folder tree and validates complete model sets.
3. Configuration changes require an idle service. Loading applies the model
   and runtime settings; a failure attempts restoration of the previous state.
4. Requests validate their model ID, prompt/output budget and supported options
   before inference. Independent sequence slots share the loaded weights.
5. Busy requests wait in a bounded queue. Saturated queues reject additional
   work. Cancellation and request state remain per sequence.
6. Successful settings changes can persist to the explicit profile path.
   Credentials are not persisted by the configuration API.

The default UI and API listen on loopback at port 9070. The embedded UI has no
frontend build or CDN dependency. Tool-call output is passed to the client;
the inference service does not execute arbitrary client tools.

## Runtime boundaries

| Boundary | Meaning |
| --- | --- |
| One loaded checkpoint | A model ID in a chat request does not download or silently switch models. |
| One to four active slots | Independent sequence capacity, subject to state and device budgets; not continuous batching. |
| Vulkan adapter | Capability-based eligibility is distinct from actual kernel execution and model fit. |
| Automatic policy | Resolves before loading; records its choice and does not mask load or execution failures. |
| WSL process type | A Linux process uses Linux paths and drivers; a Windows executable launched from WSL remains a Windows process. |
| CPU/software Vulkan tests | Useful numerical and service checks, not physical GPU performance evidence. |
| Model file size | Disk footprint; not the same quantity as owned cache size or peak memory use. |

## Source map

The native crate is [`crates/forge-engine`](../crates/forge-engine).
Server control and UI code are under its `src/server` and `src/ui` directories.
The root [`src`](../src) holds the compatibility frontend and hardware profile;
[`vendor/rusting-strata`](../vendor/rusting-strata) retains the pinned upstream
engine and its original terms. [Third-party notices](../THIRD_PARTY_NOTICES.md)
explain those licensing boundaries.
