# Nextora licences and third-party notices

The Nextora native Rust engine and the legacy command alias are covered by the
[MIT project licence](LICENSE.md). The historic Strata-derived compatibility
runtime has been removed from the build and repository source tree; it is
**not** part of the current inference service or shipped binaries.

| Component | Licence and provenance |
|---|---|
| Native Nextora engine and alias | [MIT](LICENSE.md), plus [engine licence](crates/nextora-engine/LICENSE-MIT) |
| GGML quantization codebook values | MIT, see [GGML notice](crates/nextora-engine/GGML-NOTICE.md) |
| Referenced Avalon cache/memory policy | MIT, see [Avalon notice](vendor/avalon-policy-NOTICE.md) |
| RustingStrata (historical inspiration, no included runtime) | Vasyl Trefilov; [original licence preserved for history](LICENSE-RUSTINGSTRATA-HISTORICAL.md), [pinned upstream metadata](UPSTREAM.json) |

**Credit: RustingStrata by Vasyl Trefilov.** Retaining credit does not imply
the current engine contains or is endorsed by its original implementation.
The earlier restrictive RustingStrata licence still governs copies of
RustingStrata, but it does **not** replace Nextora's MIT licence.

The dependency graph is pinned by `Cargo.lock`. Native release packaging
includes resolved dependency notices and Rust standard-library licences;
preserve every applicable third-party licence when distributing copies.
Bundled tiny GGUF fixtures are synthetic, untrained test data, not pretrained
model weights. Model publishers retain their own licences.

See [source references](SOURCES.md) and [model tests](tests/README.md).
