# Licenses and third-party notices

Forge9070 contains components with different licenses. **The repository as a
whole is not offered under a single MIT license.** Preserve the notices that
apply to each component when redistributing source or binaries.

| Component | License and provenance |
|---|---|
| Root compatibility application and modified RustingStrata distribution | [RustingStrata License 1.0](LICENSE.md); original author Vasyl Trefilov |
| `vendor/rusting-strata/` | [Original retained license](vendor/rusting-strata/LICENSE.md); pinned source in [UPSTREAM.json](UPSTREAM.json) |
| Independent `crates/forge-engine/` | [MIT](crates/forge-engine/LICENSE-MIT); native Rust implementation and Vulkan shaders |
| GGML quantization codebook values used by the native engine | MIT; source receipts and full notice in [GGML-NOTICE.md](crates/forge-engine/GGML-NOTICE.md) |
| Referenced Avalon memory/cache policies | Pinned reference and MIT notice in [avalon-policy-NOTICE.md](vendor/avalon-policy-NOTICE.md) |

**Uses RustingStrata by Vasyl Trefilov.**

The RustingStrata license permits use, modification and redistribution subject
to its conditions, including attribution and restrictions on selling the engine
itself as a substitute product. Read its full text; the native crate's MIT license
does not replace the license of the surrounding compatibility distribution.

The standalone native engine has no RustingStrata execution dependency. Its
original implementation does not remove obligations for the separate vendored
runtime retained in this repository. [MODIFICATIONS.md](MODIFICATIONS.md)
describes significant project changes, and [SOURCES.md](SOURCES.md) records
reference material. No upstream project, GPU vendor or model publisher is
represented as endorsing Forge9070.

## Rust dependencies and build tools

`Cargo.lock` pins resolved dependency versions. Dependencies and any bundled
third-party material retain their own licenses and copyright notices; the table
above is not an exhaustive dependency license inventory. Review the applicable
package notices when distributing a compiled application or modifying dependencies.

## Models and test data

Pretrained model weights are not bundled. Downloaded models retain their
publishers' licenses and access conditions independently of this application's
license. Model links do not transfer ownership or imply that Forge9070 created
or trained the weights.

The small GGUF files in `tests/fixtures/` are original, untrained numerical and
integration fixtures, not pretrained conversational models. See
[tests/README.md](tests/README.md) for their purpose.
