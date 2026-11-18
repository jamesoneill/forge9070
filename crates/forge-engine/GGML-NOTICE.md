# GGML quantization codebook attribution

`src/quant_tables.rs` contains the IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL and sign
codebook values from GGML. They are format data used by the independent Rust
block decoders. The scalar/SIMD execution and tensor/cache code are Forge code.
No RustingStrata code or runtime is incorporated in this crate.

Primary source fetched 17 November 2026:
https://raw.githubusercontent.com/ggml-org/ggml/master/src/ggml-common.h
SHA-256: `0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c`

Block format equations checked against:
https://raw.githubusercontent.com/ggml-org/ggml/master/src/ggml-quants.c
SHA-256: `bcda33625707fdfcb1367cdd2bb1a2f017430162ec9623abc84ad17459698f73`

The original MIT permission and copyright notice follows.

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
