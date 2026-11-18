#!/usr/bin/env python3
"""Generate deterministic, untrained Qwen3 GGUFs used by Forge9070 tests.

The default is a one-layer dense transformer; --moe builds a four-expert,
top-2-routed MoE. Both have nonzero attention and FFN weights. The output
projection intentionally favours the byte token ``A``.
It exercises the engine's GGUF reader, tokenizer, CPU attention/KV state,
normalization, FFN, sampling and serving paths. It is not a useful language
model, a performance benchmark, or a test of GPU numerical correctness.
No third-party model weights or Python packages are needed.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import struct


def u32(value: int) -> bytes:
    return struct.pack("<I", value)


def u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return u64(len(raw)) + raw


def aligned(data: bytes) -> bytes:
    return data + bytes((-len(data)) % 32)


def byte_token(byte: int) -> str:
    # GPT-2's reversible byte-to-Unicode alphabet, indexed here by raw byte.
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    missing = [value for value in range(256) if value not in visible]
    return chr(byte if byte in visible else 256 + missing.index(byte))


def build_fixture(moe: bool = False) -> bytes:
    dim, hidden = 32, 64
    arch = "qwen3moe" if moe else "qwen3"
    tokens = [byte_token(i) for i in range(256)] + [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"
    ]
    meta: list[bytes] = []

    def add_string(key: str, value: str) -> None:
        meta.append(string(key) + u32(8) + string(value))

    def add_int(key: str, value: int) -> None:
        meta.append(string(key) + u32(4) + u32(value))

    def add_float(key: str, value: float) -> None:
        meta.append(string(key) + u32(6) + struct.pack("<f", value))

    def add_strings(key: str, values: list[str]) -> None:
        meta.append(string(key) + u32(9) + u32(8) + u64(len(values))
                    + b"".join(string(value) for value in values))

    add_string("general.architecture", arch)
    add_string("general.name", "Forge9070 synthetic CPU test fixture (untrained)")
    add_int("general.alignment", 32)
    add_int(f"{arch}.block_count", 1)
    add_int(f"{arch}.embedding_length", dim)
    add_int(f"{arch}.feed_forward_length", hidden)
    add_int(f"{arch}.attention.head_count", 1)
    add_int(f"{arch}.attention.head_count_kv", 1)
    add_int(f"{arch}.attention.key_length", dim)
    add_int(f"{arch}.rope.dimension_count", dim)
    add_int(f"{arch}.context_length", 512)
    add_float(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6)
    add_float(f"{arch}.rope.freq_base", 10000.0)
    if moe:
        add_int(f"{arch}.expert_count", 4)
        add_int(f"{arch}.expert_used_count", 2)
        add_int(f"{arch}.expert_feed_forward_length", hidden)
        add_int(f"{arch}.expert_weights_norm", 1)
    add_string("tokenizer.ggml.model", "gpt2")
    add_string("tokenizer.ggml.pre", "qwen2")
    add_strings("tokenizer.ggml.tokens", tokens)
    add_strings("tokenizer.ggml.merges", [])
    types = [1] * 256 + [3] * 5
    meta.append(string("tokenizer.ggml.token_type") + u32(9) + u32(4)
                + u64(len(types)) + b"".join(u32(value) for value in types))
    add_int("tokenizer.ggml.eos_token_id", 256)

    tensors: list[tuple[str, list[int], bytes]] = []

    def tensor(name: str, dims: list[int], values: list[float]) -> None:
        count = 1
        for size in dims:
            count *= size
        assert count == len(values), name
        tensors.append((name, dims, struct.pack(f"<{count}f", *values)))

    def diagonal(rows: int, cols: int, scale: float) -> list[float]:
        return [scale if col == row % cols else 0.0
                for row in range(rows) for col in range(cols)]

    tensor("token_embd.weight", [dim, len(tokens)], [
        0.5 + ((token * 7 + col * 3) % 13) * 0.025
        for token in range(len(tokens)) for col in range(dim)
    ])
    tensor("output_norm.weight", [dim], [1.0] * dim)
    tensor("output.weight", [dim, len(tokens)], [
        1.0 if token == ord("A") else 0.0
        for token in range(len(tokens)) for _ in range(dim)
    ])
    for name in ["attn_norm", "attn_q_norm", "attn_k_norm", "ffn_norm"]:
        tensor(f"blk.0.{name}.weight", [dim], [1.0] * dim)
    for name in ["attn_q", "attn_k", "attn_v", "attn_output"]:
        tensor(f"blk.0.{name}.weight", [dim, dim], diagonal(dim, dim, 0.125))
    if moe:
        # Four nonidentical experts; actual top-2 routing and weighted FFNs run.
        tensor("blk.0.ffn_gate_inp.weight", [dim, 4], [
            (expert + 1) * 0.015625 if col % 4 == expert else 0.0
            for expert in range(4) for col in range(dim)
        ])
        for name in ["ffn_gate_exps", "ffn_up_exps"]:
            tensor(f"blk.0.{name}.weight", [dim, hidden, 4], [
                value for expert in range(4)
                for value in diagonal(hidden, dim, 0.0625 * (expert + 1))
            ])
        tensor("blk.0.ffn_down_exps.weight", [hidden, dim, 4], [
            value for expert in range(4)
            for value in diagonal(dim, hidden, 0.0625 * (expert + 1))
        ])
    else:
        for name in ["ffn_gate", "ffn_up"]:
            tensor(f"blk.0.{name}.weight", [dim, hidden], diagonal(hidden, dim, 0.125))
        tensor("blk.0.ffn_down.weight", [hidden, dim], diagonal(dim, hidden, 0.125))

    data = b""
    descriptions = b""
    for name, dims, values in tensors:
        descriptions += string(name) + u32(len(dims))
        descriptions += b"".join(u64(size) for size in dims)
        descriptions += u32(0) + u64(len(data))  # GGML F32
        data += aligned(values)
    header = b"GGUF" + u32(3) + u64(len(tensors)) + u64(len(meta))
    return aligned(header + b"".join(meta) + descriptions) + data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "tests" / "fixtures" / "tiny-qwen3.gguf")
    parser.add_argument("--moe", action="store_true", help="Build the four-expert, top-2 routed Qwen3MoE fixture")
    args = parser.parse_args()
    data = build_fixture(args.moe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    print(f"{args.output}: {len(data)} bytes; sha256={hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
