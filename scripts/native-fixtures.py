"""Deterministic synthetic GGUF fixtures shared by native validation scripts.

These helpers preserve the fixture transformations previously used by the API
and scheduling checks. They do not import or launch the retired compatibility
runtime. Untrained fixture output tests serialization and admission behavior;
it does not establish pretrained-model quality or hardware performance.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]

TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read one source file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}


TOOL_TEXT = (
    '<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/main.rs\n'
    '</parameter>\n</function>\n</tool_call>'
)


def check(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def fixture_copy(source: Path, destination: Path, tool_response: bool) -> None:
    """Rewrite only this known synthetic fixture's context and optional output row."""
    raw = source.read_bytes()
    check(raw[:8] == b"GGUF" + struct.pack("<I", 3), "expected a GGUF v3 fixture")
    tensors, metadata = struct.unpack_from("<QQ", raw, 8)
    pos = 24

    def string() -> str:
        nonlocal pos
        length, = struct.unpack_from("<Q", raw, pos)
        pos += 8
        value = raw[pos:pos + length].decode()
        pos += length
        return value

    def pack_string(value: str) -> bytes:
        text = value.encode()
        return struct.pack("<Q", len(text)) + text

    def value(kind: int):
        nonlocal pos
        if kind == 8:
            return string()
        if kind == 9:
            child, count = struct.unpack_from("<IQ", raw, pos)
            pos += 12
            return [value(child) for _ in range(count)]
        formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
                   6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        fmt = "<" + formats[kind]
        result, = struct.unpack_from(fmt, raw, pos)
        pos += struct.calcsize(fmt)
        return result

    entries = []
    output_token = None
    for _ in range(metadata):
        start = pos
        key = string()
        kind, = struct.unpack_from("<I", raw, pos)
        pos += 4
        data = value(kind)
        if key.endswith(".context_length"):
            entries.append(pack_string(key) + struct.pack("<II", 4, 4096))
        elif key == "tokenizer.ggml.tokens" and tool_response:
            check(data[-1] == "</think>", "unexpected fixture vocabulary")
            output_token = len(data) - 1
            data[-1] = TOOL_TEXT
            entries.append(pack_string(key) + struct.pack("<IIQ", 9, 8, len(data))
                           + b"".join(pack_string(token) for token in data))
        else:
            entries.append(raw[start:pos])
    descriptors = pos
    output = None
    for _ in range(tensors):
        name = string()
        ndims, = struct.unpack_from("<I", raw, pos)
        pos += 4
        dims = struct.unpack_from("<" + "Q" * ndims, raw, pos)
        pos += 8 * ndims
        dtype, offset = struct.unpack_from("<IQ", raw, pos)
        pos += 12
        if name == "output.weight":
            output = (dims, dtype, offset)
    tensor_descriptions = raw[descriptors:pos]
    data_start = (pos + 31) // 32 * 32
    payload = bytearray(raw[data_start:])
    if tool_response:
        check(output is not None and output_token is not None, "missing fixture output")
        dims, dtype, offset = output
        check(dtype == 0 and len(dims) == 2, "fixture output must be F32")
        width, vocab = dims
        for row in range(vocab):
            struct.pack_into("<" + "f" * width, payload, offset + row * width * 4,
                             *([1.0 if row == output_token else 0.0] * width))
    header = raw[:24] + b"".join(entries) + tensor_descriptions
    destination.write_bytes(header + bytes((-len(header)) % 32) + payload)


def events(text: str) -> list[dict]:
    chunks = [line[6:] for line in text.splitlines() if line.startswith("data: ")]
    check(chunks and chunks[-1] == "[DONE]", f"stream did not finish: {text}")
    return [json.loads(chunk) for chunk in chunks[:-1]]


CONTEXT = 32768


LAYERS = 4


def expand_fixture(source: Path, destination: Path) -> dict:
    """Repeat only this known F32 fixture's layer tensors; retain its A output."""
    raw = source.read_bytes()
    check(raw[:8] == b"GGUF" + struct.pack("<I", 3), "fixture must be GGUF v3")
    tensors, metadata = struct.unpack_from("<QQ", raw, 8)
    pos = 24

    def string() -> str:
        nonlocal pos
        length, = struct.unpack_from("<Q", raw, pos)
        pos += 8
        result = raw[pos:pos + length].decode()
        pos += length
        return result

    def packed_string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    def value(kind: int):
        nonlocal pos
        if kind == 8:
            return string()
        if kind == 9:
            child, count = struct.unpack_from("<IQ", raw, pos)
            pos += 12
            return [value(child) for _ in range(count)]
        formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
                   6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        fmt = "<" + formats[kind]
        result, = struct.unpack_from(fmt, raw, pos)
        pos += struct.calcsize(fmt)
        return result

    entries = []
    architecture = None
    for _ in range(metadata):
        start = pos
        key = string()
        kind, = struct.unpack_from("<I", raw, pos)
        pos += 4
        data = value(kind)
        if key == "general.architecture":
            architecture = data
        if key.endswith(".block_count"):
            check(kind == 4 and data == 1, "expected the one-layer synthetic fixture")
            entries.append(packed_string(key) + struct.pack("<II", 4, LAYERS))
        elif key.endswith(".context_length"):
            entries.append(packed_string(key) + struct.pack("<II", 4, CONTEXT))
        else:
            entries.append(raw[start:pos])
    check(architecture == "qwen3moe", "scheduling fixture must use the synthetic Qwen3MoE model")
    descriptions = []
    for _ in range(tensors):
        name = string()
        ndims, = struct.unpack_from("<I", raw, pos)
        pos += 4
        dims = struct.unpack_from("<" + "Q" * ndims, raw, pos)
        pos += 8 * ndims
        dtype, offset = struct.unpack_from("<IQ", raw, pos)
        pos += 12
        check(dtype == 0, "only the known synthetic F32 tensors may be expanded")
        descriptions.append((name, dims, offset))
    data_start = (pos + 31) // 32 * 32
    payload = bytearray()
    rewritten = []
    for name, dims, offset in descriptions:
        size = math.prod(dims) * 4
        data = raw[data_start + offset:data_start + offset + size]
        check(len(data) == size, f"truncated fixture tensor {name}")
        names = [name.replace("blk.0.", f"blk.{layer}.", 1) for layer in range(LAYERS)] \
            if name.startswith("blk.0.") else [name]
        for new_name in names:
            rewritten.append(packed_string(new_name) + struct.pack("<I", len(dims))
                             + struct.pack("<" + "Q" * len(dims), *dims)
                             + struct.pack("<IQ", 0, len(payload)))
            payload.extend(data)
            payload.extend(bytes((-len(payload)) % 32))
    header = b"GGUF" + struct.pack("<IQQ", 3, len(rewritten), metadata) \
        + b"".join(entries) + b"".join(rewritten)
    expanded = header + bytes((-len(header)) % 32) + payload
    destination.write_bytes(expanded)
    return {
        "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "expanded_sha256": hashlib.sha256(expanded).hexdigest(),
        "expanded_bytes": len(expanded), "layers": LAYERS,
        "declared_context_tokens": CONTEXT,
        "description": "Untrained F32 CPU Qwen3MoE; repeated real nonzero fixture layers, deterministic A output",
    }
