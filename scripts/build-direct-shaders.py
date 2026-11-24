#!/usr/bin/env python3
"""Reproduce Nextora's direct expert shaders without modifying legacy modules.

The AMD driver compiled the same four-factor Q6_K expression differently in
legacy and descriptor-array modules. An isolated hardware comparison established
that two rounded products (activation scale * integer dot, then signed subscale)
followed by fma(product, half scale, accumulator) reproduces the original shader.
This narrowly scoped SPIR-V build pass fixes that evaluation order. GLSL `precise`
propagates through storage/reduction dataflow in glslang 15, so it cannot express
this local constraint without also changing unrelated operations.

Requires glslangValidator and SPIRV-Tools. The default expected hash is the
hardware-validated module, so another compiler version fails closed rather than
silently replacing the checked shader. Updating that hash requires validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FAST_SHA256 = "e633b35f03cd65e1d30f61ee9ab62a484c620f4f91bca6e1fa980a031333bfab"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def patch_q6(text: str) -> str:
    """Only rewrite the verified fast_q6_k_partial floating-point expression."""
    marker = "%fast_q6_k_partial_u1_u1_u1_u1_ = OpFunction "
    require(text.count(marker) == 1, "expected one named Q6_K fast-dot function")
    start = text.index(marker)
    end = text.index("OpFunctionEnd", start) + len("OpFunctionEnd")
    body = text[start:end]
    require("NoContraction" not in text, "input already has precision annotations")
    definitions = {
        m.group(1): (m.group(2), m.group(3), m.group(4).split())
        for m in re.finditer(r"^\s*(%\S+) = (Op\S+) (%\S+) (.*)$", body, re.M)
    }
    counts = Counter(op for op, _, _ in definitions.values() if op.startswith("OpF") and op != "OpFunction")
    require(counts["OpFMul"] == 3 and counts["OpFAdd"] == 1,
            "Q6_K function no longer has the expected three products and one accumulation")
    require(not any(op in counts for op in ("OpFSub", "OpFDiv", "OpFmaKHR")),
            "unexpected Q6_K floating-point arithmetic")
    require(" Fma " not in body, "Q6_K function already contains a fused operation")
    stores = re.findall(r"OpStore (%partial_\d+) (%\S+)\s*\n\s*OpBranch", body)
    require(len(stores) == 1, "expected one Q6_K loop accumulator update")
    partial_variable, output = stores[0]
    op, float_type, args = definitions[output]
    require(op == "OpFAdd" and len(args) == 2, "unexpected Q6_K accumulator operation")
    partial, last = args
    require(definitions[partial] == ("OpLoad", float_type, [partial_variable]),
            "Q6_K accumulation no longer loads its original partial")

    def product(identifier: str) -> tuple[str, str]:
        opcode, ty, operands = definitions[identifier]
        require(opcode == "OpFMul" and ty == float_type and len(operands) == 2,
                "unexpected Q6_K product chain")
        return operands[0], operands[1]

    scaled_activation, products = product(last)
    scaled_half, alpha = product(scaled_activation)
    half, scale = product(scaled_half)
    require(definitions[half][0] == "OpFunctionCall"
            and definitions[half][2][0] == "%half_at_u1_",
            "Q6_K global scale is no longer obtained from the encoded half")
    require(definitions[scale][0] == "OpConvertSToF"
            and definitions[products][0] == "OpConvertSToF"
            and definitions[alpha][0] == "OpLoad",
            "unexpected Q6_K signed subscale, integer dot, or activation scale")
    alpha_pointer = definitions[alpha][2][0]
    require(definitions[alpha_pointer][0] == "OpAccessChain", "unexpected activation-scale pointer")
    alpha_buffer = definitions[alpha_pointer][2][0]
    require(re.search(r"OpDecorate " + re.escape(alpha_buffer) + r" Binding 4\b", text) is not None,
            "Q6_K activation scale must come from binding4")
    imports = re.findall(r'(%\S+) = OpExtInstImport "GLSL.std.450"', text)
    require(len(imports) == 1, "expected one GLSL standard instruction set")
    require("%factor0" not in text and "%factor1" not in text,
            "reserved Q6_K build-pass IDs already exist")

    target = re.search(r"^\s*" + re.escape(last) + r" = OpFMul .*$", body, re.M)
    require(target is not None, "missing original Q6_K final product")
    # Keep the original dead products in the module for exact, reproducible
    # compiler input; the driver eliminates them. Their annotations are local.
    replacement = (f" %factor0 = OpFMul {float_type} {alpha} {products}\n"
                   f" %factor1 = OpFMul {float_type} %factor0 {scale}\n"
                   f" {last} = OpFMul {float_type} %factor1 {half}")
    body = body.replace(target.group(), replacement)
    body, changed = re.subn(r"^\s*" + re.escape(output) + r" = OpFAdd .*$",
                           f" {output} = OpExtInst {float_type} {imports[0]} Fma %factor1 {half} {partial}",
                           body, flags=re.M)
    require(changed == 1, "Q6_K accumulation replacement was not unique")
    result = text[:start] + body + text[end:]
    annotated = re.findall(r"^\s*(%\S+) = OpF(?:Mul|Add|Sub)\b", body, re.M) + [output]
    require(len(annotated) == len(set(annotated)) == 6, "unexpected Q6_K precision annotation count")
    annotations = "".join(f" OpDecorate {identifier} NoContraction\n" for identifier in annotated)
    types = re.search(r"^.*= OpType\S+.*$", result, re.M)
    require(types is not None, "missing SPIR-V type section")
    return result[:types.start()] + annotations + result[types.start():]


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shader-dir", type=Path, default=ROOT / "crates/nextora-engine/shaders")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "crates/nextora-engine/src/vulkan")
    parser.add_argument("--tool-dir", type=Path, help="Directory containing glslangValidator and SPIRV-Tools")
    parser.add_argument("--expected-fast-sha256", default=EXPECTED_FAST_SHA256)
    args = parser.parse_args()
    shader_dir = args.shader_dir.resolve()
    output_dir = args.output_dir.resolve()
    require(re.fullmatch(r"[0-9a-f]{64}", args.expected_fast_sha256) is not None, "invalid expected shader hash")

    def tool(name: str) -> str:
        location = str(args.tool_dir / name) if args.tool_dir else shutil.which(name)
        require(location is not None, f"{name} is not installed")
        return str(location)

    with tempfile.TemporaryDirectory(prefix="nextora-direct-shaders-") as directory:
        work = Path(directory)
        base = work / "ops_direct.spv"
        run(tool("glslangValidator"), "-V", "--target-env", "vulkan1.1",
            "-I" + str(shader_dir), "-DNEXTORA_DIRECT_EXPERTS",
            str(shader_dir / "ops.comp"), "-o", str(base))
        outputs = [base]
        for name, extra_macros, expected in [
            ("ops_fast_direct", [], args.expected_fast_sha256),
            ("expert_words", ["-DNEXTORA_EXPERT_WORDS"], "cf76d4f031bd549279185fda79a8600a1f6cc1e66ad3e9e62c8e83015fa266e3"),
        ]:
            raw = work / (name + ".raw.spv")
            run(tool("glslangValidator"), "-V", "--target-env", "vulkan1.1",
                "-I" + str(shader_dir), "-DNEXTORA_FAST_DOT", "-DNEXTORA_DIRECT_EXPERTS",
                *extra_macros, str(shader_dir / "ops.comp"), "-o", str(raw))
            assembly, revised = work / "original.spvasm", work / "q6_exact.spvasm"
            run(tool("spirv-dis"), str(raw), "-o", str(assembly))
            revised.write_text(patch_q6(assembly.read_text()))
            final = work / (name + ".spv")
            run(tool("spirv-as"), "--target-env", "vulkan1.1", str(revised), "-o", str(final))
            run(tool("spirv-val"), "--target-env", "vulkan1.1", str(final))
            actual = hashlib.sha256(final.read_bytes()).hexdigest()
            require(actual == expected,
                    f"{name} shader hash changed: {actual}; expected {expected}. "
                    "Validate new compiler output before updating the expected hash.")
            outputs.append(final)
        for name, expected in [
            ("router_f32", "a55f1e9307c720109d102bcfc790c12cdae1c68ba4d3eee9e65a1961dfde4bdb"),
            ("tiled_scores", "0cd2d1db2b8b6a04d132111383d66f4942cb783baebbae27246d9bbf712318b3"),
            ("long_attention", "5510348002d515b30bbb3da9b3e24a91fa2214f85c540f9c6e4847af882f6abe"),
        ]:
            binary = work / (name + ".spv")
            run(tool("glslangValidator"), "-V", "--target-env", "vulkan1.1",
                str(shader_dir / (name + ".comp")), "-o", str(binary))
            run(tool("spirv-val"), "--target-env", "vulkan1.1", str(binary))
            require(hashlib.sha256(binary.read_bytes()).hexdigest() == expected,
                    f"{name} shader changed; validate it before updating its hash")
            outputs.append(binary)
        run(tool("spirv-val"), "--target-env", "vulkan1.1", str(base))
        output_dir.mkdir(parents=True, exist_ok=True)
        for binary in outputs:
            destination = output_dir / binary.name
            shutil.copyfile(binary, destination)
            print(f"{destination}: sha256={hashlib.sha256(binary.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
