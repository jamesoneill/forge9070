#!/usr/bin/env python3
"""Offline Nextora benchmark matrix; never downloads models or changes settings."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import time

def run(argv, dest, timeout):
    start = time.monotonic()
    try:
        p = subprocess.run([str(x) for x in argv], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        result = {"exit": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except subprocess.TimeoutExpired:
        result = {"exit": None, "error": "timeout"}
    result["seconds"] = time.monotonic() - start
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result["exit"] == 0

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=pathlib.Path, required=True)
    p.add_argument("--model", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--backend", choices=["cpu", "vulkan", "hybrid"], default="vulkan")
    p.add_argument("--contexts", nargs="+", type=int, default=[4096,8192])
    p.add_argument("--requests", nargs="+", type=int, default=[1,2,4])
    p.add_argument("--timeout", type=int, default=1800)
    a = p.parse_args()
    engine, model = a.engine.resolve(strict=True), a.model.resolve(strict=True)
    if not engine.is_file() or not model.is_file():
        p.error("engine and model must be files")
    if not all(1 <= x <= 4 for x in a.requests) or not all(x >= 256 for x in a.contexts):
        p.error("requests must be 1..4 and contexts at least 256")
    if a.timeout <= 0:
        p.error("timeout must be positive")
    a.output.mkdir(parents=True, exist_ok=False)
    out = a.output.resolve()
    prompt = "Write a Rust function that returns the first duplicate integer in a slice. Explain its complexity and include tests."
    manifest = {"backend": a.backend, "model": model.name, "model_bytes_first_shard": model.stat().st_size,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "contexts": a.contexts, "requests": a.requests,
                "note": "Context is allocated capacity, not occupied prompt length. No physical VRAM or coding correctness claims."}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    run([engine,"doctor"], out/"doctor.json", a.timeout)
    run([engine,"capabilities"], out/"capabilities.json", a.timeout)
    if not run([engine,"inspect","--model",model,"--ctx",min(a.contexts)],out/"inspect.json",a.timeout):
        print("Checkpoint rejected; inspect.json contains the reason. No benchmark attempted.")
        return 2
    failures = 0
    for ctx in dict.fromkeys(a.contexts):
        for requests in dict.fromkeys(a.requests):
            name = f"ctx{ctx}-requests{requests}"
            ok = run([engine,"bench","--model",model,"--backend",a.backend,
                      "--ctx",str(ctx),"--prompt",prompt,"--decode-steps","128",
                      "--reps","5","--warmup","1","--requests",str(requests),
                      "--json-out",out/(name+".report.json")],out/(name+".process.json"),a.timeout)
            failures += not ok
            print(name, "OK" if ok else "FAILED")
    return int(failures > 0)

if __name__ == "__main__":
    raise SystemExit(main())
