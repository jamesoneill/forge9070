#!/usr/bin/env python3
"""Fixed one, two and four request sweep using the native Nextora benchmark."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

SLOTS=(1,2,4)

def validate_run(report,slots):
    if report.get("engine")!="nextora-native" or report.get("identical_greedy_checksums") is not True:
        raise ValueError("invalid native benchmark identity or repeated greedy checksums")
    if report.get("concurrent_requests")!=slots:
        raise ValueError("concurrency does not match requested slots")
    backend=report.get("runtime",{}).get("execution",{}).get("backend")
    if backend not in ("cpu","vulkan"):
        raise ValueError("actual backend is not CPU or Vulkan")
    def positive(key):
        value=report.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError("invalid measured value for "+key)
        return float(value)
    return {
        "slots":slots,"backend":backend,
        "adapter":report.get("runtime",{}).get("execution",{}).get("adapter",{}).get("name"),
        "prompt_tokens_per_request":report.get("prompt_tokens_per_request"),
        "decode_steps_per_request":report.get("decode_steps_per_request"),
        "repetitions":report.get("repetitions"),
        "prefill_tokens_per_second":positive("median_aggregate_prefill_tokens_per_second"),
        "decode_steps_per_second":positive("median_aggregate_decode_steps_per_second"),
        "p95_completion_seconds":positive("p95_request_decode_completion_seconds"),
    }

def compare(runs):
    if [r["slots"] for r in runs]!=list(SLOTS):
        raise ValueError("expected ordered 1/2/4 slot samples")
    for field in ("backend","adapter","prompt_tokens_per_request","decode_steps_per_request","repetitions"):
        if len({str(r[field]) for r in runs})!=1:
            raise ValueError("different "+field+" across workloads")
    base=runs[0]
    return {
        "decode_rate_ratios_vs_one":{str(r["slots"]):r["decode_steps_per_second"]/base["decode_steps_per_second"] for r in runs},
        "prefill_rate_ratios_vs_one":{str(r["slots"]):r["prefill_tokens_per_second"]/base["prefill_tokens_per_second"] for r in runs},
        "highest_observed_aggregate_decode_throughput_slots":max(runs,key=lambda r:r["decode_steps_per_second"])["slots"],
        "lowest_observed_p95_completion_slots":min(runs,key=lambda r:r["p95_completion_seconds"])["slots"],
        "proven_optimal":False,
    }

def save_report(path,report):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",dir=path.parent,prefix=".nextora-sweep-",suffix=".tmp",delete=False) as f:
        temporary=Path(f.name)
        try:
            json.dump(report,f,indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--prompt",required=True)
    parser.add_argument("--backend",choices=("cpu","vulkan"),required=True)
    parser.add_argument("--ctx",type=int,default=4096)
    parser.add_argument("--decode-steps",type=int,default=128)
    parser.add_argument("--reps",type=int,default=5)
    parser.add_argument("--warmup",type=int,default=1)
    parser.add_argument("--timeout-seconds",type=int,default=3600)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if not args.prompt or len(args.prompt)>16384:
        parser.error("prompt must be nonempty and <=16384 characters")
    if not(16<=args.ctx<=131072 and 1<=args.decode_steps<args.ctx):
        parser.error("context and decode steps are inconsistent")
    if not(3<=args.reps<=100 and 1<=args.warmup<=20 and 1<=args.timeout_seconds<=86400):
        parser.error("reps/warmup/timeout out of range")
    binary=args.binary.resolve(strict=True)
    model=args.model.resolve(strict=True)
    output=args.output.resolve()
    if not binary.is_file() or not model.is_file() or model.stat().st_size==0:
        parser.error("binary and checkpoint must be valid files")
    if output in (binary,model):
        parser.error("output must not overwrite executable or checkpoint")
    started=time.monotonic()
    runs=[]
    for slots in SLOTS:
        cmd=[str(binary),"bench","--model",str(model),"--backend",args.backend,
             "--ctx",str(args.ctx),"--prompt",args.prompt,
             "--decode-steps",str(args.decode_steps),"--reps",str(args.reps),
             "--warmup",str(args.warmup),"--requests",str(slots)]
        print("Running native Nextora benchmark with "+str(slots)+" active request slots",file=sys.stderr)
        try:
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=args.timeout_seconds,check=False)
        except subprocess.TimeoutExpired:
            raise SystemExit("benchmark timed out with "+str(slots)+" slots")
        if result.returncode:
            raise SystemExit("benchmark failed: "+result.stderr[-3000:])
        try:
            raw=json.loads(result.stdout)
            record=validate_run(raw,slots)
            if record["backend"]!=args.backend:
                raise ValueError("actual backend changed unexpectedly")
        except (ValueError,TypeError,KeyError) as exc:
            raise SystemExit("invalid native benchmark evidence: "+str(exc))
        runs.append(record)
    stat=model.stat()
    result={
        "schema":"nextora.benchmark_sweep.v1",
        "engine":"nextora-native",
        "host_os":platform.system(),
        "host_architecture":platform.machine(),
        "checkpoint_filename":model.name,
        "checkpoint_selected_shard_bytes":stat.st_size,
        "checkpoint_content_hash":None,
        "driver_and_full_shard_set_fingerprint_verified":False,
        "workload":{"ctx":args.ctx,"decode_steps":args.decode_steps,"reps":args.reps,"warmup":args.warmup,
                    "prompt_text_exported":False},
        "runs":runs,"comparison":compare(runs),
        "elapsed_seconds":time.monotonic()-started,
        "scope":"fixed-step native throughput observations, not quality, GPU utilization or guaranteed performance",
        "no_telemetry_upload":True,
    }
    save_report(output,result)
    print(json.dumps({"report":str(output),"comparison":result["comparison"]},indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
