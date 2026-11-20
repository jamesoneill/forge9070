#!/usr/bin/env python3
"""Explicit software-Vulkan Qwen API/sequence concurrency correctness gate.

Set NEXTORA_VULKAN_TEST=1 and select the intended ICD. This is not a speed test.
"""
import argparse, concurrent.futures, hashlib, importlib.util, json, math, os, subprocess, threading, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_api_helpers',ROOT/'scripts/validate-native-api.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)

def positive_seconds(value):
    try:
        seconds=float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('request timeout must be a positive finite number of seconds') from error
    if not math.isfinite(seconds) or seconds<=0:
        raise argparse.ArgumentTypeError('request timeout must be a positive finite number of seconds')
    return seconds

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--request-timeout',type=positive_seconds,default=120.0,metavar='SECONDS',
                        help='HTTP response deadline for software-Vulkan requests (default: 120 seconds)')
    args=parser.parse_args()
    assert os.environ.get('NEXTORA_VULKAN_TEST')=='1','Explicit NEXTORA_VULKAN_TEST=1 is required'
    binary=args.binary.resolve();model=ROOT/'tests/fixtures/tiny-qwen38.gguf'
    bodies=[helpers.body(model.name,max_tokens=4,messages=[{'role':'user','content':f'{i}'}]) for i in range(4)]
    expected=[];started=time.monotonic()
    with helpers.server(binary,model,parallel=1,context=64,max_tokens=8) as (_,request):
        for body in bodies:
            status,response=helpers.json_call(request,'/v1/chat/completions',body)
            assert status==200,response
            expected.append((response['choices'][0]['message'],response['usage']['completion_tokens']))
    options=['--backend','vulkan','--allow-software-vulkan','--vram-mib','256']
    with helpers.server(binary,model,parallel=4,context=64,max_tokens=8,backend=options,request_timeout=args.request_timeout) as (_,request):
        barrier=threading.Barrier(5)
        def run(body):
            barrier.wait(timeout=10)
            return helpers.json_call(request,'/v1/chat/completions',body)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(run,body) for body in bodies]
            barrier.wait(timeout=10)
            actual=[future.result(timeout=args.request_timeout+30) for future in futures]
        for i,(status,response) in enumerate(actual):
            assert status==200,response
            assert (response['choices'][0]['message'],response['usage']['completion_tokens'])==expected[i],(i,response,expected[i])
        stats=helpers.wait_for(request,lambda x:x['admission']['active']==0 and x['admission']['queued']==0)
        assert stats['admission']['active_high_water']==4,stats
        runtime=stats['runtime'];execution=runtime['execution'];memory=execution['memory']
        assert execution['backend']=='vulkan' and memory['cpu_state_operations'] is False,runtime
        assert memory['selected_experts_omitted']==0 and memory['cpu_expert_fallbacks']==0,runtime
        assert memory['completed_tokens']>0 and memory['gpu_expert_executions']>0,runtime
    project_version=next(line.split('"')[1] for line in (ROOT/'Cargo.toml').read_text().splitlines() if line.startswith('version = "'))
    engine_version=subprocess.run([str(binary),'--version'],capture_output=True,text=True,encoding='utf-8',check=True,timeout=10).stdout.strip()
    report={'status':'passed','version':project_version,'engine_version':engine_version,
        'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
        'cases':4,'elapsed_seconds':time.monotonic()-started,
        'request_timeout_seconds':args.request_timeout,
        'scope':'Four simultaneous Qwen3.8 HTTP requests on independent device states; exact synthetic CPU answer/usage comparison; no hardware speed or trained quality claim',
        'software_vulkan_allowed_for_test':True,'target_amd_hardware_verified':False,
        'checks':['Four actual active requests observed','Each response/usage matches independent CPU execution',
                  'Device HC/DeltaNet/PLE/QSA state executes','All selected experts execute on GPU in fitting budget',
                  'All permits released after completion'],
        'runtime':runtime,'admission':stats['admission']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
