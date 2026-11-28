#!/usr/bin/env python3
"""Compare forced-token full-vocabulary traces exported by two engines.
Input JSON: {"model_sha256": "...", "tokenizer_sha256": "...",
"prompt_ids": [...], "forced_ids": [...], "logits": [[...], ...]}.
Each row scores the prompt plus the preceding forced IDs. No generation alignment
by text, and no claims about runtime speed. Exporters must use identical weights.
"""
import argparse
import json
import math
from pathlib import Path

def compare(a,b,atol,rtol):
    for key in ("model_sha256","tokenizer_sha256","prompt_ids","forced_ids"):
        if key not in a or key not in b or a[key]!=b[key]:
            raise ValueError("trace identity mismatch: "+key)
    for key in ("model_sha256","tokenizer_sha256"):
        if not isinstance(a[key],str) or len(a[key])!=64 or any(c not in "0123456789abcdef" for c in a[key]):
            raise ValueError("invalid digest: "+key)
    x,y=a["logits"],b["logits"]
    if not x or len(x)!=len(y) or len(x)!=len(a["forced_ids"])+1:
        raise ValueError("trace length mismatch")
    worst=0.; failures=0; agreed=0; total=0
    vocab=len(x[0])
    if vocab==0: raise ValueError("empty vocabulary")
    for u,v in zip(x,y):
        if len(u)!=vocab or len(v)!=vocab: raise ValueError("vocabulary mismatch")
        if not all(math.isfinite(z) for z in u+v): raise ValueError("nonfinite logits")
        agreed+=max(range(vocab),key=u.__getitem__)==max(range(vocab),key=v.__getitem__)
        for i,j in zip(u,v):
            err=abs(i-j)
            worst=max(worst,err)
            failures+=err>atol+rtol*abs(i)
            total+=1
    return {"positions":len(x),"logits_compared":total,"max_absolute_error":worst,
            "outside_tolerance":failures,"top1_agreement":agreed/len(x),
            "passed":failures==0 and agreed==len(x),"atol":atol,"rtol":rtol}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference",type=Path); p.add_argument("candidate",type=Path)
    p.add_argument("--atol",type=float,default=0.001)
    p.add_argument("--rtol",type=float,default=0.001)
    a=p.parse_args()
    if not all(math.isfinite(t) and t>=0 for t in (a.atol,a.rtol)): p.error("invalid tolerance")
    result=compare(json.loads(a.reference.read_text()),json.loads(a.candidate.read_text()),a.atol,a.rtol)
    print(json.dumps(result,indent=2))
    return 0 if result["passed"] else 1

if __name__=="__main__": raise SystemExit(main())
