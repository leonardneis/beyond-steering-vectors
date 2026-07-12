"""Paired prompt bootstrap for the full subliminal-minus-neutral target-logprob shift."""
from __future__ import annotations
import argparse, json
import numpy as np
from _bootstrap import bootstrap, repo_path
bootstrap()
from slgeo.analysis.split_stability import bootstrap_mean_ci  # noqa: E402
from slgeo.io import ensure_parent  # noqa: E402

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--subliminal",required=True);p.add_argument("--neutral",required=True);p.add_argument("--output",required=True);p.add_argument("--bootstrap-samples",type=int,default=5000);p.add_argument("--seed",type=int,default=42);p.add_argument("--require-positive",action="store_true");a=p.parse_args()
    s=json.loads(repo_path(a.subliminal).read_text());n=json.loads(repo_path(a.neutral).read_text())
    sv=np.asarray([r["target_logprob"] for r in s["token_metrics"]["rows"]]);nv=np.asarray([r["target_logprob"] for r in n["token_metrics"]["rows"]]);effect=sv-nv;lo,hi=bootstrap_mean_ci(effect,samples=a.bootstrap_samples,seed=a.seed)
    result={"schema_version":1,"analysis":"paired_full_target_logprob","mean":float(effect.mean()),"median":float(np.median(effect)),"ci95":[lo,hi],"per_prompt":effect.tolist(),"subliminal_source":str(repo_path(a.subliminal)),"neutral_source":str(repo_path(a.neutral))}
    out=ensure_parent(repo_path(a.output));out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps({k:v for k,v in result.items() if k!="per_prompt"},indent=2))
    if a.require_positive and lo <= 0: raise SystemExit("Confirmatory gate failed: target-logprob CI does not exclude zero positively")
if __name__=="__main__":main()
