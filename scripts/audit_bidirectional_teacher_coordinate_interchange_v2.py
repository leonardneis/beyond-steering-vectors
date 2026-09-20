"""Independent fail-closed audit of released C18-v2 cells and aggregate."""

from __future__ import annotations

import argparse
import io
import json
import os

from _bootstrap import bootstrap, repo_path

bootstrap()

import numpy as np  # noqa: E402

from slgeo.analysis.c18_v2_independent_audit import (  # noqa: E402
    bootstrap as independent_bootstrap, conflict, decision, interval,
    recompute_w, recompute_y, recompute_z,
)
from slgeo.analysis.c18_v2_manifest import (  # noqa: E402
    apply_storage_overrides, expected_raw_ids, sha256_file,
    validate_manifest_contract, validate_public_inputs,
)
from slgeo.analysis.teacher_coordinate_interchange import atomic_json, unseal_bytes  # noqa: E402
from slgeo.io import load_yaml  # noqa: E402


def decrypt(path, key: bytes) -> dict:
    with np.load(io.BytesIO(unseal_bytes(path.read_bytes(), key)), allow_pickle=False) as data:
        return json.loads(str(data["payload_json"].item()))


def compare(left, right, path="root") -> None:
    if isinstance(left, dict):
        if not isinstance(right, dict): raise RuntimeError(f"audit type mismatch: {path}")
        for key, value in left.items():
            if key not in right: raise RuntimeError(f"audit missing field: {path}.{key}")
            compare(value, right[key], f"{path}.{key}")
    elif isinstance(left, list):
        if not isinstance(right, list) or len(left) != len(right):
            raise RuntimeError(f"audit list mismatch: {path}")
        for index, value in enumerate(left): compare(value, right[index], f"{path}[{index}]")
    elif isinstance(left, (float, int)) and not isinstance(left, bool):
        if not np.isclose(float(left), float(right), rtol=0, atol=1e-12, equal_nan=False):
            raise RuntimeError(f"audit numeric mismatch: {path}")
    elif left != right:
        raise RuntimeError(f"audit mismatch: {path}")


def independent_specificity(y: dict, payloads: list[dict]) -> dict:
    natural = [dict(r) for p in payloads for r in p["Y"] if r["cell"] in ("Y00", "Y11")]
    indices = independent_bootstrap(y["families"])
    t0=np.asarray(y["quantities"]["B0"]["per_prompt"]); t1=np.asarray(y["quantities"]["B1"]["per_prompt"])
    td=np.maximum(np.abs(np.mean(t0[indices],axis=1)),np.abs(np.mean(t1[indices],axis=1)))
    controls=[]
    for family in range(5):
        cross=[dict(r) for p in payloads for r in p["orthogonal"] if int(r["dose_family"])==family]
        result=recompute_y(natural+cross)
        c0=np.asarray(result["quantities"]["B0"]["per_prompt"]); c1=np.asarray(result["quantities"]["B1"]["per_prompt"])
        cd=np.maximum(np.abs(np.mean(c0[indices],axis=1)),np.abs(np.mean(c1[indices],axis=1)))
        controls.append({"dose_family":family,
                         "D":float(max(abs(result["quantities"]["B0"]["G"]),abs(result["quantities"]["B1"]["G"]))
                                   -max(abs(y["quantities"]["B0"]["G"]),abs(y["quantities"]["B1"]["G"]))),
                         "interval99":interval(cd-td,.99),
                         "G_B0":result["quantities"]["B0"]["G"],"G_B1":result["quantities"]["B1"]["G"]})
    return {"status":"single_secondary_specificity_family","controls":controls,
            "teacher_specificity_modifier_pass":all(item["interval99"][0]>0 for item in controls)}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",default="configs/validation/cat_bidirectional_teacher_coordinate_interchange_v2.yaml")
    parser.add_argument("--release-token",required=True); parser.add_argument("--subliminal",required=True)
    parser.add_argument("--neutral",required=True); parser.add_argument("--aggregate",required=True)
    parser.add_argument("--output",required=True); args=parser.parse_args()
    manifest_path=repo_path(args.manifest); manifest=apply_storage_overrides(load_yaml(manifest_path))
    validate_manifest_contract(manifest)
    frozen=validate_public_inputs(manifest,repo_path("."),require_runtime_inputs=True,read_sensitive=True)
    release=json.loads(repo_path(args.release_token).read_text(encoding="utf-8"))
    if release != {"experiment_id":manifest["experiment_id"],"manifest_sha256":sha256_file(manifest_path),
                   "outcome_release_authorized":True}:
        raise RuntimeError("STOP: independent audit release token differs")
    key=os.environ.get("SLGEO_C18_V2_SEAL_KEY","").encode("ascii")
    if not key: raise RuntimeError("STOP: C18-v2 sealing key absent")
    paths=[repo_path(args.subliminal),repo_path(args.neutral)]; payloads=[decrypt(path,key) for path in paths]
    expected=expected_raw_ids(manifest,repo_path("."))
    for category in ("Y","Z","W","orthogonal"):
        rows=[r for p in payloads for r in p[category]]
        if category=="orthogonal":
            observed={f"{r['condition']}|{r['set_id']}|{r['prompt_id']}|O{r['dose_family']}|{r['cell']}" for r in rows}
        else: observed={f"{r['condition']}|{r['set_id']}|{r['prompt_id']}|{r['cell']}" for r in rows}
        if observed!=expected[category] or len(observed)!=len(rows): raise RuntimeError(f"audit {category} inventory failed")
    y=recompute_y([r for p in payloads for r in p["Y"]]); z=recompute_z([r for p in payloads for r in p["Z"]])
    w=recompute_w([r for p in payloads for r in p["W"]]); suffix,reasons=conflict(z,w)
    classification=decision(y,suffix); specificity=independent_specificity(y,payloads)
    stored=json.loads(repo_path(args.aggregate).read_text(encoding="utf-8"))
    compare({k:v for k,v in y.items() if k!="families"}, stored["Y"], "Y")
    compare(z,stored["Z"],"Z"); compare(w,stored["W"],"W")
    compare(suffix,stored["suffix_conflict"],"suffix_conflict"); compare(reasons,stored["suffix_conflict_reasons"],"suffix_reasons")
    compare(classification,stored["classification"],"classification"); compare(specificity,stored["specificity"],"specificity")
    atomic_json(repo_path(args.output),{"schema_version":2,"status":"PASS","experiment_id":manifest["experiment_id"],
        "manifest_sha256":sha256_file(manifest_path),"independent_reconstruction":{
            "estimands":"PASS","algebra":"PASS","cancellation_25x72":"PASS","Z_W_allocations":"PASS",
            "bootstrap":"PASS","decision_matrix":"PASS","specificity_family":"PASS"},
        "frozen_inputs":frozen,"checksums":{str(path):sha256_file(path) for path in [*paths,repo_path(args.aggregate)]}})


if __name__=="__main__": main()
