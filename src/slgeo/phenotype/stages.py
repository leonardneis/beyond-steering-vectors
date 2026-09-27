"""Stage layout of the Phenotype Anchor (standard library only: shared by execution, analysis and the submit host).

A stage is one preregistered unit of execution and analysis (PREREGISTRATION §11):

  p1          base, teachers, library personas and students N/S 1-3; analysed first.
  p2          dog-teacher students D1-D3 (P2, §9); its analysis also reads the p1 plan.
  p1-seeds45  fresh-seed students N4, S4, N5, S5 (§6.1, only after the fresh-seed trigger); its analysis also reads
              the p1 plan.

Which arms belong to a stage is fixed in the execution manifest (``stages`` in
``configs/validation/phenotype_anchor_v1.yaml``); the file names of each stage's artifacts are fixed here.
"""

from __future__ import annotations

STAGES = ("p1", "p2", "p1-seeds45")
PLAN = {"p1": "plan.json", "p2": "plan_p2.json", "p1-seeds45": "plan_p1-seeds45.json"}
# every plan a stage's analysis reads (its own plan and, for p2 / p1-seeds45, the p1 plan)
READS = {"p1": (PLAN["p1"],), "p2": (PLAN["p1"], PLAN["p2"]), "p1-seeds45": (PLAN["p1"], PLAN["p1-seeds45"])}
LOCK = {"p1": "adapters.lock.json", "p2": "adapters_p2.lock.json", "p1-seeds45": "adapters_p1-seeds45.lock.json"}
RESULT = {"p1": "p1_analysis.json", "p2": "p2_analysis.json", "p1-seeds45": "p1_seeds45_analysis.json"}
CAP = {"p1": "cap.json", "p2": "cap_p2.json", "p1-seeds45": "cap_p1-seeds45.json"}
PROJECTION = {"p1": "tv_projection.json", "p2": "tv_projection_p2.json",
              "p1-seeds45": "tv_projection_p1-seeds45.json"}  # written by tv-project into the scientific root
RUN_TAG = {"p1": "sci-p1", "p2": "sci-p2", "p1-seeds45": "sci-p1-seeds45"}
AUTHORIZATION = {"p1": "SCIENTIFIC_EXECUTION_AUTHORIZATION.json",
                 "p2": "SCIENTIFIC_EXECUTION_AUTHORIZATION_p2.json",
                 "p1-seeds45": "SCIENTIFIC_EXECUTION_AUTHORIZATION_p1-seeds45.json"}
DATA_ENTROPY = "p2_data_entropy.json"  # CPU number-entropy record of §9.1 / §9.4 (written before any P2 forward)


def check(stage: str) -> str:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r} (one of {', '.join(STAGES)})")
    return stage
