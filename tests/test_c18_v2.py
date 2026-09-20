from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from slgeo.analysis.c18_v2_execution import (
    EXPERIMENT_ID, HARD_GATES, logical_position_ids, physical_cache_position,
    validate_batch_plan, validate_technical_report,
)
from slgeo.analysis.c18_v2_independent_audit import recompute_w as audit_w
from slgeo.analysis.c18_v2_statistics import (
    aggregate_w, aggregate_y, classify, suffix_conflict,
)


FAMILIES = ["direct_preference"] * 24 + ["identity_affinity"] * 24 + ["hypothetical_choice"] * 24
SETS = ["top", *[f"norm_{index:02d}" for index in range(25)]]


def y_rows(cross_effect=0.0):
    rows=[]
    for condition in ("subliminal","neutral"):
        for set_id in SETS:
            for prompt in range(72):
                base=.01*prompt + (.1 if condition=="subliminal" else 0) + (.001*SETS.index(set_id))
                for cell,value in {"Y00":base,"Y01":base+cross_effect,"Y10":base+cross_effect,"Y11":base+2*cross_effect}.items():
                    rows.append({"condition":condition,"set_id":set_id,"prompt_id":f"p{prompt:02d}",
                                 "family":FAMILIES[prompt],"cell":cell,"margin":value})
    return rows


def w_rows(interaction=0.0):
    rows=[]
    for condition in ("subliminal","neutral"):
        for set_id in SETS:
            for prompt in range(72):
                base=.01*prompt + (.1 if condition=="subliminal" else 0) + (.001*SETS.index(set_id))
                for a in (0,1):
                    for b in (0,1):
                        for r in (0,1):
                            value=base + .02*a + .03*r + interaction*a*r
                            rows.append({"condition":condition,"set_id":set_id,"prompt_id":f"p{prompt:02d}",
                                         "family":FAMILIES[prompt],"cell":f"W{a}{b}{r}","margin":value})
    return rows


def summary(i90=(-.01,.01),i99=(-.02,.02),g=0):
    return {"interval90":i90,"interval99":i99,"G":g}


def test_logical_positions_and_physical_cache():
    mask=torch.tensor([[0,0,1,1],[0,1,1,1]])
    assert logical_position_ids(mask).tolist()==[[1,1,0,1],[1,0,1,2]]
    assert physical_cache_position(4,torch.device("cpu")).tolist()==[0,1,2,3]


def test_batch_plan_membership_validation():
    tokens={"rows":[{"prompt_id":f"p{i:02d}"} for i in range(72)]}
    batches={"batches":[{"batch_id":b,"batch_size":6,"rows":[{"prompt_id":f"p{i:02d}"} for i in range(6*b,6*b+6)]} for b in range(12)]}
    validate_batch_plan(tokens,batches)
    batches["batches"][0]["rows"].reverse()
    with pytest.raises(ValueError): validate_batch_plan(tokens,batches)


@pytest.mark.parametrize(("kwargs","expected"),[
    ({"integrity":False,"reconstruction":True,"suffix_conflict_value":False},"F_TECHNICAL_IDENTIFICATION_FAILURE"),
    ({"integrity":True,"reconstruction":True,"suffix_conflict_value":False},"A_BIDIRECTIONAL_FUNCTIONAL_TRANSPORT"),
])
def test_basic_classification(kwargs,expected):
    y={"quantities":{"B0":summary(),"B1":summary(),"I":summary()},"cancellation":{"B0":[],"B1":[]}}
    assert classify(y=y,**kwargs)==expected


def test_ordered_classes_b_to_g():
    base={"cancellation":{"B0":[],"B1":[]}}
    def call(b0,b1,i=summary(),cancel=False,suffix=False):
        y={**base,"quantities":{"B0":b0,"B1":b1,"I":i},
           "cancellation":{"B0":["x"] if cancel else [],"B1":[]}}
        return classify(integrity=True,reconstruction=True,y=y,suffix_conflict_value=suffix)
    material=summary((.06,.08),(.05,.09),.07); uncertain=summary((-.06,.02),(-.07,.03),-.02)
    assert call(summary(),material)=="B_ASYMMETRIC_TRANSPORT"
    assert call(uncertain,uncertain,i=material)=="C_STATE_PARAMETER_INTERACTION_COADAPTATION"
    assert call(summary(),summary(),cancel=True)=="E_CANCELLATION_HETEROGENEITY_LIMITED"
    assert call(material,material)=="D_TEACHER_COORDINATE_INSUFFICIENCY"
    assert call(uncertain,uncertain)=="G_INSUFFICIENT_PRECISION_UNRESOLVED"


def test_full_25x72_cancellation_is_preserved():
    rows=y_rows()
    # Opposing top-vs-norm atoms cancel only after averaging norm sets.
    for row in rows:
        if row["cell"]=="Y10" and row["condition"]=="subliminal" and row["set_id"]!="top":
            index=int(row["set_id"].split("_")[1]); row["margin"] += .2 if index<12 else -.2
    result=aggregate_y(rows)
    assert len(result["quantities"]["B0"]["atoms"])==25
    assert result["quantities"]["B0"]["mean_absolute_atoms"] > .1
    assert "mean_absolute_25x72" in result["cancellation"]["B0"]


def test_w_allocations_match_independent_auditor():
    rows=w_rows(interaction=.07)
    production=aggregate_w(rows); independent=audit_w(rows)
    assert production["donor_0"]["J"]["G"]==pytest.approx(independent["donor_0"]["J"]["G"])
    # S-N/top-norm cancels a common interaction, so this is not material at G.
    z={"I":summary()}; flagged,reasons=suffix_conflict(z,production)
    assert not flagged and reasons==[]


def test_technical_schema_rejects_scientific_content():
    report={"experiment_id":EXPERIMENT_ID,"status":"PASS","hard_gates":{name:"PASS" for name in HARD_GATES},
            "outcome_blindness":{"scientific_prompts_loaded":False,"scientific_selection_plan_loaded":False,
                "teacher_tensors_loaded":0,"teacher_interchanges":0,"cross_cells_computed":0,
                "estimands_computed":0,"raw_logits_persisted":False,"forbidden_open_attempts":0}}
    validate_technical_report(report)
    report["forbidden"]="Y01"
    with pytest.raises(ValueError): validate_technical_report(report)
