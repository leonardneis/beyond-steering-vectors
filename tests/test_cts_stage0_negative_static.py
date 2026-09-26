"""Red-team static tests over the CTS Stage-0 implementation sources (no import side effects, no model).

The checks use the AST where docstrings legitimately mention a forbidden construct (e.g. ``residual_intervention``
or ``hidden[:, -1:]`` in the steering module docstring).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "slgeo" / "cts_stage0"
SOURCES = sorted(PACKAGE.glob("*.py")) + [ROOT / "scripts" / "cts_stage0.py"]
SOURCE_IDS = [path.relative_to(ROOT).as_posix() for path in SOURCES]


def _public_files() -> list[Path]:
    files = list(SOURCES)
    files.append(ROOT / "configs" / "validation" / "cts_stage0_v1.yaml")
    files += sorted(p for p in (ROOT / "research" / "cts_stage0_v1_execution").rglob("*") if p.is_file())
    files += sorted(p for p in (ROOT / "condor").glob("*cts_stage0*") if p.is_file())
    return [p for p in files if p.exists() and "__pycache__" not in p.parts]


PUBLIC = _public_files()
# Known hit (see report): the submit script defaults SLGEO_SHARED_ROOT to the cluster scratch path.
KNOWN_PRIVATE_HITS: set[str] = set()


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _identifiers(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            out.add(child.id)
        elif isinstance(child, ast.Attribute):
            out.add(child.attr)
        elif isinstance(child, ast.alias):
            out.add(child.name)
            if child.asname:
                out.add(child.asname)
        elif isinstance(child, ast.ImportFrom) and child.module:
            out.add(child.module)
    return out


def test_sources_found():
    assert len(SOURCES) >= 24
    assert (PACKAGE / "guards.py") in SOURCES


@pytest.mark.parametrize("path", SOURCES, ids=SOURCE_IDS)
def test_no_peft_import_or_peftmodel(path):
    tree = _tree(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(alias.name.split(".")[0] == "peft" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "peft"
        if isinstance(node, ast.Call) and _identifiers(node.func) & {"import_module", "__import__"}:
            args = [a.value for a in node.args if isinstance(a, ast.Constant)]
            assert not any(str(a).split(".")[0] == "peft" for a in args)
    assert not {"PeftModel", "get_peft_model", "PeftConfig", "LoraConfig"} & _identifiers(tree)


@pytest.mark.parametrize("path", SOURCES, ids=SOURCE_IDS)
def test_no_residual_intervention_or_legacy_helpers(path):
    identifiers = _identifiers(_tree(path))
    forbidden = {"residual_intervention", "hidden_state_statistics", "capture_block_outputs", "slgeo.analysis.interventions",
                 "slgeo.analysis.activations", "reference_animal_system_prompt"}
    assert not identifiers & forbidden


def _is_minus_one(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return node.operand.value == 1
    if isinstance(node, ast.Slice):
        return node.lower is not None and _is_minus_one(node.lower)
    return False


@pytest.mark.parametrize("path", SOURCES, ids=SOURCE_IDS)
def test_no_last_position_slicing_of_hidden(path):
    """RT-02: no ``hidden[:, -1...]`` anywhere, and no ``-1`` position index inside any hook function."""
    tree = _tree(path)
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in {"hidden", "new_hidden"}:
            index = node.slice
            elements = index.elts if isinstance(index, ast.Tuple) else [index]
            if len(elements) >= 2 and _is_minus_one(elements[1]):
                offenders.append(node.lineno)
        if isinstance(node, ast.FunctionDef) and "hook" in node.name:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Subscript) and _identifiers(sub.value) & {"hidden", "new_hidden", "output", "last_hidden_state"}:
                    index = sub.slice
                    elements = index.elts if isinstance(index, ast.Tuple) else []
                    if len(elements) >= 2 and _is_minus_one(elements[1]):
                        offenders.append(sub.lineno)
    assert not offenders, offenders
    assert "hidden[:, -1" not in "\n".join(
        line for line in path.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith(("#", '"', "``"))
        and "deliberately not used" not in line
    )


@pytest.mark.parametrize("path", SOURCES, ids=SOURCE_IDS)
def test_authoring_only_referenced_in_guards(path):
    if path.name == "guards.py":
        return
    assert "authoring/" not in path.read_text(encoding="utf-8")
    assert "authoring" not in _identifiers(_tree(path))


def test_analysis_never_reads_prompt_text():
    text = (PACKAGE / "analysis.py").read_text(encoding="utf-8")
    identifiers = _identifiers(_tree(PACKAGE / "analysis.py"))
    assert "cts_stage0_prompts.jsonl" not in text
    assert "cts_stage0_validation_prompts" not in text
    assert not identifiers & {"s0_prompts", "validation_prompts", "dc_fingerprints", "load_extraction_prompts"}


def test_integrity_and_criteria_never_read_prompt_text():
    for name in ("integrity.py", "criteria.py", "statistics.py", "conditions.py", "plan.py"):
        identifiers = _identifiers(_tree(PACKAGE / name))
        assert not identifiers & {"s0_prompts", "validation_prompts", "load_extraction_prompts"}, name


PRIVATE = re.compile(r"beyond-steering-vectors-private|[A-Za-z]:\\Users\\|C:/Users/|/home/|/scratch/compuling/|ntfy\.sh/|sic-hpc",
                     re.IGNORECASE)


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(
            p,
            marks=pytest.mark.xfail(strict=True, reason="GAP: condor/submit_cts_stage0.sh:24 hard-codes the cluster "
                                    "path /scratch/compuling/$USER/... as SLGEO_SHARED_ROOT default"),
        )
        if p.relative_to(ROOT).as_posix() in KNOWN_PRIVATE_HITS
        else p
        for p in PUBLIC
    ],
    ids=[p.relative_to(ROOT).as_posix() for p in PUBLIC],
)
def test_no_private_paths_or_topics_in_public_files(path):
    hits = [i + 1 for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()) if PRIVATE.search(line)]
    assert not hits, f"{path.relative_to(ROOT).as_posix()} lines {hits}"


NAN_REDUCTION = re.compile(r"\bnan(mean|sum|quantile|percentile|median|max|min|std|var|prod|argmax|argmin|cumsum)\b")


@pytest.mark.parametrize("name", ["statistics.py", "criteria.py", "analysis.py", "integrity.py"])
def test_no_nan_reductions(name):
    assert not NAN_REDUCTION.search((PACKAGE / name).read_text(encoding="utf-8"))


FORBIDDEN_LOG_TOKENS = {
    "tau", "delta", "ell", "point", "ci", "low", "high", "reliability", "reliabilities", "norm", "norms", "score",
    "scores", "decision", "cosine", "statistic", "statistics", "criteria", "criterion", "magnitude", "magnitudes",
    "rate", "effect", "effects", "logp", "L", "R", "kl", "mean", "result", "outcome", "label", "labels",
}


def _log_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
            if name in {"log", "print"}:
                yield node


@pytest.mark.parametrize("name", ["pipeline.py", "analysis.py", "integrity.py", "preflight.py", "../../../scripts/cts_stage0.py"])
def test_log_calls_format_no_outcome_variables(name):
    """RT-56: every log()/print() argument is built only from counts, ids, timings, hashes and check names."""
    path = (PACKAGE / name).resolve()
    offenders = []
    calls = 0
    for call in _log_calls(_tree(path)):
        calls += 1
        tokens = set()
        for arg in list(call.args) + [kw.value for kw in call.keywords if kw.arg not in {"file", "flush"}]:
            for identifier in _identifiers(arg):
                tokens.update(part for part in identifier.split("_") if part)
                tokens.add(identifier)
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Constant) and isinstance(sub.slice.value, str):
                    tokens.update(part for part in sub.slice.value.split("_") if part)
        if tokens & FORBIDDEN_LOG_TOKENS or "p_value" in tokens:
            offenders.append((call.lineno, sorted(tokens & FORBIDDEN_LOG_TOKENS)))
    assert calls > 0
    assert not offenders, offenders


def test_only_pipeline_log_prints_inside_the_package():
    for path in SOURCES:
        if path.parent != PACKAGE:
            continue
        prints = [c.lineno for c in _log_calls(_tree(path)) if isinstance(c.func, ast.Name) and c.func.id == "print"]
        if path.name == "pipeline.py":
            assert len(prints) == 1  # the log() helper itself
        else:
            assert not prints, (path.name, prints)


def test_analysis_output_file_names_are_class_independent():
    """The set of analysis file names is the constant OUTPUT_FILES on both the TECHNICAL_FAIL and computed paths."""
    tree = _tree(PACKAGE / "analysis.py")
    constant = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "OUTPUT_FILES" for t in node.targets):
            constant = ast.literal_eval(node.value)
    assert constant and len(set(constant)) == len(constant)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "stage_analysis")
    literal_key_sets = [
        {k.value for k in node.keys} for node in ast.walk(function)
        if isinstance(node, ast.Dict) and node.keys and all(isinstance(k, ast.Constant) and str(k.value).endswith((".json", ".npz")) for k in node.keys)
    ]
    assert set(constant) in literal_key_sets
    comprehensions = [node for node in ast.walk(function) if isinstance(node, ast.DictComp)]
    assert any("OUTPUT_FILES" in _identifiers(node) for node in comprehensions)
    npz_assigned = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and t.slice.value == "stage2a_handoff.npz"
            for t in node.targets
        )
    ]
    assert npz_assigned
    # No branch on the decision class decides what is written or how the process exits.
    source = (PACKAGE / "analysis.py").read_text(encoding="utf-8")
    assert "sys.exit" not in source and "SystemExit" not in source
    for decision_class in ("GO_X", "GO_CAT_ONLY", "PIVOT", "STOP_INSTRUMENT", "INCONCLUSIVE"):
        assert decision_class not in source


def test_render_uses_text_path_only():
    tree = _tree(PACKAGE / "render.py")
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "apply_chat_template"]
    assert calls
    for call in calls:
        keywords = {kw.arg: kw.value for kw in call.keywords}
        assert isinstance(keywords.get("tokenize"), ast.Constant) and keywords["tokenize"].value is False


@pytest.mark.parametrize("path", SOURCES, ids=SOURCE_IDS)
def test_loading_calls_are_offline_and_safe(path):
    for node in ast.walk(_tree(path)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        if node.func.attr == "from_pretrained":
            assert isinstance(keywords.get("local_files_only"), ast.Constant) and keywords["local_files_only"].value is True
        if node.func.attr == "load" and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch":
            assert isinstance(keywords.get("weights_only"), ast.Constant) and keywords["weights_only"].value is True
        if node.func.attr == "load" and isinstance(node.func.value, ast.Name) and node.func.value.id == "np":
            assert isinstance(keywords.get("allow_pickle"), ast.Constant) and keywords["allow_pickle"].value is False
        assert node.func.attr != "generate", f"{path.name}:{node.lineno} uses generate()"


@pytest.mark.parametrize("path", [p for p in SOURCES if p.name != "techval.py"], ids=[i for p, i in zip(SOURCES, SOURCE_IDS) if p.name != "techval.py"])
def test_no_swallowed_exceptions(path):
    """RT-44: broad exception handlers must re-raise (no per-condition 'skip and continue')."""
    offenders = []
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.ExceptHandler):
            broad = node.type is None or (isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"})
            if broad and not any(isinstance(sub, ast.Raise) for sub in ast.walk(node)):
                offenders.append(node.lineno)
    assert not offenders, offenders
