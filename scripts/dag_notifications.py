"""Create ignored runtime DAGs with a reusable ntfy FINAL node."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from _bootstrap import repo_path
from notify import validate_topic


def _quote(value: str) -> str:
    return value.replace("\\", "/").replace('"', '\\"')


def append_final_notification(
    dag: str, *, study: str, git_commit: str, result_path: str,
    ntfy_topic: str = "", start_epoch: int | None = None,
    container_image: str = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
    repo_root: str = "$ENV(HOME)/beyond-steering-vectors",
    node_name: str = "bsv_notify",
) -> str:
    if any(line.lstrip().startswith("FINAL ") for line in dag.splitlines()):
        raise ValueError("DAG already contains a FINAL node")
    topic = validate_topic(ntfy_topic)
    values = {
        "BsvStudyName": study.replace(" ", "_"),
        "BsvExecutionGitCommit": git_commit,
        "BsvResultPath": result_path,
        "BsvNtfyTopic": topic,
        "BsvDockerImage": container_image,
        "BsvStartEpoch": str(start_epoch if start_epoch is not None else int(time.time())),
        "BsvRepoRoot": repo_root,
    }
    suffix = [
        "",
        f"FINAL {node_name} condor/dag_notification.sub",
        "VARS " + node_name + " " + " ".join(
            f'{key}=\"{_quote(value)}\"' for key, value in values.items()
        ),
        "",
    ]
    return dag.rstrip() + "\n" + "\n".join(suffix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--ntfy-topic", default="")
    parser.add_argument("--start-epoch", type=int)
    parser.add_argument("--container-image", default="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    parser.add_argument("--repo-root", default="$ENV(HOME)/beyond-steering-vectors")
    args = parser.parse_args()
    source, output = repo_path(args.source), repo_path(args.output)
    if source.resolve() == output.resolve():
        raise ValueError("Runtime DAG must not overwrite its versioned source")
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = append_final_notification(
        source.read_text(encoding="utf-8"), study=args.study,
        git_commit=args.git_commit, result_path=args.result_path,
        ntfy_topic=args.ntfy_topic, start_epoch=args.start_epoch,
        container_image=args.container_image,
        repo_root=args.repo_root,
    )
    output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Prepared runtime DAG at {output}")


if __name__ == "__main__":
    main()
