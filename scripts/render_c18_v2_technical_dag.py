"""Render the versioned C18-v2 technical DAG using only the standard library."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import urllib.parse


def validate_topic(value: str) -> str:
    topic = value.strip().rstrip("/")
    if not topic:
        return ""
    parsed = urllib.parse.urlparse(topic)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.path in {"", "/"}
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
        or any(character in topic for character in ('"', "\\", "\n", "\r"))
    ):
        raise ValueError("NTFY_TOPIC must be a safely encodable private HTTPS topic URL")
    return topic


def render(source: str, *, commit: str, start_epoch: int, ntfy_topic: str = "") -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("execution commit must be a full lowercase Git SHA-1")
    if source.count('BsvExecutionGitCommit="UNFROZEN"') != 4:
        raise ValueError("versioned C18-v2 DAG has an unexpected execution-commit inventory")
    if source.count('BsvStartEpoch="0"') != 1 or source.count('BsvNtfyTopic=""') != 1:
        raise ValueError("versioned C18-v2 DAG has unexpected notification placeholders")
    return (
        source.replace('BsvExecutionGitCommit="UNFROZEN"', f'BsvExecutionGitCommit="{commit}"')
        .replace('BsvStartEpoch="0"', f'BsvStartEpoch="{int(start_epoch)}"')
        .replace('BsvNtfyTopic=""', f'BsvNtfyTopic="{validate_topic(ntfy_topic)}"')
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execution-git-commit", required=True)
    parser.add_argument("--start-epoch", required=True, type=int)
    parser.add_argument("--ntfy-topic", default="")
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    if source.resolve() == output.resolve():
        raise ValueError("runtime DAG must not overwrite the versioned source")
    rendered = render(
        source.read_text(encoding="utf-8"), commit=args.execution_git_commit,
        start_epoch=args.start_epoch, ntfy_topic=args.ntfy_topic,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8", newline="\n")
    temporary.replace(output)


if __name__ == "__main__":
    main()
