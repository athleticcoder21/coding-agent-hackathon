#!/usr/bin/env python3
"""Fetch small official competition files after Kaggle authentication."""
import argparse
from pathlib import Path
import subprocess
import sys

COMPETITION = "gemma-4-developer-agent"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/competition"))
    parser.add_argument("--task-id", help="also download one snapshot, e.g. fastapi_11194")
    args = parser.parse_args()
    kaggle = Path(sys.executable).with_name("kaggle")
    if not kaggle.is_file():
        parser.error("Kaggle CLI missing; install requirements-dev.txt")
    files = ["HARNESS_README.md", "tasks.jsonl"]
    if args.task_id:
        if "/" in args.task_id or ".." in args.task_id:
            parser.error("--task-id must be a plain instance ID")
        files.append(f"snapshots/{args.task_id}.tgz")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        target = args.output_dir / name
        if target.exists():
            print(f"Already present: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [str(kaggle), "competitions", "download", "-c", COMPETITION,
             "-f", name, "-p", str(args.output_dir)],
            text=True,
        )
        if result.returncode:
            raise SystemExit(
                "Competition download failed. Sign in to Kaggle and accept its "
                "competition rules, then configure the Kaggle CLI."
            )
        if not target.exists():
            flattened = args.output_dir / Path(name).name
            if flattened.exists():
                flattened.rename(target)
            else:
                raise SystemExit(f"Kaggle reported success, but {target} was not found")
    print("Competition files ready in", args.output_dir)


if __name__ == "__main__":
    main()
