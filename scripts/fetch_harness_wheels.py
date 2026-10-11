#!/usr/bin/env python3
"""Download only the organizer's four pure-Python harness wheels."""
import argparse
import csv
from pathlib import Path
import re
import subprocess
import sys

DATASET = "metric/gemma-4-developer-agent-wheelhouse"
PACKAGES = ("swegemma", "adk_submission", "adk_eval_core", "google_adk")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/harness-wheels"))
    args = parser.parse_args()
    kaggle = Path(sys.executable).with_name("kaggle")
    if not kaggle.is_file():
        parser.error("Kaggle CLI missing; install requirements-dev.txt in this Python environment")
    result = subprocess.run(
        [str(kaggle), "datasets", "files", DATASET, "--csv", "--page-size", "200"],
        check=True,
        capture_output=True,
        text=True,
    )
    names = [row[0] for row in csv.reader(result.stdout.splitlines()) if row]
    selected = []
    for package in PACKAGES:
        matching = [
            name for name in names
            if re.fullmatch(rf"{package}-[^/]+-py3-none-any\.whl", name)
        ]
        if len(matching) != 1:
            raise SystemExit(f"Expected one official {package} wheel; found {matching}")
        selected.append(matching[0])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in selected:
        destination = args.output_dir / name
        if destination.exists():
            print(f"Already present: {destination}")
            continue
        subprocess.run(
            [str(kaggle), "datasets", "download", DATASET, "-f", name, "-p", str(args.output_dir)],
            check=True,
        )
    print("Official harness wheels ready in", args.output_dir)


if __name__ == "__main__":
    main()
