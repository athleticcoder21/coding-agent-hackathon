#!/usr/bin/env python3
"""Build text-only SFT examples from the competition's public development tasks."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile

SYSTEM = (
    "You are a software engineer. Fix the issue in the supplied repository context. "
    "Return only a unified git diff for source code. Do not edit tests."
)
DIFF_HEADER = re.compile(r"^diff --git a/(.+?) b/(.+)$")


def patch_blocks(patch: str):
    current = []
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git ") and current:
            yield "".join(current)
            current = []
        current.append(line)
    if current:
        yield "".join(current)


def is_test_or_config(path: str) -> bool:
    parts = Path(path).parts
    name = Path(path).name
    return (
        any(part in {"tests", "test", "testing"} for part in parts)
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name in {"conftest.py", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml"}
    )


def source_blocks(patch: str):
    result = []
    for block in patch_blocks(patch):
        first = block.splitlines()[0]
        match = DIFF_HEADER.match(first)
        if not match:
            continue
        path = match.group(2)
        if not path.endswith(".py") or is_test_or_config(path):
            continue
        result.append((path, block))
    return result


def snapshot_text(archive: Path, wanted: list[str], max_file_bytes: int) -> dict[str, str]:
    found = {}
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.size <= max_file_bytes]
        for path in wanted:
            matches = [m for m in members if m.name == path or m.name.endswith("/" + path)]
            if not matches:
                continue
            member = min(matches, key=lambda m: len(m.name))
            stream = tar.extractfile(member)
            if stream:
                found[path] = stream.read(max_file_bytes + 1).decode("utf-8", errors="replace")
    return found


def split_for(instance_id: str, validation_percent: int) -> str:
    bucket = int(hashlib.sha256(instance_id.encode()).hexdigest()[:8], 16) % 100
    return "validation" if bucket < validation_percent else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-percent", type=int, default=20)
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument("--max-file-bytes", type=int, default=12_000)
    parser.add_argument("--max-patch-bytes", type=int, default=30_000)
    args = parser.parse_args()
    if not 1 <= args.validation_percent <= 50:
        parser.error("--validation-percent must be between 1 and 50")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples = {"train": [], "validation": []}
    skipped = []
    for line in args.tasks.read_text().splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        instance_id = task["instance_id"]
        blocks = source_blocks(task.get("patch") or "")
        if not blocks:
            skipped.append({"instance_id": instance_id, "reason": "no source patch"})
            continue
        patch = "".join(block for _, block in blocks)
        archive = args.snapshots / (instance_id + ".tgz")
        if not archive.is_file() or len(patch.encode()) > args.max_patch_bytes:
            reason = "snapshot missing" if not archive.is_file() else "patch too large"
            skipped.append({"instance_id": instance_id, "reason": reason})
            continue
        paths = list(dict.fromkeys(path for path, _ in blocks))[: args.max_files]
        context = snapshot_text(archive, paths, args.max_file_bytes)
        if not context:
            skipped.append({"instance_id": instance_id, "reason": "no source context"})
            continue
        issue = task.get("problem_statement") or ""
        if not issue.strip():
            skipped.append({"instance_id": instance_id, "reason": "no issue"})
            continue
        hints = task.get("hints_text") or ""
        files = "\n\n".join(f"### {path}\n```python\n{body}\n```" for path, body in context.items())
        user = f"Issue:\n{issue}\n\nHints:\n{hints}\n\nRelevant files at base commit:\n{files}"
        example = {
            "instance_id": instance_id,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": patch},
            ],
        }
        examples[split_for(instance_id, args.validation_percent)].append(example)
    for split, rows in examples.items():
        with (args.output_dir / f"{split}.jsonl").open("w") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"counts": {k: len(v) for k, v in examples.items()}, "skipped": skipped}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"]))
    print(f"Skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
