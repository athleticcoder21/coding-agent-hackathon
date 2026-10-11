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
FILE_HEADER = re.compile(r"(?m)^--- a/(.+)\n\+\+\+ b/(.+)\n")
HUNK_HEADER = re.compile(r"(?m)^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")


def patch_blocks(patch: str):
    headers = list(FILE_HEADER.finditer(patch))
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(patch)
        yield header.group(2), patch[header.start():end]


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
    for path, block in patch_blocks(patch):
        if not path.endswith(".py") or is_test_or_config(path):
            continue
        result.append((path, block))
    return result


def source_excerpt(body: str, patch: str, max_file_bytes: int) -> str:
    if len(body.encode()) <= max_file_bytes:
        return body
    lines = body.splitlines(keepends=True)
    starts = [int(match.group(1)) for match in HUNK_HEADER.finditer(patch)]
    if not starts:
        return body.encode()[:max_file_bytes].decode("utf-8", errors="ignore")
    pieces = []
    used = 0
    for line_number in starts:
        start = max(0, line_number - 21)
        end = min(len(lines), line_number + 25)
        piece = f"# Base lines {start + 1}-{end}\n" + "".join(lines[start:end])
        encoded = piece.encode()
        if used + len(encoded) > max_file_bytes:
            break
        pieces.append(piece)
        used += len(encoded)
    return "\n# ...\n".join(pieces)


def snapshot_text(archive: Path, wanted: dict[str, str], max_file_bytes: int) -> dict[str, str]:
    found = {}
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.size <= 1_000_000]
        for path, patch in wanted.items():
            matches = [m for m in members if m.name == path or m.name.endswith("/" + path)]
            if not matches:
                continue
            member = min(matches, key=lambda m: len(m.name))
            stream = tar.extractfile(member)
            if stream:
                body = stream.read().decode("utf-8", errors="replace")
                excerpt = source_excerpt(body, patch, max_file_bytes)
                if excerpt:
                    found[path] = excerpt
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
        wanted = {path: "".join(block for block_path, block in blocks if block_path == path)
                  for path in paths}
        context = snapshot_text(archive, wanted, args.max_file_bytes)
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
