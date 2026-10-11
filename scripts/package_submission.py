#!/usr/bin/env python3
"""Package the declarative agent and optional LoRA into submission.zip."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 3 * 1024**3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, help="PEFT adapter directory")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/submission.zip")
    args = parser.parse_args()
    submission = ROOT / "submission"
    agent = (submission / "agent.yaml").read_text()
    files = {}
    for path in submission.rglob("*"):
        if path.is_symlink():
            parser.error(f"symlink not allowed in submission: {path}")
        if path.is_file() and "adapters" not in path.relative_to(submission).parts:
            files[path.relative_to(submission).as_posix()] = path.read_bytes()
    if args.adapter:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            path = args.adapter / name
            if not path.is_file():
                parser.error(f"missing adapter file: {path}")
            files[f"adapters/main_lora/{name}"] = path.read_bytes()
        agent = agent.replace(
            "model: gemma-4-31b-it-qat-w4a16-ct\n",
            "model: gemma-4-31b-it-qat-w4a16-ct\nadapter: main_lora\n",
            1,
        )
        if "adapter: main_lora" not in agent:
            parser.error("could not add adapter field to agent.yaml")
    files["agent.yaml"] = agent.encode()
    if sum(map(len, files.values())) >= LIMIT:
        parser.error("unpacked submission exceeds the 3 GiB competition limit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, contents in sorted(files.items()):
            archive.writestr(name, contents)
    with zipfile.ZipFile(args.output) as archive:
        assert "agent.yaml" in archive.namelist()
        assert all(not name.startswith("/") and ".." not in Path(name).parts for name in archive.namelist())
    print(f"Created {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
