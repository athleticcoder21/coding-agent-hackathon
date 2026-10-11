#!/usr/bin/env python3
"""Offline structural checks for the local competition project."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        import yaml
    except ImportError:
        print("PyYAML missing: run uv sync --locked", file=sys.stderr)
        return 2

    submission = ROOT / "submission"
    agent_text = (submission / "agent.yaml").read_text()
    assert "model: gemma-4-31b-it-qat-w4a16-ct" in agent_text
    assert (submission / "prompts/system.md").is_file()

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor("!include", lambda loader, node: loader.construct_scalar(node))
    agent = yaml.load(agent_text, Loader=Loader)
    assert agent["agent_class"] == "LlmAgent"
    assert agent["instruction"] == "prompts/system.md"
    assert "submit_patch" in agent["tools"]
    for path in submission.rglob("*"):
        assert not path.is_symlink(), f"symlink in submission: {path}"
    print("Project structure: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
