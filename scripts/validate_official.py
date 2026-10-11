#!/usr/bin/env python3
"""Validate and compile an unpacked agent with the organizer's harness wheels."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_dir", type=Path, nargs="?", default=Path("submission"))
    args = parser.parse_args()

    from adk_submission import ModelRegistry, compile_submission, validate_directory
    from google.adk.models.lite_llm import LiteLlm
    from swegemma.config import build_submission_limits
    from swegemma.context import SwegemmaContext

    limits, constraints = build_submission_limits()
    validate_directory(args.submission_dir, limits)
    models = ModelRegistry()
    models.register(
        "gemma-4-31b-it-qat-w4a16-ct",
        LiteLlm(
            model="openai/gemma-4-31b-it-qat-w4a16-ct",
            api_base="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
        ),
    )
    tools = SwegemmaContext().create_tools()
    agent = compile_submission(
        args.submission_dir,
        tool_registry=tools,
        model_registry=models,
        limits=limits,
        generation_constraints=constraints,
    )
    print(f"Official harness validation and compile: OK ({agent.name})")


if __name__ == "__main__":
    main()
