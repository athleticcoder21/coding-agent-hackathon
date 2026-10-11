# Gemma 4 Developer Agent workspace

This folder contains a Kaggle submission skeleton, a supervised LoRA training recipe, and scripts to install and run the official evaluation harness. The competition accepts only the base model named `gemma-4-31b-it-qat-w4a16-ct`; the optional LoRA is packaged separately.

## Desktop setup

The local `.venv` contains the official `swegemma`, `adk-submission`, `adk-eval-core`, and Google ADK wheels when the organizer wheelhouse is available. On macOS it supports config checks and data preparation. This 16 GB Apple Silicon Mac cannot train or serve the 31B model or run the CUDA scoring path.

1. Run `uv sync --locked`. This creates `.venv` from `pyproject.toml` and `uv.lock` using Python 3.12.
2. Run `uv run --locked python scripts/fetch_harness_wheels.py` to download the four pure-Python wheels from the [organizer wheelhouse](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse).
3. Run `scripts/setup_harness.sh data/harness-wheels`.
4. Run `uv run --locked python scripts/validate_official.py`.

The local environment also includes the official Kaggle CLI for authenticated downloads after you accept the competition rules. Its credentials remain outside this project. The four pure-Python organizer wheels can be downloaded individually with the CLI; a bulk wheelhouse download is unnecessary on macOS.

The official harness directory validator and compiler passed for the baseline agent on this Mac. After Kaggle authentication, run `uv run --locked python scripts/fetch_competition_data.py` for the readme and task list, or add `--task-id INSTANCE_ID` to fetch one snapshot. To fetch the complete dataset, run `uv run --locked kaggle competitions download gemma-4-developer-agent -p data/competition` and extract the resulting archive into `data/competition/`.

The [competition files](https://www.kaggle.com/competitions/gemma-4-developer-agent/data) require Kaggle sign-in and acceptance of the competition rules. Download them into `data/competition/` so `tasks.jsonl`, `snapshots/`, and `HARNESS_README.md` sit directly there. Full Docker evaluation also needs its `docker/`, `sandbox/`, `graphs/`, `embeddings/`, and `wheels/` directories. Do not commit this data or credentials.

For CLI sign-in, run `uv run --locked kaggle auth login`. The OAuth flow keeps credentials outside this project. Check free disk space before downloading the full dataset and extracting its archive.

## Build training examples

Run:

    uv run --locked python scripts/prepare_training_data.py --tasks data/competition/tasks.jsonl --snapshots data/competition/snapshots --output-dir data/processed

The script uses only public development issues, reference source patches, and source files from the corresponding base snapshot. It excludes test patches and test files. It writes `train.jsonl`, `validation.jsonl`, and a skip manifest. The split is deterministic by instance ID. These 129 public cases are a small supervised starting point, so measure generalization on held-out cases.

## Train on a CUDA host

Run `uv sync --locked --extra train` on a CUDA host with a compatible PyTorch build. Attach or pre-download Google's `google/gemma-4-31B-it-qat-q4_0-unquantized` checkpoint, then run:

    uv run --locked --extra train python scripts/train_lora.py --train-file data/processed/train.jsonl --validation-file data/processed/validation.jsonl --model-id google/gemma-4-31B-it-qat-q4_0-unquantized --output-dir artifacts/main_lora --max-length 2048 --epochs 1

The training script loads the **unquantized QAT** weights with bitsandbytes NF4 and trains a rank-16 PEFT LoRA. The requested W4A16 compressed-tensors variation is Google's serving format; its corresponding unquantized QAT weights are used as the trainable source. The saved adapter targets the competition's W4A16 model, but this compatibility and its quality must be checked with vLLM and the official harness before submission. On an offline Kaggle GPU notebook, point `--model-id` at the attached local checkpoint path and install dependencies from attached wheels.

## Package and evaluate

Run `uv run --locked python scripts/package_submission.py` for the prompt-only baseline, or add `--adapter artifacts/main_lora` after training. The zip is written to `outputs/submission.zip`; it contains `agent.yaml` at its root. Then, on a compatible CUDA/Docker host with the official data, run `scripts/evaluate.sh INSTANCE_ID`. This evaluates the unpacked prompt-only `submission/` directory. To evaluate a LoRA package, unzip it into a temporary directory and run `scripts/evaluate.sh INSTANCE_ID /path/to/unpacked/submission`.

This workspace has no verified score until the official evaluator completes a run against the real model.

## Sources

- [Competition overview](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview)
- [Public data schema](https://www.kaggle.com/competitions/gemma-4-developer-agent/data)
- [Gemma 4 QAT format guide](https://ai.google.dev/gemma/docs/core)
- [Google W4A16 model card](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct)
