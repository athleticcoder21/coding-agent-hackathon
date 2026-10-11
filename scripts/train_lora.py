#!/usr/bin/env python3
"""Train a PEFT LoRA for Gemma 4 31B from its unquantized QAT checkpoint."""
import argparse
import json
from pathlib import Path

DEFAULT_MODEL = "google/gemma-4-31B-it-qat-q4_0-unquantized"
DEPLOYMENT_MODEL = "gemma-4-31b-it-qat-w4a16-ct"
# Restrict LoRA to the language model. A bare q_proj target can also match
# Gemma 4's vision tower, which the competition's text agent does not use.
TARGET_MODULES = (
    r".*language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj))"
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=-1)
    return parser.parse_args()


def load_examples(path, tokenizer, max_length):
    from datasets import Dataset

    rows = []
    skipped = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        messages = record["messages"]
        if len(messages) < 2 or messages[-1]["role"] != "assistant":
            raise ValueError(f"Bad conversation in {path}: {record.get('instance_id')}")
        prompt_ids = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=True
        )["input_ids"]
        full_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=True
        )["input_ids"]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("Chat template prefix mismatch; update masking before training")
        if len(full_ids) > max_length or len(full_ids) == len(prompt_ids):
            skipped += 1
            continue
        rows.append({
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):],
        })
    if not rows:
        raise ValueError(f"No usable examples in {path}; increase --max-length or shorten context")
    print(f"{path}: {len(rows)} accepted, {skipped} too long/empty")
    return Dataset.from_list(rows)


class CompletionCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        import torch

        width = max(len(row["input_ids"]) for row in features)
        batch = {}
        for key, pad in (("input_ids", self.pad_id), ("attention_mask", 0), ("labels", -100)):
            batch[key] = torch.tensor(
                [row[key] + [pad] * (width - len(row[key])) for row in features],
                dtype=torch.long,
            )
        return batch


def main():
    args = arguments()
    if args.rank < 1 or args.rank > 128:
        raise SystemExit("LoRA rank must be 1..128 for the competition harness")
    if not args.train_file.is_file():
        raise SystemExit(f"Training file missing: {args.train_file}")

    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForMultimodalLM,
        AutoProcessor,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise SystemExit("This 31B training recipe requires an NVIDIA CUDA GPU")
    set_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train = load_examples(args.train_file, tokenizer, args.max_length)
    validation = (
        load_examples(args.validation_file, tokenizer, args.max_length)
        if args.validation_file and args.validation_file.is_file()
        else None
    )

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if getattr(model.config, "quantization_config", None) and "w4a16" in args.model_id.lower():
        raise RuntimeError("Use the unquantized QAT checkpoint for training, not W4A16 compressed tensors")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch" if validation is not None else "no",
        save_strategy="no",
        logging_steps=1,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=validation,
        data_collator=CompletionCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    model.save_pretrained(args.output_dir, safe_serialization=True)
    metadata = {
        "training_model": args.model_id,
        "competition_deployment_model": DEPLOYMENT_MODEL,
        "train_examples": len(train),
        "validation_examples": len(validation) if validation is not None else 0,
        "max_length": args.max_length,
        "rank": args.rank,
        "seed": args.seed,
    }
    (args.output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved PEFT adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
