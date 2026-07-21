#!/usr/bin/env python3
"""LoRA SFT trainer for Qwen3.5 on ReFrameJudge-v1 with multi-GPU support."""

import argparse
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    set_seed,
)


PROMPT = """You are an expert evaluator for photographic recomposition.

You will receive two images:
1. Source image: the original image.
2. Candidate image: a recomposed, cropped, edited, or generated version.

Judge whether the candidate is better than the source for composition-oriented image improvement. Consider:
- Composition: framing, subject placement, balance, crop, empty space, visual focus, leading lines.
- Content preservation: whether the main subject, identity, important objects, and scene semantics are preserved.
- Visual naturalness: artifacts, lighting, perspective, texture, realism, and awkward generated details.

Return only one valid JSON object:
{
  "overall_label": "win|tie|lose",
  "improvement_score": 0,
  "composition_gain": 3,
  "content_preservation": 5,
  "visual_naturalness": 5,
  "issue_tags": [],
  "reason": ""
}

Rules:
- "win": the candidate is clearly better overall.
- "tie": no clear winner, or composition gain is offset by content/quality loss.
- "lose": the candidate is worse overall.
- improvement_score must be an integer or float from -2 to 2.
- composition_gain, content_preservation, and visual_naturalness must be integers from 1 to 5.
- Keep reason concise.
"""


def read_jsonl(path, max_samples=None, seed=42):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if max_samples is not None and max_samples < len(records):
        rng = random.Random(seed)
        records = rng.sample(records, max_samples)
        records.sort(key=lambda item: item["id"])
    return records


def resolve_image_path(project_root, image_path):
    path = Path(image_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def image_uri(path):
    return str(path)


def target_json(record, include_reason):
    output = {
        "overall_label": record["overall_label"],
        "improvement_score": record.get("improvement_score", 0),
        "composition_gain": record.get("composition_gain", 3),
        "content_preservation": record.get("content_preservation", 5),
        "visual_naturalness": record.get("visual_naturalness", 5),
        "issue_tags": record.get("issue_tags", []),
        "reason": record.get("reason", "") if include_reason else "",
    }
    return json.dumps(output, ensure_ascii=False)


class ReFrameJudgeDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def build_messages(record, project_root, prompt, include_reason):
    user_content = [
        {"type": "image", "url": image_uri(resolve_image_path(project_root, record["source_image"]))},
        {"type": "image", "url": image_uri(resolve_image_path(project_root, record["edited_image"]))},
        {"type": "text", "text": prompt},
    ]
    prompt_messages = [{"role": "user", "content": user_content}]
    full_messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": target_json(record, include_reason)},
    ]
    return prompt_messages, full_messages


class Qwen35Collator:
    def __init__(self, processor, project_root, prompt, include_reason):
        self.processor = processor
        self.project_root = project_root
        self.prompt = prompt
        self.include_reason = include_reason

    def __call__(self, records):
        if len(records) != 1:
            raise ValueError("qwen35_lora_train.py currently supports --train-batch-size 1 only.")
        prompt_messages, full_messages = build_messages(
            records[0],
            self.project_root,
            self.prompt,
            self.include_reason,
        )
        prompt_inputs = self.processor.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        full_inputs = self.processor.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        labels = full_inputs["input_ids"].clone()
        prompt_len = min(prompt_inputs["input_ids"].shape[1], labels.shape[1])
        labels[:, :prompt_len] = -100
        if "attention_mask" in full_inputs:
            labels[full_inputs["attention_mask"] == 0] = -100
        full_inputs["labels"] = labels
        return full_inputs


def lora_target_modules(value):
    value = value.strip()
    if value == "all-linear":
        return "all-linear"
    return [item.strip() for item in value.split(",") if item.strip()]


def disable_peft_adapter_probe(local_only):
    """Work around transformers/huggingface_hub making a Hub HEAD request for
    adapter_config.json even when local_files_only=True.  This script loads a
    base model and attaches LoRA itself, so the adapter probe is unnecessary
    and fails when the configured mirror endpoint is unreachable.
    """
    if not local_only:
        return
    import transformers.utils.peft_utils as peft_utils

    peft_utils.find_adapter_config_file = lambda *args, **kwargs: None


def load_model(args):
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        device_map = {"": local_rank}
        print(f"Distributed mode: local_rank={local_rank}, device_map={device_map}")
    else:
        device_map = "auto"
    
    model_kwargs = {
        "device_map": device_map,
        "cache_dir": args.hf_cache_dir,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    model_kwargs["dtype"] = "auto" if args.torch_dtype == "auto" else getattr(torch, args.torch_dtype)
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForImageTextToText.from_pretrained(args.model_name, **model_kwargs)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "config"):
            model.config.use_cache = False
        model.enable_input_require_grads()
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules(args.lora_target_modules),
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--hf-cache-dir", type=Path, default=Path("data/cache/huggingface"))
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--include-reason", action="store_true")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="all-linear")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", choices=["fp16", "bf16", "no"], default="no")
    args = parser.parse_args()

    if args.train_batch_size != 1:
        raise ValueError("Only --train-batch-size 1 is currently supported.")

    set_seed(args.seed)
    project_root = args.project_root.resolve()
    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else PROMPT

    disable_peft_adapter_probe(args.local_files_only)

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        cache_dir=args.hf_cache_dir,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    model = load_model(args)

    train_records = read_jsonl(args.train_jsonl, args.max_train_samples, args.seed)
    val_records = read_jsonl(args.val_jsonl, args.max_val_samples, args.seed)
    collator = Qwen35Collator(processor, project_root, prompt, args.include_reason)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        eval_strategy="steps" if args.eval_steps else "no",
        logging_strategy="steps",
        seed=args.seed,
        fp16=args.mixed_precision == "fp16",
        bf16=args.mixed_precision == "bf16",
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ReFrameJudgeDataset(train_records),
        eval_dataset=ReFrameJudgeDataset(val_records),
        data_collator=collator,
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    if args.eval_steps:
        eval_results = trainer.evaluate()
        print(f"Final eval results: {eval_results}")


if __name__ == "__main__":
    main()
