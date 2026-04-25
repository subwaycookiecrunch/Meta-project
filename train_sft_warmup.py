#!/usr/bin/env python3
"""
train_sft_warmup.py
====================
SFT warmup phase for metacognitive format training.

Teaches Qwen3-1.7B the output format (budget_prediction + think + tool_call)
via supervised fine-tuning on 50 demonstration trajectories BEFORE running
GRPO. This eliminates the ~30% zero-reward rate in GRPO caused by the model
not knowing how to produce valid tool calls.

Two-phase training:
  1. SFT warmup (this script) — 2-3 epochs, ~30-60 minutes on A10G
  2. GRPO refinement (train_grpo.py) — loads the SFT adapter, continues
     with RL to learn the allocation policy

Usage:
    python train_sft_warmup.py
    # Then: ADAPTER_PATH=./grpo_output/sft_adapter python train_grpo.py

Environment variables:
    SFT_EPOCHS      Number of SFT epochs (default: 3)
    SFT_LR          Learning rate (default: 2e-5)
    SFT_BATCH_SIZE  Batch size (default: 1)
    SFT_GRAD_ACCUM  Gradient accumulation steps (default: 4)
    MODEL_NAME      Base model (default: Qwen/Qwen3-1.7B)
"""
from __future__ import annotations

import json
import os
import sys
import torch
from datetime import datetime

# ── Configuration ───────────────────────────────────────────────────
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-1.7B")
ROOT = os.path.dirname(os.path.abspath(__file__))
SFT_DATA = os.path.join(ROOT, "data", "sft_demonstrations.json")
OUTPUT_DIR = os.path.join(ROOT, "grpo_output", "sft_adapter")

SFT_EPOCHS = int(os.environ.get("SFT_EPOCHS", "3"))
SFT_LR = float(os.environ.get("SFT_LR", "2e-5"))
SFT_BATCH_SIZE = int(os.environ.get("SFT_BATCH_SIZE", "1"))
SFT_GRAD_ACCUM = int(os.environ.get("SFT_GRAD_ACCUM", "4"))
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 32


def main():
    print("=" * 60)
    print("SFT Warmup — Metacognitive Format Training")
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {SFT_EPOCHS} | LR: {SFT_LR}")
    print(f"Batch: {SFT_BATCH_SIZE} x {SFT_GRAD_ACCUM} grad accum")
    print(f"LoRA: r={LORA_R}, α={LORA_ALPHA}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    # ── Load SFT data ──────────────────────────────────────────────
    if not os.path.exists(SFT_DATA):
        print(f"❌ SFT data not found at {SFT_DATA}")
        print("   Run: python scripts/generate_sft_data.py")
        sys.exit(1)

    with open(SFT_DATA) as f:
        sft_examples = json.load(f)
    print(f"📊 Loaded {len(sft_examples)} SFT demonstrations")

    # ── Load model ────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig

    print(f"\n🔄 Loading {MODEL_NAME}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Force LM head to bfloat16 (same fix as train_grpo.py)
    if hasattr(model, "lm_head"):
        model.lm_head.to(torch.bfloat16)
        def _cast_hook(module, args):
            return tuple(a.to(torch.bfloat16) if hasattr(a, "to") else a for a in args)
        model.lm_head.register_forward_pre_hook(_cast_hook)

    # ── LoRA ─────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Format data for SFT ──────────────────────────────────────
    # Convert messages to the chat template format
    formatted_data = []
    for ex in sft_examples:
        try:
            text = tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=True,
            )
            formatted_data.append({"text": text})
        except Exception:
            text = tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            formatted_data.append({"text": text})

    from datasets import Dataset
    dataset = Dataset.from_list(formatted_data)
    print(f"📊 Dataset: {len(dataset)} examples")

    # ── Training config ──────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=SFT_EPOCHS,
        per_device_train_batch_size=SFT_BATCH_SIZE,
        gradient_accumulation_steps=SFT_GRAD_ACCUM,
        learning_rate=SFT_LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        fp16=False,
        bf16=True,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        report_to="none",
        optim="adamw_8bit",
        gradient_checkpointing=True,
    )

    # ── Train ────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"\n🚀 Starting SFT warmup at {datetime.now().strftime('%H:%M:%S')}...")
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ SFT adapter saved to {OUTPUT_DIR}")
    print(f"   Next step: ADAPTER_PATH={OUTPUT_DIR} python train_grpo.py")


if __name__ == "__main__":
    main()
