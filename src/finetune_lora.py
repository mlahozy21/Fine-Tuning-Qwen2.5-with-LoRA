"""LoRA instruction fine-tuning of Qwen2.5-1.5B (PEFT + Transformers).

Takes the *base* Qwen2.5-1.5B model and teaches it to follow instructions by
training small LoRA adapters on the Alpaca dataset. Only the adapters are
trained (~0.5% of the parameters), so it fits comfortably on a single GPU and
runs in ~30 minutes on a Colab A100/L4.

Run from the repository root:
    python src/finetune_lora.py --max-samples 3000 --epochs 1
Add --load-4bit to use QLoRA (4-bit) on smaller GPUs (e.g. a free T4).
"""

import argparse
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# Pure-python prompt formatting + response-only masking (offline-testable).
from prompt_utils import (  # noqa: E402
    IGNORE_INDEX,
    build_labels,
    format_example,
    format_prompt,
)


def set_seed(seed: int = 42) -> None:
    """Seed every RNG used in training for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_precision():
    """Use bf16 where supported (A100/L4), otherwise fp16 (e.g. T4)."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    return torch.float16, False, True


def main():
    ap = argparse.ArgumentParser(description="LoRA instruction tuning of Qwen2.5-1.5B.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--dataset", default="tatsu-lab/alpaca")
    ap.add_argument("--max-samples", type=int, default=3000,
                    help="Subset size for a fast run (0 = full dataset).")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--output-dir", default="outputs/qwen2.5-1.5b-lora")
    ap.add_argument("--load-4bit", action="store_true", help="Use QLoRA (4-bit).")
    ap.add_argument("--seed", type=int, default=42, help="Global RNG seed.")
    ap.add_argument("--mask-prompt", dest="mask_prompt", action="store_true",
                    default=True,
                    help="Compute the loss on the response only (default).")
    ap.add_argument("--no-mask-prompt", dest="mask_prompt", action="store_false",
                    help="Compute the loss over prompt+response (legacy behaviour).")
    args = ap.parse_args()

    set_seed(args.seed)

    dtype, use_bf16, use_fp16 = pick_precision()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"torch_dtype": dtype}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False
    if args.load_4bit:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset(args.dataset, split="train")
    if args.max_samples and args.max_samples < len(ds):
        ds = ds.shuffle(seed=args.seed).select(range(args.max_samples))

    eos = tokenizer.eos_token or ""

    def tokenize(ex):
        # Tokenize the prompt alone to find the response boundary, then the full
        # sequence. The number of prompt tokens is exactly how many leading
        # labels we mask out (response-only loss).
        prompt_ids = tokenizer(format_prompt(ex), truncation=True,
                               max_length=args.max_len, add_special_tokens=False)["input_ids"]
        full = tokenizer(format_example(ex) + eos, truncation=True,
                         max_length=args.max_len, add_special_tokens=False)
        input_ids = full["input_ids"]
        full["labels"] = build_labels(input_ids, len(prompt_ids),
                                      mask_prompt=args.mask_prompt)
        return full

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    if args.mask_prompt:
        # Seq2Seq collator pads `input_ids` with pad_token and `labels` with -100,
        # preserving the per-token response-only mask we built above.
        collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=IGNORE_INDEX,
                                          padding=True)
    else:
        # Legacy behaviour: labels == input_ids, loss over the whole sequence.
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        optim="paged_adamw_8bit" if args.load_4bit else "adamw_torch",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nLoRA adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
