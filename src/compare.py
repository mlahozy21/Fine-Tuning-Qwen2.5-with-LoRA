"""Qualitative comparison: base Qwen2.5-1.5B vs. the LoRA-fine-tuned model.

Generates answers for a few held-out instructions with both models so the
effect of instruction tuning is visible side by side.

    python src/compare.py --adapter outputs/qwen2.5-1.5b-lora
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "### Instruction:\n{instruction}\n\n### Response:\n"
SAMPLES = [
    "Explain what overfitting is in one short paragraph.",
    "Write a haiku about gradient descent.",
    "List three practical uses of LoRA for fine-tuning large language models.",
]


def load(model_name, adapter=None):
    dtype = torch.bfloat16 if (torch.cuda.is_available() and
                               torch.cuda.is_bf16_supported()) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval()


def generate(model, tokenizer, instruction, max_new_tokens=200):
    inputs = tokenizer(PROMPT.format(instruction=instruction),
                       return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                            skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser(description="Compare base vs LoRA-tuned model.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--adapter", default="outputs/qwen2.5-1.5b-lora")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = load(args.model)
    tuned = load(args.model, adapter=args.adapter)

    for s in SAMPLES:
        print("=" * 72)
        print("INSTRUCTION:", s)
        print("\n[BASE]\n" + generate(base, tokenizer, s))
        print("\n[LoRA FINE-TUNED]\n" + generate(tuned, tokenizer, s))


if __name__ == "__main__":
    main()
