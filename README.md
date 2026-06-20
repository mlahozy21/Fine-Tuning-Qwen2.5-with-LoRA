# Fine-Tuning Qwen2.5-1.5B with LoRA

> Companion repo: [Interpreting-LoRA-Fine-Tuning](https://github.com/mlahozy21/Interpreting-LoRA-Fine-Tuning)
> uses this same training setup and analyses *what* the LoRA update actually changes
> inside the model (update norms, effective rank, representation drift).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlahozy21/Fine-Tuning-Qwen2.5-with-LoRA/blob/main/notebooks/finetune_qwen2.5_lora.ipynb)

**Instruction-tuning a large language model with LoRA.** Starting from the *base*
`Qwen2.5-1.5B`, I train lightweight **LoRA** adapters (PEFT) on the Alpaca dataset so the
model learns to follow instructions — training only ~0.5% of the parameters. The pipeline
is compact and **Colab-ready**: a fast run finishes in **~30 minutes on an A100/L4** (and
runs on a free T4 with `--load-4bit` / QLoRA).

## Why LoRA

Full fine-tuning of an LLM updates every weight (expensive, large checkpoints). **LoRA**
freezes the pretrained model and learns small low-rank update matrices on the attention
and MLP projections. This trains a tiny fraction of the parameters, fits on a single
consumer GPU, and produces an adapter of a few MB instead of a multi-GB checkpoint.

## What it does

1. Loads the base `Qwen2.5-1.5B` and attaches LoRA adapters to the attention/MLP
   projections (`q/k/v/o_proj`, `gate/up/down_proj`).
2. Trains on Alpaca with an instruction prompt template. By default the
   cross-entropy loss is computed **on the response tokens only** — the prompt /
   instruction tokens are masked with `-100` (response-only masking). Pass
   `--no-mask-prompt` to fall back to the legacy full-sequence causal-LM
   objective. A global `--seed` (default 42) seeds Python/NumPy/torch for
   reproducibility.
3. Compares the **base** vs. the **fine-tuned** model on held-out instructions, so the
   effect of instruction tuning is visible side by side.

## Quick start (Colab, one click)

Click the badge above (or open `notebooks/finetune_qwen2.5_lora.ipynb`), set
**Runtime → GPU**, and run all cells.

## Run locally

```bash
pip install -r requirements.txt

# Train LoRA adapters (fast run: 3k examples, 1 epoch)
python src/finetune_lora.py --max-samples 3000 --epochs 1
# On a small GPU (e.g. T4), use QLoRA:
python src/finetune_lora.py --max-samples 3000 --epochs 1 --load-4bit

# Compare the base model with the fine-tuned one
python src/compare.py --adapter outputs/qwen2.5-1.5b-lora
```

Key flags: `--model`, `--dataset`, `--max-samples`, `--epochs`, `--lora-r`, `--load-4bit`.

## Result

Trained on a Colab **A100** with the fast settings above (3k Alpaca examples, 1 epoch,
~95 s). LoRA touches **18.5 M parameters — 1.18 %** of the 1.56 B-parameter model; the
saved adapter is only a few MB. Training loss fell from ~1.51 to ~1.37.

### Sample output (real, greedy decoding)

The clearest effect of instruction tuning is **format adherence**. Asked for a *haiku*,
the base model ignores the format and explains the algorithm in prose, while the
fine-tuned model attempts an actual short poem:

> **Instruction:** *Write a haiku about gradient descent.*
>
> **Base:** "Gradient descent is a powerful optimization algorithm that is used to
> minimize the cost function in machine learning models. It works by iteratively
> adjusting the model's parameters in the direction of steepest descent... " *(prose,
> ignores the haiku request)*
>
> **LoRA fine-tuned:**
> "Gradient descent is a powerful tool, / It's a method to find the minimum of a
> function, / It's a journey of discovery." *(three short lines — follows the form)*

On factual questions the gap is smaller, because **Qwen2.5-1.5B is already partly
instruction-capable out of the box**, so both give reasonable answers (e.g. both define
overfitting correctly). This is expected for a strong modern base model and a short,
single-epoch run — the point of the demo is the pipeline and the *visible shift toward
the trained response format*, not a state-of-the-art assistant.

The comparison uses greedy decoding for reproducibility; with greedy decoding the small
fine-tuned model can fall into **repetition** on open-ended list prompts, which sampling
(`do_sample=True`, `temperature≈0.7`) or a longer run mitigates.

Reproduce it yourself with the one-click notebook `notebooks/compare_qwen2.5_lora.ipynb`
(train + compare end-to-end on a Colab GPU).

## Repository layout

```
.
├── src/
│   ├── finetune_lora.py   # LoRA/QLoRA instruction fine-tuning (PEFT + Transformers)
│   └── compare.py         # base vs. fine-tuned generation comparison
├── notebooks/
│   ├── finetune_qwen2.5_lora.ipynb   # one-click Colab pipeline (train)
│   └── compare_qwen2.5_lora.ipynb    # train + base-vs-fine-tuned comparison
├── requirements.txt  LICENSE  .gitignore
```

## References

- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*.
- Qwen2.5 (Alibaba) · Alpaca dataset (Taori et al., 2023) · PEFT / Transformers (Hugging Face).

## Troubleshooting

On Google Colab, `peft` may raise an `ImportError` about an incompatible `torchao`
version (Colab preinstalls an old one). `torchao` is **not used** here, so just remove it:

```bash
pip uninstall -y torchao
```

(The Colab notebook already does this in the install cell.)

## License

Released under the MIT License — see `LICENSE`.
