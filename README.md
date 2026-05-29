# Fine-Tuning Qwen2.5-1.5B with LoRA

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
2. Trains on Alpaca with an instruction prompt template (causal-LM objective).
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

## Expected result

After tuning, the model stops *continuing* the prompt and instead **answers in the
instruction/response format** it was trained on (concise, on-task responses), whereas the
base model tends to ramble or echo the prompt. The LoRA adapter saved under `outputs/` is
only a few MB.

## Repository layout

```
.
├── src/
│   ├── finetune_lora.py   # LoRA/QLoRA instruction fine-tuning (PEFT + Transformers)
│   └── compare.py         # base vs. fine-tuned generation comparison
├── notebooks/
│   └── finetune_qwen2.5_lora.ipynb   # one-click Colab pipeline
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
