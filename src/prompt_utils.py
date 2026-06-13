"""Dependency-free prompt formatting and response-only label masking.

These helpers contain the *logic* that decides which tokens contribute to the
training loss. They deliberately have no torch / transformers / datasets import
so they can be unit-tested offline with a fake tokenizer (no model download).
"""

from __future__ import annotations

# Label id that PyTorch's cross-entropy ignores.
IGNORE_INDEX = -100

# Alpaca-style prompt template the model learns to complete.
PROMPT_NO_INPUT = "### Instruction:\n{instruction}\n\n### Response:\n"
PROMPT_INPUT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"


def format_prompt(ex: dict) -> str:
    """Render only the prompt (instruction [+ input] + the Response header)."""
    if ex.get("input", "").strip():
        return PROMPT_INPUT.format(instruction=ex["instruction"], input=ex["input"])
    return PROMPT_NO_INPUT.format(instruction=ex["instruction"])


def format_example(ex: dict) -> str:
    """Render one Alpaca row as prompt + response."""
    return format_prompt(ex) + ex["output"]


def build_labels(input_ids, prompt_len, mask_prompt=True):
    """Build the training ``labels`` for one tokenized example.

    With ``mask_prompt=True`` (the default) the first ``prompt_len`` tokens -
    i.e. the instruction / prompt - are set to ``IGNORE_INDEX`` (-100) so the
    loss is computed *only* over the response tokens. With ``mask_prompt=False``
    every token contributes to the loss (the standard causal-LM behaviour where
    labels == input_ids).

    Pure-python (no tokenizer / torch dependency) so it can be unit-tested
    directly with a fake tokenizer.
    """
    labels = list(input_ids)
    if mask_prompt:
        cut = min(prompt_len, len(labels))
        for i in range(cut):
            labels[i] = IGNORE_INDEX
    return labels
