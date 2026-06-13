"""Offline tests for response-only label masking and tokenization shapes.

These tests do NOT download any model. They exercise the pure-python masking
logic (`build_labels`, `format_prompt`) against a tiny deterministic *fake*
tokenizer, so they run in CI with no network access and no torch/transformers.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prompt_utils import (  # noqa: E402
    IGNORE_INDEX,
    build_labels,
    format_example,
    format_prompt,
)


class FakeTokenizer:
    """A whitespace tokenizer with a tiny growing vocab. Mimics the bits of the
    HF tokenizer API that `finetune_lora.tokenize` relies on."""

    def __init__(self):
        self.vocab = {}
        self.eos_token = "</s>"

    def _id(self, tok):
        return self.vocab.setdefault(tok, len(self.vocab) + 1)

    def __call__(self, text, truncation=True, max_length=512,
                 add_special_tokens=False):
        ids = [self._id(t) for t in text.split()]
        if truncation:
            ids = ids[:max_length]
        return {"input_ids": ids}


EXAMPLE_NO_INPUT = {
    "instruction": "Name three colours .",
    "input": "",
    "output": "Red green blue .",
}
EXAMPLE_WITH_INPUT = {
    "instruction": "Sort these numbers .",
    "input": "3 1 2",
    "output": "1 2 3 .",
}


def _tokenize(ex, tok, max_len=512, mask_prompt=True):
    """Mirror of finetune_lora.tokenize using a fake tokenizer."""
    prompt_ids = tok(format_prompt(ex), truncation=True, max_length=max_len,
                     add_special_tokens=False)["input_ids"]
    full = tok(format_example(ex) + tok.eos_token, truncation=True,
               max_length=max_len, add_special_tokens=False)
    full["labels"] = build_labels(full["input_ids"], len(prompt_ids),
                                  mask_prompt=mask_prompt)
    return full, len(prompt_ids)


def test_prompt_tokens_masked_response_tokens_kept():
    tok = FakeTokenizer()
    out, prompt_len = _tokenize(EXAMPLE_NO_INPUT, tok)
    labels = out["labels"]
    ids = out["input_ids"]

    # All prompt tokens must be ignored.
    assert all(l == IGNORE_INDEX for l in labels[:prompt_len])
    # No response token may be ignored; each must equal the real input id.
    assert prompt_len < len(labels)
    for i in range(prompt_len, len(labels)):
        assert labels[i] != IGNORE_INDEX
        assert labels[i] == ids[i]
    # At least one response token actually contributes to the loss.
    assert any(l != IGNORE_INDEX for l in labels)


def test_prompt_boundary_uses_prompt_only_length():
    """The number of masked tokens must equal the prompt-only token count."""
    tok = FakeTokenizer()
    out, prompt_len = _tokenize(EXAMPLE_WITH_INPUT, tok)
    n_masked = sum(1 for l in out["labels"] if l == IGNORE_INDEX)
    assert n_masked == prompt_len


def test_no_mask_prompt_keeps_all_labels():
    tok = FakeTokenizer()
    out, _ = _tokenize(EXAMPLE_NO_INPUT, tok, mask_prompt=False)
    assert out["labels"] == out["input_ids"]
    assert IGNORE_INDEX not in out["labels"]


def test_tokenization_shapes():
    """input_ids and labels must be equal-length 1-D sequences, response present."""
    tok = FakeTokenizer()
    for ex in (EXAMPLE_NO_INPUT, EXAMPLE_WITH_INPUT):
        out, prompt_len = _tokenize(ex, tok)
        assert len(out["input_ids"]) == len(out["labels"])
        assert len(out["input_ids"]) > prompt_len  # response + eos present
        assert all(isinstance(i, int) for i in out["input_ids"])


def test_truncation_respects_max_length():
    tok = FakeTokenizer()
    out, _ = _tokenize(EXAMPLE_WITH_INPUT, tok, max_len=4)
    assert len(out["input_ids"]) <= 4
    assert len(out["labels"]) == len(out["input_ids"])


def test_build_labels_handles_prompt_longer_than_sequence():
    # Defensive: if truncation cut the response off, masking must not overrun.
    labels = build_labels([1, 2, 3], prompt_len=10, mask_prompt=True)
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]
