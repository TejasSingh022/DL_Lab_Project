"""Single-note equivalent of run_icd.py's nested preprocessing/collator."""

import torch

MAX_LENGTH = 3072
CHUNK_SIZE = 128


def preprocess(text, tokenizer):
    """Tokenize once, then right-pad to a multiple of the chunk size."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a non-empty discharge summary.")
    # Match the original LAAT path: special tokens surround the whole note,
    # not each chunk. Do not introduce additional text cleaning or lowercasing.
    encoded = tokenizer(
        text, padding=False, max_length=MAX_LENGTH,
        truncation=True, add_special_tokens=True,
    )
    length = len(encoded["input_ids"])
    padded_length = ((length + CHUNK_SIZE - 1) // CHUNK_SIZE) * CHUNK_SIZE
    batch = {}
    for name in ("input_ids", "attention_mask", "token_type_ids"):
        if name in encoded:
            pad = tokenizer.pad_token_id if name == "input_ids" else 0
            values = encoded[name] + [pad] * (padded_length - length)
            batch[name] = torch.tensor(values, dtype=torch.long).view(1, -1, CHUNK_SIZE)
    return batch
