"""Load the original PLM-ICD architecture and local trained weights only."""

import importlib.util
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from src.preprocess import CHUNK_SIZE, MAX_LENGTH

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "roberta-mimic3-full"


@dataclass
class Predictor:
    model: torch.nn.Module
    tokenizer: object
    labels: list[str]
    descriptions: dict[str, str]
    device: torch.device


def load_labels():
    path = ROOT / "data" / "ALL_CODES.txt"
    if not path.is_file():
        raise ValueError(f"Missing original checkpoint label vocabulary: {path}")
    # Exactly the set + lexicographic sort used by PLM-ICD/src/run_icd.py.
    labels = sorted({line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()})
    if len(labels) != 8921:
        raise ValueError(f"MIMIC-III Full requires 8,921 original labels; found {len(labels)}. Restore the original ALL_CODES.txt.")
    digest = hashlib.sha256("\n".join(labels).encode("utf-8")).hexdigest()
    if digest != "e2a8b7574a398eaccd4fa7efd0bf12106b5ab394775b7400162d1d6d51f1a0c2":
        raise ValueError("ALL_CODES.txt differs from your confirmed original vocabulary. Restore that file; do not regenerate it from a subset.")
    return labels


def load_descriptions():
    path = ROOT / "data" / "icd9_descriptions.json"
    if not path.is_file():
        raise ValueError(f"Missing description file: {path}")
    descriptions = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(descriptions, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) or not v.strip()
        for k, v in descriptions.items()
    ):
        raise ValueError("icd9_descriptions.json must map code strings to non-empty description strings.")
    return descriptions


def load_model():
    source = ROOT / "PLM-ICD" / "src" / "modeling_roberta.py"
    required = [source] + [MODEL_DIR / name for name in (
        "pytorch_model.bin", "config.json", "vocab.json", "merges.txt",
    )]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise ValueError(
            "Required local files are missing:\n" + "\n".join(missing)
            + "\nPlace the trained MIMIC-III Full checkpoint and its matching tokenizer in models/roberta-mimic3-full. Nothing is downloaded automatically."
        )
    labels = load_labels()
    descriptions = load_descriptions()
    raw_config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
    if config.model_type != "roberta":
        raise ValueError("Expected a RoBERTa checkpoint.")
    if getattr(config, "model_mode", None) != "laat":
        raise ValueError("Checkpoint config.json must specify model_mode='laat'. Use the trained PLM-ICD Full checkpoint.")
    if config.num_labels != len(labels):
        raise ValueError(f"Checkpoint has {config.num_labels} labels, but vocabulary has {len(labels)}. Do not override num_labels.")
    # Generic LABEL_n entries are normal in the original saved checkpoints.
    id2label = config.id2label
    if id2label:
        mapped = [id2label.get(i) for i in range(len(labels))]
        generic = [f"LABEL_{i}" for i in range(len(labels))]
        if mapped != generic and mapped != labels:
            raise ValueError("Checkpoint id2label disagrees with the original vocabulary ordering.")
    # max_length on HF configs may be a generation default, not note length.
    # chunk_size, when explicitly saved, is meaningful for this architecture.
    if "chunk_size" in raw_config and raw_config["chunk_size"] != CHUNK_SIZE:
        raise ValueError("Checkpoint chunk_size differs from 128; verify its original inference settings.")
    if CHUNK_SIZE + config.pad_token_id >= config.max_position_embeddings:
        raise ValueError("Checkpoint position embeddings cannot accommodate 128-token chunks.")
    if MAX_LENGTH % CHUNK_SIZE:
        raise ValueError("max_length must be divisible by chunk_size.")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True, use_fast=True, do_lower_case=True,
    )
    if len(tokenizer) != config.vocab_size or tokenizer.pad_token_id != config.pad_token_id:
        raise ValueError("Tokenizer vocabulary size or padding ID does not match the checkpoint.")
    for name in ("bos_token_id", "eos_token_id"):
        if getattr(tokenizer, name) != getattr(config, name):
            raise ValueError(f"Tokenizer {name} does not match the checkpoint.")

    spec = importlib.util.spec_from_file_location("plm_icd_roberta", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Use the original attention implementation rather than newer SDPA defaults.
    config._attn_implementation = "eager"
    model = module.RobertaForMultilabelClassification(config)
    try:
        state = torch.load(MODEL_DIR / "pytorch_model.bin", map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("Expected the original save_pretrained state dictionary.")
        # Transformers 4.5 persisted this deterministic buffer; 4.44 does not.
        # Validate it against the original value before removing it, never weights.
        key = "roberta.embeddings.position_ids"
        if key in state:
            expected = model.roberta.embeddings.position_ids
            if not torch.equal(state[key], expected):
                raise ValueError("Checkpoint position_ids buffer is incompatible.")
            del state[key]
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ValueError(f"Cannot load the complete trained checkpoint: {exc}") from exc
    del state
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model.requires_grad_(False)
    return Predictor(model, tokenizer, labels, descriptions, device)
