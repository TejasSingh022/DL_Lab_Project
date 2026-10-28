"""Top ten multi-label predictions; sigmoid scores are independent."""

import torch

from src.preprocess import preprocess


def predict(text, predictor):
    batch = {key: value.to(predictor.device) for key, value in preprocess(text, predictor.tokenizer).items()}
    with torch.inference_mode():
        logits = predictor.model(**batch, return_dict=True).logits
        if logits.shape != (1, len(predictor.labels)) or not torch.isfinite(logits).all():
            raise ValueError("Model returned invalid logits; verify the checkpoint.")
        scores, indices = torch.topk(logits[0].sigmoid(), k=10)
    rows = []
    for score, index in zip(scores.cpu().tolist(), indices.cpu().tolist()):
        code = predictor.labels[index]
        rows.append({
            "ICD-9 code": code,
            "Description": predictor.descriptions.get(code, "Description unavailable"),
            "Prediction score": score,
        })
    return rows
