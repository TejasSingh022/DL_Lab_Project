"""Run: python -m unittest discover -s tests -v. No weights or downloads."""

import ast
import importlib.util
import unittest
from types import SimpleNamespace

import torch
from streamlit.testing.v1 import AppTest

from src.model_loader import ROOT, load_descriptions, load_labels, load_model
from src.preprocess import preprocess


class InferenceTests(unittest.TestCase):
    def test_original_model_imports(self):
        spec = importlib.util.spec_from_file_location(
            "original_roberta", ROOT / "PLM-ICD/src/modeling_roberta.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(issubclass(module.RobertaForMultilabelClassification, torch.nn.Module))

    def test_original_collator_equivalence(self):
        # Extract only the original nested collator, avoiding the training runner.
        tree = ast.parse((ROOT / "PLM-ICD/src/run_icd.py").read_text(encoding="utf-8"))
        collator_node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "data_collator")
        module = ast.Module(body=[collator_node], type_ignores=[])
        for length in (2, 127, 128, 129, 3071, 3072):
            feature = {"input_ids": list(range(length)), "attention_mask": [1] * length}

            class TokenizerFixture:
                pad_token_id = 1

                def __call__(self, text, **kwargs):
                    self.kwargs = kwargs
                    return {key: value[:] for key, value in feature.items()}

            tokenizer = TokenizerFixture()
            namespace = {
                "torch": torch, "tokenizer": tokenizer, "label_list": [],
                "args": SimpleNamespace(model_mode="laat", chunk_size=128),
            }
            exec(compile(module, "original_collator", "exec"), namespace)
            original = namespace["data_collator"]([{**feature, "label_ids": []}])
            actual = preprocess("a note", tokenizer)
            for key in actual:
                self.assertTrue(torch.equal(actual[key], original[key]), (length, key))
            self.assertEqual(tokenizer.kwargs, {
                "padding": False, "max_length": 3072,
                "truncation": True, "add_special_tokens": True,
            })

    def test_labels_and_description_coverage(self):
        labels, descriptions = load_labels(), load_descriptions()
        self.assertEqual(len(labels), 8921)
        self.assertEqual(sum(code not in descriptions for code in labels), 2171)

    def test_blank_note(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            preprocess(" \n", None)

    @unittest.skipIf((ROOT / "models/roberta-mimic3-full/pytorch_model.bin").exists(), "Checkpoint has since been supplied")
    def test_missing_checkpoint_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "pytorch_model.bin"):
            load_model()

    def test_streamlit_form(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertFalse(app.exception)
        app.button[0].click().run()
        self.assertIn("Enter a discharge summary", app.warning[0].value)
        if not (ROOT / "models/roberta-mimic3-full/pytorch_model.bin").exists():
            app.text_area[0].set_value("Discharge summary for missing-file handling test.")
            app.button[0].click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertIn("pytorch_model.bin", app.error[0].value)


if __name__ == "__main__":
    unittest.main()
