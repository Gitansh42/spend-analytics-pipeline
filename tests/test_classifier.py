import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import pandas as pd

from src.classifier import classify_spend


class FakeOpenAI:
    response_category = 1
    calls = []

    def __init__(self, api_key):
        if not api_key:
            raise AssertionError("API key was not passed")
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        items = json.loads(prompt.split("Items: ", 1)[1])
        content = json.dumps({
            "results": [
                {"id": item["id"], "category": self.response_category, "confidence": 0.8}
                for item in items
            ]
        })
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content=content)
            )]
        )


class ClassifierApiTests(unittest.TestCase):
    def setUp(self):
        FakeOpenAI.calls = []
        FakeOpenAI.response_category = 1
        self.taxonomy = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", encoding="utf-8", delete=False
        )
        self.taxonomy.write(
            "category_l1,category_l2,category_l3\n"
            "Materials,Metals,Steel\n"
            "Services,Freight,Courier\n"
        )
        self.taxonomy.close()

    def tearDown(self):
        os.unlink(self.taxonomy.name)

    def classify(self, descriptions):
        fake_module = types.SimpleNamespace(OpenAI=FakeOpenAI)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), \
                patch.dict(sys.modules, {"openai": fake_module}):
            return classify_spend(
                pd.DataFrame({"description": descriptions}),
                self.taxonomy.name,
                "description",
                provider="openai",
                batch_size=1,
            )

    def test_batches_unique_descriptions_and_maps_taxonomy_path(self):
        result = self.classify(["steel plate", "steel plate", "courier service"])

        self.assertEqual(len(FakeOpenAI.calls), 2)
        self.assertEqual(result["l1"].tolist(), ["Services", "Services", "Services"])
        self.assertEqual(result["l2"].tolist(), ["Freight"] * 3)
        self.assertEqual(result["l3"].tolist(), ["Courier"] * 3)
        self.assertEqual(result["confidence"].tolist(), [0.8] * 3)

    def test_rejects_out_of_range_category(self):
        FakeOpenAI.response_category = 99
        with self.assertRaisesRegex(RuntimeError, "out-of-range"):
            self.classify(["steel plate"])

    def test_local_classifier_remains_available(self):
        result = classify_spend(
            pd.DataFrame({"description": ["steel plate"]}),
            self.taxonomy.name,
            "description",
            provider="local",
        )

        self.assertIn(result.loc[0, "l1"], {"Materials", "Services"})
        self.assertGreaterEqual(result.loc[0, "confidence"], 0)
        self.assertLessEqual(result.loc[0, "confidence"], 1)


if __name__ == "__main__":
    unittest.main()
