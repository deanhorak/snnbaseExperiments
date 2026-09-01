from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools"))

from qwen_checkpoint import _safe_tensor_filename, tensor_record  # noqa: E402
from qwen_contract import ContractError  # noqa: E402


class FakeTensor:
    def __init__(self, raw: bytes):
        self._raw = raw

    def raw_bytes_for_test(self) -> bytes:
        return self._raw


class QwenCheckpointTests(unittest.TestCase):
    def test_tensor_record_hashes_canonical_fake_bytes(self) -> None:
        raw = bytes(range(16))
        record = tensor_record(
            "layer.weight", "fake.safetensors", [2, 2], "F32", tensor=FakeTensor(raw)
        )
        self.assertEqual(record["numel"], 4)
        self.assertEqual(record["storage_bytes"], 16)
        self.assertEqual(
            record["raw_little_endian_sha256"], hashlib.sha256(raw).hexdigest()
        )

    def test_tensor_byte_count_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ContractError, "raw byte size mismatch"):
            tensor_record(
                "bad", "fake.safetensors", [2], "F32", tensor=FakeTensor(b"short")
            )

    def test_export_filename_cannot_create_a_path(self) -> None:
        name = _safe_tensor_filename(2, "../../model/layer:weight")
        self.assertNotIn("/", name)
        self.assertTrue(name.startswith("0002-"))

    def test_fake_reference_logit_fixture_has_stable_contract(self) -> None:
        path = (
            REPOSITORY
            / "tests"
            / "fixtures"
            / "chatbot"
            / "fake_qwen"
            / "reference_logits.json"
        )
        with path.open(encoding="utf-8") as stream:
            fixture = json.load(stream)
        self.assertEqual(fixture["kind"], "snnbase.qwen-reference-logits")
        self.assertEqual(fixture["execution"]["seed"], 0)
        self.assertEqual(fixture["cases"][0]["final_logits"]["shape"], [256])


if __name__ == "__main__":
    unittest.main()
