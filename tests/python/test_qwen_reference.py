from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools"))

from qwen_contract import (  # noqa: E402
    PHASE0_MODEL_ID,
    PHASE0_REVISION,
    PHASE0_TOKENIZER_FINGERPRINT,
    require_sha256,
    verify_asset_manifest,
)
from qwen_reference import generate_tokenizer_golden  # noqa: E402

FIXTURES = REPOSITORY / "tests" / "fixtures" / "chatbot"


class FakeByteTokenizer:
    is_fast = True
    vocab_size = 256
    bos_token_id = None
    eos_token_id = None
    pad_token_id = None
    unk_token_id = None

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int], **_: object) -> str:
        return bytes(token_ids).decode("utf-8")

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str | list[int]:
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>\n"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
            if not enable_thinking:
                rendered += "<think>\n\n</think>\n\n"
        return self.encode(rendered) if tokenize else rendered


class QwenReferenceTests(unittest.TestCase):
    def test_tiny_fake_tokenizer_generates_encode_decode_and_chat_goldens(self) -> None:
        assets = verify_asset_manifest(
            FIXTURES / "fake_qwen" / "assets",
            "tests/FakeQwen",
            "0123456789abcdef0123456789abcdef01234567",
            "e2f448429ba76678d32ed7cf99e8d9adb031944a11a170771365a0f76f29a742",
        )
        cases = {
            "schema_version": 1,
            "encode_cases": [
                {"id": "ascii", "text": "SNN", "add_special_tokens": False}
            ],
            "chat_cases": [
                {
                    "id": "chat",
                    "messages": [{"role": "user", "content": "Ping"}],
                    "add_generation_prompt": True,
                    "enable_thinking": False,
                }
            ],
        }
        result = generate_tokenizer_golden(FakeByteTokenizer(), assets, cases)
        self.assertEqual(result["encode_cases"][0]["token_ids"], [83, 78, 78])
        self.assertEqual(result["encode_cases"][0]["decoded"], "SNN")
        self.assertTrue(result["chat_cases"][0]["rendered"].endswith("</think>\n\n"))

    def test_checked_in_qwen_golden_is_self_consistent(self) -> None:
        with (FIXTURES / "qwen3_phase0_tokenizer_golden.json").open(
            encoding="utf-8"
        ) as stream:
            golden = json.load(stream)
        self.assertEqual(golden["oracle"]["model_id"], PHASE0_MODEL_ID)
        self.assertEqual(golden["oracle"]["revision"], PHASE0_REVISION)
        self.assertEqual(
            golden["oracle"]["tokenizer_fingerprint_sha256"],
            PHASE0_TOKENIZER_FINGERPRINT,
        )
        self.assertEqual(golden["tokenizer"]["class"], "Qwen2TokenizerFast")
        self.assertEqual(golden["tokenizer"]["size_with_added_tokens"], 151669)
        for case in golden["encode_cases"]:
            self.assertEqual(case["text"], case["decoded"])

    def test_checked_in_reference_logits_identify_the_exact_oracle(self) -> None:
        with (FIXTURES / "qwen3_phase0_reference_logits.json").open(
            encoding="utf-8"
        ) as stream:
            golden = json.load(stream)

        self.assertEqual(golden["schema_version"], 1)
        self.assertEqual(golden["kind"], "snnbase.qwen-reference-logits")
        oracle = golden["oracle"]
        self.assertEqual(oracle["model_id"], PHASE0_MODEL_ID)
        self.assertEqual(oracle["revision"], PHASE0_REVISION)
        self.assertEqual(
            oracle["tokenizer_fingerprint_sha256"],
            PHASE0_TOKENIZER_FINGERPRINT,
        )
        self.assertEqual(
            oracle["checkpoint_files"],
            [
                {
                    "path": "model.safetensors",
                    "sha256": (
                        "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"
                    ),
                    "size_bytes": 1_192_135_096,
                }
            ],
        )

        expected_hashes = {
            "next-token-ascii": (
                "d3bc6c1c21075d6bccb38a6f7b706553bfba226b9f048796728b9be187fe993d"
            ),
            "next-token-snn": (
                "9e92a6aa80e2795e0b512187d4a2376522dc27779af4642924b0e45b8074fa01"
            ),
            "next-token-unicode": (
                "43a6fdede3904db5fbac19b51933270d3a1a3e37029bc4bac194558c1e388caa"
            ),
        }
        self.assertEqual(golden["execution"]["attention_implementation"], "eager")
        cases = {case["id"]: case for case in golden["cases"]}
        self.assertEqual(set(cases), set(expected_hashes))
        for identifier, expected_hash in expected_hashes.items():
            case = cases[identifier]
            logits = case["final_logits"]
            self.assertEqual(logits["shape"], [151_936])
            self.assertEqual(logits["canonical_dtype"], "float32")
            self.assertEqual(logits["byte_order"], "little")
            self.assertEqual(require_sha256(logits["sha256"]), expected_hash)
            self.assertEqual(len(logits["top_k"]), 16)
            top_logits = [entry["logit"] for entry in logits["top_k"]]
            self.assertEqual(top_logits, sorted(top_logits, reverse=True))
            for entry in [*logits["top_k"], *logits["probes"]]:
                self.assertGreaterEqual(entry["token_id"], 0)
                self.assertLess(entry["token_id"], 151_936)

    @unittest.skipUnless(
        os.environ.get("SNNBASE_QWEN_ASSETS"), "real tokenizer snapshot not configured"
    )
    def test_real_snapshot_exactly_regenerates_checked_in_golden(self) -> None:
        from qwen_contract import load_json, load_transformers_tokenizer

        assets = verify_asset_manifest(
            Path(os.environ["SNNBASE_QWEN_ASSETS"]),
            PHASE0_MODEL_ID,
            PHASE0_REVISION,
            PHASE0_TOKENIZER_FINGERPRINT,
        )
        result = generate_tokenizer_golden(
            load_transformers_tokenizer(assets),
            assets,
            load_json(FIXTURES / "qwen3_phase0_cases.json"),
        )
        with (FIXTURES / "qwen3_phase0_tokenizer_golden.json").open(
            encoding="utf-8"
        ) as stream:
            expected = json.load(stream)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
