from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools"))

import qwen_contract  # noqa: E402
from qwen_contract import (  # noqa: E402
    ContractError,
    PHASE0_MODEL_ID,
    PHASE0_REVISION,
    PHASE0_TOKENIZER_FINGERPRINT,
    require_revision,
    tokenizer_fingerprint,
    verify_asset_manifest,
)

FIXTURE = REPOSITORY / "tests" / "fixtures" / "chatbot" / "fake_qwen" / "assets"
FAKE_MODEL_ID = "tests/FakeQwen"
FAKE_REVISION = "0123456789abcdef0123456789abcdef01234567"
FAKE_FINGERPRINT = "e2f448429ba76678d32ed7cf99e8d9adb031944a11a170771365a0f76f29a742"


class QwenContractTests(unittest.TestCase):
    def test_phase0_constants_are_immutable_values(self) -> None:
        self.assertEqual(PHASE0_MODEL_ID, "Qwen/Qwen3-0.6B-Base")
        self.assertEqual(PHASE0_REVISION, "da87bfb608c14b7cf20ba1ce41287e8de496c0cd")
        self.assertEqual(
            PHASE0_TOKENIZER_FINGERPRINT,
            "6a4dac166a643066a1af821907cfd1c998c8a195e6458a90979953e848b6e237",
        )

    def test_floating_or_missing_revision_is_rejected(self) -> None:
        for value in ("", "main", "v1", "A" * 40, "a" * 39):
            with self.subTest(value=value), self.assertRaises(ContractError):
                require_revision(value)

    def test_tiny_local_snapshot_verifies_offline(self) -> None:
        verified = verify_asset_manifest(
            FIXTURE, FAKE_MODEL_ID, FAKE_REVISION, FAKE_FINGERPRINT
        )
        self.assertEqual(verified.tokenizer_fingerprint, FAKE_FINGERPRINT)
        fingerprint, records = tokenizer_fingerprint(FIXTURE)
        self.assertEqual(fingerprint, FAKE_FINGERPRINT)
        self.assertEqual(
            [record["path"] for record in records],
            ["tokenizer.json", "tokenizer_config.json"],
        )

    def test_content_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "assets"
            shutil.copytree(FIXTURE, copied)
            (copied / "tokenizer.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "content mismatch"):
                verify_asset_manifest(
                    copied, FAKE_MODEL_ID, FAKE_REVISION, FAKE_FINGERPRINT
                )

    def test_unmanifested_chat_template_override_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "assets"
            shutil.copytree(FIXTURE, copied)
            (copied / "chat_template.jinja").write_text(
                "{{ messages | length }}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ContractError, "behavior-determining local files"
            ):
                verify_asset_manifest(
                    copied, FAKE_MODEL_ID, FAKE_REVISION, FAKE_FINGERPRINT
                )

    def test_weight_requiring_commands_reject_tokenizer_only_snapshot(self) -> None:
        with self.assertRaisesRegex(ContractError, "tokenizer-only snapshot"):
            verify_asset_manifest(
                FIXTURE,
                FAKE_MODEL_ID,
                FAKE_REVISION,
                FAKE_FINGERPRINT,
                require_weights=True,
            )

    def test_phase0_rejects_an_alternate_weight_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "assets"
            shutil.copytree(FIXTURE, copied)
            alternate = copied / "pytorch_model.bin"
            alternate.write_bytes(b"not-the-pinned-phase0-weight")
            manifest_path = copied / "qwen-assets.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["model_id"] = PHASE0_MODEL_ID
            manifest["revision"] = PHASE0_REVISION
            alternate_record = qwen_contract.file_record(copied, "pytorch_model.bin")
            manifest["files"].append(alternate_record)
            manifest["weight_files"] = [alternate_record]
            manifest["weights_present"] = True
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            patched_hashes = {
                "PHASE0_CONFIG_SHA256": qwen_contract.sha256_file(
                    copied / "config.json"
                ),
                "PHASE0_TOKENIZER_JSON_SHA256": qwen_contract.sha256_file(
                    copied / "tokenizer.json"
                ),
                "PHASE0_TOKENIZER_CONFIG_SHA256": qwen_contract.sha256_file(
                    copied / "tokenizer_config.json"
                ),
            }
            with mock.patch.multiple(qwen_contract, **patched_hashes):
                with self.assertRaisesRegex(
                    ContractError, "Phase 0 asset hash mismatch for model.safetensors"
                ):
                    verify_asset_manifest(
                        copied,
                        PHASE0_MODEL_ID,
                        PHASE0_REVISION,
                        FAKE_FINGERPRINT,
                        require_weights=True,
                    )


if __name__ == "__main__":
    unittest.main()
