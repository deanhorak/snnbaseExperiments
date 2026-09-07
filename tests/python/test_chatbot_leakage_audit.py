#!/usr/bin/env python3
"""Offline tests for the chatbot cross-split leakage audit."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

from chatbot_leakage_audit import (  # noqa: E402
    CONVERSION_KIND,
    Conversation,
    LeakageAuditError,
    _split_name,
    lexical_collisions,
    load_conversations,
    merge_collisions,
    recommended_exclusions,
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _identifier(split: str, start: int) -> str:
    for index in range(start, start + 100_000):
        value = f"00000000-0000-4000-8000-{index:012d}"
        if _split_name(value) == split:
            return value
    raise AssertionError("could not find a deterministic split fixture")


TRAIN_ID = _identifier("train", 1)
TRAIN_ID_2 = _identifier("train", 100_001)
VALIDATION_ID = _identifier("validation", 200_001)
TEST_ID = _identifier("test", 300_001)
TEST_ID_2 = _identifier("test", 400_001)


def _record(identifier: str, prompt: str, answer: str = "A useful answer.") -> dict:
    return {
        "id": identifier,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def _bundle(root: Path, records: list[dict]) -> Path:
    conversations = b"".join(_canonical(record) for record in records)
    (root / "conversations.jsonl").write_bytes(conversations)
    counts = {"train": 0, "validation": 0, "test": 0}
    for record in records:
        counts[_split_name(record["id"])] += 1
    manifest = {
        "schema_version": 1,
        "kind": CONVERSION_KIND,
        "status": "completed",
        "output": {
            "path": "conversations.jsonl",
            "sha256": hashlib.sha256(conversations).hexdigest(),
            "size_bytes": len(conversations),
            "record_count": len(records),
        },
        "split": {
            "algorithm": "sha256-seed-bucket-v1",
            "seed": 42,
            "bucket_count": 10_000,
            "record_counts": counts,
        },
    }
    path = root / "conversion-manifest.json"
    path.write_bytes(_canonical(manifest))
    return path


class LeakageAuditTests(unittest.TestCase):
    def test_loader_verifies_manifest_hash_counts_and_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _bundle(
                root,
                [
                    _record(TRAIN_ID, "Train prompt"),
                    _record(VALIDATION_ID, "Validation prompt"),
                    _record(TEST_ID, "Test prompt"),
                ],
            )
            records, _, provenance = load_conversations(path)
            self.assertEqual(len(records), 3)
            self.assertEqual(provenance["canonical_conversations"]["record_count"], 3)
            with (root / "conversations.jsonl").open("ab") as stream:
                stream.write(b"{}\n")
            with self.assertRaisesRegex(LeakageAuditError, "canonical conversations"):
                load_conversations(path)

    def test_loader_rejects_duplicate_json_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _bundle(
                root,
                [
                    _record(TRAIN_ID, "Train prompt"),
                    _record(VALIDATION_ID, "Validation prompt"),
                    _record(TEST_ID, "Test prompt"),
                ],
            )
            manifest = path.read_text(encoding="utf-8")
            path.write_text(
                manifest.replace('"kind":', '"kind":"duplicate","kind":', 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(LeakageAuditError, "duplicate JSON field"):
                load_conversations(path)

    def test_lexical_audit_finds_exact_and_contained_cross_split_text(self) -> None:
        prompt = "How can I safely back up my computer before an upgrade?"
        records = [
            Conversation(
                TRAIN_ID,
                "train",
                {
                    "root_prompt": prompt,
                    "user_context": prompt,
                    "assistant_targets": "Use two independent backups and verify them.",
                },
            ),
            Conversation(
                TEST_ID,
                "test",
                {
                    "root_prompt": f"  {prompt.upper()}  ",
                    "user_context": f"  {prompt.upper()}  ",
                    "assistant_targets": "Keep an offline copy and test restoration.",
                },
            ),
            Conversation(
                VALIDATION_ID,
                "validation",
                {
                    "root_prompt": prompt + " Please include detailed numbered steps.",
                    "user_context": prompt + " Please include detailed numbered steps.",
                    "assistant_targets": "Make a backup, verify it, and then upgrade.",
                },
            ),
        ]
        pairs = merge_collisions(lexical_collisions(records))
        reasons = {reason for pair in pairs for reason in pair["reasons"]}
        self.assertIn("exact_normalized", reasons)
        self.assertIn("fuzzy_ngram_containment", reasons)

    def test_remediation_retains_all_records_in_highest_priority_split(self) -> None:
        records = [
            Conversation(identifier, _split_name(identifier), {})
            for identifier in (
                TRAIN_ID,
                TRAIN_ID_2,
                VALIDATION_ID,
                TEST_ID,
                TEST_ID_2,
            )
        ]
        collisions = [
            {"ids": [TRAIN_ID, TEST_ID]},
            {"ids": [TRAIN_ID, TEST_ID_2]},
            {"ids": [TRAIN_ID_2, VALIDATION_ID]},
        ]
        self.assertEqual(
            recommended_exclusions(records, collisions),
            sorted([TRAIN_ID, TRAIN_ID_2]),
        )


if __name__ == "__main__":
    unittest.main()
