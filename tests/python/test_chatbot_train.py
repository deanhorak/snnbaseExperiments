from __future__ import annotations

import json
import contextlib
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools"))

from chatbot_prepare import (  # noqa: E402
    DATASET_KIND,
    DATASET_SCHEMA_VERSION,
    DERIVATION_KIND,
    DERIVATION_SCHEMA_VERSION,
    PREPARED_CONVERSION_MANIFEST_FILENAME,
    PREPARED_LINEAGE_FILENAME,
    SPLIT_ALGORITHM,
    SPLIT_BUCKET_COUNT,
    SPLIT_SEED,
)
from chatbot_token_protocol import PROTOCOL  # noqa: E402
from chatbot_train import load_prepared_dataset, main, run_training  # noqa: E402
from qwen_contract import (  # noqa: E402
    ContractError,
    PHASE0_MODEL_ID,
    PHASE0_REVISION,
    PHASE0_TOKENIZER_FINGERPRINT,
    sha256_file,
)

DERIVED_IDS = {
    "train": "00000000-0000-4000-8000-000000000002",
    "validation": "00000000-0000-4000-8000-000000000001",
    "test": "00000000-0000-4000-8000-000000000005",
}
DERIVED_ASSISTANT_IDS = {
    "train": "00000000-0000-4000-8000-000000000003",
    "validation": "00000000-0000-4000-8000-000000000004",
    "test": "00000000-0000-4000-8000-000000000006",
}


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _dataset(root: Path, identifiers: dict[str, str] | None = None) -> None:
    root.mkdir()
    shards: dict[str, object] = {}
    selected_identifiers = identifiers or {
        "train": "fixture-0",
        "validation": "fixture-14",
        "test": "fixture-12",
    }
    for split in ("train", "validation", "test"):
        identifier = selected_identifiers[split]
        path = root / f"{split}.jsonl"
        records = [
            {
                "schema_version": DATASET_SCHEMA_VERSION,
                "id": identifier,
                "input_ids": [1, 2, 3, 4],
                "loss_mask": [0, 0, 1, 1],
            }
        ]
        _write_jsonl(path, records)
        shards[split] = {
            "path": path.name,
            "record_count": 1,
            "input_token_count": 4,
            "target_token_count": 2,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "kind": DATASET_KIND,
        "token_protocol": PROTOCOL,
        "source": {
            "filename": "source.jsonl",
            "size_bytes": 123,
            "sha256": "b" * 64,
            "record_schema": "exactly {id,messages}",
            "source_uri": "https://example.invalid/conversations-v1.jsonl",
            "version": "conversations-v1",
            "license": "CC0-1.0",
        },
        "tokenizer": {
            "model_id": PHASE0_MODEL_ID,
            "revision": PHASE0_REVISION,
            "fingerprint_sha256": PHASE0_TOKENIZER_FINGERPRINT,
        },
        "split": {
            "algorithm": SPLIT_ALGORITHM,
            "seed": SPLIT_SEED,
            "hash_payload": "seed u64 big-endian || id UTF-8",
            "bucket_count": SPLIT_BUCKET_COUNT,
            "test_buckets_inclusive": [0, 999],
            "validation_buckets_inclusive": [1000, 1999],
            "train_buckets_inclusive": [2000, 9999],
        },
        "shards": shards,
    }
    (root / "dataset-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _derived_dataset(root: Path, *, source_sha256: str = "b" * 64) -> None:
    _dataset(root, DERIVED_IDS)
    lineage = "".join(
        json.dumps(
            {
                "tree_id": DERIVED_IDS[name],
                "message_ids": [
                    DERIVED_IDS[name],
                    DERIVED_ASSISTANT_IDS[name],
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for name in ("train", "validation", "test")
    ).encode("utf-8")
    lineage_path = root / PREPARED_LINEAGE_FILENAME
    lineage_path.write_bytes(lineage)
    raw_source = {
        "filename": "ready.trees.jsonl.gz",
        "source_uri": "https://example.invalid/oasst2/resolve/" + "2" * 40,
        "version": "2" * 40,
        "license": "Apache-2.0",
        "license_reviewed": True,
        "policy_reviewed": True,
        "compression": "gzip",
        "compressed": {"size_bytes": 100, "sha256": "3" * 64},
        "decompressed": {"size_bytes": 200, "sha256": "4" * 64},
    }
    converter = {
        "filename": "oasst2_convert.py",
        "version": "1.0.0",
        "size_bytes": 1234,
        "sha256": "5" * 64,
    }
    conversion_document = {
        "schema_version": 1,
        "kind": "snnbase.oasst2-conversion-manifest",
        "status": "completed",
        "canonical_serialization": "utf8-json-sort-keys-compact-lf-v1",
        "converter": converter,
        "source": raw_source,
        "output": {
            "path": "source.jsonl",
            "size_bytes": 123,
            "sha256": source_sha256,
            "record_count": 3,
            "message_count": 6,
            "content_bytes": 6,
            "fields": ["id", "messages"],
        },
        "lineage": {
            "path": "lineage.jsonl",
            "size_bytes": len(lineage),
            "sha256": hashlib.sha256(lineage).hexdigest(),
            "record_count": 3,
            "fields": ["tree_id", "message_ids"],
            "contains_user_ids_or_text": False,
        },
        "tokenizer": {
            "model_id": PHASE0_MODEL_ID,
            "revision": PHASE0_REVISION,
            "fingerprint_sha256": PHASE0_TOKENIZER_FINGERPRINT,
            "template": {
                "add_generation_prompt": False,
                "enable_thinking": True,
            },
            "maximum_input_tokens": 512,
            "selected_maximum_token_count": 4,
        },
        "counts": {"selected_tree_count": 3},
        "filter_reasons": {},
        "policy": {
            "profile": "quality05",
            "conservative_zero": {"active": False},
            "quality05": {"active": True},
        },
        "split": {
            "algorithm": SPLIT_ALGORITHM,
            "seed": SPLIT_SEED,
            "bucket_count": SPLIT_BUCKET_COUNT,
            "test_buckets_inclusive": [0, 999],
            "validation_buckets_inclusive": [1000, 1999],
            "train_buckets_inclusive": [2000, 9999],
            "record_counts": {"train": 1, "validation": 1, "test": 1},
        },
    }
    conversion_bytes = (
        json.dumps(
            conversion_document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    conversion_path = root / PREPARED_CONVERSION_MANIFEST_FILENAME
    conversion_path.write_bytes(conversion_bytes)
    manifest_path = root / "dataset-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"].update(
        {
            "sha256": source_sha256,
            "source_uri": f"urn:sha256:{source_sha256}",
            "version": f"sha256:{source_sha256}",
            "license": "Apache-2.0",
        }
    )
    manifest["derivation"] = {
        "schema_version": DERIVATION_SCHEMA_VERSION,
        "kind": DERIVATION_KIND,
        "profile": "quality05",
        "canonical_input": {
            "filename": "source.jsonl",
            "size_bytes": 123,
            "sha256": source_sha256,
        },
        "conversion_manifest": {
            "path": PREPARED_CONVERSION_MANIFEST_FILENAME,
            "source_filename": "conversion-manifest.json",
            "size_bytes": len(conversion_bytes),
            "sha256": hashlib.sha256(conversion_bytes).hexdigest(),
        },
        "lineage": {
            "path": PREPARED_LINEAGE_FILENAME,
            "source_filename": "lineage.jsonl",
            "size_bytes": len(lineage),
            "sha256": hashlib.sha256(lineage).hexdigest(),
            "record_count": 3,
        },
        "raw_source": raw_source,
        "converter": converter,
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


class ChatbotTrainTests(unittest.TestCase):
    def test_persistent_train_validation_test_run_is_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            _dataset(temporary / "dataset")
            dataset = load_prepared_dataset(temporary / "dataset")
            summary = run_training(
                dataset,
                temporary / "run",
                [
                    sys.executable,
                    str(REPOSITORY / "tests" / "fixtures" / "chatbot" / "fake_core.py"),
                    "serve",
                ],
                epochs=2,
                seed=42,
                max_line_bytes=1_048_576,
                checkpoint_filename="final-checkpoint.pt",
            )
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["model_selection"]["best_validation_epoch"], 2)
            self.assertFalse(
                summary["model_selection"]["test_split_observed_during_selection"]
            )
            self.assertEqual(
                summary["model_selection"]["selected_training_state"][
                    "optimizer_step_count"
                ],
                2,
            )
            self.assertEqual(
                summary["model_selection"]["best_validation_loss"],
                summary["model_selection"]["reloaded_validation_metrics"]["loss"],
            )
            self.assertEqual(
                [item["phase"] for item in summary["aggregates"]],
                [
                    "validation-baseline",
                    "train",
                    "validation",
                    "train",
                    "validation",
                    "test",
                ],
            )
            checkpoint = json.loads(
                (temporary / "run" / "final-checkpoint.pt").read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["micro_batches"], 2)
            self.assertEqual(checkpoint["optimizer_steps"], 2)
            self.assertEqual(checkpoint["operations"][-1], "flush")
            metric_lines = [
                json.loads(line)
                for line in (temporary / "run" / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [line["sequence"] for line in metric_lines],
                list(range(len(metric_lines))),
            )
            self.assertEqual(metric_lines[-1]["phase"], "test")
            self.assertEqual(
                len(
                    [
                        line
                        for line in metric_lines
                        if line.get("phase") == "validation-selected-reload"
                        and line["event"] == "aggregate"
                    ]
                ),
                1,
            )
            self.assertEqual(
                [
                    line["phase"]
                    for line in metric_lines
                    if line["event"] == "record" and line["phase"] == "test"
                ],
                ["test"],
            )
            with self.assertRaisesRegex(ContractError, "already exists"):
                run_training(
                    dataset,
                    temporary / "run",
                    [sys.executable, "unused.py", "serve"],
                    epochs=1,
                    seed=42,
                    max_line_bytes=1_048_576,
                    checkpoint_filename="checkpoint.pt",
                )

    def test_corrupt_shard_is_rejected_before_core_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            _dataset(root)
            with (root / "train.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ContractError, "hash/size mismatch"):
                load_prepared_dataset(root)

    def test_derived_dataset_artifacts_validate_before_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            _derived_dataset(root)
            dataset = load_prepared_dataset(root)
            self.assertEqual(dataset.manifest["derivation"]["profile"], "quality05")
            self.assertEqual(
                dataset.manifest["derivation"]["conversion_manifest"]["sha256"],
                sha256_file(root / PREPARED_CONVERSION_MANIFEST_FILENAME),
            )
            self.assertEqual(
                dataset.manifest["derivation"]["lineage"]["sha256"],
                sha256_file(root / PREPARED_LINEAGE_FILENAME),
            )

    def test_derived_dataset_rejects_tamper_swap_and_extra_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)

            tampered = temporary / "tampered"
            _derived_dataset(tampered)
            with (tampered / PREPARED_LINEAGE_FILENAME).open("ab") as lineage_stream:
                lineage_stream.write(b"{}\n")
            with self.assertRaisesRegex(ContractError, "lineage"):
                load_prepared_dataset(tampered)

            extra = temporary / "extra"
            _derived_dataset(extra)
            extra_manifest_path = extra / "dataset-manifest.json"
            extra_manifest = json.loads(extra_manifest_path.read_text("utf-8"))
            extra_manifest["derivation"]["unexpected"] = True
            extra_manifest_path.write_text(
                json.dumps(extra_manifest, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContractError, "unexpected fields"):
                load_prepared_dataset(extra)

            first = temporary / "first"
            second = temporary / "second"
            _derived_dataset(first, source_sha256="b" * 64)
            _derived_dataset(second, source_sha256="c" * 64)
            swapped_bytes = (
                second / PREPARED_CONVERSION_MANIFEST_FILENAME
            ).read_bytes()
            (first / PREPARED_CONVERSION_MANIFEST_FILENAME).write_bytes(swapped_bytes)
            first_manifest_path = first / "dataset-manifest.json"
            first_manifest = json.loads(first_manifest_path.read_text("utf-8"))
            first_manifest["derivation"]["conversion_manifest"].update(
                {
                    "size_bytes": len(swapped_bytes),
                    "sha256": hashlib.sha256(swapped_bytes).hexdigest(),
                }
            )
            first_manifest_path.write_text(
                json.dumps(first_manifest, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContractError, "canonical input"):
                load_prepared_dataset(first)

            wrong_tree = temporary / "wrong-tree"
            _derived_dataset(wrong_tree)
            lineage_path = wrong_tree / PREPARED_LINEAGE_FILENAME
            lineage_rows = [
                json.loads(line)
                for line in lineage_path.read_text(encoding="utf-8").splitlines()
            ]
            wrong_id = "00000000-0000-4000-8000-000000000009"
            lineage_rows[0]["tree_id"] = wrong_id
            lineage_rows[0]["message_ids"][0] = wrong_id
            _write_jsonl(lineage_path, lineage_rows)
            conversion_path = wrong_tree / PREPARED_CONVERSION_MANIFEST_FILENAME
            conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
            conversion["lineage"]["size_bytes"] = lineage_path.stat().st_size
            conversion["lineage"]["sha256"] = sha256_file(lineage_path)
            conversion_path.write_text(
                json.dumps(conversion, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            prepared_manifest_path = wrong_tree / "dataset-manifest.json"
            prepared_manifest = json.loads(
                prepared_manifest_path.read_text(encoding="utf-8")
            )
            prepared_manifest["derivation"]["lineage"].update(
                {
                    "size_bytes": lineage_path.stat().st_size,
                    "sha256": sha256_file(lineage_path),
                }
            )
            prepared_manifest["derivation"]["conversion_manifest"].update(
                {
                    "size_bytes": conversion_path.stat().st_size,
                    "sha256": sha256_file(conversion_path),
                }
            )
            prepared_manifest_path.write_text(
                json.dumps(prepared_manifest, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContractError, "lineage tree IDs"):
                load_prepared_dataset(wrong_tree)

    def test_derived_dataset_rejects_self_consistent_provenance_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            _derived_dataset(root)
            manifest_path = root / "dataset-manifest.json"
            manifest = json.loads(manifest_path.read_text("utf-8"))
            manifest["derivation"]["raw_source"]["license"] = "CC0-1.0"
            manifest["source"]["license"] = "CC0-1.0"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContractError, "provenance mismatch"):
                load_prepared_dataset(root)

    def test_self_consistent_but_misbucketed_shard_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            _dataset(root)
            train_path = root / "train.jsonl"
            test_path = root / "test.jsonl"
            train_content = train_path.read_text(encoding="utf-8")
            test_content = test_path.read_text(encoding="utf-8")
            train_path.write_text(test_content, encoding="utf-8")
            test_path.write_text(train_content, encoding="utf-8")
            manifest_path = root / "dataset-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for name, path in (("train", train_path), ("test", test_path)):
                manifest["shards"][name]["size_bytes"] = path.stat().st_size
                manifest["shards"][name]["sha256"] = sha256_file(path)
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContractError, "deterministic split"):
                load_prepared_dataset(root)

    def test_tampered_source_provenance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            _dataset(root)
            manifest_path = root / "dataset-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source"]["version"] = "latest"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContractError, "version must be immutable"):
                load_prepared_dataset(root)

    def test_config_seed_mismatch_fails_before_core_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            dataset = temporary / "dataset"
            _dataset(dataset)
            run = temporary / "run"
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                status = main(
                    [
                        "--dataset-dir",
                        str(dataset),
                        "--run-dir",
                        str(run),
                        "--core-executable",
                        sys.executable,
                        "--config",
                        str(REPOSITORY / "configs" / "chatbot" / "snn-baseline.json"),
                        "--seed",
                        "43",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("--seed does not match", error.getvalue())
            self.assertFalse(run.exists())

    def test_selected_checkpoint_state_must_actually_reload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            _dataset(temporary / "dataset")
            dataset = load_prepared_dataset(temporary / "dataset")
            with self.assertRaisesRegex(ContractError, "changed training state"):
                run_training(
                    dataset,
                    temporary / "bad-run",
                    [
                        sys.executable,
                        str(
                            REPOSITORY
                            / "tests"
                            / "fixtures"
                            / "chatbot"
                            / "fake_core.py"
                        ),
                        "serve",
                        "--ignore-checkpoint-state",
                    ],
                    epochs=1,
                    seed=42,
                    max_line_bytes=1_048_576,
                    checkpoint_filename="checkpoint.pt",
                )

    def test_selected_checkpoint_must_reproduce_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            _dataset(temporary / "dataset")
            dataset = load_prepared_dataset(temporary / "dataset")
            with self.assertRaisesRegex(ContractError, "reproduce validation metric"):
                run_training(
                    dataset,
                    temporary / "bad-run",
                    [
                        sys.executable,
                        str(
                            REPOSITORY
                            / "tests"
                            / "fixtures"
                            / "chatbot"
                            / "fake_core.py"
                        ),
                        "serve",
                        "--perturb-reloaded-loss",
                    ],
                    epochs=1,
                    seed=42,
                    max_line_bytes=1_048_576,
                    checkpoint_filename="checkpoint.pt",
                )


if __name__ == "__main__":
    unittest.main()
