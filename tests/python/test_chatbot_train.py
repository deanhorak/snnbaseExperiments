from __future__ import annotations

import json
import contextlib
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


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _dataset(root: Path) -> None:
    root.mkdir()
    shards: dict[str, object] = {}
    for split, identifier in (
        ("train", "fixture-0"),
        ("validation", "fixture-14"),
        ("test", "fixture-12"),
    ):
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
