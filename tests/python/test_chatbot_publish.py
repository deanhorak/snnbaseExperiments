from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools"))

import chatbot_publish  # noqa: E402
from chatbot_config import EXACT_TRACK, load_config  # noqa: E402
from chatbot_publish import (  # noqa: E402
    PublicationError,
    assemble_publication_manifest,
    validate_publication_file,
    validate_publication_manifest,
    write_manifest_exclusive,
)
from qwen_contract import (  # noqa: E402
    PHASE0_CONFIG_SHA256,
    PHASE0_MODEL_ID,
    PHASE0_REVISION,
    PHASE0_TOKENIZER_CONFIG_SHA256,
    PHASE0_TOKENIZER_FINGERPRINT,
    PHASE0_TOKENIZER_JSON_SHA256,
    PHASE0_WEIGHT_SHA256,
)

PUBLICATION_SOURCE_IDS = frozenset(
    {
        "00000000-0000-4000-8000-000000000001",
        "00000000-0000-4000-8000-000000000002",
        "00000000-0000-4000-8000-000000000005",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _initialize_repository(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "tests@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Offline Tests"],
        check=True,
    )
    (path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)


class PublicationFixture:
    def __init__(self, root: Path, *, config_name: str = "snn-baseline.json"):
        self.root = root
        self.experiments_repo = root / "experiments-repo"
        self.snnbase_repo = root / "snnbase-repo"
        _initialize_repository(self.experiments_repo)
        _initialize_repository(self.snnbase_repo)

        self.executable = root / "chatbot_experiment"
        self.executable.write_bytes(b"#!/bin/sh\nexit 0\n")
        self.executable.chmod(0o755)

        self.qwen_assets = root / "qwen-assets"
        self.qwen_assets.mkdir()
        (self.qwen_assets / "config.json").write_text("{}\n", encoding="utf-8")
        (self.qwen_assets / "tokenizer.json").write_text(
            '{"version":"fixture"}\n', encoding="utf-8"
        )
        (self.qwen_assets / "tokenizer_config.json").write_text(
            '{"tokenizer_class":"Fixture"}\n', encoding="utf-8"
        )
        (self.qwen_assets / "model.safetensors").write_bytes(b"fixture weights\n")

        self.source = root / "source.jsonl"
        self.source.write_text(
            "".join(
                json.dumps(
                    {
                        "id": identifier,
                        "messages": [
                            {"role": "user", "content": "u"},
                            {"role": "assistant", "content": "a"},
                        ],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for identifier in sorted(PUBLICATION_SOURCE_IDS)
            ),
            encoding="utf-8",
        )
        self.dataset = root / "dataset"
        self.dataset.mkdir()
        shards: dict[str, object] = {}
        for name in ("train", "validation", "test"):
            path = self.dataset / f"{name}.jsonl"
            path.write_text(f'{{"fixture":"{name}"}}\n', encoding="utf-8")
            shards[name] = {
                "path": path.name,
                "record_count": 1,
                "input_token_count": 4,
                "target_token_count": 2,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        tokenizer = {
            "model_id": PHASE0_MODEL_ID,
            "revision": PHASE0_REVISION,
            "fingerprint_sha256": PHASE0_TOKENIZER_FINGERPRINT,
        }
        self.dataset_manifest = self.dataset / "dataset-manifest.json"
        dataset_manifest = {
            "schema_version": 1,
            "kind": "snnbase.chatbot-token-dataset",
            "token_protocol": "snnbase.chatbot.tokens/v1",
            "source": {
                "filename": self.source.name,
                "size_bytes": self.source.stat().st_size,
                "sha256": _sha256(self.source),
                "record_schema": "exactly {id,messages}",
                "source_uri": "https://example.invalid/dataset/fixture-v1.jsonl",
                "version": "fixture-v1",
                "license": "CC0-1.0",
            },
            "tokenizer": tokenizer,
            "split": {
                "algorithm": "sha256-seed-bucket-v1",
                "seed": 42,
                "hash_payload": "seed u64 big-endian || id UTF-8",
                "bucket_count": 10_000,
                "test_buckets_inclusive": [0, 999],
                "validation_buckets_inclusive": [1000, 1999],
                "train_buckets_inclusive": [2000, 9999],
            },
            "shards": shards,
        }
        _write_json(self.dataset_manifest, dataset_manifest)

        self.config = root / "resolved-config.json"
        config = json.loads(
            (REPOSITORY / "configs" / "chatbot" / config_name).read_text(
                encoding="utf-8"
            )
        )
        _write_json(self.config, config)
        selected_config = load_config(self.config)
        runner_arguments = list(config["runner_binding"]["arguments"])
        self.exact = selected_config.track == EXACT_TRACK
        self.archive: Path | None = None
        if self.exact:
            self.archive = root / "qwen-dense.snnq"
            with self.archive.open("wb") as stream:
                stream.truncate(config["archive"]["size_bytes"])

        self.environment = root / "environment.json"
        _write_json(
            self.environment,
            {
                "toolchain": {
                    "compiler": "gcc 13.3.0",
                    "cmake": "3.31.4",
                    "torch": "2.5.1+cpu",
                    "cuda": "none",
                },
                "hardware": {"cpu": "fixture CPU", "gpu": "none"},
            },
        )

        self.run = root / "run"
        self.run.mkdir()
        self.checkpoint = self.run / "selected-checkpoint.pt"
        self.checkpoint.write_bytes(b"checkpoint fixture\n")
        validation_metrics = {
            "record_count": 1,
            "token_count": 2,
            "correct_token_count": 1,
            "loss": 1.25,
            "perplexity": 3.4903429574618414,
            "token_accuracy": 0.5,
            "mean_spike_rate": 0.125,
        }
        test_metrics = {
            "record_count": 1,
            "token_count": 2,
            "correct_token_count": 0,
            "loss": 1.5,
            "perplexity": 4.4816890703380645,
            "token_accuracy": 0.0,
            "mean_spike_rate": 0.25,
        }
        self.metrics = self.run / "metrics.jsonl"
        metric_rows = [
            {
                "schema_version": 1,
                "kind": "snnbase.chatbot-training-metric",
                "sequence": 0,
                "event": "record",
                "phase": "validation-selected-reload",
            },
            {
                "schema_version": 1,
                "kind": "snnbase.chatbot-training-metric",
                "sequence": 1,
                "event": "aggregate",
                "phase": "validation-selected-reload",
                "metrics": validation_metrics,
            },
            {
                "schema_version": 1,
                "kind": "snnbase.chatbot-training-metric",
                "sequence": 2,
                "event": "record",
                "phase": "test",
            },
            {
                "schema_version": 1,
                "kind": "snnbase.chatbot-training-metric",
                "sequence": 3,
                "event": "aggregate",
                "phase": "test",
                "metrics": test_metrics,
            },
        ]
        self.metrics.write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in metric_rows
            ),
            encoding="utf-8",
        )
        model = config["model"]
        qwen_weights = None
        if self.exact:
            archive = config["archive"]
            qwen_weights = {
                "model_id": PHASE0_MODEL_ID,
                "revision": PHASE0_REVISION,
                "archive_sha256": archive["sha256"],
                "source_checkpoint_sha256": PHASE0_WEIGHT_SHA256,
                "config_sha256": PHASE0_CONFIG_SHA256,
                "tokenizer_fingerprint_sha256": PHASE0_TOKENIZER_FINGERPRINT,
                "metadata_sha256": archive["metadata_sha256"],
                "payload_sha256": archive["payload_sha256"],
                "loaded_tensor_count": archive["dense_tensor_count"],
                "loaded_payload_bytes": archive["dense_tensor_bytes"],
            }
        metadata = {
            "vocabulary_size": model["vocabulary_size"],
            "maximum_sequence_length": model["maximum_sequence_length"],
            "model_dimension": model["model_dimension"],
            "layer_count": model["layer_count"],
            "head_dimension": model["head_dimension"],
            "simulation_steps": model["simulation_steps"],
            "query_key_normalization": model["query_key_normalization"],
            "spiking": model["spiking"],
            "qwen_weights": qwen_weights,
            "build_provenance": {
                "experiments": {
                    "revision": subprocess.run(
                        [
                            "git",
                            "-C",
                            str(self.experiments_repo),
                            "rev-parse",
                            "HEAD",
                        ],
                        check=True,
                        stdout=subprocess.PIPE,
                        text=True,
                    ).stdout.strip(),
                    "dirty": False,
                },
                "snnbase": {
                    "revision": subprocess.run(
                        [
                            "git",
                            "-C",
                            str(self.snnbase_repo),
                            "rev-parse",
                            "HEAD",
                        ],
                        check=True,
                        stdout=subprocess.PIPE,
                        text=True,
                    ).stdout.strip(),
                    "dirty": False,
                },
            },
            "training_state": {
                "micro_batch_count": 0,
                "optimizer_step_count": 0,
                "pending_accumulation_steps": 0,
                "learning_rate": config["training"]["learning_rate"],
            },
        }
        command = [
            str(self.executable.resolve()),
            "serve",
            *runner_arguments,
            "--save-checkpoint",
            str(self.run / ".latest-checkpoint.pt"),
        ]
        run_summary = {
            "schema_version": 1,
            "kind": "snnbase.chatbot-training-run",
            "status": "completed",
            "completed_utc": "2026-09-01T12:34:56.789123+00:00",
            "seed": config["seed"],
            "epochs": config["training"]["epochs"],
            "config": {
                **_file_record(self.config),
                "schema_version": config["schema_version"],
                "experiment_id": config["experiment_id"],
            },
            "dataset": {
                "path": str(self.dataset.resolve()),
                "manifest_sha256": _sha256(self.dataset_manifest),
                "tokenizer": tokenizer,
                "shards": shards,
            },
            "core": {
                "initialization": {
                    "mode": (
                        "verified_qwen_import" if self.exact else "deterministic_random"
                    ),
                    "executable": _file_record(self.executable),
                    **(
                        {
                            "qwen_archive": {
                                "path": str(self.archive.resolve()),
                                "sha256": config["archive"]["sha256"],
                                "size_bytes": config["archive"]["size_bytes"],
                            }
                        }
                        if self.exact and self.archive is not None
                        else {}
                    ),
                },
                "training_command": command,
                "training_metadata": metadata,
                "selected_checkpoint_metadata": {
                    **metadata,
                    "training_state": {
                        "micro_batch_count": 1,
                        "optimizer_step_count": 1,
                        "pending_accumulation_steps": 0,
                        "learning_rate": config["training"]["learning_rate"],
                    },
                },
            },
            "checkpoint": {
                "path": self.checkpoint.name,
                "sha256": _sha256(self.checkpoint),
                "size_bytes": self.checkpoint.stat().st_size,
            },
            "metrics_log": {
                "path": self.metrics.name,
                "sha256": _sha256(self.metrics),
                "size_bytes": self.metrics.stat().st_size,
            },
            "model_selection": {
                "best_validation_epoch": 1,
                "best_validation_loss": validation_metrics["loss"],
                "reloaded_validation_metrics": validation_metrics,
                "test_split_observed_during_selection": False,
            },
            "aggregates": [
                {"phase": "validation", "epoch": 1, "metrics": validation_metrics},
                {"phase": "test", "epoch": 1, "metrics": test_metrics},
            ],
        }
        self.summary = self.run / "run-summary.json"
        _write_json(self.summary, run_summary)

    @contextmanager
    def phase0_hashes(self):
        actual_hasher = chatbot_publish.sha256_file
        assets_root = self.qwen_assets.resolve()
        expected = {
            "config.json": PHASE0_CONFIG_SHA256,
            "tokenizer.json": PHASE0_TOKENIZER_JSON_SHA256,
            "tokenizer_config.json": PHASE0_TOKENIZER_CONFIG_SHA256,
            "model.safetensors": PHASE0_WEIGHT_SHA256,
        }

        def selected_hasher(path: Path) -> str:
            resolved = path.resolve()
            if resolved.parent == assets_root:
                return expected[resolved.name]
            if self.archive is not None and resolved == self.archive.resolve():
                config = json.loads(self.config.read_text(encoding="utf-8"))
                return config["archive"]["sha256"]
            return actual_hasher(resolved)

        with mock.patch.object(
            chatbot_publish, "sha256_file", side_effect=selected_hasher
        ):
            yield

    def assemble(self, *, publishable: bool = True) -> dict[str, object]:
        return assemble_publication_manifest(
            run_dir=self.run,
            config_path=self.config,
            dataset_source=self.source,
            experiments_repository=self.experiments_repo,
            snnbase_repository=self.snnbase_repo,
            environment_path=self.environment,
            qwen_assets_dir=self.qwen_assets,
            run_id="offline-publication-fixture",
            dataset_name="offline fixture conversations",
            publishable=publishable,
            notes=["offline unit-test fixture"],
        )


@unittest.skipUnless(shutil.which("git"), "git is required for provenance tests")
class ChatbotPublishTests(unittest.TestCase):
    def test_publishable_manifest_hashes_inputs_and_validates_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PublicationFixture(Path(temporary))
            with fixture.phase0_hashes():
                manifest = fixture.assemble()
            validate_publication_manifest(manifest, require_publishable=True)
            self.assertTrue(manifest["publishable"])
            self.assertEqual(manifest["created_utc"], "2026-09-01T12:34:56Z")
            self.assertEqual(manifest["config"]["sha256"], _sha256(fixture.config))
            self.assertEqual(
                manifest["executable"]["sha256"], _sha256(fixture.executable)
            )
            self.assertEqual(
                manifest["dataset"]["source"]["sha256"], _sha256(fixture.source)
            )
            artifacts = {item["kind"]: item["file"] for item in manifest["artifacts"]}
            self.assertEqual(
                artifacts["dataset_manifest"]["sha256"],
                _sha256(fixture.dataset_manifest),
            )
            self.assertEqual(
                artifacts["selected_checkpoint"]["sha256"],
                _sha256(fixture.checkpoint),
            )
            self.assertEqual(
                artifacts["training_metrics_jsonl"]["sha256"],
                _sha256(fixture.metrics),
            )
            self.assertEqual(manifest["reference_model"]["revision"], PHASE0_REVISION)
            tokenizer_hashes = {
                item["path"]: item["sha256"] for item in manifest["tokenizer"]["files"]
            }
            self.assertEqual(
                tokenizer_hashes,
                {
                    "tokenizer.json": PHASE0_TOKENIZER_JSON_SHA256,
                    "tokenizer_config.json": PHASE0_TOKENIZER_CONFIG_SHA256,
                },
            )
            self.assertFalse(manifest["repositories"]["experiments"]["dirty"])
            self.assertFalse(manifest["repositories"]["snnbase"]["dirty"])
            self.assertEqual(len(manifest["metrics"]), 8)

            output = Path(temporary) / "publication.json"
            write_manifest_exclusive(output, manifest)
            validated = validate_publication_file(output, require_publishable=True)
            self.assertEqual(validated, manifest)
            with self.assertRaisesRegex(PublicationError, "immutable"):
                write_manifest_exclusive(output, manifest)

    def test_publishable_mode_rejects_dirty_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PublicationFixture(Path(temporary))
            (fixture.experiments_repo / "untracked.txt").write_text(
                "dirty\n", encoding="utf-8"
            )
            with fixture.phase0_hashes(), self.assertRaisesRegex(
                PublicationError, "clean repository state.*experiments"
            ):
                fixture.assemble(publishable=True)
            with fixture.phase0_hashes():
                draft = fixture.assemble(publishable=False)
            self.assertFalse(draft["publishable"])
            self.assertFalse(draft["repositories"]["experiments"]["dirty"])
            validate_publication_manifest(draft)

    def test_publishable_mode_rejects_clean_head_newer_than_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PublicationFixture(Path(temporary))
            embedded_revision = json.loads(fixture.summary.read_text(encoding="utf-8"))[
                "core"
            ]["training_metadata"]["build_provenance"]["experiments"]["revision"]
            (fixture.experiments_repo / "tracked.txt").write_text(
                "new clean revision\n", encoding="utf-8"
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(fixture.experiments_repo),
                    "commit",
                    "-qam",
                    "newer source",
                ],
                check=True,
            )
            with fixture.phase0_hashes(), self.assertRaisesRegex(
                PublicationError, "clean repository state.*experiments"
            ):
                fixture.assemble(publishable=True)
            with fixture.phase0_hashes():
                draft = fixture.assemble(publishable=False)
            self.assertEqual(
                draft["repositories"]["experiments"]["revision"],
                embedded_revision,
            )

    def test_exact_config_is_validated_and_preserves_archive_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PublicationFixture(
                Path(temporary), config_name="qwen3-0.6b-ann-import.json"
            )
            with fixture.phase0_hashes():
                manifest = fixture.assemble()
            self.assertEqual(manifest["architecture"], "ann")
            artifacts = {item["kind"]: item["file"] for item in manifest["artifacts"]}
            self.assertEqual(
                artifacts["qwen_dense_archive"]["sha256"],
                json.loads(fixture.config.read_text(encoding="utf-8"))["archive"][
                    "sha256"
                ],
            )

    def test_completed_run_must_record_the_exact_validated_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PublicationFixture(Path(temporary))
            replacement = json.loads(fixture.config.read_text(encoding="utf-8"))
            replacement["dataset"]["path"] = "data/chatbot/different.jsonl"
            _write_json(fixture.config, replacement)
            with fixture.phase0_hashes(), self.assertRaisesRegex(
                PublicationError, "consumed chatbot config does not match"
            ):
                fixture.assemble()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = PublicationFixture(Path(temporary))
            summary = json.loads(fixture.summary.read_text(encoding="utf-8"))
            summary["config"] = None
            _write_json(fixture.summary, summary)
            with fixture.phase0_hashes(), self.assertRaisesRegex(
                PublicationError, "consumed config provenance"
            ):
                fixture.assemble()

    def test_metrics_log_streaming_bounds_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oversized = root / "oversized.jsonl"
            oversized.write_bytes(b'{"padding":"' + b"x" * 80 + b'"}\n')
            with self.assertRaisesRegex(PublicationError, "32-byte limit"):
                chatbot_publish._load_metrics_log(
                    oversized,
                    0,
                    {},
                    {},
                    maximum_line_bytes=32,
                    maximum_records=10,
                )

            too_many = root / "too-many.jsonl"
            rows = [
                {
                    "schema_version": 1,
                    "kind": "snnbase.chatbot-training-metric",
                    "sequence": sequence,
                    "event": "record",
                    "phase": "train",
                }
                for sequence in range(2)
            ]
            too_many.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(PublicationError, "1-record limit"):
                chatbot_publish._load_metrics_log(
                    too_many,
                    0,
                    {},
                    {},
                    maximum_line_bytes=1024,
                    maximum_records=1,
                )

    def test_missing_or_changed_retained_artifacts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PublicationFixture(Path(temporary))
            fixture.metrics.unlink()
            with fixture.phase0_hashes(), self.assertRaisesRegex(
                PublicationError, "training metrics log is missing"
            ):
                fixture.assemble()
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PublicationFixture(Path(temporary))
            fixture.checkpoint.unlink()
            with fixture.phase0_hashes(), self.assertRaisesRegex(
                PublicationError, "selected checkpoint is missing"
            ):
                fixture.assemble()
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PublicationFixture(Path(temporary))
            with fixture.metrics.open("a", encoding="utf-8") as stream:
                stream.write("{}\n")
            with fixture.phase0_hashes(), self.assertRaisesRegex(
                PublicationError, "does not match its recorded hash/size"
            ):
                fixture.assemble()

    def test_dataset_derivation_artifacts_are_retained_and_rehashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PublicationFixture(Path(temporary))
            conversion = fixture.dataset / "source-conversion-manifest.json"
            lineage = fixture.dataset / "source-lineage.jsonl"
            conversion.write_text('{"kind":"conversion-fixture"}\n', encoding="utf-8")
            lineage.write_text('{"tree_id":"fixture"}\n', encoding="utf-8")
            dataset_manifest = json.loads(
                fixture.dataset_manifest.read_text(encoding="utf-8")
            )
            dataset_manifest["derivation"] = {
                "conversion_manifest": {
                    "path": conversion.name,
                    "source_filename": "conversion-manifest.json",
                    "size_bytes": conversion.stat().st_size,
                    "sha256": _sha256(conversion),
                },
                "lineage": {
                    "path": lineage.name,
                    "source_filename": "lineage.jsonl",
                    "size_bytes": lineage.stat().st_size,
                    "sha256": _sha256(lineage),
                    "record_count": 3,
                },
            }
            _write_json(fixture.dataset_manifest, dataset_manifest)
            summary = json.loads(fixture.summary.read_text(encoding="utf-8"))
            summary["dataset"]["manifest_sha256"] = _sha256(fixture.dataset_manifest)
            _write_json(fixture.summary, summary)

            with mock.patch.object(
                chatbot_publish, "validate_dataset_derivation"
            ) as validator, fixture.phase0_hashes():
                validator.return_value = PUBLICATION_SOURCE_IDS
                manifest = fixture.assemble()
            validator.assert_called_once()
            artifacts = {item["kind"]: item["file"] for item in manifest["artifacts"]}
            self.assertEqual(
                artifacts["dataset_conversion_manifest"]["sha256"],
                _sha256(conversion),
            )
            self.assertEqual(
                artifacts["dataset_conversion_lineage"]["sha256"],
                _sha256(lineage),
            )

            with mock.patch.object(
                chatbot_publish,
                "validate_dataset_derivation",
                return_value=frozenset({"00000000-0000-4000-8000-000000000009"}),
            ), fixture.phase0_hashes(), self.assertRaisesRegex(
                PublicationError, "tree IDs differ"
            ):
                fixture.assemble()

            lineage.write_text("tampered\n", encoding="utf-8")
            with mock.patch.object(
                chatbot_publish,
                "validate_dataset_derivation",
                return_value=PUBLICATION_SOURCE_IDS,
            ), fixture.phase0_hashes(), self.assertRaisesRegex(
                PublicationError, "conversion lineage.*hash/size"
            ):
                fixture.assemble()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = PublicationFixture(Path(temporary))
            dataset_manifest = json.loads(
                fixture.dataset_manifest.read_text(encoding="utf-8")
            )
            dataset_manifest["derivation"] = None
            _write_json(fixture.dataset_manifest, dataset_manifest)
            summary = json.loads(fixture.summary.read_text(encoding="utf-8"))
            summary["dataset"]["manifest_sha256"] = _sha256(fixture.dataset_manifest)
            _write_json(fixture.summary, summary)
            with fixture.phase0_hashes(), self.assertRaisesRegex(
                PublicationError, "derivation"
            ):
                fixture.assemble()

    def test_exact_pins_and_schema_additional_properties_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PublicationFixture(Path(temporary))
            with fixture.phase0_hashes():
                manifest = fixture.assemble()
            invalid = copy.deepcopy(manifest)
            invalid["tokenizer"]["revision"] = "0" * 40
            with self.assertRaisesRegex(PublicationError, "const mismatch"):
                validate_publication_manifest(invalid)
            invalid = copy.deepcopy(manifest)
            invalid["unexpected"] = True
            with self.assertRaisesRegex(PublicationError, "unexpected"):
                validate_publication_manifest(invalid)

            config = json.loads(fixture.config.read_text(encoding="utf-8"))
            config["reference"]["tokenizer_fingerprint_sha256"] = "0" * 64
            _write_json(fixture.config, config)
            with fixture.phase0_hashes(), self.assertRaisesRegex(
                PublicationError, "resolved config is invalid"
            ):
                fixture.assemble()


if __name__ == "__main__":
    unittest.main()
