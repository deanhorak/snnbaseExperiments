#!/usr/bin/env python3
"""Run deterministic chatbot training/validation/test over one persistent core."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence
from urllib.parse import urlsplit

from chatbot_config import load_config
from chatbot_prepare import (
    DATASET_KIND,
    DATASET_SCHEMA_VERSION,
    MAX_PROVENANCE_FIELD_BYTES,
    MAX_SOURCE_ID_BYTES_DEFAULT,
    SPLIT_ALGORITHM,
    SPLIT_BUCKET_COUNT,
    SPLIT_SEED,
    bounded_binary_lines,
    split_name,
)
from chatbot_token_protocol import MAX_LINE_BYTES_DEFAULT, PROTOCOL
from qwen_contract import (
    ContractError,
    PHASE0_MODEL_ID,
    PHASE0_REVISION,
    PHASE0_TOKENIZER_FINGERPRINT,
    load_json,
    require_sha256,
    sha256_file,
    write_json_atomic,
)

RUN_SCHEMA_VERSION = 1
RUN_KIND = "snnbase.chatbot-training-run"
METRICS_KIND = "snnbase.chatbot-training-metric"
ORDER_ALGORITHM = "sha256-seed-epoch-id-v1"
SHARD_NAMES = ("train", "validation", "test")
MAX_RECORD_ID_BYTES = MAX_SOURCE_ID_BYTES_DEFAULT
MAX_TOKEN_COUNT = 32_768


@dataclass(frozen=True)
class PreparedRecord:
    identifier: str
    input_ids: list[int]
    loss_mask: list[int]


@dataclass(frozen=True)
class PreparedDataset:
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    shards: Mapping[str, list[PreparedRecord]]


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _regular_child(root: Path, filename: str) -> Path:
    if not filename or Path(filename).is_absolute() or Path(filename).name != filename:
        raise ContractError(f"dataset shard path is not a plain filename: {filename!r}")
    path = root / filename
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ContractError(
            f"dataset shard is missing or escapes its root: {filename}"
        ) from error
    if not path.is_file() or path.is_symlink():
        raise ContractError(
            f"dataset shard must be a regular non-symlink file: {filename}"
        )
    return path


def _load_shard(path: Path) -> list[PreparedRecord]:
    records: list[PreparedRecord] = []
    with path.open("rb") as stream:
        for line_number, encoded in enumerate(
            bounded_binary_lines(stream, maximum_line_bytes=MAX_LINE_BYTES_DEFAULT), 1
        ):
            if not encoded.endswith(b"\n"):
                raise ContractError(
                    f"{path.name}:{line_number} is not newline terminated"
                )
            try:
                line = encoded.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ContractError(
                    f"invalid UTF-8 in {path.name}:{line_number}: {error}"
                ) from error
            if not line.strip():
                raise ContractError(f"{path.name}:{line_number} is blank")
            try:
                raw = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
            except (json.JSONDecodeError, UnicodeError) as error:
                raise ContractError(
                    f"invalid JSON in {path.name}:{line_number}: {error}"
                ) from error
            if not isinstance(raw, Mapping) or set(raw) != {
                "schema_version",
                "id",
                "input_ids",
                "loss_mask",
            }:
                raise ContractError(
                    f"{path.name}:{line_number} must contain exactly schema_version/id/input_ids/loss_mask"
                )
            identifier = raw["id"]
            input_ids = raw["input_ids"]
            loss_mask = raw["loss_mask"]
            try:
                identifier_bytes = (
                    len(identifier.encode("utf-8"))
                    if isinstance(identifier, str)
                    else 0
                )
            except UnicodeEncodeError as error:
                raise ContractError(
                    f"invalid UTF-8 id in {path.name}:{line_number}"
                ) from error
            if (
                not isinstance(raw["schema_version"], int)
                or isinstance(raw["schema_version"], bool)
                or raw["schema_version"] != DATASET_SCHEMA_VERSION
                or not isinstance(identifier, str)
                or not identifier
                or identifier_bytes > MAX_RECORD_ID_BYTES
                or any(ord(character) < 0x20 for character in identifier)
                or not isinstance(input_ids, list)
                or not 2 <= len(input_ids) <= MAX_TOKEN_COUNT
                or any(
                    not isinstance(token, int)
                    or isinstance(token, bool)
                    or token < 0
                    or token > 0xFFFFFFFF
                    for token in input_ids
                )
                or not isinstance(loss_mask, list)
                or len(loss_mask) != len(input_ids)
                or any(
                    value not in (0, 1) or isinstance(value, bool)
                    for value in loss_mask
                )
                or not any(loss_mask[1:])
            ):
                raise ContractError(
                    f"invalid prepared record in {path.name}:{line_number}"
                )
            records.append(PreparedRecord(identifier, input_ids, loss_mask))
    return records


def _provenance_text(value: Any, label: str) -> str:
    try:
        encoded_size = len(value.encode("utf-8")) if isinstance(value, str) else 0
    except UnicodeEncodeError as error:
        raise ContractError(f"dataset source {label} is not valid UTF-8") from error
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 0x20 for character in value)
        or encoded_size > MAX_PROVENANCE_FIELD_BYTES
    ):
        raise ContractError(f"dataset source {label} is invalid")
    return value


def _validate_source_provenance(value: Any) -> None:
    expected_fields = {
        "filename",
        "size_bytes",
        "sha256",
        "record_schema",
        "source_uri",
        "version",
        "license",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ContractError("dataset source provenance has unexpected fields")
    filename = _provenance_text(value["filename"], "filename")
    if Path(filename).name != filename:
        raise ContractError("dataset source filename must be a plain filename")
    size_bytes = value["size_bytes"]
    if (
        not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes < 0
    ):
        raise ContractError("dataset source size_bytes is invalid")
    require_sha256(value["sha256"], "dataset source SHA-256")
    if value["record_schema"] != "exactly {id,messages}":
        raise ContractError("dataset source record schema mismatch")
    source_uri = _provenance_text(value["source_uri"], "source_uri")
    try:
        parsed = urlsplit(source_uri)
    except ValueError as error:
        raise ContractError(f"dataset source URI is invalid: {error}") from error
    if not parsed.scheme or not (parsed.netloc or parsed.path):
        raise ContractError("dataset source URI must be absolute")
    version = _provenance_text(value["version"], "version")
    if version.casefold() in {
        "current",
        "head",
        "latest",
        "main",
        "master",
        "stable",
        "tip",
        "trunk",
    }:
        raise ContractError("dataset source version must be immutable")
    _provenance_text(value["license"], "license")


def load_prepared_dataset(root: Path) -> PreparedDataset:
    root = root.resolve(strict=True)
    manifest_path = _regular_child(root, "dataset-manifest.json")
    manifest = load_json(manifest_path)
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "schema_version",
        "kind",
        "token_protocol",
        "source",
        "tokenizer",
        "split",
        "shards",
    }:
        raise ContractError("dataset manifest has unexpected top-level fields")
    if (
        not isinstance(manifest["schema_version"], int)
        or isinstance(manifest["schema_version"], bool)
        or manifest["schema_version"] != DATASET_SCHEMA_VERSION
        or manifest["kind"] != DATASET_KIND
        or manifest["token_protocol"] != PROTOCOL
    ):
        raise ContractError("dataset manifest schema/kind/protocol mismatch")
    _validate_source_provenance(manifest["source"])
    split = manifest["split"]
    expected_split_fields = {
        "algorithm",
        "seed",
        "hash_payload",
        "bucket_count",
        "test_buckets_inclusive",
        "validation_buckets_inclusive",
        "train_buckets_inclusive",
    }
    if (
        not isinstance(split, Mapping)
        or set(split) != expected_split_fields
        or (
            split.get("algorithm"),
            split.get("seed"),
            split.get("hash_payload"),
            split.get("bucket_count"),
            split.get("test_buckets_inclusive"),
            split.get("validation_buckets_inclusive"),
            split.get("train_buckets_inclusive"),
        )
        != (
            SPLIT_ALGORITHM,
            SPLIT_SEED,
            "seed u64 big-endian || id UTF-8",
            SPLIT_BUCKET_COUNT,
            [0, 999],
            [1000, 1999],
            [2000, 9999],
        )
    ):
        raise ContractError("dataset split contract mismatch")
    tokenizer = manifest["tokenizer"]
    if (
        not isinstance(tokenizer, Mapping)
        or set(tokenizer) != {"model_id", "revision", "fingerprint_sha256"}
        or (
            tokenizer.get("model_id"),
            tokenizer.get("revision"),
            tokenizer.get("fingerprint_sha256"),
        )
        != (
            PHASE0_MODEL_ID,
            PHASE0_REVISION,
            PHASE0_TOKENIZER_FINGERPRINT,
        )
    ):
        raise ContractError("dataset tokenizer does not match the Phase 0 pin")
    shard_manifest = manifest["shards"]
    if not isinstance(shard_manifest, Mapping) or set(shard_manifest) != set(
        SHARD_NAMES
    ):
        raise ContractError(
            "dataset manifest must declare train/validation/test shards"
        )
    shards: dict[str, list[PreparedRecord]] = {}
    seen: set[str] = set()
    for name in SHARD_NAMES:
        record = shard_manifest[name]
        expected_fields = {
            "path",
            "record_count",
            "input_token_count",
            "target_token_count",
            "size_bytes",
            "sha256",
        }
        if not isinstance(record, Mapping) or set(record) != expected_fields:
            raise ContractError(f"dataset manifest shard {name} has unexpected fields")
        if record["path"] != f"{name}.jsonl":
            raise ContractError(f"dataset shard {name} has a noncanonical filename")
        for field in (
            "record_count",
            "input_token_count",
            "target_token_count",
            "size_bytes",
        ):
            value = record[field]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ContractError(f"dataset manifest shard {name} {field} is invalid")
        path = _regular_child(root, record["path"])
        expected_sha = require_sha256(record["sha256"], f"{name} shard SHA-256")
        if (
            path.stat().st_size != record["size_bytes"]
            or sha256_file(path) != expected_sha
        ):
            raise ContractError(f"dataset shard {name} content hash/size mismatch")
        records = _load_shard(path)
        if not records:
            raise ContractError(
                f"dataset shard {name} is empty; train/validation/test are required"
            )
        if (
            len(records) != record["record_count"]
            or sum(len(item.input_ids) for item in records)
            != record["input_token_count"]
            or sum(sum(item.loss_mask) for item in records)
            != record["target_token_count"]
        ):
            raise ContractError(f"dataset shard {name} aggregate counts mismatch")
        for item in records:
            if item.identifier in seen:
                raise ContractError(
                    f"duplicate prepared id across shards: {item.identifier}"
                )
            if split_name(item.identifier) != name:
                raise ContractError(
                    f"prepared id {item.identifier!r} is in {name}, not its deterministic split"
                )
            seen.add(item.identifier)
        shards[name] = records
    return PreparedDataset(root, manifest, sha256_file(manifest_path), shards)


class PersistentCore:
    def __init__(self, command: Sequence[str], max_line_bytes: int):
        if not command or any(
            not isinstance(value, str) or not value for value in command
        ):
            raise ContractError("core command must contain non-empty argv entries")
        self.command = list(command)
        self.process = subprocess.Popen(  # noqa: S603 - exact argv, no shell.
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            errors="strict",
            bufsize=1,
            shell=False,
        )
        self.max_line_bytes = max_line_bytes

    def transact(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        request_id = request.get("request_id")
        if self.process.stdin is None or self.process.stdout is None:
            raise ContractError("core pipes are unavailable")
        if self.process.poll() is not None:
            raise ContractError(f"core exited with status {self.process.returncode}")
        line = json.dumps(
            request, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if len(line.encode("utf-8")) > self.max_line_bytes:
            raise ContractError("core request exceeds configured JSONL line limit")
        try:
            self.process.stdin.write(line + "\n")
            self.process.stdin.flush()
            response_line = self.process.stdout.readline(self.max_line_bytes + 2)
        except (BrokenPipeError, OSError, UnicodeError) as error:
            raise ContractError(f"core transport failed: {error}") from error
        if not response_line or not response_line.endswith("\n"):
            raise ContractError(
                "core closed stdout or emitted an unterminated response"
            )
        if len(response_line.encode("utf-8")) > self.max_line_bytes + 1:
            raise ContractError("core response exceeds configured JSONL line limit")
        try:
            response = json.loads(
                response_line, object_pairs_hook=_reject_duplicate_pairs
            )
        except (json.JSONDecodeError, UnicodeError) as error:
            raise ContractError(f"core emitted invalid JSON: {error}") from error
        if (
            not isinstance(response, Mapping)
            or response.get("protocol") != PROTOCOL
            or response.get("request_id") != request_id
            or not isinstance(response.get("ok"), bool)
        ):
            raise ContractError("core response correlation/schema mismatch")
        if not response["ok"]:
            error = response.get("error")
            if not isinstance(error, Mapping):
                raise ContractError("core returned a malformed error response")
            raise ContractError(
                f"core rejected {request.get('op')}: {error.get('code')}: {error.get('message')}"
            )
        return response

    def wait_after_shutdown(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            status = self.process.wait(timeout=30.0)
        except subprocess.TimeoutExpired as error:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)
            raise ContractError("core did not exit after shutdown") from error
        if status != 0:
            raise ContractError(f"core exited with status {status} after shutdown")
        if self.process.stdout is not None:
            self.process.stdout.close()

    def abort(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)
        if self.process.stdout is not None:
            self.process.stdout.close()


class MetricsLog:
    def __init__(self, path: Path):
        self.path = path
        self.sequence = 0
        self.stream: BinaryIO = path.open("xb", buffering=0)

    def append(self, event: Mapping[str, Any]) -> None:
        document = {
            "schema_version": RUN_SCHEMA_VERSION,
            "kind": METRICS_KIND,
            "sequence": self.sequence,
            **event,
        }
        encoded = (
            json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        self.stream.write(encoded)
        os.fsync(self.stream.fileno())
        self.sequence += 1

    def close(self) -> None:
        self.stream.close()


def _ordered(
    records: Sequence[PreparedRecord], seed: int, epoch: int
) -> list[PreparedRecord]:
    prefix = seed.to_bytes(8, "big") + epoch.to_bytes(8, "big")
    return sorted(
        records,
        key=lambda record: (
            hashlib.sha256(prefix + record.identifier.encode()).digest(),
            record.identifier,
        ),
    )


def _finite_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ContractError(f"core metric {label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"core metric {label} must be finite")
    return result


def _metrics(response: Mapping[str, Any], expected_targets: int) -> dict[str, Any]:
    raw = response.get("metrics")
    if not isinstance(raw, Mapping):
        raise ContractError("core train/evaluate response is missing metrics")
    token_count = raw.get("token_count")
    correct = raw.get("correct_token_count", 0)
    if (
        not isinstance(token_count, int)
        or isinstance(token_count, bool)
        or token_count != expected_targets
        or not isinstance(correct, int)
        or isinstance(correct, bool)
        or not 0 <= correct <= token_count
    ):
        raise ContractError("core metric token counts disagree with loss_mask")
    result = dict(raw)
    for name in ("loss", "perplexity", "mean_spike_rate"):
        result[name] = _finite_number(raw.get(name), name)
    if (
        result["loss"] < 0.0
        or result["perplexity"] < 1.0
        or not 0.0 <= result["mean_spike_rate"] <= 1.0
    ):
        raise ContractError("core loss/perplexity/spike metrics are outside bounds")
    result["token_count"] = token_count
    result["correct_token_count"] = correct
    return result


def _training_state(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "micro_batch_count",
        "optimizer_step_count",
        "pending_accumulation_steps",
        "learning_rate",
    }:
        raise ContractError(f"{label} training_state has unexpected fields")
    result = dict(value)
    for field in (
        "micro_batch_count",
        "optimizer_step_count",
        "pending_accumulation_steps",
    ):
        field_value = result[field]
        if (
            not isinstance(field_value, int)
            or isinstance(field_value, bool)
            or field_value < 0
        ):
            raise ContractError(f"{label} training_state {field} is invalid")
    result["learning_rate"] = _finite_number(
        result["learning_rate"], f"{label} learning_rate"
    )
    if result["learning_rate"] < 0.0:
        raise ContractError(f"{label} training_state learning_rate is negative")
    return result


def _flush(
    core: PersistentCore,
    log: MetricsLog,
    request_id: str,
    phase: str,
    epoch: int,
) -> dict[str, Any]:
    response = core.transact(
        {"protocol": PROTOCOL, "request_id": request_id, "op": "flush"}
    )
    state = _training_state(response.get("training_state"), "core flush")
    if state["pending_accumulation_steps"] != 0:
        raise ContractError("core flush left pending gradient accumulation")
    log.append(
        {
            "event": "flush",
            "phase": phase,
            "epoch": epoch,
            "request_id": request_id,
            "training_state": state,
        }
    )
    return state


def _copy_file_atomic(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise ContractError(f"checkpoint source is not a regular file: {source.name}")
    temporary = destination.with_name(destination.name + ".partial")
    if destination.exists() or temporary.exists():
        raise ContractError(
            f"checkpoint destination already exists: {destination.name}"
        )
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _replace_core_checkpoint(command: Sequence[str], checkpoint: Path) -> list[str]:
    """Replace initialization artifacts with one selected runner checkpoint."""

    value_options = {"--checkpoint", "--qwen-archive", "--qwen-archive-sha256"}
    result: list[str] = []
    index = 0
    while index < len(command):
        value = command[index]
        if value in value_options:
            if index + 1 >= len(command):
                raise ContractError(f"core command option {value} is missing its value")
            index += 2
            continue
        result.append(value)
        index += 1
    return [*result, "--checkpoint", str(checkpoint)]


def _single_option_value(command: Sequence[str], option: str) -> str | None:
    positions = [index for index, value in enumerate(command) if value == option]
    if not positions:
        return None
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ContractError(
            f"core command must provide {option} exactly once with a value"
        )
    return command[positions[0] + 1]


def _file_provenance(value: str, label: str) -> dict[str, Any]:
    try:
        path = Path(value).resolve(strict=True)
    except OSError as error:
        raise ContractError(f"{label} does not exist: {value}") from error
    if not path.is_file():
        raise ContractError(f"{label} is not a regular file: {value}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _initialization_provenance(command: Sequence[str]) -> dict[str, Any]:
    if not command or any(not isinstance(value, str) or not value for value in command):
        raise ContractError("core command must contain non-empty argv entries")
    executable = _file_provenance(command[0], "core executable")
    checkpoint_value = _single_option_value(command, "--checkpoint")
    archive_value = _single_option_value(command, "--qwen-archive")
    archive_sha_value = _single_option_value(command, "--qwen-archive-sha256")
    if checkpoint_value is not None and archive_value is not None:
        raise ContractError("core command cannot combine checkpoint and Qwen import")
    if (archive_value is None) != (archive_sha_value is None):
        raise ContractError(
            "core command must pair --qwen-archive and --qwen-archive-sha256"
        )
    result: dict[str, Any] = {"executable": executable}
    if archive_value is not None and archive_sha_value is not None:
        expected = require_sha256(archive_sha_value, "Qwen archive SHA-256")
        archive = _file_provenance(archive_value, "Qwen archive")
        if archive["sha256"] != expected:
            raise ContractError("Qwen archive does not match its command-line SHA-256")
        result.update({"mode": "verified_qwen_import", "qwen_archive": archive})
    elif checkpoint_value is not None:
        result.update(
            {
                "mode": "runner_checkpoint",
                "runner_checkpoint": _file_provenance(
                    checkpoint_value, "initial runner checkpoint"
                ),
            }
        )
    else:
        result["mode"] = "deterministic_random"
    return result


def _validate_tokenizer_provenance(
    dataset: PreparedDataset,
    metadata: Mapping[str, Any],
    initialization: Mapping[str, Any],
) -> None:
    qwen = metadata.get("qwen_weights")
    if qwen is None:
        if initialization.get("mode") == "verified_qwen_import":
            raise ContractError("Qwen-import core metadata omitted Qwen provenance")
        return
    tokenizer = dataset.manifest["tokenizer"]
    if not isinstance(qwen, Mapping) or (
        qwen.get("model_id"),
        qwen.get("revision"),
        qwen.get("tokenizer_fingerprint_sha256"),
    ) != (
        tokenizer.get("model_id"),
        tokenizer.get("revision"),
        tokenizer.get("fingerprint_sha256"),
    ):
        raise ContractError("dataset tokenizer does not match loaded Qwen provenance")
    archive = initialization.get("qwen_archive")
    if isinstance(archive, Mapping) and qwen.get("archive_sha256") != archive.get(
        "sha256"
    ):
        raise ContractError(
            "loaded Qwen provenance does not match the imported archive"
        )


def _validate_reload_metadata(
    training: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    selected_state: Mapping[str, Any],
) -> None:
    immutable_fields = {
        "vocabulary_size",
        "maximum_sequence_length",
        "model_dimension",
        "layer_count",
        "simulation_steps",
        "head_dimension",
        "query_key_normalization",
        "spiking",
        "build_provenance",
        "qwen_weights",
    }
    for field in immutable_fields:
        if training.get(field) != evaluation.get(field):
            raise ContractError(
                f"selected checkpoint changed core metadata field: {field}"
            )
    restored = _training_state(
        evaluation.get("training_state"), "selected-checkpoint metadata"
    )
    for field in (
        "micro_batch_count",
        "optimizer_step_count",
        "pending_accumulation_steps",
    ):
        if restored[field] != selected_state[field]:
            raise ContractError(
                f"selected checkpoint changed training state field: {field}"
            )
    if not math.isclose(
        restored["learning_rate"],
        selected_state["learning_rate"],
        rel_tol=1.0e-15,
        abs_tol=1.0e-15,
    ):
        raise ContractError(
            "selected checkpoint changed training state field: learning_rate"
        )


def _require_reproduced_metrics(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> None:
    for field in ("record_count", "token_count", "correct_token_count"):
        if expected.get(field) != actual.get(field):
            raise ContractError(
                f"selected checkpoint did not reproduce validation metric: {field}"
            )
    for field in (
        "loss",
        "perplexity",
        "token_accuracy",
        "mean_spike_rate",
    ):
        expected_value = _finite_number(expected.get(field), f"selected {field}")
        actual_value = _finite_number(actual.get(field), f"reloaded {field}")
        if not math.isclose(
            expected_value, actual_value, rel_tol=1.0e-7, abs_tol=1.0e-7
        ):
            raise ContractError(
                f"selected checkpoint did not reproduce validation metric: {field}"
            )


def _run_phase(
    core: PersistentCore,
    log: MetricsLog,
    records: Sequence[PreparedRecord],
    *,
    operation: str,
    phase: str,
    epoch: int,
    seed: int,
) -> dict[str, Any]:
    ordered = _ordered(records, seed, epoch if phase == "train" else 0)
    loss_sum = 0.0
    spike_sum = 0.0
    correct = 0
    tokens = 0
    for index, record in enumerate(ordered):
        request_id = f"{phase}-e{epoch:04d}-r{index:08d}"
        response = core.transact(
            {
                "protocol": PROTOCOL,
                "request_id": request_id,
                "op": operation,
                "input_ids": record.input_ids,
                "loss_mask": record.loss_mask,
                "reset_state": True,
            }
        )
        metrics = _metrics(response, sum(record.loss_mask[1:]))
        count = metrics["token_count"]
        loss_sum += metrics["loss"] * count
        spike_sum += metrics["mean_spike_rate"] * count
        correct += metrics["correct_token_count"]
        tokens += count
        log.append(
            {
                "event": "record",
                "phase": phase,
                "epoch": epoch,
                "record_index": index,
                "record_id": record.identifier,
                "request_id": request_id,
                "metrics": metrics,
            }
        )
    loss = loss_sum / tokens
    aggregate = {
        "record_count": len(ordered),
        "token_count": tokens,
        "correct_token_count": correct,
        "loss": loss,
        "perplexity": math.exp(min(loss, 80.0)),
        "token_accuracy": correct / tokens,
        "mean_spike_rate": spike_sum / tokens,
    }
    log.append(
        {"event": "aggregate", "phase": phase, "epoch": epoch, "metrics": aggregate}
    )
    return aggregate


def run_training(
    dataset: PreparedDataset,
    run_dir: Path,
    core_command: Sequence[str],
    *,
    epochs: int,
    seed: int,
    max_line_bytes: int,
    checkpoint_filename: str,
    config_provenance: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if epochs <= 0 or seed < 0 or seed > (1 << 64) - 1:
        raise ContractError("epochs must be positive and seed must be uint64")
    if Path(checkpoint_filename).name != checkpoint_filename or not checkpoint_filename:
        raise ContractError("checkpoint-filename must be a plain filename")
    if "--save-checkpoint" in core_command:
        raise ContractError(
            "core-command must not provide --save-checkpoint; the driver owns it"
        )
    initialization = _initialization_provenance(core_command)
    run_dir = run_dir.resolve()
    if run_dir.exists():
        raise ContractError("run-dir already exists; run artifacts are immutable")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()
    checkpoint = run_dir / checkpoint_filename
    latest_checkpoint = run_dir / ".latest-checkpoint.pt"
    selected_checkpoint = run_dir / ".selected-validation.pt"
    if checkpoint in (latest_checkpoint, selected_checkpoint):
        raise ContractError("checkpoint-filename conflicts with a reserved driver path")
    command = [*core_command, "--save-checkpoint", str(latest_checkpoint)]
    log = MetricsLog(run_dir / "metrics.jsonl")
    core: PersistentCore | None = None
    aggregates: list[dict[str, Any]] = []
    selection: list[dict[str, Any]] = []
    flush_states: list[dict[str, Any]] = []
    best_validation_loss = math.inf
    best_validation_epoch: int | None = None
    best_validation_metrics: dict[str, Any] | None = None
    selected_training_state: dict[str, Any] | None = None
    selected_checkpoint_sha256: str | None = None
    try:
        core = PersistentCore(command, max_line_bytes)
        metadata_response = core.transact(
            {"protocol": PROTOCOL, "request_id": "metadata-00000000", "op": "metadata"}
        )
        metadata = metadata_response.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ContractError("core metadata response is missing metadata")
        vocabulary_size = metadata.get("vocabulary_size")
        context_size = metadata.get("maximum_sequence_length")
        if (
            not isinstance(vocabulary_size, int)
            or isinstance(vocabulary_size, bool)
            or vocabulary_size <= 0
            or not isinstance(context_size, int)
            or isinstance(context_size, bool)
            or context_size <= 0
        ):
            raise ContractError("core metadata has invalid vocabulary/context bounds")
        _validate_tokenizer_provenance(dataset, metadata, initialization)
        for records in dataset.shards.values():
            for record in records:
                if (
                    len(record.input_ids) > context_size
                    or max(record.input_ids) >= vocabulary_size
                ):
                    raise ContractError(
                        f"prepared record {record.identifier!r} exceeds core vocabulary/context"
                    )
        log.append({"event": "metadata", "metadata": dict(metadata)})

        baseline_state = _flush(
            core, log, "flush-validation-e0000", "validation-baseline", 0
        )
        flush_states.append(baseline_state)
        baseline_metrics = _run_phase(
            core,
            log,
            dataset.shards["validation"],
            operation="evaluate",
            phase="validation-baseline",
            epoch=0,
            seed=seed,
        )
        aggregates.append(
            {"phase": "validation-baseline", "epoch": 0, "metrics": baseline_metrics}
        )
        _copy_file_atomic(latest_checkpoint, selected_checkpoint)
        best_validation_loss = baseline_metrics["loss"]
        best_validation_epoch = 0
        best_validation_metrics = baseline_metrics
        selected_training_state = baseline_state
        selected_checkpoint_sha256 = sha256_file(selected_checkpoint)
        selection.append(
            {
                "epoch": 0,
                "validation_loss": best_validation_loss,
                "selected": True,
                "checkpoint_sha256": selected_checkpoint_sha256,
            }
        )

        for epoch in range(1, epochs + 1):
            aggregates.append(
                {
                    "phase": "train",
                    "epoch": epoch,
                    "metrics": _run_phase(
                        core,
                        log,
                        dataset.shards["train"],
                        operation="train",
                        phase="train",
                        epoch=epoch,
                        seed=seed,
                    ),
                }
            )
            epoch_state = _flush(
                core,
                log,
                f"flush-validation-e{epoch:04d}",
                "train",
                epoch,
            )
            flush_states.append(epoch_state)
            validation_metrics = _run_phase(
                core,
                log,
                dataset.shards["validation"],
                operation="evaluate",
                phase="validation",
                epoch=epoch,
                seed=seed,
            )
            aggregates.append(
                {"phase": "validation", "epoch": epoch, "metrics": validation_metrics}
            )
            improved = validation_metrics["loss"] < best_validation_loss
            selection_record: dict[str, Any] = {
                "epoch": epoch,
                "validation_loss": validation_metrics["loss"],
                "selected": improved,
            }
            if improved:
                if selected_checkpoint.exists():
                    selected_checkpoint.unlink()
                _copy_file_atomic(latest_checkpoint, selected_checkpoint)
                best_validation_loss = validation_metrics["loss"]
                best_validation_epoch = epoch
                best_validation_metrics = validation_metrics
                selected_training_state = epoch_state
                selected_checkpoint_sha256 = sha256_file(selected_checkpoint)
                selection_record["checkpoint_sha256"] = selected_checkpoint_sha256
            selection.append(selection_record)

        core.transact(
            {"protocol": PROTOCOL, "request_id": "shutdown-final", "op": "shutdown"}
        )
        core.wait_after_shutdown()
        core = None
        if (
            best_validation_epoch is None
            or best_validation_metrics is None
            or selected_training_state is None
            or selected_checkpoint_sha256 is None
            or not selected_checkpoint.is_file()
            or selected_checkpoint.is_symlink()
        ):
            raise ContractError("best-validation checkpoint selection failed")
        os.replace(selected_checkpoint, checkpoint)
        if sha256_file(checkpoint) != selected_checkpoint_sha256:
            raise ContractError("selected checkpoint hash changed during finalization")
        if latest_checkpoint.exists():
            latest_checkpoint.unlink()

        evaluation_command = _replace_core_checkpoint(core_command, checkpoint)
        core = PersistentCore(evaluation_command, max_line_bytes)
        selected_metadata_response = core.transact(
            {"protocol": PROTOCOL, "request_id": "metadata-selected", "op": "metadata"}
        )
        selected_metadata = selected_metadata_response.get("metadata")
        if not isinstance(selected_metadata, Mapping):
            raise ContractError("selected-checkpoint core metadata is missing")
        _validate_reload_metadata(metadata, selected_metadata, selected_training_state)
        reproduced_validation_metrics = _run_phase(
            core,
            log,
            dataset.shards["validation"],
            operation="evaluate",
            phase="validation-selected-reload",
            epoch=best_validation_epoch,
            seed=seed,
        )
        _require_reproduced_metrics(
            best_validation_metrics, reproduced_validation_metrics
        )
        test_metrics = _run_phase(
            core,
            log,
            dataset.shards["test"],
            operation="evaluate",
            phase="test",
            epoch=best_validation_epoch,
            seed=seed,
        )
        aggregates.append(
            {"phase": "test", "epoch": best_validation_epoch, "metrics": test_metrics}
        )
        core.transact(
            {"protocol": PROTOCOL, "request_id": "shutdown-selected", "op": "shutdown"}
        )
        core.wait_after_shutdown()
        core = None

        log.close()
        metrics_path = run_dir / "metrics.jsonl"
        summary = {
            "schema_version": RUN_SCHEMA_VERSION,
            "kind": RUN_KIND,
            "status": "completed",
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "seed": seed,
            "epochs": epochs,
            "record_order_algorithm": ORDER_ALGORITHM,
            "config": dict(config_provenance) if config_provenance else None,
            "dataset": {
                "path": str(dataset.root),
                "manifest_sha256": dataset.manifest_sha256,
                "tokenizer": dataset.manifest["tokenizer"],
                "shards": dataset.manifest["shards"],
            },
            "core": {
                "initialization": initialization,
                "training_command": command,
                "selected_checkpoint_evaluation_command": evaluation_command,
                "training_metadata": dict(metadata),
                "selected_checkpoint_metadata": dict(selected_metadata),
            },
            "model_selection": {
                "metric": "validation_assistant_token_nll",
                "direction": "minimize",
                "test_split_observed_during_selection": False,
                "best_validation_epoch": best_validation_epoch,
                "best_validation_loss": best_validation_loss,
                "selected_training_state": selected_training_state,
                "reloaded_validation_metrics": reproduced_validation_metrics,
                "candidates": selection,
            },
            "flush_training_states": flush_states,
            "aggregates": aggregates,
            "metrics_log": {
                "path": metrics_path.name,
                "size_bytes": metrics_path.stat().st_size,
                "sha256": sha256_file(metrics_path),
            },
            "checkpoint": {
                "path": checkpoint.name,
                "size_bytes": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
            },
        }
        write_json_atomic(run_dir / "run-summary.json", summary)
        return summary
    except BaseException:
        if core is not None:
            core.abort()
        if not log.stream.closed:
            log.close()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--core-executable", required=True, type=Path)
    core = parser.add_mutually_exclusive_group(required=True)
    core.add_argument(
        "--config",
        type=Path,
        help="validated chatbot config whose runner_binding supplies core argv",
    )
    core.add_argument("--core-arg", action="append")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-line-bytes", type=int, default=MAX_LINE_BYTES_DEFAULT)
    parser.add_argument("--checkpoint-filename", default="final-checkpoint.pt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.max_line_bytes <= 0:
            raise ContractError("max-line-bytes must be positive")
        config_provenance: Mapping[str, Any] | None = None
        if arguments.config is not None:
            selected_config = load_config(arguments.config)
            configured_seed = selected_config.document["seed"]
            if arguments.seed is not None and arguments.seed != configured_seed:
                raise ContractError("--seed does not match the selected config")
            training = selected_config.document.get("training")
            configured_epochs = (
                training.get("epochs") if isinstance(training, Mapping) else None
            )
            if (
                arguments.epochs is not None
                and configured_epochs is not None
                and arguments.epochs != configured_epochs
            ):
                raise ContractError("--epochs does not match the selected config")
            core_arguments = list(selected_config.core_argv)
            seed = configured_seed
            epochs = (
                configured_epochs
                if configured_epochs is not None
                else (arguments.epochs if arguments.epochs is not None else 1)
            )
            config_provenance = {
                **_file_provenance(str(arguments.config), "chatbot config"),
                "schema_version": selected_config.document["schema_version"],
                "experiment_id": selected_config.experiment_id,
            }
        else:
            core_arguments = list(arguments.core_arg or [])
            seed = arguments.seed if arguments.seed is not None else 42
            epochs = arguments.epochs if arguments.epochs is not None else 1
        dataset = load_prepared_dataset(arguments.dataset_dir)
        summary = run_training(
            dataset,
            arguments.run_dir,
            [str(arguments.core_executable), *core_arguments],
            epochs=epochs,
            seed=seed,
            max_line_bytes=arguments.max_line_bytes,
            checkpoint_filename=arguments.checkpoint_filename,
            config_provenance=config_provenance,
        )
    except (ContractError, OSError, UnicodeError, ValueError) as error:
        print(f"chatbot_train: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "run_summary": str(arguments.run_dir.resolve() / "run-summary.json"),
                "checkpoint_sha256": summary["checkpoint"]["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
