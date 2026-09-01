#!/usr/bin/env python3
"""Assemble and validate immutable chatbot publication manifests."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from chatbot_config import (
    COMPACT_TRACK,
    EXACT_TRACK,
    ConfigError,
    ValidatedConfig,
    load_config,
)
from chatbot_prepare import load_canonical_conversation_ids
from chatbot_train import validate_dataset_derivation
from qwen_contract import (
    ContractError,
    PHASE0_CONFIG_SHA256,
    PHASE0_MODEL_ID,
    PHASE0_REVISION,
    PHASE0_TOKENIZER_CONFIG_SHA256,
    PHASE0_TOKENIZER_FINGERPRINT,
    PHASE0_TOKENIZER_JSON_SHA256,
    PHASE0_WEIGHT_SHA256,
)

PUBLICATION_SCHEMA_VERSION = "snnbase-chatbot-run-v1"
TRAINING_RUN_KIND = "snnbase.chatbot-training-run"
TRAINING_METRIC_KIND = "snnbase.chatbot-training-metric"
QWEN_LICENSE = "Apache-2.0"
DEFAULT_SCHEMA = (
    Path(__file__).resolve().parents[1] / "schemas" / "chatbot-run-v1.schema.json"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
MAX_METRICS_LINE_BYTES = 4 * 1024 * 1024
MAX_METRICS_RECORDS = 10_000_000


class PublicationError(RuntimeError):
    """Raised when a publication artifact is incomplete or inconsistent."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PublicationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise PublicationError(f"non-standard JSON constant: {value}")


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise PublicationError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PublicationError(f"{label} is missing: {path}") from error
    if not resolved.is_file():
        raise PublicationError(f"{label} is not a regular file: {path}")
    return resolved


def _load_json(path: Path, label: str) -> Any:
    resolved = _regular_file(path, label)
    try:
        with resolved.open("r", encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_constant,
            )
    except PublicationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PublicationError(f"could not read {label} JSON: {error}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_digest(
    path: Path, label: str, *, display_path: str | None = None
) -> dict[str, Any]:
    resolved = _regular_file(path, label)
    return {
        "path": display_path if display_path is not None else str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _recorded_file_digest(path: Path, recorded: Any, label: str) -> dict[str, Any]:
    if not isinstance(recorded, Mapping):
        raise PublicationError(f"{label} provenance is missing")
    digest = _file_digest(path, label)
    if (
        recorded.get("sha256") != digest["sha256"]
        or recorded.get("size_bytes") != digest["bytes"]
    ):
        raise PublicationError(f"{label} does not match its recorded hash/size")
    return digest


def _safe_child(root: Path, filename: Any, label: str) -> Path:
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).is_absolute()
        or Path(filename).name != filename
    ):
        raise PublicationError(f"{label} must be a plain filename")
    candidate = root / filename
    resolved = _regular_file(candidate, label)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise PublicationError(f"{label} escapes its owning directory") from error
    return resolved


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicationError(f"{label} must be an object")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PublicationError(f"{label} must be a non-empty trimmed string")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise PublicationError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str) -> float | int:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise PublicationError(f"{label} must be a finite number")
    return value


def _run_git(repository: Path, arguments: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
        )
    except (OSError, UnicodeError, subprocess.CalledProcessError) as error:
        detail = ""
        if isinstance(error, subprocess.CalledProcessError):
            detail = error.stderr.strip()
        raise PublicationError(
            f"could not inspect git repository {repository}: {detail or error}"
        ) from error
    return result.stdout.strip()


def repository_provenance(path: Path) -> dict[str, Any]:
    try:
        requested = path.resolve(strict=True)
    except OSError as error:
        raise PublicationError(f"git repository is missing: {path}") from error
    top_level_text = _run_git(requested, ["rev-parse", "--show-toplevel"])
    try:
        top_level = Path(top_level_text).resolve(strict=True)
    except OSError as error:
        raise PublicationError(
            f"git reported a missing repository root: {top_level_text}"
        ) from error
    revision = _run_git(top_level, ["rev-parse", "--verify", "HEAD"])
    if not GIT_REVISION_RE.fullmatch(revision):
        raise PublicationError(f"git repository has invalid HEAD: {top_level}")
    dirty = bool(
        _run_git(top_level, ["status", "--porcelain=v1", "--untracked-files=all"])
    )
    return {"path": str(top_level), "revision": revision, "dirty": dirty}


def _repositories_from_build(
    training_metadata: Mapping[str, Any],
    experiments_repository: Path,
    snnbase_repository: Path,
    publishable: bool,
) -> dict[str, dict[str, Any]]:
    embedded = _mapping(
        training_metadata.get("build_provenance"), "core build provenance"
    )
    if set(embedded) != {"experiments", "snnbase"}:
        raise PublicationError(
            "core build provenance must contain exactly experiments and snnbase"
        )
    current = {
        "experiments": repository_provenance(experiments_repository),
        "snnbase": repository_provenance(snnbase_repository),
    }
    result: dict[str, dict[str, Any]] = {}
    for name in ("experiments", "snnbase"):
        record = _mapping(embedded[name], f"core build provenance {name}")
        if set(record) != {"revision", "dirty"}:
            raise PublicationError(
                f"core build provenance {name} has unexpected fields"
            )
        revision = record.get("revision")
        dirty = record.get("dirty")
        if not isinstance(revision, str) or not GIT_REVISION_RE.fullmatch(revision):
            raise PublicationError(f"core build provenance {name} revision is invalid")
        if not isinstance(dirty, bool):
            raise PublicationError(
                f"core build provenance {name} dirty flag is invalid"
            )
        result[name] = {
            "path": current[name]["path"],
            "revision": revision,
            "dirty": dirty,
        }
        if publishable and (
            dirty or current[name]["dirty"] or current[name]["revision"] != revision
        ):
            raise PublicationError(
                "publishable manifest requires the clean repository state "
                f"embedded in the executable: {name}"
            )
    return result


def _resolve_json_pointer(document: Any, reference: str) -> Any:
    if not reference.startswith("#/"):
        raise PublicationError(f"unsupported JSON Schema reference: {reference}")
    value = document
    for encoded in reference[2:].split("/"):
        key = encoded.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, Mapping) or key not in value:
            raise PublicationError(f"unresolved JSON Schema reference: {reference}")
        value = value[key]
    return value


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right


def _validate_schema_node(
    instance: Any,
    node: Any,
    root_schema: Mapping[str, Any],
    path: str,
) -> None:
    if not isinstance(node, Mapping):
        raise PublicationError(f"invalid JSON Schema node at {path}")
    if "$ref" in node:
        referenced = _resolve_json_pointer(root_schema, node["$ref"])
        _validate_schema_node(instance, referenced, root_schema, path)
    for index, child in enumerate(node.get("allOf", [])):
        _validate_schema_node(instance, child, root_schema, f"{path}.allOf[{index}]")
    if "if" in node:
        try:
            _validate_schema_node(instance, node["if"], root_schema, path)
            matched = True
        except PublicationError:
            matched = False
        branch = node.get("then") if matched else node.get("else")
        if branch is not None:
            _validate_schema_node(instance, branch, root_schema, path)

    expected_type = node.get("type")
    type_matches = {
        "object": isinstance(instance, Mapping),
        "array": isinstance(instance, list),
        "string": isinstance(instance, str),
        "integer": isinstance(instance, int) and not isinstance(instance, bool),
        "number": isinstance(instance, (int, float))
        and not isinstance(instance, bool)
        and math.isfinite(float(instance)),
        "boolean": isinstance(instance, bool),
    }
    if expected_type is not None and not type_matches.get(expected_type, False):
        raise PublicationError(
            f"schema validation failed at {path}: expected {expected_type}"
        )
    if "const" in node and not _json_equal(instance, node["const"]):
        raise PublicationError(f"schema validation failed at {path}: const mismatch")
    if "enum" in node and not any(
        _json_equal(instance, candidate) for candidate in node["enum"]
    ):
        raise PublicationError(f"schema validation failed at {path}: enum mismatch")

    if isinstance(instance, Mapping):
        required = node.get("required", [])
        for key in required:
            if key not in instance:
                raise PublicationError(
                    f"schema validation failed at {path}: missing {key}"
                )
        properties = node.get("properties", {})
        if node.get("additionalProperties") is False:
            extras = sorted(set(instance) - set(properties))
            if extras:
                raise PublicationError(
                    f"schema validation failed at {path}: unexpected {extras[0]}"
                )
        for key, child in properties.items():
            if key in instance:
                _validate_schema_node(
                    instance[key], child, root_schema, f"{path}.{key}"
                )
    if isinstance(instance, list):
        if len(instance) < node.get("minItems", 0):
            raise PublicationError(f"schema validation failed at {path}: too few items")
        if "items" in node:
            for index, item in enumerate(instance):
                _validate_schema_node(
                    item, node["items"], root_schema, f"{path}[{index}]"
                )
        if "contains" in node:
            for item in instance:
                try:
                    _validate_schema_node(item, node["contains"], root_schema, path)
                    break
                except PublicationError:
                    continue
            else:
                raise PublicationError(
                    f"schema validation failed at {path}: contains did not match"
                )
    if isinstance(instance, str):
        if len(instance) < node.get("minLength", 0):
            raise PublicationError(
                f"schema validation failed at {path}: string is too short"
            )
        if "pattern" in node and re.search(node["pattern"], instance) is None:
            raise PublicationError(
                f"schema validation failed at {path}: pattern mismatch"
            )
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in node and instance < node["minimum"]:
            raise PublicationError(f"schema validation failed at {path}: below minimum")
        if "maximum" in node and instance > node["maximum"]:
            raise PublicationError(f"schema validation failed at {path}: above maximum")


def validate_publication_manifest(
    manifest: Any,
    *,
    schema_path: Path = DEFAULT_SCHEMA,
    require_publishable: bool = False,
) -> None:
    schema = _load_json(schema_path, "publication schema")
    if not isinstance(schema, Mapping):
        raise PublicationError("publication schema must be an object")
    _validate_schema_node(manifest, schema, schema, "$")
    document = _mapping(manifest, "publication manifest")
    if require_publishable and document.get("publishable") is not True:
        raise PublicationError("manifest is valid but is not marked publishable")
    for resource_name in ("reference_model", "tokenizer"):
        resource = _mapping(document[resource_name], resource_name)
        paths = [item["path"] for item in resource["files"]]
        if len(paths) != len(set(paths)):
            raise PublicationError(f"{resource_name} contains duplicate file paths")
    artifact_kinds = [item["kind"] for item in document["artifacts"]]
    if len(artifact_kinds) != len(set(artifact_kinds)):
        raise PublicationError("publication manifest contains duplicate artifact kinds")
    metric_ids = [item["id"] for item in document["metrics"]]
    if len(metric_ids) != len(set(metric_ids)):
        raise PublicationError("publication manifest contains duplicate metric ids")


def validate_publication_file(
    path: Path,
    *,
    schema_path: Path = DEFAULT_SCHEMA,
    require_publishable: bool = False,
) -> Mapping[str, Any]:
    document = _load_json(path, "publication manifest")
    validate_publication_manifest(
        document,
        schema_path=schema_path,
        require_publishable=require_publishable,
    )
    return _mapping(document, "publication manifest")


def _phase0_resources(
    assets_dir: Path,
    config: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    training_metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference = _mapping(config.get("reference"), "config reference")
    expected_reference = {
        "repository": PHASE0_MODEL_ID,
        "revision": PHASE0_REVISION,
        "license": QWEN_LICENSE,
        "tokenizer_fingerprint_sha256": PHASE0_TOKENIZER_FINGERPRINT,
    }
    for field, expected in expected_reference.items():
        if reference.get(field) != expected:
            raise PublicationError(f"config reference {field} is not the Phase 0 pin")
    reference_files = _mapping(reference.get("files"), "config reference files")
    expected_hashes = {
        "config.json": PHASE0_CONFIG_SHA256,
        "tokenizer.json": PHASE0_TOKENIZER_JSON_SHA256,
        "tokenizer_config.json": PHASE0_TOKENIZER_CONFIG_SHA256,
    }
    for filename, expected in expected_hashes.items():
        if reference_files.get(filename) != expected:
            raise PublicationError(
                f"config reference hash is not pinned for {filename}"
            )
    tokenizer_pin = _mapping(
        dataset_manifest.get("tokenizer"), "dataset tokenizer provenance"
    )
    if (
        tokenizer_pin.get("model_id"),
        tokenizer_pin.get("revision"),
        tokenizer_pin.get("fingerprint_sha256"),
    ) != (PHASE0_MODEL_ID, PHASE0_REVISION, PHASE0_TOKENIZER_FINGERPRINT):
        raise PublicationError("dataset tokenizer is not the exact Phase 0 pin")

    try:
        root = assets_dir.resolve(strict=True)
    except OSError as error:
        raise PublicationError(
            f"Qwen asset directory is missing: {assets_dir}"
        ) from error
    if not root.is_dir() or root.is_symlink():
        raise PublicationError("Qwen asset directory must be a non-symlink directory")
    phase_files: dict[str, dict[str, Any]] = {}
    for filename, expected in expected_hashes.items():
        digest = _file_digest(
            root / filename, f"Qwen asset {filename}", display_path=filename
        )
        if digest["sha256"] != expected:
            raise PublicationError(f"Qwen asset hash mismatch for {filename}")
        phase_files[filename] = digest

    tokenizer_files = [
        phase_files["tokenizer.json"],
        phase_files["tokenizer_config.json"],
    ]
    for optional in ("vocab.json", "merges.txt"):
        candidate = root / optional
        if candidate.exists():
            tokenizer_files.append(
                _file_digest(candidate, f"Qwen asset {optional}", display_path=optional)
            )

    model_files = [phase_files["config.json"]]
    qwen = training_metadata.get("qwen_weights")
    if qwen is not None:
        qwen = _mapping(qwen, "core Qwen provenance")
        expected_core = {
            "model_id": PHASE0_MODEL_ID,
            "revision": PHASE0_REVISION,
            "source_checkpoint_sha256": PHASE0_WEIGHT_SHA256,
            "config_sha256": PHASE0_CONFIG_SHA256,
            "tokenizer_fingerprint_sha256": PHASE0_TOKENIZER_FINGERPRINT,
        }
        for field, expected in expected_core.items():
            if qwen.get(field) != expected:
                raise PublicationError(
                    f"core Qwen provenance is not pinned for {field}"
                )
        weight = _file_digest(
            root / "model.safetensors",
            "Qwen source checkpoint",
            display_path="model.safetensors",
        )
        if weight["sha256"] != PHASE0_WEIGHT_SHA256:
            raise PublicationError("Qwen source checkpoint hash mismatch")
        model_files.append(weight)

    common = {
        "repository": PHASE0_MODEL_ID,
        "revision": PHASE0_REVISION,
        "license": QWEN_LICENSE,
    }
    return (
        {**common, "files": model_files},
        {**common, "files": tokenizer_files},
    )


def _validate_environment(
    value: Any, publishable: bool
) -> tuple[dict[str, str], dict[str, str]]:
    environment = _mapping(value, "environment")
    if set(environment) != {"toolchain", "hardware"}:
        raise PublicationError(
            "environment must contain exactly toolchain and hardware"
        )
    toolchain_raw = _mapping(environment["toolchain"], "environment toolchain")
    hardware_raw = _mapping(environment["hardware"], "environment hardware")
    if set(toolchain_raw) != {"compiler", "cmake", "torch", "cuda"}:
        raise PublicationError("toolchain has unexpected fields")
    if set(hardware_raw) != {"cpu", "gpu"}:
        raise PublicationError("hardware has unexpected fields")
    toolchain: dict[str, str] = {}
    for field in ("compiler", "cmake", "torch", "cuda"):
        value = toolchain_raw[field]
        if not isinstance(value, str):
            raise PublicationError(f"toolchain {field} must be a string")
        toolchain[field] = value
    hardware: dict[str, str] = {}
    for field in ("cpu", "gpu"):
        value = hardware_raw[field]
        if not isinstance(value, str):
            raise PublicationError(f"hardware {field} must be a string")
        hardware[field] = value
    if publishable:
        for field in ("compiler", "cmake", "torch"):
            _nonempty_string(toolchain[field], f"toolchain {field}")
        for field in ("cpu", "gpu"):
            _nonempty_string(hardware[field], f"hardware {field}")
    return toolchain, hardware


def _completed_timestamp(value: Any) -> str:
    text = _nonempty_string(value, "run completed_utc")
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise PublicationError(
            "run completed_utc is not an ISO-8601 timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PublicationError("run completed_utc must include a timezone")
    return parsed.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _strip_driver_checkpoint(command: Sequence[str]) -> list[str]:
    result: list[str] = []
    found = 0
    index = 0
    while index < len(command):
        if command[index] == "--save-checkpoint":
            if index + 1 >= len(command):
                raise PublicationError(
                    "training command has a truncated --save-checkpoint"
                )
            found += 1
            index += 2
            continue
        result.append(command[index])
        index += 1
    if found != 1:
        raise PublicationError(
            "completed training command must contain exactly one --save-checkpoint"
        )
    return result


def _validate_consumed_config(
    run: Mapping[str, Any],
    selected: ValidatedConfig,
    config_path: Path,
) -> dict[str, Any]:
    recorded = _mapping(run.get("config"), "consumed config provenance")
    expected_fields = {
        "path",
        "size_bytes",
        "sha256",
        "schema_version",
        "experiment_id",
    }
    if set(recorded) != expected_fields:
        raise PublicationError(
            "consumed config provenance has unexpected or missing fields"
        )
    recorded_path = _nonempty_string(recorded.get("path"), "consumed config path")
    if not Path(recorded_path).is_absolute():
        raise PublicationError("consumed config path must be absolute")
    if (
        recorded.get("schema_version") != selected.document.get("schema_version")
        or recorded.get("experiment_id") != selected.experiment_id
    ):
        raise PublicationError(
            "selected config schema/experiment ID differs from the completed run"
        )
    digest = _recorded_file_digest(config_path, recorded, "consumed chatbot config")
    if not SHA256_RE.fullmatch(str(recorded.get("sha256", ""))):
        raise PublicationError("consumed config SHA-256 is malformed")
    if (
        not isinstance(recorded.get("size_bytes"), int)
        or isinstance(recorded.get("size_bytes"), bool)
        or recorded["size_bytes"] <= 0
    ):
        raise PublicationError("consumed config size is invalid")
    return digest


def _validate_config_binding(
    selected: ValidatedConfig,
    run: Mapping[str, Any],
    executable: Path,
) -> tuple[str, int, list[str]]:
    config = selected.document
    architecture = selected.architecture
    seed = _integer(config["seed"], "config seed")
    if run.get("seed") != seed:
        raise PublicationError("run seed does not match the selected config")
    training = _mapping(config["training"], "config training")
    if training.get("epochs") != run.get("epochs"):
        raise PublicationError("run epoch count does not match the selected config")

    core = _mapping(run.get("core"), "run core")
    initialization = _mapping(core.get("initialization"), "core initialization")
    metadata = _mapping(core.get("training_metadata"), "training core metadata")
    if metadata.get("spiking") is not (architecture == "snn"):
        raise PublicationError("run architecture does not match the selected config")
    model = _mapping(config["model"], "config model")
    metadata_fields = {
        "vocabulary_size": "vocabulary_size",
        "maximum_sequence_length": "maximum_sequence_length",
        "model_dimension": "model_dimension",
        "layer_count": "layer_count",
        "head_dimension": "head_dimension",
        "simulation_steps": "simulation_steps",
        "query_key_normalization": "query_key_normalization",
        "spiking": "spiking",
    }
    for config_field, metadata_field in metadata_fields.items():
        if model.get(config_field) != metadata.get(metadata_field):
            raise PublicationError(
                f"run metadata does not match config model field {config_field}"
            )

    expected_initialization = {
        COMPACT_TRACK: "deterministic_random",
        EXACT_TRACK: "verified_qwen_import",
    }[selected.track]
    if initialization.get("mode") != expected_initialization:
        raise PublicationError(
            "run initialization mode does not match the selected config track"
        )
    if selected.track == COMPACT_TRACK:
        if metadata.get("qwen_weights") is not None:
            raise PublicationError("compact config unexpectedly loaded Qwen weights")
    else:
        archive = _mapping(config["archive"], "config archive")
        recorded_archive = _mapping(
            initialization.get("qwen_archive"), "Qwen archive provenance"
        )
        if recorded_archive.get("sha256") != archive.get(
            "sha256"
        ) or recorded_archive.get("size_bytes") != archive.get("size_bytes"):
            raise PublicationError(
                "imported Qwen archive differs from the selected exact config"
            )
        qwen = _mapping(metadata.get("qwen_weights"), "core Qwen provenance")
        qwen_archive_fields = {
            "archive_sha256": "sha256",
            "metadata_sha256": "metadata_sha256",
            "payload_sha256": "payload_sha256",
            "loaded_tensor_count": "dense_tensor_count",
            "loaded_payload_bytes": "dense_tensor_bytes",
        }
        for metadata_field, archive_field in qwen_archive_fields.items():
            if qwen.get(metadata_field) != archive.get(archive_field):
                raise PublicationError(
                    "core Qwen provenance differs from exact config field "
                    f"archive.{archive_field}"
                )

    command = core.get("training_command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise PublicationError("run training command is invalid")
    try:
        command_executable = Path(command[0]).resolve(strict=True)
    except OSError as error:
        raise PublicationError("run training executable is missing") from error
    if command_executable != executable:
        raise PublicationError(
            "run command executable differs from recorded executable"
        )
    stripped = _strip_driver_checkpoint(command)
    if stripped[1:] != list(selected.core_argv):
        raise PublicationError("run command does not exactly bind the selected config")
    return architecture, seed, list(command)


def _load_dataset(
    run: Mapping[str, Any],
    config: Mapping[str, Any],
    source_path: Path,
    dataset_name: str,
) -> tuple[dict[str, Any], Mapping[str, Any], list[dict[str, Any]]]:
    run_dataset = _mapping(run.get("dataset"), "run dataset")
    try:
        root = Path(
            _nonempty_string(run_dataset.get("path"), "run dataset path")
        ).resolve(strict=True)
    except OSError as error:
        raise PublicationError("run dataset directory is missing") from error
    if not root.is_dir() or root.is_symlink():
        raise PublicationError("run dataset path must be a non-symlink directory")
    manifest_path = _safe_child(root, "dataset-manifest.json", "dataset manifest")
    manifest = _mapping(
        _load_json(manifest_path, "dataset manifest"), "dataset manifest"
    )
    manifest_digest = _file_digest(manifest_path, "dataset manifest")
    if run_dataset.get("manifest_sha256") != manifest_digest["sha256"]:
        raise PublicationError("dataset manifest hash differs from the completed run")
    if run_dataset.get("shards") != manifest.get("shards"):
        raise PublicationError("dataset shard manifest differs from the completed run")
    if run_dataset.get("tokenizer") != manifest.get("tokenizer"):
        raise PublicationError("dataset tokenizer differs from the completed run")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "snnbase.chatbot-token-dataset"
        or manifest.get("token_protocol") != "snnbase.chatbot.tokens/v1"
    ):
        raise PublicationError("dataset manifest contract is unsupported")
    required_dataset_fields = {
        "schema_version",
        "kind",
        "token_protocol",
        "source",
        "tokenizer",
        "split",
        "shards",
    }
    if set(manifest) not in (
        required_dataset_fields,
        required_dataset_fields | {"derivation"},
    ):
        raise PublicationError("dataset manifest has unexpected top-level fields")

    source = _mapping(manifest.get("source"), "dataset source provenance")
    source_digest = _recorded_file_digest(source_path, source, "dataset source")
    if Path(source_digest["path"]).name != source.get("filename"):
        raise PublicationError("dataset source filename differs from its manifest")
    source_uri = _nonempty_string(source.get("source_uri"), "dataset source_uri")
    version = _nonempty_string(source.get("version"), "dataset version")
    license_name = _nonempty_string(source.get("license"), "dataset license")

    split = _mapping(manifest.get("split"), "dataset split")
    expected_split = {
        "algorithm": "sha256-seed-bucket-v1",
        "seed": 42,
        "bucket_count": 10_000,
        "test_buckets_inclusive": [0, 999],
        "validation_buckets_inclusive": [1000, 1999],
        "train_buckets_inclusive": [2000, 9999],
    }
    for field, expected in expected_split.items():
        if split.get(field) != expected:
            raise PublicationError(f"dataset split field {field} is unsupported")
    config_dataset = _mapping(config.get("dataset"), "config dataset")
    config_split = _mapping(config_dataset.get("split"), "config dataset split")
    if (
        config_dataset.get("format") != "snnbase-chatbot-jsonl-v1"
        or config_split.get("algorithm") != expected_split["algorithm"]
        or config_split.get("seed") != expected_split["seed"]
        or config_split.get("validation_basis_points") != 1000
        or config_split.get("test_basis_points") != 1000
    ):
        raise PublicationError(
            "config dataset/split does not match the prepared dataset"
        )

    shard_manifest = _mapping(manifest.get("shards"), "dataset shards")
    if set(shard_manifest) != {"train", "validation", "test"}:
        raise PublicationError("dataset must contain train/validation/test shards")
    artifacts = [
        {"kind": "dataset_manifest", "file": manifest_digest},
    ]
    if "derivation" in manifest:
        derivation = manifest["derivation"]
        derivation_record = _mapping(derivation, "dataset derivation")
        lineage_record = _mapping(
            derivation_record.get("lineage"),
            "dataset conversion lineage provenance",
        )
        try:
            canonical_ids = load_canonical_conversation_ids(
                source_path,
                expected_size_bytes=source_digest["bytes"],
                expected_sha256=source_digest["sha256"],
                expected_record_count=_integer(
                    lineage_record.get("record_count"),
                    "dataset conversion lineage record_count",
                ),
            )
            lineage_ids = validate_dataset_derivation(
                root,
                derivation,
                source,
                _mapping(manifest.get("tokenizer"), "dataset tokenizer"),
                shard_manifest,
            )
        except ContractError as error:
            raise PublicationError(
                f"dataset conversion derivation is invalid: {error}"
            ) from error
        if lineage_ids != canonical_ids:
            raise PublicationError(
                "dataset conversion lineage tree IDs differ from canonical source IDs"
            )
        conversion_record = _mapping(
            derivation_record.get("conversion_manifest"),
            "dataset conversion manifest provenance",
        )
        conversion_path = _safe_child(
            root,
            conversion_record.get("path"),
            "dataset conversion manifest",
        )
        lineage_path = _safe_child(
            root,
            lineage_record.get("path"),
            "dataset conversion lineage",
        )
        artifacts.extend(
            [
                {
                    "kind": "dataset_conversion_manifest",
                    "file": _recorded_file_digest(
                        conversion_path,
                        conversion_record,
                        "dataset conversion manifest",
                    ),
                },
                {
                    "kind": "dataset_conversion_lineage",
                    "file": _recorded_file_digest(
                        lineage_path,
                        lineage_record,
                        "dataset conversion lineage",
                    ),
                },
            ]
        )
    record_counts: dict[str, int] = {}
    for name in ("train", "validation", "test"):
        record = _mapping(shard_manifest[name], f"dataset {name} shard")
        shard_path = _safe_child(root, record.get("path"), f"dataset {name} shard")
        digest = _recorded_file_digest(shard_path, record, f"dataset {name} shard")
        artifacts.append({"kind": f"dataset_{name}_shard", "file": digest})
        record_counts[name] = _integer(
            record.get("record_count"), f"dataset {name} record_count"
        )
    total_records = sum(record_counts.values())
    if total_records <= 0:
        raise PublicationError("dataset contains no records")
    dataset = {
        "name": _nonempty_string(dataset_name, "dataset name"),
        "version": version,
        "source_uri": source_uri,
        "license": license_name,
        "source": source_digest,
        "format": "snnbase-chatbot-jsonl-v1",
        "records": total_records,
        "split": {
            "algorithm": expected_split["algorithm"],
            "seed": expected_split["seed"],
            "validation_basis_points": 1000,
            "test_basis_points": 1000,
            "train_records": record_counts["train"],
            "validation_records": record_counts["validation"],
            "test_records": record_counts["test"],
        },
    }
    return dataset, manifest, artifacts


def _load_metrics_log(
    path: Path,
    expected_test_records: int,
    expected_validation: Mapping[str, Any],
    expected_test: Mapping[str, Any],
    *,
    maximum_line_bytes: int = MAX_METRICS_LINE_BYTES,
    maximum_records: int = MAX_METRICS_RECORDS,
) -> None:
    resolved = _regular_file(path, "training metrics log")
    if maximum_line_bytes <= 0 or maximum_records <= 0:
        raise PublicationError("training metrics log bounds must be positive")
    sequence = 0
    test_record_count = 0
    test_aggregate_count = 0
    test_aggregate: Any = None
    reload_aggregate_count = 0
    reload_aggregate: Any = None
    try:
        with resolved.open("rb") as stream:
            while True:
                encoded = stream.readline(maximum_line_bytes + 2)
                if not encoded:
                    break
                line_number = sequence + 1
                if sequence >= maximum_records:
                    raise PublicationError(
                        "training metrics log exceeds "
                        f"the {maximum_records}-record limit"
                    )
                if not encoded.endswith(b"\n"):
                    if len(encoded) > maximum_line_bytes:
                        raise PublicationError(
                            f"training metrics log line {line_number} exceeds "
                            f"the {maximum_line_bytes}-byte limit"
                        )
                    raise PublicationError(
                        "training metrics log is not newline terminated"
                    )
                payload = encoded[:-1]
                if len(payload) > maximum_line_bytes:
                    raise PublicationError(
                        f"training metrics log line {line_number} exceeds "
                        f"the {maximum_line_bytes}-byte limit"
                    )
                if payload.endswith(b"\r"):
                    payload = payload[:-1]
                if not payload:
                    raise PublicationError(
                        f"training metrics log line {line_number} is blank"
                    )
                try:
                    line = payload.decode("utf-8", errors="strict")
                    value = json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_pairs,
                        parse_constant=_reject_constant,
                    )
                except (UnicodeError, json.JSONDecodeError, PublicationError) as error:
                    raise PublicationError(
                        f"training metrics log line {line_number} is invalid: {error}"
                    ) from error
                record = _mapping(value, f"training metric line {line_number}")
                if (
                    record.get("schema_version") != 1
                    or record.get("kind") != TRAINING_METRIC_KIND
                    or record.get("sequence") != sequence
                ):
                    raise PublicationError(
                        f"training metrics log line {line_number} has invalid "
                        "provenance/sequence"
                    )
                sequence += 1
                if record.get("event") == "record" and record.get("phase") == "test":
                    test_record_count += 1
                elif record.get("event") == "aggregate":
                    if record.get("phase") == "test":
                        test_aggregate_count += 1
                        test_aggregate = record.get("metrics")
                    elif record.get("phase") == "validation-selected-reload":
                        reload_aggregate_count += 1
                        reload_aggregate = record.get("metrics")
    except PublicationError:
        raise
    except OSError as error:
        raise PublicationError(
            f"could not read training metrics log: {error}"
        ) from error
    if sequence == 0:
        raise PublicationError("training metrics log is empty")
    if test_record_count != expected_test_records:
        raise PublicationError(
            "training metrics log has the wrong number of test records"
        )
    if test_aggregate_count != 1 or test_aggregate != expected_test:
        raise PublicationError(
            "training metrics log test aggregate is missing or changed"
        )
    if reload_aggregate_count != 1 or reload_aggregate != expected_validation:
        raise PublicationError(
            "training metrics log selected validation aggregate is missing or changed"
        )


def _publication_metrics(
    run: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Mapping[str, Any], Mapping[str, Any]]:
    model_selection = _mapping(run.get("model_selection"), "run model_selection")
    validation = _mapping(
        model_selection.get("reloaded_validation_metrics"),
        "reloaded validation metrics",
    )
    if model_selection.get("test_split_observed_during_selection") is not False:
        raise PublicationError("completed run observed test during model selection")
    aggregates = run.get("aggregates")
    if not isinstance(aggregates, list):
        raise PublicationError("run aggregates must be an array")
    tests = [
        item
        for item in aggregates
        if isinstance(item, Mapping) and item.get("phase") == "test"
    ]
    if len(tests) != 1:
        raise PublicationError("completed run must contain exactly one test aggregate")
    test = _mapping(tests[0].get("metrics"), "test metrics")
    fields = (
        ("loss", "assistant_token_nll", "nats/token"),
        ("perplexity", "assistant_token_perplexity", "ratio"),
        ("token_accuracy", "assistant_token_accuracy", "fraction"),
        ("mean_spike_rate", "mean_spike_rate", "fraction"),
    )
    metrics: list[dict[str, Any]] = []
    for phase, source in (("validation", validation), ("test", test)):
        for field, metric_name, unit in fields:
            metrics.append(
                {
                    "id": f"{phase}.{metric_name}",
                    "value": _number(source.get(field), f"{phase} {field}"),
                    "unit": unit,
                }
            )
    return metrics, validation, test


def assemble_publication_manifest(
    *,
    run_dir: Path,
    config_path: Path,
    dataset_source: Path,
    experiments_repository: Path,
    snnbase_repository: Path,
    environment_path: Path,
    qwen_assets_dir: Path,
    run_id: str,
    dataset_name: str,
    publishable: bool,
    notes: Sequence[str] = (),
    schema_path: Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    try:
        resolved_run_dir = run_dir.resolve(strict=True)
    except OSError as error:
        raise PublicationError(
            f"completed run directory is missing: {run_dir}"
        ) from error
    if not resolved_run_dir.is_dir() or resolved_run_dir.is_symlink():
        raise PublicationError("completed run path must be a non-symlink directory")
    summary_path = _safe_child(
        resolved_run_dir, "run-summary.json", "training run summary"
    )
    run = _mapping(
        _load_json(summary_path, "training run summary"), "training run summary"
    )
    if (
        run.get("schema_version") != 1
        or run.get("kind") != TRAINING_RUN_KIND
        or run.get("status") != "completed"
    ):
        raise PublicationError(
            "chatbot_train run is not completed or has an unsupported schema"
        )
    try:
        selected_config = load_config(config_path)
    except ConfigError as error:
        raise PublicationError(f"resolved config is invalid: {error}") from error
    config = selected_config.document
    config_digest = _validate_consumed_config(run, selected_config, config_path)
    environment_value = _load_json(environment_path, "environment input")
    toolchain, hardware = _validate_environment(environment_value, publishable)

    initialization = _mapping(
        _mapping(run.get("core"), "run core").get("initialization"),
        "core initialization",
    )
    executable_record = _mapping(
        initialization.get("executable"), "core executable provenance"
    )
    executable_path = _regular_file(
        Path(_nonempty_string(executable_record.get("path"), "core executable path")),
        "core executable",
    )
    executable_digest = _recorded_file_digest(
        executable_path, executable_record, "core executable"
    )
    architecture, seed, command = _validate_config_binding(
        selected_config, run, executable_path
    )

    dataset, dataset_manifest, dataset_artifacts = _load_dataset(
        run, config, dataset_source, dataset_name
    )
    core_metadata = _mapping(
        _mapping(run.get("core"), "run core").get("training_metadata"),
        "training core metadata",
    )
    reference_model, tokenizer = _phase0_resources(
        qwen_assets_dir, config, dataset_manifest, core_metadata
    )

    checkpoint_record = _mapping(run.get("checkpoint"), "run checkpoint")
    checkpoint_path = _safe_child(
        resolved_run_dir,
        checkpoint_record.get("path"),
        "selected checkpoint",
    )
    checkpoint_digest = _recorded_file_digest(
        checkpoint_path, checkpoint_record, "selected checkpoint"
    )
    metrics_record = _mapping(run.get("metrics_log"), "run metrics_log")
    metrics_path = _safe_child(
        resolved_run_dir, metrics_record.get("path"), "training metrics log"
    )
    metrics_digest = _recorded_file_digest(
        metrics_path, metrics_record, "training metrics log"
    )
    metrics, validation_metrics, test_metrics = _publication_metrics(run)
    _load_metrics_log(
        metrics_path,
        dataset["split"]["test_records"],
        validation_metrics,
        test_metrics,
    )

    repositories = _repositories_from_build(
        core_metadata,
        experiments_repository,
        snnbase_repository,
        publishable,
    )
    if publishable:
        if dataset["split"]["test_records"] <= 0:
            raise PublicationError(
                "publishable manifest requires a nonempty test split"
            )
        if metrics_digest["bytes"] <= 0 or not metrics:
            raise PublicationError("publishable manifest requires retained metrics")

    artifacts = [
        {
            "kind": "training_run_summary",
            "file": _file_digest(summary_path, "training run summary"),
        },
        *dataset_artifacts,
        {"kind": "selected_checkpoint", "file": checkpoint_digest},
        {"kind": "training_metrics_jsonl", "file": metrics_digest},
        {
            "kind": "publication_environment_input",
            "file": _file_digest(environment_path, "environment input"),
        },
    ]
    if initialization.get("mode") == "verified_qwen_import":
        archive = _mapping(
            initialization.get("qwen_archive"), "Qwen archive provenance"
        )
        archive_path = Path(_nonempty_string(archive.get("path"), "Qwen archive path"))
        artifacts.append(
            {
                "kind": "qwen_dense_archive",
                "file": _recorded_file_digest(
                    archive_path, archive, "Qwen dense archive"
                ),
            }
        )

    training_config = _mapping(config.get("training"), "config training")
    manifest = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "run_id": _nonempty_string(run_id, "run id"),
        "created_utc": _completed_timestamp(run.get("completed_utc")),
        "status": "completed",
        "architecture": architecture,
        "seed": seed,
        "publishable": bool(publishable),
        "repositories": repositories,
        "executable": executable_digest,
        "config": config_digest,
        "dataset": dataset,
        "reference_model": reference_model,
        "tokenizer": tokenizer,
        "training": {
            "objective": training_config["objective"],
            "loss_scope": training_config["loss_scope"],
            "assistant_only_loss": training_config["assistant_only_loss"],
            "ignore_index": -100,
        },
        "command": command,
        "toolchain": toolchain,
        "hardware": hardware,
        "artifacts": artifacts,
        "metrics": metrics,
        "notes": [
            _nonempty_string(note, f"note {index}") for index, note in enumerate(notes)
        ],
    }
    validate_publication_manifest(
        manifest,
        schema_path=schema_path,
        require_publishable=publishable,
    )
    return manifest


def write_manifest_exclusive(path: Path, manifest: Mapping[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise PublicationError(
            f"publication output already exists and is immutable: {destination}"
        ) from error
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        try:
            destination.unlink()
        except OSError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--run-dir", required=True, type=Path)
    assemble.add_argument("--config", required=True, type=Path)
    assemble.add_argument("--dataset-source", required=True, type=Path)
    assemble.add_argument("--experiments-repo", required=True, type=Path)
    assemble.add_argument("--snnbase-repo", required=True, type=Path)
    assemble.add_argument("--environment", required=True, type=Path)
    assemble.add_argument("--qwen-assets-dir", required=True, type=Path)
    assemble.add_argument("--run-id", required=True)
    assemble.add_argument("--dataset-name", required=True)
    assemble.add_argument("--output", required=True, type=Path)
    assemble.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    assemble.add_argument("--publishable", action="store_true")
    assemble.add_argument("--note", action="append", default=[])

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    validate.add_argument("--require-publishable", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "validate":
            validate_publication_file(
                arguments.manifest,
                schema_path=arguments.schema,
                require_publishable=arguments.require_publishable,
            )
            print(
                json.dumps({"ok": True, "manifest": str(arguments.manifest.resolve())})
            )
            return 0
        manifest = assemble_publication_manifest(
            run_dir=arguments.run_dir,
            config_path=arguments.config,
            dataset_source=arguments.dataset_source,
            experiments_repository=arguments.experiments_repo,
            snnbase_repository=arguments.snnbase_repo,
            environment_path=arguments.environment,
            qwen_assets_dir=arguments.qwen_assets_dir,
            run_id=arguments.run_id,
            dataset_name=arguments.dataset_name,
            publishable=arguments.publishable,
            notes=arguments.note,
            schema_path=arguments.schema,
        )
        write_manifest_exclusive(arguments.output, manifest)
        print(
            json.dumps(
                {
                    "ok": True,
                    "output": str(arguments.output.resolve()),
                    "sha256": sha256_file(arguments.output.resolve()),
                    "publishable": manifest["publishable"],
                },
                sort_keys=True,
            )
        )
        return 0
    except (PublicationError, OSError, UnicodeError, ValueError) as error:
        print(f"chatbot_publish: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
