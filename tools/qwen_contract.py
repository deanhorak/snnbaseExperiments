#!/usr/bin/env python3
"""Fail-closed asset contracts shared by the Qwen reference tools.

The tools deliberately load from a verified local snapshot.  Network access and
floating Hugging Face revisions are kept out of the reference path so a golden
file always identifies the exact tokenizer and checkpoint that produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PHASE0_MODEL_ID = "Qwen/Qwen3-0.6B-Base"
PHASE0_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
PHASE0_TOKENIZER_FINGERPRINT = (
    "6a4dac166a643066a1af821907cfd1c998c8a195e6458a90979953e848b6e237"
)
PHASE0_TOKENIZER_JSON_SHA256 = (
    "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
)
PHASE0_TOKENIZER_CONFIG_SHA256 = (
    "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5"
)
PHASE0_CONFIG_SHA256 = (
    "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59"
)
PHASE0_WEIGHT_SHA256 = (
    "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"
)

ASSET_MANIFEST_NAME = "qwen-assets.json"
ASSET_MANIFEST_KIND = "snnbase.qwen-assets"
ASSET_MANIFEST_SCHEMA_VERSION = 1

TOKENIZER_REQUIRED_FILES = ("tokenizer.json", "tokenizer_config.json")
TOKENIZER_OPTIONAL_FILES = (
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)
CONFIG_FILES = ("config.json", "generation_config.json")
DOCUMENTATION_FILES = ("LICENSE", "README.md")
WEIGHT_INDEX_FILES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


class ContractError(RuntimeError):
    """Raised when provenance or content does not match the declared contract."""


@dataclass(frozen=True)
class VerifiedAssets:
    root: Path
    manifest_path: Path
    model_id: str
    revision: str
    tokenizer_fingerprint: str
    manifest: Mapping[str, Any]


def require_model_id(value: str) -> str:
    if not isinstance(value, str) or not _MODEL_ID_RE.fullmatch(value):
        raise ContractError(
            "model_id must be an explicit Hugging Face owner/repository id"
        )
    return value


def require_revision(value: str) -> str:
    if not isinstance(value, str) or not _REVISION_RE.fullmatch(value):
        raise ContractError(
            "revision must be an explicit lowercase 40-character commit SHA; "
            "branches, tags, and omitted revisions are forbidden"
        )
    return value


def require_sha256(value: str, label: str = "sha256") -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ContractError(
            f"{label} must be an explicit lowercase 64-character SHA-256"
        )
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream, object_pairs_hook=_reject_duplicate_keys)
    except ContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"could not read JSON from {path}: {error}") from error


def write_json_atomic(path: Path, value: Any) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _safe_regular_file(root: Path, relative_path: str) -> Path:
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or Path(relative_path).is_absolute()
    ):
        raise ContractError(
            f"asset path must be a non-empty relative path: {relative_path!r}"
        )
    candidate = root / relative_path
    try:
        candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError, OSError) as error:
        raise ContractError(
            f"asset escapes its snapshot or is missing: {relative_path}"
        ) from error
    if not candidate.is_file() or candidate.is_symlink():
        raise ContractError(
            f"asset must be a regular non-symlink file: {relative_path}"
        )
    return candidate


def file_record(root: Path, relative_path: str) -> dict[str, Any]:
    path = _safe_regular_file(root, relative_path)
    return {
        "path": relative_path,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def tokenizer_file_names(root: Path) -> tuple[str, ...]:
    missing = [name for name in TOKENIZER_REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise ContractError(
            f"tokenizer snapshot is missing required files: {', '.join(missing)}"
        )
    present = list(TOKENIZER_REQUIRED_FILES)
    present.extend(name for name in TOKENIZER_OPTIONAL_FILES if (root / name).is_file())
    return tuple(present)


def tokenizer_records(root: Path) -> list[dict[str, Any]]:
    return [file_record(root, name) for name in tokenizer_file_names(root)]


def tokenizer_fingerprint_from_records(records: Sequence[Mapping[str, Any]]) -> str:
    normalized = {
        "algorithm": "sha256",
        "files": [
            {
                "path": record["path"],
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
            }
            for record in records
        ],
        "schema_version": 1,
    }
    serialized = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return sha256_bytes(serialized)


def tokenizer_fingerprint(root: Path) -> tuple[str, list[dict[str, Any]]]:
    records = tokenizer_records(root)
    return tokenizer_fingerprint_from_records(records), records


def _download_metadata_revision(root: Path, relative_path: str) -> str:
    metadata = (
        root / ".cache" / "huggingface" / "download" / f"{relative_path}.metadata"
    )
    try:
        first_line = metadata.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, UnicodeError, IndexError) as error:
        raise ContractError(
            f"missing Hugging Face revision proof for {relative_path}; "
            "download the pinned file with `hf download --revision <commit>`"
        ) from error
    return require_revision(first_line)


def _candidate_snapshot_files(root: Path) -> tuple[str, ...]:
    names: list[str] = list(tokenizer_file_names(root))
    names.extend(
        name for name in CONFIG_FILES + DOCUMENTATION_FILES if (root / name).is_file()
    )
    names.extend(name for name in WEIGHT_INDEX_FILES if (root / name).is_file())
    names.extend(
        path.name for path in sorted(root.glob("*.safetensors")) if path.is_file()
    )
    names.extend(
        path.name for path in sorted(root.glob("pytorch_model*.bin")) if path.is_file()
    )
    return tuple(dict.fromkeys(names))


def _require_phase0_file_hashes(
    model_id: str,
    revision: str,
    records: Sequence[Mapping[str, Any]],
    *,
    require_weights: bool = False,
) -> None:
    if (model_id, revision) != (PHASE0_MODEL_ID, PHASE0_REVISION):
        return
    actual = {record["path"]: record["sha256"] for record in records}
    expected = {
        "config.json": PHASE0_CONFIG_SHA256,
        "tokenizer.json": PHASE0_TOKENIZER_JSON_SHA256,
        "tokenizer_config.json": PHASE0_TOKENIZER_CONFIG_SHA256,
    }
    if require_weights or "model.safetensors" in actual:
        expected["model.safetensors"] = PHASE0_WEIGHT_SHA256
    for path, expected_hash in expected.items():
        if actual.get(path) != expected_hash:
            raise ContractError(
                f"Phase 0 asset hash mismatch for {path}: expected {expected_hash}, "
                f"found {actual.get(path)!r}"
            )


def create_asset_manifest(
    root: Path,
    model_id: str,
    revision: str,
    expected_tokenizer_fingerprint: str,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    model_id = require_model_id(model_id)
    revision = require_revision(revision)
    expected = require_sha256(expected_tokenizer_fingerprint, "tokenizer fingerprint")

    fingerprint, tokenizer_assets = tokenizer_fingerprint(root)
    if fingerprint != expected:
        raise ContractError(
            f"tokenizer fingerprint mismatch: expected {expected}, computed {fingerprint}"
        )

    names = _candidate_snapshot_files(root)
    if "config.json" not in names:
        raise ContractError("snapshot is missing config.json")
    for name in names:
        proved_revision = _download_metadata_revision(root, name)
        if proved_revision != revision:
            raise ContractError(
                f"revision proof mismatch for {name}: expected {revision}, found {proved_revision}"
            )

    records = [file_record(root, name) for name in names]
    _require_phase0_file_hashes(model_id, revision, records)
    weight_records = [
        record
        for record in records
        if record["path"].endswith((".safetensors", ".bin"))
        or record["path"] in WEIGHT_INDEX_FILES
    ]
    weight_payloads = [
        record
        for record in weight_records
        if record["path"].endswith(".safetensors")
        or record["path"].startswith("pytorch_model")
        and record["path"].endswith(".bin")
    ]
    return {
        "schema_version": ASSET_MANIFEST_SCHEMA_VERSION,
        "kind": ASSET_MANIFEST_KIND,
        "model_id": model_id,
        "revision": revision,
        "tokenizer": {
            "fingerprint_algorithm": "sha256-canonical-file-manifest-v1",
            "fingerprint_sha256": fingerprint,
            "tokenizer_json_sha256": next(
                record["sha256"]
                for record in tokenizer_assets
                if record["path"] == "tokenizer.json"
            ),
            "files": tokenizer_assets,
        },
        "files": records,
        "weights_present": bool(weight_payloads),
        "weight_files": weight_records,
    }


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a JSON object")
    return value


def _verify_records(
    root: Path, records_value: Any, label: str
) -> list[Mapping[str, Any]]:
    if not isinstance(records_value, list) or not records_value:
        raise ContractError(f"{label} must be a non-empty JSON array")
    verified: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for index, raw_record in enumerate(records_value):
        record = _require_mapping(raw_record, f"{label}[{index}]")
        path = record.get("path")
        if not isinstance(path, str) or path in seen:
            raise ContractError(f"{label}[{index}].path must be unique")
        seen.add(path)
        expected_hash = require_sha256(record.get("sha256"), f"hash for {path}")
        expected_size = record.get("size_bytes")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
        ):
            raise ContractError(f"size for {path} must be a nonnegative integer")
        actual = file_record(root, path)
        if actual["size_bytes"] != expected_size or actual["sha256"] != expected_hash:
            raise ContractError(f"asset content mismatch: {path}")
        verified.append(record)
    return verified


def verify_asset_manifest(
    root: Path,
    model_id: str,
    revision: str,
    expected_tokenizer_fingerprint: str,
    *,
    require_weights: bool = False,
    manifest_path: Path | None = None,
) -> VerifiedAssets:
    root = root.resolve(strict=True)
    model_id = require_model_id(model_id)
    revision = require_revision(revision)
    expected = require_sha256(expected_tokenizer_fingerprint, "tokenizer fingerprint")
    selected_manifest = manifest_path or root / ASSET_MANIFEST_NAME
    selected_manifest = selected_manifest.resolve(strict=True)
    manifest = _require_mapping(load_json(selected_manifest), "asset manifest")

    if manifest.get("schema_version") != ASSET_MANIFEST_SCHEMA_VERSION:
        raise ContractError("unsupported Qwen asset manifest schema_version")
    if manifest.get("kind") != ASSET_MANIFEST_KIND:
        raise ContractError("asset manifest kind is not snnbase.qwen-assets")
    if manifest.get("model_id") != model_id:
        raise ContractError(
            f"model id mismatch: expected {model_id}, found {manifest.get('model_id')!r}"
        )
    if manifest.get("revision") != revision:
        raise ContractError(
            f"revision mismatch: expected {revision}, found {manifest.get('revision')!r}"
        )

    tokenizer = _require_mapping(manifest.get("tokenizer"), "manifest.tokenizer")
    manifest_fingerprint = require_sha256(
        tokenizer.get("fingerprint_sha256"), "manifest tokenizer fingerprint"
    )
    if manifest_fingerprint != expected:
        raise ContractError(
            f"tokenizer fingerprint mismatch: expected {expected}, manifest has {manifest_fingerprint}"
        )

    tokenizer_manifest_records = _verify_records(
        root, tokenizer.get("files"), "tokenizer files"
    )
    local_names = tokenizer_file_names(root)
    manifest_names = tuple(record["path"] for record in tokenizer_manifest_records)
    if manifest_names != local_names:
        raise ContractError(
            "tokenizer manifest file order/set does not match behavior-determining local files"
        )
    computed = tokenizer_fingerprint_from_records(tokenizer_manifest_records)
    if computed != expected:
        raise ContractError(
            f"tokenizer content fingerprint mismatch: expected {expected}, computed {computed}"
        )

    all_records = _verify_records(root, manifest.get("files"), "snapshot files")
    _require_phase0_file_hashes(
        model_id, revision, all_records, require_weights=require_weights
    )
    all_paths = {record["path"] for record in all_records}
    if "config.json" not in all_paths:
        raise ContractError("asset manifest does not cover config.json")

    weight_records_value = manifest.get("weight_files")
    if not isinstance(weight_records_value, list):
        raise ContractError("manifest.weight_files must be a JSON array")
    weight_records = (
        _verify_records(root, weight_records_value, "weight files")
        if weight_records_value
        else []
    )
    weight_payloads = [
        record
        for record in weight_records
        if record["path"].endswith(".safetensors")
        or record["path"].startswith("pytorch_model")
        and record["path"].endswith(".bin")
    ]
    if bool(weight_payloads) != bool(manifest.get("weights_present")):
        raise ContractError("weights_present disagrees with weight_files")
    if require_weights and not weight_payloads:
        raise ContractError(
            "verified model weights are required for this command, but this is a tokenizer-only snapshot"
        )
    if require_weights and (model_id, revision) == (
        PHASE0_MODEL_ID,
        PHASE0_REVISION,
    ):
        phase0_payloads = [
            (record["path"], record["sha256"]) for record in weight_payloads
        ]
        if phase0_payloads != [("model.safetensors", PHASE0_WEIGHT_SHA256)]:
            raise ContractError(
                "Phase 0 weight payload must be exactly the pinned model.safetensors"
            )

    return VerifiedAssets(
        root=root,
        manifest_path=selected_manifest,
        model_id=model_id,
        revision=revision,
        tokenizer_fingerprint=expected,
        manifest=manifest,
    )


def package_versions(names: Iterable[str]) -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "unavailable"
    return result


def load_transformers_tokenizer(assets: VerifiedAssets) -> Any:
    try:
        from transformers import AutoTokenizer
    except (
        Exception
    ) as error:  # Transformers can fail on incompatible optional packages.
        raise ContractError(
            "could not import transformers.AutoTokenizer; install the exact reference lock"
        ) from error
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(assets.root),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
    except Exception as error:
        raise ContractError(
            f"could not load the verified local tokenizer: {error}"
        ) from error
    if not getattr(tokenizer, "is_fast", False):
        raise ContractError(
            "the Qwen reference contract requires the fast tokenizer implementation"
        )
    return tokenizer
