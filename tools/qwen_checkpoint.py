#!/usr/bin/env python3
"""Inventory and selectively export tensors from a verified Qwen checkpoint."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from qwen_contract import (
    ContractError,
    PHASE0_MODEL_ID,
    PHASE0_REVISION,
    PHASE0_TOKENIZER_FINGERPRINT,
    VerifiedAssets,
    package_versions,
    sha256_bytes,
    verify_asset_manifest,
    write_json_atomic,
)

CHECKPOINT_MANIFEST_KIND = "snnbase.qwen-checkpoint-tensors"
TENSOR_EXPORT_KIND = "snnbase.qwen-tensor-export"
SCHEMA_VERSION = 1

_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


def _add_contract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--assets-dir", required=True, type=Path)
    parser.add_argument("--model-id", required=True, help=f"Phase 0: {PHASE0_MODEL_ID}")
    parser.add_argument("--revision", required=True, help=f"Phase 0: {PHASE0_REVISION}")
    parser.add_argument(
        "--expected-tokenizer-fingerprint",
        required=True,
        help=f"Phase 0: {PHASE0_TOKENIZER_FINGERPRINT}",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser(
        "manifest", description="List every checkpoint tensor."
    )
    _add_contract_arguments(manifest)
    manifest.add_argument("--output", required=True, type=Path)
    manifest.add_argument(
        "--hash-tensors",
        action="store_true",
        help="also read and hash canonical raw bytes for every tensor",
    )

    export = subparsers.add_parser(
        "export", description="Export explicitly selected raw tensors."
    )
    _add_contract_arguments(export)
    export.add_argument("--tensor", required=True, action="append", dest="tensors")
    export.add_argument("--output-dir", required=True, type=Path)
    return parser


def _safe_tensor_filename(index: int, name: str) -> str:
    sanitized = _SAFE_COMPONENT_RE.sub("_", name).strip("._")
    if not sanitized:
        sanitized = "tensor"
    return f"{index:04d}-{sanitized[:180]}.bin"


def _numel(shape: Sequence[int]) -> int:
    result = 1
    for extent in shape:
        if not isinstance(extent, int) or isinstance(extent, bool) or extent < 0:
            raise ContractError(f"invalid tensor shape: {shape!r}")
        result *= extent
    return result


def _tensor_bytes(tensor: Any) -> bytes:
    if sys.byteorder != "little":
        raise ContractError("raw tensor export currently requires a little-endian host")
    if hasattr(tensor, "raw_bytes_for_test"):
        value = tensor.raw_bytes_for_test()
        if not isinstance(value, bytes):
            raise ContractError("test tensor raw byte hook returned a non-bytes value")
        return value
    try:
        import torch
    except Exception as error:
        raise ContractError(
            "PyTorch is required for checkpoint tensor export"
        ) from error
    try:
        contiguous = tensor.detach().cpu().contiguous()
        return contiguous.view(torch.uint8).numpy().tobytes(order="C")
    except Exception as error:
        raise ContractError(
            f"could not canonicalize checkpoint tensor bytes: {error}"
        ) from error


def tensor_record(
    name: str,
    shard: str,
    shape: Sequence[int],
    dtype: str,
    *,
    tensor: Any | None = None,
) -> dict[str, Any]:
    if not isinstance(name, str) or not name:
        raise ContractError("tensor name must be non-empty")
    if dtype not in _DTYPE_BYTES:
        raise ContractError(f"unsupported safetensors dtype {dtype!r} for {name}")
    normalized_shape = list(shape)
    element_count = _numel(normalized_shape)
    record: dict[str, Any] = {
        "name": name,
        "shard": shard,
        "shape": normalized_shape,
        "dtype": dtype,
        "numel": element_count,
        "storage_bytes": element_count * _DTYPE_BYTES[dtype],
    }
    if tensor is not None:
        raw = _tensor_bytes(tensor)
        if len(raw) != record["storage_bytes"]:
            raise ContractError(
                f"raw byte size mismatch for {name}: expected {record['storage_bytes']}, got {len(raw)}"
            )
        record["raw_little_endian_sha256"] = sha256_bytes(raw)
    return record


def _weight_map(assets: VerifiedAssets) -> dict[str, str] | None:
    index_path = assets.root / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    try:
        with index_path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"could not load safetensors index: {error}") from error
    raw_map = document.get("weight_map") if isinstance(document, Mapping) else None
    if not isinstance(raw_map, Mapping) or not raw_map:
        raise ContractError("model.safetensors.index.json has no weight_map")
    result: dict[str, str] = {}
    for name, shard in raw_map.items():
        if not isinstance(name, str) or not isinstance(shard, str):
            raise ContractError("safetensors weight_map must map strings to strings")
        result[name] = shard
    return result


def _safetensor_shards(assets: VerifiedAssets) -> list[Path]:
    shards = sorted(assets.root.glob("*.safetensors"))
    if not shards:
        raise ContractError("checkpoint export currently requires safetensors weights")
    allowed = {record["path"] for record in assets.manifest["weight_files"]}
    for shard in shards:
        if shard.name not in allowed:
            raise ContractError(f"unmanifested safetensors shard: {shard.name}")
    return shards


def iter_checkpoint_tensors(
    assets: VerifiedAssets, *, load_tensors: bool
) -> Iterator[tuple[dict[str, Any], Any | None]]:
    try:
        from safetensors import safe_open
    except Exception as error:
        raise ContractError(
            "safetensors is required for checkpoint inspection"
        ) from error

    expected_map = _weight_map(assets)
    observed_names: set[str] = set()
    for shard in _safetensor_shards(assets):
        try:
            with safe_open(shard, framework="pt", device="cpu") as checkpoint:
                for name in sorted(checkpoint.keys()):
                    if name in observed_names:
                        raise ContractError(f"duplicate checkpoint tensor: {name}")
                    observed_names.add(name)
                    view = checkpoint.get_slice(name)
                    shape = list(view.get_shape())
                    dtype = view.get_dtype()
                    if (
                        expected_map is not None
                        and expected_map.get(name) != shard.name
                    ):
                        raise ContractError(
                            f"checkpoint index disagrees for tensor {name}"
                        )
                    tensor = checkpoint.get_tensor(name) if load_tensors else None
                    yield (
                        tensor_record(name, shard.name, shape, dtype, tensor=tensor),
                        tensor,
                    )
        except ContractError:
            raise
        except Exception as error:
            raise ContractError(f"could not inspect {shard.name}: {error}") from error
    if expected_map is not None and observed_names != set(expected_map):
        missing = sorted(set(expected_map) - observed_names)
        extra = sorted(observed_names - set(expected_map))
        raise ContractError(
            f"checkpoint index mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )


def checkpoint_manifest(
    assets: VerifiedAssets, *, hash_tensors: bool
) -> dict[str, Any]:
    tensors = [
        record
        for record, _ in iter_checkpoint_tensors(assets, load_tensors=hash_tensors)
    ]
    total_parameters = sum(record["numel"] for record in tensors)
    total_bytes = sum(record["storage_bytes"] for record in tensors)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CHECKPOINT_MANIFEST_KIND,
        "oracle": {
            "model_id": assets.model_id,
            "revision": assets.revision,
            "tokenizer_fingerprint_sha256": assets.tokenizer_fingerprint,
        },
        "implementation": package_versions(("torch", "safetensors")),
        "checkpoint_files": assets.manifest["weight_files"],
        "tensor_hashes_included": hash_tensors,
        "tensor_count": len(tensors),
        "total_parameters": total_parameters,
        "total_storage_bytes": total_bytes,
        "tensors": tensors,
    }


def export_tensors(
    assets: VerifiedAssets, selected_names: Sequence[str], output_dir: Path
) -> dict[str, Any]:
    if not selected_names or any(not name for name in selected_names):
        raise ContractError("at least one non-empty --tensor is required")
    if len(set(selected_names)) != len(selected_names):
        raise ContractError("duplicate --tensor selections are forbidden")
    selected = set(selected_names)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ContractError("tensor export directory must be empty")

    records: dict[str, tuple[dict[str, Any], bytes]] = {}
    for record, tensor in iter_checkpoint_tensors(assets, load_tensors=True):
        if record["name"] in selected:
            assert tensor is not None
            raw = _tensor_bytes(tensor)
            records[record["name"]] = (record, raw)
    missing = [name for name in selected_names if name not in records]
    if missing:
        raise ContractError(f"requested tensors were not found: {', '.join(missing)}")

    exports: list[dict[str, Any]] = []
    for index, name in enumerate(selected_names):
        record, raw = records[name]
        filename = _safe_tensor_filename(index, name)
        path = output_dir / filename
        path.write_bytes(raw)
        exports.append(
            {
                **record,
                "file": filename,
                "file_sha256": sha256_bytes(raw),
                "encoding": "contiguous-raw-little-endian",
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": TENSOR_EXPORT_KIND,
        "oracle": {
            "model_id": assets.model_id,
            "revision": assets.revision,
            "tokenizer_fingerprint_sha256": assets.tokenizer_fingerprint,
        },
        "exports": exports,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    assets = verify_asset_manifest(
        arguments.assets_dir,
        arguments.model_id,
        arguments.revision,
        arguments.expected_tokenizer_fingerprint,
        require_weights=True,
    )
    if arguments.command == "manifest":
        result = checkpoint_manifest(assets, hash_tensors=arguments.hash_tensors)
        write_json_atomic(arguments.output, result)
        return result
    return export_tensors(assets, arguments.tensors, arguments.output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        result = run(parser.parse_args(argv))
    except (ContractError, OSError, ValueError) as error:
        print(f"qwen_checkpoint: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "kind": result["kind"],
                "tensor_count": result.get(
                    "tensor_count", len(result.get("exports", []))
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
