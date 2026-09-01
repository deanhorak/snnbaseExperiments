#!/usr/bin/env python3
"""Validate and convert dense Qwen3 safetensors to the snnbase archive format."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

from qwen_contract import (
    ContractError,
    PHASE0_MODEL_ID,
    PHASE0_REVISION,
    PHASE0_TOKENIZER_FINGERPRINT,
    VerifiedAssets,
    load_json,
    sha256_file,
    verify_asset_manifest,
    write_json_atomic,
)

ARCHIVE_MAGIC = b"SNNQWEN\0"
ARCHIVE_VERSION = 1
ENDIAN_MARKER = 0x01020304
ARCHIVE_HEADER = struct.Struct("<8sIIIIQQQQ32s32s32s32s32s40s")
CONFIG_BLOCK = struct.Struct("<QQQQQQQQddII")
TENSOR_ENTRY = struct.Struct("<HHBBHQQ32s")
ARCHIVE_ALIGNMENT = 64
DTYPE_BFLOAT16 = 1
FLAG_TIED_EMBEDDINGS = 1 << 0
FLAG_QUERY_KEY_NORMALIZATION = 1 << 1
FLAG_SILU = 1 << 2
FLAG_NO_ATTENTION_BIAS = 1 << 3
FLAG_NO_SLIDING_WINDOW = 1 << 4
QWEN_REQUIRED_FLAGS = (
    FLAG_TIED_EMBEDDINGS
    | FLAG_QUERY_KEY_NORMALIZATION
    | FLAG_SILU
    | FLAG_NO_ATTENTION_BIAS
    | FLAG_NO_SLIDING_WINDOW
)

ARCHIVE_FILENAME = "qwen-dense.snnq"
MANIFEST_FILENAME = "manifest.json"
INVENTORY_KIND = "snnbase.qwen3-dense-inventory"
CONVERSION_KIND = "snnbase.qwen3-dense-conversion"
SCHEMA_VERSION = 1

MAX_SAFETENSORS_HEADER = 64 * 1024 * 1024
MAX_TENSOR_COUNT = 4096
MAX_RANK = 8
MAX_NAME_BYTES = 1024
COPY_BLOCK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class QwenDenseConfig:
    vocabulary_size: int
    maximum_sequence_length: int
    model_dimension: int
    layer_count: int
    query_head_count: int
    key_value_head_count: int
    head_dimension: int
    feed_forward_dimension: int
    rms_epsilon: float
    rope_base: float
    flags: int = QWEN_REQUIRED_FLAGS

    def archive_block(self) -> bytes:
        return CONFIG_BLOCK.pack(
            self.vocabulary_size,
            self.maximum_sequence_length,
            self.model_dimension,
            self.layer_count,
            self.query_head_count,
            self.key_value_head_count,
            self.head_dimension,
            self.feed_forward_dimension,
            self.rms_epsilon,
            self.rope_base,
            self.flags,
            0,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "vocabulary_size": self.vocabulary_size,
            "maximum_sequence_length": self.maximum_sequence_length,
            "model_dimension": self.model_dimension,
            "layer_count": self.layer_count,
            "query_head_count": self.query_head_count,
            "key_value_head_count": self.key_value_head_count,
            "head_dimension": self.head_dimension,
            "feed_forward_dimension": self.feed_forward_dimension,
            "rms_epsilon": self.rms_epsilon,
            "rope_base": self.rope_base,
            "query_key_normalization": True,
            "hidden_activation": "silu",
            "attention_bias": False,
            "sliding_window": False,
            "tied_output_projection": True,
        }


@dataclass(frozen=True)
class TensorMapping:
    source_name: str
    destination_name: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class SafeTensorEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    absolute_offset: int
    length: int


@dataclass(frozen=True)
class ArchiveTensor:
    mapping: TensorMapping
    source: SafeTensorEntry
    raw_sha256: str
    archive_offset: int


def _positive_int(document: Mapping[str, Any], name: str) -> int:
    value = document.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"Qwen config {name} must be a positive integer")
    return value


def parse_qwen_dense_config(path: Path) -> QwenDenseConfig:
    raw = load_json(path)
    if not isinstance(raw, Mapping):
        raise ContractError("Qwen config must be a JSON object")
    exact_values = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "attention_bias": False,
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "use_sliding_window": False,
        "sliding_window": None,
        "rope_scaling": None,
    }
    for key, expected in exact_values.items():
        if raw.get(key) != expected:
            raise ContractError(
                f"unsupported Qwen config {key}: expected {expected!r}, found {raw.get(key)!r}"
            )
    if raw.get("torch_dtype") not in ("bfloat16", "bf16"):
        raise ContractError(
            "the dense import contract requires a BF16 source checkpoint"
        )

    config = QwenDenseConfig(
        vocabulary_size=_positive_int(raw, "vocab_size"),
        maximum_sequence_length=_positive_int(raw, "max_position_embeddings"),
        model_dimension=_positive_int(raw, "hidden_size"),
        layer_count=_positive_int(raw, "num_hidden_layers"),
        query_head_count=_positive_int(raw, "num_attention_heads"),
        key_value_head_count=_positive_int(raw, "num_key_value_heads"),
        head_dimension=_positive_int(raw, "head_dim"),
        feed_forward_dimension=_positive_int(raw, "intermediate_size"),
        rms_epsilon=float(raw.get("rms_norm_eps", 0.0)),
        rope_base=float(raw.get("rope_theta", 0.0)),
    )
    if (
        config.query_head_count % config.key_value_head_count != 0
        or config.head_dimension % 2 != 0
        or not math.isfinite(config.rms_epsilon)
        or config.rms_epsilon <= 0.0
        or not math.isfinite(config.rope_base)
        or config.rope_base <= 1.0
    ):
        raise ContractError("Qwen attention/RMS/RoPE dimensions are incompatible")
    return config


def expected_tensor_mappings(config: QwenDenseConfig) -> list[TensorMapping]:
    hidden = config.model_dimension
    query_width = config.query_head_count * config.head_dimension
    key_value_width = config.key_value_head_count * config.head_dimension
    intermediate = config.feed_forward_dimension
    mappings = [
        TensorMapping(
            "model.embed_tokens.weight",
            "token_embedding.weight",
            (config.vocabulary_size, hidden),
        )
    ]
    for layer in range(config.layer_count):
        source = f"model.layers.{layer}"
        destination = f"block_{layer}"
        mappings.extend(
            [
                TensorMapping(
                    f"{source}.input_layernorm.weight",
                    f"{destination}.attention_norm.weight",
                    (hidden,),
                ),
                TensorMapping(
                    f"{source}.self_attn.q_proj.weight",
                    f"{destination}.attention.query.weight",
                    (query_width, hidden),
                ),
                TensorMapping(
                    f"{source}.self_attn.k_proj.weight",
                    f"{destination}.attention.key.weight",
                    (key_value_width, hidden),
                ),
                TensorMapping(
                    f"{source}.self_attn.v_proj.weight",
                    f"{destination}.attention.value.weight",
                    (key_value_width, hidden),
                ),
                TensorMapping(
                    f"{source}.self_attn.o_proj.weight",
                    f"{destination}.attention.output.weight",
                    (hidden, query_width),
                ),
                TensorMapping(
                    f"{source}.self_attn.q_norm.weight",
                    f"{destination}.attention.q_norm.weight",
                    (config.head_dimension,),
                ),
                TensorMapping(
                    f"{source}.self_attn.k_norm.weight",
                    f"{destination}.attention.k_norm.weight",
                    (config.head_dimension,),
                ),
                TensorMapping(
                    f"{source}.post_attention_layernorm.weight",
                    f"{destination}.feed_forward_norm.weight",
                    (hidden,),
                ),
                TensorMapping(
                    f"{source}.mlp.gate_proj.weight",
                    f"{destination}.feed_forward_gate.weight",
                    (intermediate, hidden),
                ),
                TensorMapping(
                    f"{source}.mlp.up_proj.weight",
                    f"{destination}.feed_forward_up.weight",
                    (intermediate, hidden),
                ),
                TensorMapping(
                    f"{source}.mlp.down_proj.weight",
                    f"{destination}.feed_forward_down.weight",
                    (hidden, intermediate),
                ),
            ]
        )
    mappings.append(TensorMapping("model.norm.weight", "final_norm.weight", (hidden,)))
    return mappings


def _shape(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value or len(value) > MAX_RANK:
        raise ContractError(f"safetensors shape for {name} is invalid")
    result: list[int] = []
    for extent in value:
        if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
            raise ContractError(f"safetensors shape for {name} is invalid")
        result.append(extent)
    return tuple(result)


def read_safetensors_inventory(path: Path) -> dict[str, SafeTensorEntry]:
    path = path.resolve(strict=True)
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ContractError(
                "safetensors file is truncated before its header length"
            )
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length == 0 or header_length > MAX_SAFETENSORS_HEADER:
            raise ContractError(
                "safetensors header length is outside the fail-closed limit"
            )
        if 8 + header_length > file_size:
            raise ContractError("safetensors header extends beyond the file")
        header_bytes = stream.read(header_length)

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"duplicate safetensors JSON key: {key}")
            result[key] = value
        return result

    try:
        header = json.loads(
            header_bytes.decode("utf-8"), object_pairs_hook=reject_duplicate_pairs
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid safetensors JSON header: {error}") from error
    if not isinstance(header, Mapping):
        raise ContractError("safetensors header must be an object")
    data_start = 8 + header_length
    result: dict[str, SafeTensorEntry] = {}
    intervals: list[tuple[int, int, str]] = []
    for name, raw_entry in header.items():
        if name == "__metadata__":
            if not isinstance(raw_entry, Mapping):
                raise ContractError("safetensors __metadata__ must be an object")
            continue
        if (
            not isinstance(name, str)
            or not name
            or len(name.encode("utf-8")) > MAX_NAME_BYTES
        ):
            raise ContractError("safetensors tensor name is invalid")
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != {
            "dtype",
            "shape",
            "data_offsets",
        }:
            raise ContractError(f"safetensors entry for {name} has unexpected fields")
        dtype = raw_entry["dtype"]
        shape = _shape(raw_entry["shape"], name)
        offsets = raw_entry["data_offsets"]
        if (
            not isinstance(dtype, str)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in offsets
            )
        ):
            raise ContractError(f"safetensors entry for {name} is malformed")
        begin, end = offsets
        expected_length = math.prod(shape) * (2 if dtype == "BF16" else 0)
        if (
            begin < 0
            or end <= begin
            or end - begin != expected_length
            or data_start + end > file_size
        ):
            raise ContractError(f"safetensors data range for {name} is invalid")
        entry = SafeTensorEntry(name, dtype, shape, data_start + begin, end - begin)
        result[name] = entry
        intervals.append((begin, end, name))
    if not result or len(result) > MAX_TENSOR_COUNT:
        raise ContractError("safetensors tensor count is outside the fail-closed limit")
    intervals.sort()
    previous_end = 0
    for begin, end, name in intervals:
        if begin != previous_end:
            raise ContractError(
                f"safetensors payload has a gap or overlap before {name}"
            )
        previous_end = end
    if data_start + previous_end != file_size:
        raise ContractError(
            "safetensors payload does not account for the complete file"
        )
    return result


def validate_inventory(
    checkpoint: Path, config: QwenDenseConfig
) -> tuple[list[TensorMapping], dict[str, SafeTensorEntry]]:
    mappings = expected_tensor_mappings(config)
    inventory = read_safetensors_inventory(checkpoint)
    expected_names = {mapping.source_name for mapping in mappings}
    actual_names = set(inventory)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ContractError(
            f"Qwen tensor inventory mismatch: missing={missing[:8]}, extra={extra[:8]}"
        )
    destinations: set[str] = set()
    for mapping in mappings:
        source = inventory[mapping.source_name]
        if source.dtype != "BF16" or source.shape != mapping.shape:
            raise ContractError(
                f"Qwen tensor mismatch for {mapping.source_name}: expected BF16 {list(mapping.shape)}, "
                f"found {source.dtype} {list(source.shape)}"
            )
        if mapping.destination_name in destinations:
            raise ContractError(
                f"duplicate destination tensor: {mapping.destination_name}"
            )
        destinations.add(mapping.destination_name)
    return mappings, inventory


def lif_initialization_contract(config: QwenDenseConfig) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for layer in range(config.layer_count):
        for site, size in (
            ("attention_lif", config.model_dimension),
            ("feed_forward_lif", config.feed_forward_dimension),
        ):
            for parameter in ("raw_threshold", "raw_leak"):
                result.append(
                    {
                        "destination_name": f"block_{layer}.{site}.{parameter}",
                        "shape": [size],
                        "dtype": "runtime-default",
                        "initialization": "DecoderConfig.lif",
                        "source": None,
                    }
                )
    return result


def inventory_document(
    assets: VerifiedAssets,
    config: QwenDenseConfig,
    mappings: Sequence[TensorMapping],
    inventory: Mapping[str, SafeTensorEntry],
) -> dict[str, Any]:
    checkpoint_record = next(
        record
        for record in assets.manifest["weight_files"]
        if record["path"] == "model.safetensors"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": INVENTORY_KIND,
        "oracle": {
            "model_id": assets.model_id,
            "revision": assets.revision,
            "tokenizer_fingerprint_sha256": assets.tokenizer_fingerprint,
            "config_sha256": sha256_file(assets.root / "config.json"),
            "checkpoint": checkpoint_record,
        },
        "decoder_config": config.as_dict(),
        "tensor_count": len(mappings),
        "total_tensor_bytes": sum(
            inventory[item.source_name].length for item in mappings
        ),
        "tied_output_projection": {
            "source_destination": "token_embedding.weight",
            "operation": "transpose-at-readout",
            "separate_lm_head": False,
        },
        "dense_tensors": [
            {
                "source_name": item.source_name,
                "destination_name": item.destination_name,
                "dtype": inventory[item.source_name].dtype,
                "shape": list(item.shape),
                "source_offset": inventory[item.source_name].absolute_offset,
                "length": inventory[item.source_name].length,
            }
            for item in mappings
        ],
        "unmapped_runtime_parameters": lif_initialization_contract(config),
    }


def _hash_region(path: Path, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = length
        while remaining:
            block = stream.read(min(remaining, COPY_BLOCK_SIZE))
            if not block:
                raise ContractError("source checkpoint ended while hashing a tensor")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _align(value: int, alignment: int = ARCHIVE_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _entry_size(mapping: TensorMapping) -> int:
    source_bytes = mapping.source_name.encode("utf-8")
    destination_bytes = mapping.destination_name.encode("utf-8")
    if len(source_bytes) > MAX_NAME_BYTES or len(destination_bytes) > MAX_NAME_BYTES:
        raise ContractError("archive tensor name exceeds its byte limit")
    return (
        TENSOR_ENTRY.size
        + 8 * len(mapping.shape)
        + len(source_bytes)
        + len(destination_bytes)
    )


def _metadata_bytes(
    config: QwenDenseConfig,
    model_id: str,
    archive_tensors: Sequence[ArchiveTensor],
) -> bytes:
    model_id_bytes = model_id.encode("utf-8")
    if not model_id_bytes or len(model_id_bytes) > MAX_NAME_BYTES:
        raise ContractError("archive model id exceeds its byte limit")
    chunks = [
        config.archive_block(),
        struct.pack("<H", len(model_id_bytes)),
        model_id_bytes,
    ]
    for tensor in archive_tensors:
        source = tensor.mapping.source_name.encode("utf-8")
        destination = tensor.mapping.destination_name.encode("utf-8")
        chunks.append(
            TENSOR_ENTRY.pack(
                len(source),
                len(destination),
                DTYPE_BFLOAT16,
                len(tensor.mapping.shape),
                0,
                tensor.archive_offset,
                tensor.source.length,
                bytes.fromhex(tensor.raw_sha256),
            )
        )
        chunks.append(
            struct.pack(f"<{len(tensor.mapping.shape)}Q", *tensor.mapping.shape)
        )
        chunks.extend((source, destination))
    return b"".join(chunks)


def _copy_region(
    source_stream: BinaryIO,
    destination_stream: BinaryIO,
    offset: int,
    length: int,
    payload_digest: Any,
) -> None:
    source_stream.seek(offset)
    remaining = length
    while remaining:
        block = source_stream.read(min(remaining, COPY_BLOCK_SIZE))
        if not block:
            raise ContractError("source checkpoint ended while copying a tensor")
        destination_stream.write(block)
        payload_digest.update(block)
        remaining -= len(block)


def convert_archive(
    assets: VerifiedAssets,
    config: QwenDenseConfig,
    mappings: Sequence[TensorMapping],
    inventory: Mapping[str, SafeTensorEntry],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ContractError(
            "output-dir already exists; converted checkpoints are immutable"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    checkpoint = assets.root / "model.safetensors"
    try:
        checkpoint_record = next(
            record
            for record in assets.manifest["weight_files"]
            if record["path"] == "model.safetensors"
        )
        source_checkpoint_sha = checkpoint_record["sha256"]
        config_sha = sha256_file(assets.root / "config.json")
        tensor_hashes = {
            mapping.source_name: _hash_region(
                checkpoint,
                inventory[mapping.source_name].absolute_offset,
                inventory[mapping.source_name].length,
            )
            for mapping in mappings
        }
        model_id_size = len(assets.model_id.encode("utf-8"))
        metadata_size = (
            CONFIG_BLOCK.size
            + 2
            + model_id_size
            + sum(_entry_size(mapping) for mapping in mappings)
        )
        payload_offset = _align(ARCHIVE_HEADER.size + metadata_size)
        selected_offset = payload_offset
        archive_tensors: list[ArchiveTensor] = []
        for mapping in mappings:
            source = inventory[mapping.source_name]
            archive_tensors.append(
                ArchiveTensor(
                    mapping, source, tensor_hashes[mapping.source_name], selected_offset
                )
            )
            selected_offset += source.length
        payload_size = selected_offset - payload_offset
        archive_size = payload_offset + payload_size
        metadata = _metadata_bytes(config, assets.model_id, archive_tensors)
        if len(metadata) != metadata_size:
            raise ContractError("internal archive metadata size calculation disagrees")
        metadata_sha = hashlib.sha256(metadata).digest()

        archive_path = temporary / ARCHIVE_FILENAME
        placeholder = ARCHIVE_HEADER.pack(
            ARCHIVE_MAGIC,
            ARCHIVE_VERSION,
            ENDIAN_MARKER,
            len(archive_tensors),
            CONFIG_BLOCK.size,
            metadata_size,
            payload_offset,
            payload_size,
            archive_size,
            bytes.fromhex(source_checkpoint_sha),
            bytes.fromhex(config_sha),
            bytes.fromhex(assets.tokenizer_fingerprint),
            metadata_sha,
            bytes(32),
            assets.revision.encode("ascii"),
        )
        with archive_path.open("xb+") as output, checkpoint.open("rb") as source_stream:
            output.write(placeholder)
            output.write(metadata)
            padding = payload_offset - output.tell()
            if padding < 0:
                raise ContractError(
                    "archive metadata exceeds its declared payload offset"
                )
            output.write(bytes(padding))
            payload_digest = hashlib.sha256()
            for tensor in archive_tensors:
                if output.tell() != tensor.archive_offset:
                    raise ContractError("archive tensor offset calculation disagrees")
                _copy_region(
                    source_stream,
                    output,
                    tensor.source.absolute_offset,
                    tensor.source.length,
                    payload_digest,
                )
            if output.tell() != archive_size:
                raise ContractError("archive size calculation disagrees")
            output.seek(0)
            output.write(
                ARCHIVE_HEADER.pack(
                    ARCHIVE_MAGIC,
                    ARCHIVE_VERSION,
                    ENDIAN_MARKER,
                    len(archive_tensors),
                    CONFIG_BLOCK.size,
                    metadata_size,
                    payload_offset,
                    payload_size,
                    archive_size,
                    bytes.fromhex(source_checkpoint_sha),
                    bytes.fromhex(config_sha),
                    bytes.fromhex(assets.tokenizer_fingerprint),
                    metadata_sha,
                    payload_digest.digest(),
                    assets.revision.encode("ascii"),
                )
            )
            output.flush()
            os.fsync(output.fileno())

        archive_sha = sha256_file(archive_path)
        inventory_base = inventory_document(assets, config, mappings, inventory)
        manifest = {
            **inventory_base,
            "kind": CONVERSION_KIND,
            "archive": {
                "path": ARCHIVE_FILENAME,
                "format": "snnbase-qwen-dense-archive",
                "format_version": ARCHIVE_VERSION,
                "byte_order": "little",
                "size_bytes": archive_path.stat().st_size,
                "sha256": archive_sha,
                "metadata_sha256": metadata_sha.hex(),
                "payload_sha256": payload_digest.hexdigest(),
                "payload_offset": payload_offset,
                "payload_size": payload_size,
            },
            "dense_tensors": [
                {
                    "source_name": tensor.mapping.source_name,
                    "destination_name": tensor.mapping.destination_name,
                    "dtype": "BF16",
                    "shape": list(tensor.mapping.shape),
                    "archive_offset": tensor.archive_offset,
                    "length": tensor.source.length,
                    "raw_sha256": tensor.raw_sha256,
                }
                for tensor in archive_tensors
            ],
        }
        write_json_atomic(temporary / MANIFEST_FILENAME, manifest)
        temporary.replace(output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _contract_arguments(parser: argparse.ArgumentParser) -> None:
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
    inventory = subparsers.add_parser(
        "inventory",
        description="Validate every source name/shape without writing tensor payloads.",
    )
    _contract_arguments(inventory)
    inventory.add_argument("--output", required=True, type=Path)
    convert = subparsers.add_parser(
        "convert", description="Write the deterministic dense archive."
    )
    _contract_arguments(convert)
    convert.add_argument("--output-dir", required=True, type=Path)
    return parser


def _load_plan(
    arguments: argparse.Namespace,
) -> tuple[
    VerifiedAssets, QwenDenseConfig, list[TensorMapping], dict[str, SafeTensorEntry]
]:
    if (
        arguments.model_id,
        arguments.revision,
        arguments.expected_tokenizer_fingerprint,
    ) != (PHASE0_MODEL_ID, PHASE0_REVISION, PHASE0_TOKENIZER_FINGERPRINT):
        raise ContractError(
            "Qwen dense archive v1 is restricted to the exact Phase 0 model pin"
        )
    assets = verify_asset_manifest(
        arguments.assets_dir,
        arguments.model_id,
        arguments.revision,
        arguments.expected_tokenizer_fingerprint,
        require_weights=True,
    )
    config = parse_qwen_dense_config(assets.root / "config.json")
    checkpoint_paths = [
        record["path"]
        for record in assets.manifest["weight_files"]
        if record["path"].endswith(".safetensors")
    ]
    if checkpoint_paths != ["model.safetensors"]:
        raise ContractError(
            "Qwen dense converter v1 requires exactly model.safetensors"
        )
    mappings, inventory = validate_inventory(assets.root / "model.safetensors", config)
    return assets, config, mappings, inventory


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        assets, config, mappings, inventory = _load_plan(arguments)
        if arguments.command == "inventory":
            if arguments.output.exists():
                raise ContractError("inventory output already exists")
            result = inventory_document(assets, config, mappings, inventory)
            write_json_atomic(arguments.output, result)
        else:
            result = convert_archive(
                assets, config, mappings, inventory, arguments.output_dir
            )
    except (ContractError, OSError, UnicodeError, ValueError) as error:
        print(f"qwen_convert: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "kind": result["kind"],
                "tensor_count": result["tensor_count"],
                "total_tensor_bytes": result["total_tensor_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
