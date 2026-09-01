from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools"))

from qwen_contract import (  # noqa: E402
    ContractError,
    file_record,
    tokenizer_fingerprint,
    verify_asset_manifest,
)
from qwen_convert import (  # noqa: E402
    ARCHIVE_FILENAME,
    ARCHIVE_MAGIC,
    ARCHIVE_VERSION,
    convert_archive,
    expected_tensor_mappings,
    parse_qwen_dense_config,
    read_safetensors_inventory,
    validate_inventory,
)

MODEL_ID = "tests/Qwen3Tiny"
REVISION = "0123456789abcdef0123456789abcdef01234567"


def _json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _config() -> dict[str, object]:
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "attention_bias": False,
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "use_sliding_window": False,
        "sliding_window": None,
        "rope_scaling": None,
        "torch_dtype": "bfloat16",
        "vocab_size": 17,
        "max_position_embeddings": 32,
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 12,
        "rms_norm_eps": 1.0e-6,
        "rope_theta": 1_000_000.0,
    }


def _write_safetensors(
    path: Path,
    mappings: list[object],
    *,
    wrong_shape_for: str | None = None,
) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for index, mapping in enumerate(mappings):
        shape = list(mapping.shape)
        if mapping.source_name == wrong_shape_for:
            shape[-1] += 1
        length = math.prod(shape) * 2
        begin = len(payload)
        # These are deterministic raw BF16 bit patterns; conversion never
        # interprets them as host floating-point values.
        payload.extend(((index * 37 + byte) % 256 for byte in range(length)))
        header[mapping.source_name] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [begin, len(payload)],
        }
    encoded = json.dumps(
        header, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _make_assets(root: Path, *, wrong_shape: bool = False) -> tuple[str, object]:
    root.mkdir()
    fixture = REPOSITORY / "tests" / "fixtures" / "chatbot" / "fake_qwen" / "assets"
    shutil.copyfile(fixture / "tokenizer.json", root / "tokenizer.json")
    shutil.copyfile(fixture / "tokenizer_config.json", root / "tokenizer_config.json")
    _json(root / "config.json", _config())
    config = parse_qwen_dense_config(root / "config.json")
    mappings = expected_tensor_mappings(config)
    _write_safetensors(
        root / "model.safetensors",
        mappings,
        wrong_shape_for=mappings[1].source_name if wrong_shape else None,
    )
    fingerprint, tokenizer_records = tokenizer_fingerprint(root)
    records = [
        *tokenizer_records,
        file_record(root, "config.json"),
        file_record(root, "model.safetensors"),
    ]
    weight = file_record(root, "model.safetensors")
    _json(
        root / "qwen-assets.json",
        {
            "schema_version": 1,
            "kind": "snnbase.qwen-assets",
            "model_id": MODEL_ID,
            "revision": REVISION,
            "tokenizer": {
                "fingerprint_algorithm": "sha256-canonical-file-manifest-v1",
                "fingerprint_sha256": fingerprint,
                "tokenizer_json_sha256": tokenizer_records[0]["sha256"],
                "files": tokenizer_records,
            },
            "files": records,
            "weights_present": True,
            "weight_files": [weight],
        },
    )
    return fingerprint, config


class QwenConvertTests(unittest.TestCase):
    def test_synthetic_conversion_is_deterministic_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            fingerprint, config = _make_assets(temporary / "assets")
            assets = verify_asset_manifest(
                temporary / "assets",
                MODEL_ID,
                REVISION,
                fingerprint,
                require_weights=True,
            )
            mappings, inventory = validate_inventory(
                assets.root / "model.safetensors", config
            )
            first = convert_archive(
                assets, config, mappings, inventory, temporary / "converted-a"
            )
            second = convert_archive(
                assets, config, mappings, inventory, temporary / "converted-b"
            )
            first_archive = temporary / "converted-a" / ARCHIVE_FILENAME
            second_archive = temporary / "converted-b" / ARCHIVE_FILENAME
            self.assertEqual(first_archive.read_bytes(), second_archive.read_bytes())
            self.assertEqual(first["archive"]["sha256"], second["archive"]["sha256"])
            self.assertEqual(first["tensor_count"], 24)
            self.assertEqual(len(first["dense_tensors"]), 24)
            self.assertEqual(len(first["unmapped_runtime_parameters"]), 8)
            self.assertTrue(
                first["tied_output_projection"]["separate_lm_head"] is False
            )
            raw = first_archive.read_bytes()
            self.assertEqual(raw[:8], ARCHIVE_MAGIC)
            self.assertEqual(struct.unpack_from("<I", raw, 8)[0], ARCHIVE_VERSION)
            self.assertEqual(
                hashlib.sha256(raw).hexdigest(), first["archive"]["sha256"]
            )

    def test_wrong_source_shape_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "assets"
            _, config = _make_assets(root, wrong_shape=True)
            with self.assertRaisesRegex(ContractError, "tensor mismatch"):
                validate_inventory(root / "model.safetensors", config)

    def test_actual_pinned_checkpoint_inventory_when_present(self) -> None:
        root = REPOSITORY / "artifacts" / "chatbot" / "qwen3-0.6b-base"
        checkpoint = root / "model.safetensors"
        if not checkpoint.is_file():
            self.skipTest("ignored pinned checkpoint is not installed")
        config = parse_qwen_dense_config(root / "config.json")
        mappings, inventory = validate_inventory(checkpoint, config)
        self.assertEqual(len(mappings), 310)
        self.assertEqual(
            sum(entry.length for entry in inventory.values()), 1_192_099_840
        )
        self.assertEqual(
            inventory["model.layers.0.self_attn.q_proj.weight"].shape,
            (2048, 1024),
        )
        self.assertEqual(
            inventory["model.layers.0.self_attn.q_norm.weight"].shape, (128,)
        )

    def test_safetensors_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.safetensors"
            entry = '{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}'
            header = (f'{{"x":{entry},"x":{entry}}}').encode("ascii")
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
            with self.assertRaisesRegex(ContractError, "duplicate"):
                read_safetensors_inventory(path)


if __name__ == "__main__":
    unittest.main()
