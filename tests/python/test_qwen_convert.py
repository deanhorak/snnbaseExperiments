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
    validate_checkpoint_shards,
    checkpoint_identity,
    build_parser,
    _load_plan,
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


def _make_assets(root: Path, *, wrong_shape: bool = False, untied: bool = False,
                 sharded: bool = False) -> tuple[str, object]:
    root.mkdir()
    fixture = REPOSITORY / "tests" / "fixtures" / "chatbot" / "fake_qwen" / "assets"
    shutil.copyfile(fixture / "tokenizer.json", root / "tokenizer.json")
    shutil.copyfile(fixture / "tokenizer_config.json", root / "tokenizer_config.json")
    raw_config = _config()
    if untied:
        raw_config.update(tie_word_embeddings=False, hidden_size=12,
                          num_attention_heads=4, num_key_value_heads=2,
                          intermediate_size=20, num_hidden_layers=3)
    _json(root / "config.json", raw_config)
    config = parse_qwen_dense_config(root / "config.json")
    mappings = expected_tensor_mappings(config)
    if sharded:
        shard_names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        sections = [mappings[::2], mappings[1::2]]
        for name, section in zip(shard_names, sections):
            _write_safetensors(root / name, section)
        _json(root / "model.safetensors.index.json", {
            "metadata": {"total_size": sum(math.prod(item.shape) * 2 for item in mappings)},
            "weight_map": {item.source_name: name for name, section in zip(shard_names, sections) for item in section},
        })
        weight_names = [*shard_names, "model.safetensors.index.json"]
    else:
        _write_safetensors(
            root / "model.safetensors", mappings,
            wrong_shape_for=mappings[1].source_name if wrong_shape else None,
        )
        weight_names = ["model.safetensors"]
    fingerprint, tokenizer_records = tokenizer_fingerprint(root)
    records = [
        *tokenizer_records,
        file_record(root, "config.json"),
        *(file_record(root, name) for name in weight_names),
    ]
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
            "weight_files": [file_record(root, name) for name in weight_names],
        },
    )
    return fingerprint, config


class QwenConvertTests(unittest.TestCase):
    def test_generic_cli_requires_opt_in_and_imports_untied_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            fingerprint, config = _make_assets(temporary / "assets", untied=True, sharded=True)
            arguments = ["inventory", "--assets-dir", str(temporary / "assets"),
                         "--model-id", MODEL_ID, "--revision", REVISION,
                         "--expected-tokenizer-fingerprint", fingerprint,
                         "--output", str(temporary / "inventory.json")]
            with self.assertRaisesRegex(ContractError, "allow-unpinned-dense-model"):
                _load_plan(build_parser().parse_args(arguments))
            assets, loaded_config, mappings, inventory = _load_plan(
                build_parser().parse_args([*arguments, "--allow-unpinned-dense-model"]))
            self.assertEqual(loaded_config, config)
            self.assertEqual(len(mappings), 36)
            self.assertEqual(mappings[-1].destination_name, "lm_head.weight")
            self.assertEqual(mappings[-1].shape, (17, 12))
            first = convert_archive(assets, config, mappings, inventory, temporary / "converted")
            self.assertTrue(first["tied_output_projection"]["separate_lm_head"])
            self.assertFalse(first["decoder_config"]["tied_output_projection"])
            identity = checkpoint_identity(assets)
            self.assertEqual(identity["fingerprint_algorithm"], "sha256-canonical-weight-file-manifest-v1")
            self.assertEqual(len(identity["files"]), 3)
            raw = (temporary / "converted" / ARCHIVE_FILENAME).read_bytes()
            self.assertEqual(raw[56:88].hex(), identity["sha256"])
            for record in first["dense_tensors"]:
                source = inventory[record["source_name"]]
                with source.source_path.open("rb") as stream:
                    stream.seek(source.absolute_offset)
                    expected = stream.read(source.length)
                self.assertEqual(raw[record["archive_offset"]:record["archive_offset"] + record["length"]], expected)

    def test_shard_index_and_duplicate_payloads_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "assets"
            fingerprint, config = _make_assets(root, untied=True, sharded=True)
            assets = verify_asset_manifest(root, MODEL_ID, REVISION, fingerprint, require_weights=True)
            index = json.loads((root / "model.safetensors.index.json").read_text())
            name = next(iter(index["weight_map"]))
            index["weight_map"][name] = "missing.safetensors"
            _json(root / "model.safetensors.index.json", index)
            with self.assertRaisesRegex(ContractError, "wrong file"):
                validate_checkpoint_shards(assets, config)
            shard = root / "model-00001-of-00002.safetensors"
            with self.assertRaisesRegex(ContractError, "duplicate tensors"):
                validate_inventory([shard, shard], config)

    def test_unsupported_architectures_fail_before_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for changed in ({"rope_scaling": {"rope_type": "yarn"}},
                            {"use_sliding_window": True}, {"num_experts": 8}):
                _json(path, {**_config(), **changed})
                with self.subTest(changed=changed), self.assertRaises(ContractError):
                    parse_qwen_dense_config(path)

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
