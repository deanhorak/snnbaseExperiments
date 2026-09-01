from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Mapping, Sequence, cast

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools"))

from chatbot_prepare import (  # noqa: E402
    CONVERSION_MANIFEST_KIND,
    CONVERSION_MANIFEST_SCHEMA_VERSION,
    DERIVATION_KIND,
    PREPARED_CONVERSION_MANIFEST_FILENAME,
    PREPARED_LINEAGE_FILENAME,
    bounded_binary_lines,
    build_parser,
    prepare_record,
    prepare_stream,
    split_bucket,
    split_name,
    tokenize_conversation,
    write_dataset,
)
from qwen_contract import (  # noqa: E402
    ContractError,
    PHASE0_MODEL_ID,
    PHASE0_REVISION,
    PHASE0_TOKENIZER_FINGERPRINT,
    load_transformers_tokenizer,
    verify_asset_manifest,
)

TEST_TREE_ID = "00000000-0000-4000-8000-000000000002"
TEST_ASSISTANT_ID = "00000000-0000-4000-8000-000000000003"
OTHER_TREE_ID = "00000000-0000-4000-8000-000000000006"
OTHER_ASSISTANT_ID = "00000000-0000-4000-8000-000000000007"


class FakeByteTokenizer:
    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str | list[int]:
        del add_generation_prompt, enable_thinking
        text = "".join(
            f"<{item['role']}>{item['content']}</{item['role']}>\n" for item in messages
        )
        return list(text.encode("utf-8")) if tokenize else text


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _conversion_bundle(
    root: Path,
    canonical_source: Path,
    *,
    model_id: str,
    revision: str,
    fingerprint: str,
) -> tuple[Path, dict[str, object]]:
    root.mkdir()
    lineage = _canonical_json(
        {
            "tree_id": TEST_TREE_ID,
            "message_ids": [TEST_TREE_ID, TEST_ASSISTANT_ID],
        }
    )
    (root / "lineage.jsonl").write_bytes(lineage)
    source_bytes = canonical_source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    document: dict[str, object] = {
        "schema_version": CONVERSION_MANIFEST_SCHEMA_VERSION,
        "kind": CONVERSION_MANIFEST_KIND,
        "status": "completed",
        "canonical_serialization": "utf8-json-sort-keys-compact-lf-v1",
        "converter": {
            "filename": "oasst2_convert.py",
            "version": "1.0.0",
            "size_bytes": 1234,
            "sha256": "1" * 64,
        },
        "source": {
            "filename": "ready.trees.jsonl.gz",
            "source_uri": "https://example.invalid/oasst2/resolve/" + "2" * 40,
            "version": "2" * 40,
            "license": "Apache-2.0",
            "license_reviewed": True,
            "policy_reviewed": True,
            "compression": "gzip",
            "compressed": {"size_bytes": 100, "sha256": "3" * 64},
            "decompressed": {"size_bytes": 200, "sha256": "4" * 64},
        },
        "output": {
            "path": canonical_source.name,
            "size_bytes": len(source_bytes),
            "sha256": source_sha256,
            "record_count": 1,
            "message_count": 2,
            "content_bytes": 2,
            "fields": ["id", "messages"],
        },
        "lineage": {
            "path": "lineage.jsonl",
            "size_bytes": len(lineage),
            "sha256": hashlib.sha256(lineage).hexdigest(),
            "record_count": 1,
            "fields": ["tree_id", "message_ids"],
            "contains_user_ids_or_text": False,
        },
        "tokenizer": {
            "model_id": model_id,
            "revision": revision,
            "fingerprint_sha256": fingerprint,
            "template": {
                "add_generation_prompt": False,
                "enable_thinking": True,
            },
            "maximum_input_tokens": 512,
            "selected_maximum_token_count": 32,
        },
        "counts": {"selected_tree_count": 1},
        "filter_reasons": {},
        "policy": {
            "profile": "quality05",
            "conservative_zero": {"active": False},
            "quality05": {"active": True},
        },
        "split": {
            "algorithm": "sha256-seed-bucket-v1",
            "seed": 42,
            "bucket_count": 10_000,
            "test_buckets_inclusive": [0, 999],
            "validation_buckets_inclusive": [1000, 1999],
            "train_buckets_inclusive": [2000, 9999],
            "record_counts": {"train": 1, "validation": 0, "test": 0},
        },
    }
    manifest = root / "conversion-manifest.json"
    manifest.write_bytes(_canonical_json(document))
    return manifest, document


class ChatbotPrepareTests(unittest.TestCase):
    @staticmethod
    def conversation(identifier: str = "example") -> dict[str, object]:
        return {
            "id": identifier,
            "messages": [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "B"},
            ],
        }

    def test_assistant_only_mask_excludes_prompt_and_marks_response(self) -> None:
        messages = [
            {"role": "user", "content": "Ping?"},
            {"role": "assistant", "content": "Pong."},
        ]
        input_ids, mask = tokenize_conversation(
            FakeByteTokenizer(), messages, assistant_only_loss=True
        )
        self.assertIsNotNone(mask)
        mask = cast(list[int], mask)
        self.assertEqual(len(input_ids), len(mask))
        rendered = bytes(input_ids).decode("utf-8")
        target_text = "".join(
            character for character, selected in zip(rendered, mask) if selected
        )
        self.assertTrue(target_text.startswith("Pong."))
        self.assertNotIn("Ping?", target_text)

    def test_multi_turn_mask_selects_every_assistant_without_users(self) -> None:
        messages = [
            {"role": "system", "content": "Rules"},
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Alpha"},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "Beta"},
        ]
        input_ids, mask = tokenize_conversation(
            FakeByteTokenizer(), messages, assistant_only_loss=True
        )
        selected = bytes(
            token for token, include in zip(input_ids, cast(list[int], mask)) if include
        ).decode("utf-8")
        self.assertIn("Alpha", selected)
        self.assertIn("Beta", selected)
        self.assertNotIn("Rules", selected)
        self.assertNotIn("First", selected)
        self.assertNotIn("Second", selected)

    @unittest.skipUnless(
        os.environ.get("SNNBASE_QWEN_ASSETS"),
        "real tokenizer snapshot not configured",
    )
    def test_real_qwen_multi_turn_assistant_mask(self) -> None:
        assets = verify_asset_manifest(
            Path(os.environ["SNNBASE_QWEN_ASSETS"]),
            PHASE0_MODEL_ID,
            PHASE0_REVISION,
            PHASE0_TOKENIZER_FINGERPRINT,
        )
        tokenizer = load_transformers_tokenizer(assets)
        messages = [
            {"role": "system", "content": "Follow the rules."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Again"},
            {"role": "assistant", "content": "Yes"},
        ]
        input_ids, mask = tokenize_conversation(
            tokenizer, messages, assistant_only_loss=True
        )
        selected_ids = [
            token for token, include in zip(input_ids, cast(list[int], mask)) if include
        ]
        selected_text = tokenizer.decode(selected_ids, skip_special_tokens=False)
        self.assertIn("Hello", selected_text)
        self.assertIn("Yes", selected_text)
        self.assertNotIn("Follow the rules.", selected_text)
        self.assertNotIn("Hi", selected_text)
        self.assertNotIn("Again", selected_text)
        self.assertEqual(selected_ids.count(151645), 2)

    def test_record_has_exact_immutable_shard_fields(self) -> None:
        record = prepare_record(
            FakeByteTokenizer(),
            {
                "id": "example",
                "messages": [
                    {"role": "user", "content": "A"},
                    {"role": "assistant", "content": "B"},
                ],
            },
            max_input_tokens=256,
        )
        self.assertEqual(
            set(record), {"schema_version", "id", "input_ids", "loss_mask"}
        )
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(len(record["input_ids"]), len(record["loss_mask"]))

    def test_split_algorithm_is_stable_and_uses_declared_ranges(self) -> None:
        vectors = REPOSITORY / "tests" / "fixtures" / "chatbot" / "split_vectors_v1.tsv"
        with vectors.open(encoding="utf-8") as stream:
            rows = [line.rstrip("\n").split("\t") for line in stream]
        self.assertEqual(rows.pop(0), ["seed", "conversation_id", "bucket"])
        for raw_seed, identifier, raw_bucket in rows:
            with self.subTest(seed=raw_seed, identifier=identifier):
                self.assertEqual(
                    split_bucket(identifier, int(raw_seed)), int(raw_bucket)
                )
        self.assertEqual(split_bucket("conversation-17"), 7457)
        self.assertEqual(split_name("conversation-1"), "test")
        self.assertEqual(split_name("conversation-2"), "train")

    def test_duplicate_ids_and_oversize_records_fail(self) -> None:
        line = (
            '{"id":"same","messages":[{"role":"user","content":"A"},'
            '{"role":"assistant","content":"B"}]}\n'
        )
        with self.assertRaisesRegex(ContractError, "duplicate"):
            prepare_stream(
                FakeByteTokenizer(),
                [line, line],
                max_input_tokens=256,
            )
        with self.assertRaisesRegex(ContractError, "fail-closed limit"):
            prepare_record(
                FakeByteTokenizer(),
                {
                    "id": "long",
                    "messages": [
                        {"role": "user", "content": "A"},
                        {"role": "assistant", "content": "B"},
                    ],
                },
                max_input_tokens=3,
            )

    def test_source_schema_and_role_order_are_strict(self) -> None:
        base = {
            "id": "bad",
            "messages": [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "B"},
            ],
        }
        with self.assertRaisesRegex(ContractError, "exactly"):
            prepare_record(
                FakeByteTokenizer(), {**base, "split": "train"}, max_input_tokens=256
            )
        with self.assertRaisesRegex(ContractError, "strictly alternate"):
            prepare_record(
                FakeByteTokenizer(),
                {
                    "id": "bad-order",
                    "messages": [
                        {"role": "user", "content": "A"},
                        {"role": "user", "content": "B"},
                        {"role": "assistant", "content": "C"},
                    ],
                },
                max_input_tokens=256,
            )

    def test_nested_message_fields_are_exact(self) -> None:
        value = self.conversation()
        messages = cast(list[dict[str, str]], value["messages"])
        messages[0] = {"role": "user", "content": "A", "name": "ignored"}
        with self.assertRaisesRegex(ContractError, "exactly role and content"):
            prepare_record(FakeByteTokenizer(), value, max_input_tokens=256)

    def test_source_json_rejects_duplicate_fields_at_every_object_level(self) -> None:
        duplicate_top_level = (
            '{"id":"first","id":"second","messages":['
            '{"role":"user","content":"A"},'
            '{"role":"assistant","content":"B"}]}\n'
        )
        duplicate_nested = (
            '{"id":"nested","messages":['
            '{"role":"user","role":"assistant","content":"A"},'
            '{"role":"assistant","content":"B"}]}\n'
        )
        for line in (duplicate_top_level, duplicate_nested):
            with self.subTest(line=line):
                with self.assertRaisesRegex(ContractError, "duplicate JSON"):
                    prepare_stream(FakeByteTokenizer(), [line], max_input_tokens=256)

    def test_ids_and_content_use_utf8_byte_limits_and_reject_nul(self) -> None:
        allowed = self.conversation("é" * 128)
        record = prepare_record(FakeByteTokenizer(), allowed, max_input_tokens=4096)
        self.assertEqual(record["id"], "é" * 128)

        with self.assertRaisesRegex(ContractError, "maximum_id_bytes"):
            prepare_record(
                FakeByteTokenizer(),
                self.conversation("é" * 129),
                max_input_tokens=4096,
            )
        with self.assertRaisesRegex(ContractError, "NUL"):
            prepare_record(
                FakeByteTokenizer(),
                self.conversation("bad\0id"),
                max_input_tokens=256,
            )
        with self.assertRaisesRegex(ContractError, "control characters"):
            prepare_record(
                FakeByteTokenizer(),
                self.conversation("bad\tid"),
                max_input_tokens=256,
            )

        content_limited = self.conversation()
        messages = cast(list[dict[str, str]], content_limited["messages"])
        messages[0] = {"role": "user", "content": "éé"}
        with self.assertRaisesRegex(ContractError, "content exceeds"):
            prepare_record(
                FakeByteTokenizer(),
                content_limited,
                max_input_tokens=256,
                maximum_content_bytes_per_message=3,
            )
        messages[0] = {"role": "user", "content": "bad\0content"}
        with self.assertRaisesRegex(ContractError, "NUL"):
            prepare_record(FakeByteTokenizer(), content_limited, max_input_tokens=256)

    def test_stream_resource_limits_and_blank_lines_fail_closed(self) -> None:
        first = (
            '{"id":"first","messages":[{"role":"user","content":"A"},'
            '{"role":"assistant","content":"B"}]}\n'
        )
        second = first.replace('"first"', '"second"')
        payload_bytes = len(first.removesuffix("\n").encode("utf-8"))
        prepared = prepare_stream(
            FakeByteTokenizer(),
            [first],
            max_input_tokens=256,
            maximum_line_bytes=payload_bytes,
        )
        self.assertEqual(sum(map(len, prepared.values())), 1)
        with self.assertRaisesRegex(ContractError, "maximum_line_bytes"):
            prepare_stream(
                FakeByteTokenizer(),
                [first],
                max_input_tokens=256,
                maximum_line_bytes=payload_bytes - 1,
            )
        with self.assertRaisesRegex(ContractError, "maximum_records"):
            prepare_stream(
                FakeByteTokenizer(),
                [first, second],
                max_input_tokens=256,
                maximum_records=1,
            )
        with self.assertRaisesRegex(ContractError, "blank input line"):
            prepare_stream(FakeByteTokenizer(), ["\n"], max_input_tokens=256)
        with self.assertRaisesRegex(ContractError, "message count"):
            prepare_stream(
                FakeByteTokenizer(),
                [first],
                max_input_tokens=256,
                maximum_messages_per_conversation=1,
            )

    def test_binary_reader_bounds_allocation_and_rejects_invalid_utf8(self) -> None:
        with self.assertRaisesRegex(ContractError, "maximum_line_bytes"):
            list(
                bounded_binary_lines(
                    io.BytesIO(b"x" * 12 + b"\n"), maximum_line_bytes=10
                )
            )
        with self.assertRaisesRegex(ContractError, "invalid UTF-8"):
            prepare_stream(FakeByteTokenizer(), [b"{\xff}\n"], max_input_tokens=256)

    def test_dataset_writer_creates_hashed_immutable_three_shard_contract(self) -> None:
        records = {
            "train": [
                {
                    "schema_version": 1,
                    "id": TEST_TREE_ID,
                    "input_ids": [1, 2],
                    "loss_mask": [0, 1],
                }
            ],
            "validation": [],
            "test": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            source.write_text('{"id":"source","messages":[]}\n', encoding="utf-8")
            output = root / "prepared"
            manifest = write_dataset(
                output,
                records,
                source_path=source,
                source_size_bytes=source.stat().st_size,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_uri="https://example.invalid/datasets/conversations",
                source_version="sha256:0123456789abcdef",
                source_license="Apache-2.0",
                model_id="tests/FakeQwen",
                revision="0123456789abcdef0123456789abcdef01234567",
                tokenizer_fingerprint="e2f448429ba76678d32ed7cf99e8d9adb031944a11a170771365a0f76f29a742",
            )
            self.assertEqual(set(manifest["shards"]), {"train", "validation", "test"})
            self.assertEqual(manifest["shards"]["train"]["record_count"], 1)
            self.assertEqual(len(manifest["shards"]["train"]["sha256"]), 64)
            self.assertEqual(
                manifest["split"]["hash_payload"],
                "seed u64 big-endian || id UTF-8",
            )
            self.assertEqual(
                set(manifest["source"]),
                {
                    "filename",
                    "size_bytes",
                    "sha256",
                    "record_schema",
                    "source_uri",
                    "version",
                    "license",
                },
            )
            self.assertEqual(
                manifest["source"]["source_uri"],
                "https://example.invalid/datasets/conversations",
            )
            self.assertEqual(manifest["source"]["version"], "sha256:0123456789abcdef")
            self.assertEqual(manifest["source"]["license"], "Apache-2.0")
            self.assertTrue((output / "dataset-manifest.json").is_file())
            with self.assertRaisesRegex(ContractError, "immutable"):
                write_dataset(
                    output,
                    records,
                    source_path=source,
                    source_size_bytes=source.stat().st_size,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    source_uri="https://example.invalid/datasets/conversations",
                    source_version="sha256:0123456789abcdef",
                    source_license="Apache-2.0",
                    model_id="tests/FakeQwen",
                    revision="0123456789abcdef0123456789abcdef01234567",
                    tokenizer_fingerprint="e2f448429ba76678d32ed7cf99e8d9adb031944a11a170771365a0f76f29a742",
                )

    def test_dataset_writer_copies_and_hash_binds_conversion_derivation(self) -> None:
        records = {
            "train": [
                {
                    "schema_version": 1,
                    "id": TEST_TREE_ID,
                    "input_ids": [1, 2],
                    "loss_mask": [0, 1],
                }
            ],
            "validation": [],
            "test": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "conversations.jsonl"
            source.write_text(
                f'{{"id":"{TEST_TREE_ID}","messages":['
                '{"content":"A","role":"user"},'
                '{"content":"B","role":"assistant"}]}\n',
                encoding="utf-8",
            )
            source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
            conversion_manifest, conversion_document = _conversion_bundle(
                root / "conversion",
                source,
                model_id="tests/FakeQwen",
                revision="5" * 40,
                fingerprint="6" * 64,
            )
            output = root / "prepared"
            manifest = write_dataset(
                output,
                records,
                source_path=source,
                source_size_bytes=source.stat().st_size,
                source_sha256=source_sha256,
                source_uri=f"urn:sha256:{source_sha256}",
                source_version=f"sha256:{source_sha256}",
                source_license="Apache-2.0",
                model_id="tests/FakeQwen",
                revision="5" * 40,
                tokenizer_fingerprint="6" * 64,
                conversion_manifest_path=conversion_manifest,
            )

            derivation = manifest["derivation"]
            self.assertEqual(derivation["kind"], DERIVATION_KIND)
            self.assertEqual(derivation["profile"], "quality05")
            self.assertEqual(derivation["raw_source"], conversion_document["source"])
            self.assertEqual(derivation["converter"], conversion_document["converter"])
            copied_manifest = output / PREPARED_CONVERSION_MANIFEST_FILENAME
            copied_lineage = output / PREPARED_LINEAGE_FILENAME
            self.assertEqual(
                copied_manifest.read_bytes(), conversion_manifest.read_bytes()
            )
            self.assertEqual(
                hashlib.sha256(copied_manifest.read_bytes()).hexdigest(),
                derivation["conversion_manifest"]["sha256"],
            )
            self.assertEqual(
                hashlib.sha256(copied_lineage.read_bytes()).hexdigest(),
                derivation["lineage"]["sha256"],
            )
            self.assertEqual(copied_manifest.stat().st_mode & 0o777, 0o444)
            self.assertEqual(copied_lineage.stat().st_mode & 0o777, 0o444)
            self.assertIn("conversion_manifest", derivation)
            self.assertIn("lineage", derivation)
            action = next(
                item
                for item in build_parser()._actions
                if item.dest == "conversion_manifest"
            )
            self.assertFalse(action.required)

    def test_conversion_derivation_rejects_mismatch_swap_and_non_strict_json(
        self,
    ) -> None:
        records = {
            "train": [
                {
                    "schema_version": 1,
                    "id": TEST_TREE_ID,
                    "input_ids": [1, 2],
                    "loss_mask": [0, 1],
                }
            ],
            "validation": [],
            "test": [],
        }
        mutations = (
            "output-hash",
            "tokenizer",
            "extra-field",
            "profile",
            "raw-version",
            "http-no-authority",
            "lineage-swap",
            "lineage-extra-field",
            "lineage-non-uuid",
            "lineage-wrong-tree",
            "manifest-rename",
            "lineage-rename",
            "duplicate-key",
            "nan",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "conversations.jsonl"
            source.write_text(
                f'{{"id":"{TEST_TREE_ID}","messages":['
                '{"content":"A","role":"user"},'
                '{"content":"B","role":"assistant"}]}\n',
                encoding="utf-8",
            )
            source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
            for index, mutation in enumerate(mutations):
                with self.subTest(mutation=mutation):
                    conversion_manifest, document = _conversion_bundle(
                        root / f"conversion-{index}",
                        source,
                        model_id="tests/FakeQwen",
                        revision="5" * 40,
                        fingerprint="6" * 64,
                    )
                    if mutation == "output-hash":
                        document["output"]["sha256"] = "7" * 64
                    elif mutation == "tokenizer":
                        document["tokenizer"]["fingerprint_sha256"] = "7" * 64
                    elif mutation == "extra-field":
                        document["unexpected"] = True
                    elif mutation == "profile":
                        document["policy"]["profile"] = "unreviewed-profile"
                    elif mutation == "raw-version":
                        document["source"]["version"] = "latest"
                    elif mutation == "http-no-authority":
                        document["source"]["source_uri"] = "https:relative"
                    elif mutation == "lineage-swap":
                        (conversion_manifest.parent / "lineage.jsonl").write_text(
                            '{"message_ids":["other"],"tree_id":"other"}\n',
                            encoding="utf-8",
                        )
                    elif mutation in (
                        "lineage-extra-field",
                        "lineage-non-uuid",
                        "lineage-wrong-tree",
                    ):
                        if mutation == "lineage-extra-field":
                            lineage_record = {
                                "tree_id": TEST_TREE_ID,
                                "message_ids": [TEST_TREE_ID, TEST_ASSISTANT_ID],
                                "text": "must never be retained",
                            }
                        elif mutation == "lineage-non-uuid":
                            lineage_record = {
                                "tree_id": "not-a-uuid",
                                "message_ids": ["not-a-uuid", TEST_ASSISTANT_ID],
                            }
                        else:
                            lineage_record = {
                                "tree_id": OTHER_TREE_ID,
                                "message_ids": [OTHER_TREE_ID, OTHER_ASSISTANT_ID],
                            }
                        lineage_bytes = _canonical_json(lineage_record)
                        (conversion_manifest.parent / "lineage.jsonl").write_bytes(
                            lineage_bytes
                        )
                        document["lineage"]["size_bytes"] = len(lineage_bytes)
                        document["lineage"]["sha256"] = hashlib.sha256(
                            lineage_bytes
                        ).hexdigest()
                    elif mutation == "manifest-rename":
                        renamed_manifest = conversion_manifest.with_name("renamed.json")
                        conversion_manifest.rename(renamed_manifest)
                        conversion_manifest = renamed_manifest
                    elif mutation == "lineage-rename":
                        lineage = conversion_manifest.parent / "lineage.jsonl"
                        lineage.rename(
                            conversion_manifest.parent / "renamed-lineage.jsonl"
                        )
                        document["lineage"]["path"] = "renamed-lineage.jsonl"
                    elif mutation == "duplicate-key":
                        encoded = _canonical_json(document)
                        conversion_manifest.write_bytes(
                            encoded.replace(
                                b'{"canonical_serialization":',
                                b'{"kind":"duplicate","canonical_serialization":',
                                1,
                            )
                        )
                    elif mutation == "nan":
                        document["counts"] = {"bad": float("nan")}
                        conversion_manifest.write_text(
                            json.dumps(
                                document,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=True,
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                    if mutation not in (
                        "lineage-swap",
                        "manifest-rename",
                        "duplicate-key",
                        "nan",
                    ):
                        conversion_manifest.write_bytes(_canonical_json(document))
                    with self.assertRaises(ContractError):
                        write_dataset(
                            root / f"prepared-{index}",
                            records,
                            source_path=source,
                            source_size_bytes=source.stat().st_size,
                            source_sha256=source_sha256,
                            source_uri=f"urn:sha256:{source_sha256}",
                            source_version=f"sha256:{source_sha256}",
                            source_license="Apache-2.0",
                            model_id="tests/FakeQwen",
                            revision="5" * 40,
                            tokenizer_fingerprint="6" * 64,
                            conversion_manifest_path=conversion_manifest,
                        )
                    self.assertFalse((root / f"prepared-{index}").exists())

    def test_derived_canonical_source_provenance_cannot_be_overridden(self) -> None:
        records = {
            "train": [
                {
                    "schema_version": 1,
                    "id": TEST_TREE_ID,
                    "input_ids": [1, 2],
                    "loss_mask": [0, 1],
                }
            ],
            "validation": [],
            "test": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "conversations.jsonl"
            source.write_text("canonical\n", encoding="utf-8")
            source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
            conversion_manifest, _ = _conversion_bundle(
                root / "conversion",
                source,
                model_id="tests/FakeQwen",
                revision="5" * 40,
                fingerprint="6" * 64,
            )
            common = {
                "records": records,
                "source_path": source,
                "source_size_bytes": source.stat().st_size,
                "source_sha256": source_sha256,
                "source_uri": f"urn:sha256:{source_sha256}",
                "source_version": f"sha256:{source_sha256}",
                "source_license": "Apache-2.0",
                "model_id": "tests/FakeQwen",
                "revision": "5" * 40,
                "tokenizer_fingerprint": "6" * 64,
                "conversion_manifest_path": conversion_manifest,
            }
            for index, override in enumerate(
                (
                    {"source_uri": "https://example.invalid/canonical"},
                    {"source_version": "sha256:wrong"},
                    {"source_license": "CC0-1.0"},
                )
            ):
                with self.subTest(override=override):
                    with self.assertRaises(ContractError):
                        write_dataset(
                            root / f"bad-provenance-{index}",
                            **{**common, **override},
                        )

    def test_dataset_writer_rejects_source_changed_after_consumption(self) -> None:
        records: dict[str, list[dict[str, object]]] = {
            "train": [
                {
                    "schema_version": 1,
                    "id": "t",
                    "input_ids": [1, 2],
                    "loss_mask": [0, 1],
                }
            ],
            "validation": [],
            "test": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            source.write_text("original source\n", encoding="utf-8")
            consumed = source.read_bytes()
            source.write_text("mutated source\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "changed after it was read"):
                write_dataset(
                    root / "prepared",
                    records,
                    source_path=source,
                    source_size_bytes=len(consumed),
                    source_sha256=hashlib.sha256(consumed).hexdigest(),
                    source_uri="https://example.invalid/datasets/conversations",
                    source_version="sha256:0123456789abcdef",
                    source_license="Apache-2.0",
                    model_id="tests/FakeQwen",
                    revision="0123456789abcdef0123456789abcdef01234567",
                    tokenizer_fingerprint=(
                        "e2f448429ba76678d32ed7cf99e8d9adb031944a11a170771365a0f76f29a742"
                    ),
                )

    def test_source_provenance_is_required_and_validated(self) -> None:
        required = {
            action.dest
            for action in build_parser()._actions
            if getattr(action, "required", False)
        }
        self.assertTrue({"source_uri", "source_version", "source_license"} <= required)
        records: dict[str, list[dict[str, object]]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            source.write_text("{}\n", encoding="utf-8")
            common = {
                "records": records,
                "source_path": source,
                "source_size_bytes": source.stat().st_size,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "source_uri": "https://example.invalid/dataset",
                "source_version": "2026-08-31",
                "source_license": "Apache-2.0",
                "model_id": "tests/FakeQwen",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "tokenizer_fingerprint": "a" * 64,
            }
            with self.assertRaisesRegex(ContractError, "absolute URI"):
                write_dataset(
                    root / "bad-uri", **{**common, "source_uri": "relative/path"}
                )
            with self.assertRaisesRegex(ContractError, "immutable snapshot"):
                write_dataset(
                    root / "moving-version",
                    **{**common, "source_version": "latest"},
                )
            with self.assertRaisesRegex(ContractError, "immutable snapshot"):
                write_dataset(
                    root / "moving-branch",
                    **{**common, "source_version": "refs/heads/release"},
                )
            with self.assertRaisesRegex(ContractError, "control characters"):
                write_dataset(
                    root / "bad-license",
                    **{**common, "source_license": "Apache\0-2.0"},
                )


if __name__ == "__main__":
    unittest.main()
