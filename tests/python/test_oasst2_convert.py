#!/usr/bin/env python3
"""Offline contract tests for the pinned OASST2 conversion boundary."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

from oasst2_convert import (  # noqa: E402
    CANONICAL_SERIALIZATION,
    CONVERSATIONS_FILENAME,
    LINEAGE_FILENAME,
    MANIFEST_FILENAME,
    PROFILE_CONSERVATIVE_ZERO,
    PROFILE_QUALITY05,
    ConversionLimits,
    ConversionPolicy,
    Oasst2ConversionError,
    TokenizerIdentity,
    build_parser,
    convert_oasst2,
)

IDENTITY = TokenizerIdentity(
    model_id="Qwen/Test",
    revision="a" * 40,
    fingerprint_sha256="b" * 64,
)
SOURCE_REVISION = "c" * 40


def _uuid(index: int) -> str:
    return f"00000000-0000-0000-0000-{index:012d}"


def _label(value: float) -> dict[str, Any]:
    return {"value": value, "count": 1}


def _zero_labels(*, assistant: bool) -> dict[str, Any]:
    names = {
        "spam",
        "lang_mismatch",
        "pii",
        "not_appropriate",
        "hate_speech",
        "sexual_content",
        "toxicity",
        "violence",
    }
    if assistant:
        names.add("fails_task")
    return {name: _label(0.0) for name in sorted(names)}


def _message(
    index: int,
    *,
    parent: str | None,
    role: str,
    text: str,
    lang: str = "es",
    rank: int | None = None,
    labels: Mapping[str, Any] | None = None,
    replies: Sequence[dict[str, Any]] = (),
    synthetic: bool = False,
) -> dict[str, Any]:
    return {
        "message_id": _uuid(index),
        "parent_id": parent,
        "user_id": _uuid(900_000 + index),
        "created_date": "2023-10-01T00:00:00+00:00",
        "text": text,
        "role": role,
        "lang": lang,
        "review_count": 1,
        "review_result": True,
        "deleted": False,
        "rank": rank,
        "synthetic": synthetic,
        "model_name": None,
        "emojis": None,
        "replies": list(replies),
        "labels": dict(labels or {}),
        "events": None,
        "detoxify": None,
        "message_tree_id": None,
        "tree_state": None,
    }


def _tree(index: int, prompt: dict[str, Any]) -> dict[str, Any]:
    tree_id = _uuid(index)
    assert prompt["message_id"] == tree_id
    return {
        "message_tree_id": tree_id,
        "origin": None,
        "prompt": prompt,
        "tree_state": "ready_for_export",
    }


def _jsonl_bytes(trees: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                tree,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for tree in trees
    )


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool, **kwargs: Any
    ) -> list[int]:
        self.calls.append(([dict(message) for message in messages], dict(kwargs)))
        if not tokenize:
            raise AssertionError("converter must request token IDs")
        rendered = "|".join(
            f"{message['role']}:{message['content']}" for message in messages
        )
        return list(rendered.encode("utf-8"))


class MutatingTokenizer(FakeTokenizer):
    def __init__(self, source: Path, replacement: bytes) -> None:
        super().__init__()
        self.source = source
        self.replacement = replacement

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool, **kwargs: Any
    ) -> list[int]:
        if not self.calls:
            self.source.unlink()
            self.source.write_bytes(self.replacement)
        return super().apply_chat_template(messages, tokenize=tokenize, **kwargs)


class Oasst2ConvertTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        for path in self.root.iterdir():
            if path.is_dir():
                os.chmod(path, 0o755)
        self.temporary.cleanup()

    def _source(self, payload: bytes, name: str = "ready.trees.jsonl.gz") -> Path:
        source = self.root / name
        source.write_bytes(gzip.compress(payload, mtime=0))
        return source

    def _convert(
        self,
        trees: Sequence[Mapping[str, Any]],
        *,
        profile: str = PROFILE_QUALITY05,
        tokenizer: Any | None = None,
        maximum_input_tokens: int = 512,
        maximum_conversation_content_bytes: int = 32 * 1024,
        output_name: str = "converted",
        limits: ConversionLimits = ConversionLimits(),
    ) -> tuple[dict[str, Any], Path, Path, FakeTokenizer]:
        payload = _jsonl_bytes(trees)
        source = self._source(payload)
        output = self.root / output_name
        selected_tokenizer = tokenizer or FakeTokenizer()
        manifest = convert_oasst2(
            source_path=source,
            output_directory=output,
            expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            expected_source_bytes=source.stat().st_size,
            source_uri="https://example.invalid/oasst2/ready.trees.jsonl.gz",
            source_version=SOURCE_REVISION,
            source_license="caller-declared-test-provenance",
            license_reviewed=True,
            policy_reviewed=True,
            tokenizer=selected_tokenizer,
            tokenizer_identity=IDENTITY,
            policy=ConversionPolicy(
                profile=profile,
                maximum_conversation_content_bytes=(maximum_conversation_content_bytes),
                maximum_input_tokens=maximum_input_tokens,
            ),
            limits=limits,
        )
        return manifest, output, source, selected_tokenizer

    def test_quality05_publishes_canonical_bundle_and_selects_longest_path(
        self,
    ) -> None:
        tree_id = 1
        root_id = _uuid(tree_id)
        shallow_bad = _message(
            2,
            parent=root_id,
            role="assistant",
            text="bad-ranked-answer",
            rank=0,
            labels={"quality": _label(0.49)},
        )
        final = _message(
            5,
            parent=_uuid(4),
            role="assistant",
            text="respuesta-final",
            rank=None,
        )
        follow_up = _message(
            4,
            parent=_uuid(3),
            role="prompter",
            text="seguimiento",
            replies=[final],
        )
        first = _message(
            3,
            parent=root_id,
            role="assistant",
            text="primera-respuesta",
            rank=1,
            replies=[follow_up],
        )
        prompt = _message(
            tree_id,
            parent=None,
            role="prompter",
            text="pregunta",
            replies=[shallow_bad, first],
        )

        manifest, output, source, tokenizer = self._convert([_tree(tree_id, prompt)])

        self.assertEqual(
            sorted(path.name for path in output.iterdir()),
            sorted([CONVERSATIONS_FILENAME, LINEAGE_FILENAME, MANIFEST_FILENAME]),
        )
        conversation_line = (output / CONVERSATIONS_FILENAME).read_bytes()
        conversation = json.loads(conversation_line)
        self.assertEqual(
            [message["content"] for message in conversation["messages"]],
            ["pregunta", "primera-respuesta", "seguimiento", "respuesta-final"],
        )
        self.assertEqual(set(conversation), {"id", "messages"})
        self.assertEqual(
            conversation_line,
            (
                json.dumps(
                    conversation,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8"),
        )
        lineage = json.loads((output / LINEAGE_FILENAME).read_text("utf-8"))
        self.assertEqual(set(lineage), {"tree_id", "message_ids"})
        self.assertEqual(lineage["message_ids"], [_uuid(i) for i in (1, 3, 4, 5)])
        self.assertNotIn("text", lineage)
        self.assertNotIn("user_id", lineage)

        manifest_bytes = (output / MANIFEST_FILENAME).read_bytes()
        self.assertEqual(
            manifest_bytes,
            (
                json.dumps(
                    manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8"),
        )
        self.assertEqual(manifest["canonical_serialization"], CANONICAL_SERIALIZATION)
        self.assertEqual(manifest["source"]["filename"], source.name)
        self.assertEqual(
            manifest["source"]["decompressed"],
            {
                "sha256": hashlib.sha256(
                    _jsonl_bytes([_tree(tree_id, prompt)])
                ).hexdigest(),
                "size_bytes": len(_jsonl_bytes([_tree(tree_id, prompt)])),
            },
        )
        self.assertNotIn(str(self.root), manifest_bytes.decode("utf-8"))
        self.assertEqual(manifest["policy"]["profile"], PROFILE_QUALITY05)
        self.assertFalse(manifest["policy"]["conservative_zero"]["active"])
        self.assertTrue(manifest["policy"]["quality05"]["active"])
        self.assertEqual(manifest["policy"]["quality05"]["threshold"], 0.5)
        self.assertNotIn("pii_maximum_value", manifest["policy"])
        self.assertEqual(manifest["tokenizer"]["model_id"], IDENTITY.model_id)
        self.assertEqual(
            manifest["tokenizer"]["template"],
            {
                "add_generation_prompt": False,
                "enable_thinking": True,
            },
        )
        self.assertEqual(sum(manifest["split"]["record_counts"].values()), 1)
        self.assertTrue(tokenizer.calls)
        self.assertTrue(
            all(
                kwargs == {"add_generation_prompt": False, "enable_thinking": True}
                for _, kwargs in tokenizer.calls
            )
        )
        with self.assertRaisesRegex(Oasst2ConversionError, "already exists"):
            self._convert([_tree(tree_id, prompt)])

    def test_conservative_zero_filters_controls_and_missing_labels_per_path(
        self,
    ) -> None:
        root_id = _uuid(10)
        controlled = _message(
            11,
            parent=root_id,
            role="assistant",
            text="bad\u0018control",
            rank=0,
            labels=_zero_labels(assistant=True),
        )
        missing_labels = _message(
            12,
            parent=root_id,
            role="assistant",
            text="missing",
            rank=1,
        )
        selected = _message(
            13,
            parent=root_id,
            role="assistant",
            text="válida",
            rank=2,
            labels=_zero_labels(assistant=True),
        )
        prompt = _message(
            10,
            parent=None,
            role="prompter",
            text="consulta",
            labels=_zero_labels(assistant=False),
            replies=[controlled, missing_labels, selected],
        )

        manifest, output, _, _ = self._convert(
            [_tree(10, prompt)], profile=PROFILE_CONSERVATIVE_ZERO
        )

        conversation = json.loads((output / CONVERSATIONS_FILENAME).read_text("utf-8"))
        self.assertEqual(conversation["messages"][-1]["content"], "válida")
        reasons = manifest["filter_reasons"]["path_primary_reason_counts"]
        self.assertEqual(reasons["disallowed_control"], 1)
        self.assertEqual(reasons["missing_label_pii"], 1)
        self.assertTrue(manifest["policy"]["conservative_zero"]["active"])
        self.assertFalse(manifest["policy"]["quality05"]["active"])

    def test_token_and_content_limits_filter_without_truncation(self) -> None:
        root_id = _uuid(20)
        token_long = _message(
            21,
            parent=root_id,
            role="assistant",
            text="x" * 100,
            rank=0,
        )
        content_long = _message(
            22,
            parent=root_id,
            role="assistant",
            text="content-is-too-long",
            rank=1,
        )
        selected = _message(
            23,
            parent=root_id,
            role="assistant",
            text="ok",
            rank=2,
        )
        prompt = _message(
            20,
            parent=None,
            role="prompter",
            text="q",
            replies=[token_long, content_long, selected],
        )

        manifest, output, _, _ = self._convert(
            [_tree(20, prompt)],
            maximum_input_tokens=30,
            maximum_conversation_content_bytes=50,
        )

        conversation = json.loads((output / CONVERSATIONS_FILENAME).read_text("utf-8"))
        self.assertEqual(conversation["messages"][-1]["content"], "ok")
        reasons = manifest["filter_reasons"]["path_primary_reason_counts"]
        self.assertEqual(reasons["conversation_content_bytes_exceeded"], 1)
        self.assertEqual(reasons["maximum_input_tokens_exceeded"], 1)

    def test_normalized_root_prompt_dedup_retains_first_source_tree(self) -> None:
        first_prompt = _message(
            70,
            parent=None,
            role="prompter",
            text="  ＨELLO\tWorld ",
            replies=[
                _message(71, parent=_uuid(70), role="assistant", text="first", rank=0)
            ],
        )
        second_prompt = _message(
            72,
            parent=None,
            role="prompter",
            text="hello world",
            replies=[
                _message(73, parent=_uuid(72), role="assistant", text="second", rank=0)
            ],
        )

        manifest, output, _, _ = self._convert(
            [_tree(70, first_prompt), _tree(72, second_prompt)]
        )

        records = [
            json.loads(line)
            for line in (output / CONVERSATIONS_FILENAME)
            .read_text("utf-8")
            .splitlines()
        ]
        self.assertEqual([record["id"] for record in records], [_uuid(70)])
        self.assertEqual(
            manifest["counts"]["duplicate_normalized_root_prompt_count"], 1
        )
        self.assertEqual(
            manifest["filter_reasons"]["filtered_tree_dominant_reason_counts"][
                "duplicate_normalized_root_prompt"
            ],
            1,
        )
        dedup = manifest["policy"]["cross_tree_root_prompt_deduplication"]
        self.assertIn("NFKC", dedup["normalization"])
        self.assertIn("may remain", dedup["near_duplicate_detection"])

    def test_duplicate_json_nan_duplicate_ids_and_gzip_limit_fail_closed(self) -> None:
        invalid_payloads = {
            "duplicate-key": b'{"x":1,"x":2}\n',
            "nan": b'{"x":NaN}\n',
            "invalid-utf8": b'{"x":"\xff"}\n',
        }
        for index, (name, payload) in enumerate(invalid_payloads.items()):
            with self.subTest(name=name):
                source = self._source(payload, f"{name}.gz")
                output = self.root / f"out-{index}"
                with self.assertRaises(Oasst2ConversionError):
                    convert_oasst2(
                        source_path=source,
                        output_directory=output,
                        expected_source_sha256=hashlib.sha256(
                            source.read_bytes()
                        ).hexdigest(),
                        expected_source_bytes=source.stat().st_size,
                        source_uri="https://example.invalid/source.gz",
                        source_version=SOURCE_REVISION,
                        source_license="declared",
                        license_reviewed=True,
                        policy_reviewed=True,
                        tokenizer=FakeTokenizer(),
                        tokenizer_identity=IDENTITY,
                        policy=ConversionPolicy(profile=PROFILE_QUALITY05),
                    )
                self.assertFalse(output.exists())

        prompt = _message(
            30,
            parent=None,
            role="prompter",
            text="q",
            replies=[
                _message(31, parent=_uuid(30), role="assistant", text="a", rank=0)
            ],
        )
        payload = _jsonl_bytes([_tree(30, prompt)])
        source = self._source(payload, "bomb-limit.gz")
        output = self.root / "bomb-output"
        with self.assertRaisesRegex(Oasst2ConversionError, "uncompressed"):
            convert_oasst2(
                source_path=source,
                output_directory=output,
                expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                expected_source_bytes=source.stat().st_size,
                source_uri="https://example.invalid/source.gz",
                source_version=SOURCE_REVISION,
                source_license="declared",
                license_reviewed=True,
                policy_reviewed=True,
                tokenizer=FakeTokenizer(),
                tokenizer_identity=IDENTITY,
                policy=ConversionPolicy(profile=PROFILE_QUALITY05),
                limits=ConversionLimits(maximum_uncompressed_bytes=16),
            )
        self.assertFalse(output.exists())

        duplicate_prompt = _message(
            60,
            parent=None,
            role="prompter",
            text="duplicate ids",
            replies=[
                _message(61, parent=_uuid(60), role="assistant", text="one", rank=0),
                _message(61, parent=_uuid(60), role="assistant", text="two", rank=1),
            ],
        )
        with self.assertRaisesRegex(Oasst2ConversionError, "duplicate message_id"):
            self._convert([_tree(60, duplicate_prompt)], output_name="duplicate-id")
        self.assertFalse((self.root / "duplicate-id").exists())

    def test_empty_selection_and_source_path_replacement_do_not_publish(self) -> None:
        root_id = _uuid(40)
        prompt = _message(
            40,
            parent=None,
            role="prompter",
            text="q",
            replies=[
                _message(
                    41,
                    parent=root_id,
                    role="assistant",
                    text="unsafe",
                    rank=0,
                    labels={"pii": _label(0.5)},
                )
            ],
        )
        with self.assertRaisesRegex(Oasst2ConversionError, "selected no"):
            self._convert([_tree(40, prompt)], output_name="empty")
        self.assertFalse((self.root / "empty").exists())

        valid_prompt = _message(
            50,
            parent=None,
            role="prompter",
            text="q",
            replies=[
                _message(51, parent=_uuid(50), role="assistant", text="ok", rank=0)
            ],
        )
        payload = _jsonl_bytes([_tree(50, valid_prompt)])
        source = self._source(payload, "mutable.gz")
        tokenizer = MutatingTokenizer(source, gzip.compress(payload, mtime=1))
        output = self.root / "mutated-output"
        with self.assertRaisesRegex(Oasst2ConversionError, "source changed"):
            convert_oasst2(
                source_path=source,
                output_directory=output,
                expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                expected_source_bytes=source.stat().st_size,
                source_uri="https://example.invalid/source.gz",
                source_version=SOURCE_REVISION,
                source_license="declared",
                license_reviewed=True,
                policy_reviewed=True,
                tokenizer=tokenizer,
                tokenizer_identity=IDENTITY,
                policy=ConversionPolicy(profile=PROFILE_QUALITY05),
            )
        self.assertFalse(output.exists())

    def test_cli_requires_explicit_profile_and_single_output_directory(self) -> None:
        help_text = build_parser().format_help()
        self.assertIn("--profile {conservative-zero,quality05}", help_text)
        self.assertIn("--output-dir", help_text)
        self.assertNotIn("--audit-output", help_text)
        self.assertNotIn("--lineage-output", help_text)


if __name__ == "__main__":
    unittest.main()
