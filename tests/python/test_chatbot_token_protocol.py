from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from typing import Any, Mapping

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools"))

from chatbot_token_protocol import (  # noqa: E402
    FrontendContext,
    Limits,
    PROTOCOL,
    ProtocolError,
    prepare_core_request,
    process_request,
    protocol_description,
    serve,
)
from qwen_contract import verify_asset_manifest  # noqa: E402

ASSETS = REPOSITORY / "tests" / "fixtures" / "chatbot" / "fake_qwen" / "assets"
FINGERPRINT = "e2f448429ba76678d32ed7cf99e8d9adb031944a11a170771365a0f76f29a742"


class FakeByteTokenizer:
    def __len__(self) -> int:
        return 256

    def encode(self, text: str, **_: object) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int], **_: object) -> str:
        return bytes(token_ids).decode("utf-8")

    def apply_chat_template(
        self, messages: list[Mapping[str, str]], *, tokenize: bool, **_: object
    ) -> str | list[int]:
        text = "".join(
            f"<{item['role']}>{item['content']}</{item['role']}>\n" for item in messages
        )
        text += "<assistant>"
        return self.encode(text) if tokenize else text


def context(**limit_overrides: int) -> FrontendContext:
    assets = verify_asset_manifest(
        ASSETS,
        "tests/FakeQwen",
        "0123456789abcdef0123456789abcdef01234567",
        FINGERPRINT,
    )
    return FrontendContext(FakeByteTokenizer(), assets, Limits(**limit_overrides))


def request(operation: str, **fields: Any) -> dict[str, Any]:
    return {"protocol": PROTOCOL, "request_id": "request-1", "op": operation, **fields}


class TokenProtocolTests(unittest.TestCase):
    def test_text_generate_is_tokenized_and_sampling_fields_are_forwarded(self) -> None:
        core, input_ids = prepare_core_request(
            request(
                "generate",
                text="Hi",
                max_new_tokens=3,
                seed=42,
                temperature=0.7,
                top_k=8,
                top_p=0.9,
                eos_token_ids=[0, 255],
            ),
            context(),
        )
        self.assertEqual(input_ids, [72, 105])
        self.assertEqual(core["input_ids"], [72, 105])
        self.assertEqual(core["seed"], 42)
        self.assertEqual(core["sampling_vocabulary_size"], 256)
        self.assertEqual(core["top_p"], 0.9)
        self.assertEqual(core["eos_token_ids"], [0, 255])
        self.assertNotIn("text", core)

    def test_seed_and_bounds_are_fail_closed(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "seed"):
            prepare_core_request(
                request("generate", text="Hi", max_new_tokens=2), context()
            )
        with self.assertRaisesRegex(ProtocolError, "eos_token_ids must be unique"):
            prepare_core_request(
                request(
                    "generate",
                    text="Hi",
                    max_new_tokens=2,
                    seed=0,
                    eos_token_ids=[7, 7],
                ),
                context(),
            )
        with self.assertRaisesRegex(ProtocolError, "encoded input_ids"):
            prepare_core_request(
                request("generate", text="too long", max_new_tokens=1, seed=0),
                context(max_input_tokens=3),
            )

    def test_request_id_limits_are_utf8_bytes_and_nul_is_rejected(self) -> None:
        allowed = request("metadata")
        allowed["request_id"] = "é" * 64
        core, _ = prepare_core_request(allowed, context())
        self.assertEqual(core["request_id"], "é" * 64)

        too_large = request("metadata")
        too_large["request_id"] = "é" * 65
        with self.assertRaisesRegex(ProtocolError, "UTF-8 bytes"):
            prepare_core_request(too_large, context())
        contains_nul = request("metadata")
        contains_nul["request_id"] = "bad\0id"
        with self.assertRaisesRegex(ProtocolError, "without NUL"):
            prepare_core_request(contains_nul, context())

    def test_generate_messages_are_exact_ordered_and_end_with_user(self) -> None:
        valid_messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": "Next"},
        ]
        core, input_ids = prepare_core_request(
            request(
                "generate",
                messages=valid_messages,
                enable_thinking=True,
                max_new_tokens=2,
                seed=0,
            ),
            context(),
        )
        self.assertEqual(core["input_ids"], input_ids)
        self.assertNotIn("messages", core)
        self.assertNotIn("enable_thinking", core)

        invalid_messages = [
            [{"role": "assistant", "content": "first"}],
            [
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
            ],
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "done"},
            ],
            [{"role": "system", "content": "only"}],
        ]
        for messages in invalid_messages:
            with self.subTest(messages=messages):
                with self.assertRaisesRegex(ProtocolError, "messages"):
                    prepare_core_request(
                        request(
                            "generate",
                            messages=messages,
                            max_new_tokens=2,
                            seed=0,
                        ),
                        context(),
                    )

    def test_generate_messages_reject_extra_fields_nul_and_byte_overflow(
        self,
    ) -> None:
        invalid_values = [
            [{"role": "user", "content": "hello", "name": "ignored"}],
            [{"role": "user", "content": "bad\0content"}],
            [{"role": "user", "content": ""}],
        ]
        for messages in invalid_values:
            with self.subTest(messages=messages):
                with self.assertRaisesRegex(ProtocolError, "messages"):
                    prepare_core_request(
                        request(
                            "generate",
                            messages=messages,
                            max_new_tokens=1,
                            seed=0,
                        ),
                        context(),
                    )
        with self.assertRaisesRegex(ProtocolError, "byte limit"):
            prepare_core_request(
                request(
                    "generate",
                    messages=[{"role": "user", "content": "éé"}],
                    max_new_tokens=1,
                    seed=0,
                ),
                context(max_text_bytes=3),
            )

    def test_text_rejects_nul_and_uses_utf8_byte_limit(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "without NUL"):
            prepare_core_request(
                request("generate", text="bad\0text", max_new_tokens=1, seed=0),
                context(),
            )
        with self.assertRaisesRegex(ProtocolError, "byte limit"):
            prepare_core_request(
                request("generate", text="éé", max_new_tokens=1, seed=0),
                context(max_text_bytes=3),
            )

    def test_each_operation_rejects_ignored_fields(self) -> None:
        cases = [
            request(
                "generate",
                text="hello",
                max_new_tokens=1,
                seed=0,
                ignored=True,
            ),
            request(
                "generate",
                text="hello",
                enable_thinking=True,
                max_new_tokens=1,
                seed=0,
            ),
            request(
                "generate",
                input_ids=[1],
                enable_thinking=False,
                max_new_tokens=1,
                seed=0,
            ),
            request("train", input_ids=[1], ignored=True),
            request("evaluate", input_ids=[1], ignored=True),
            request(
                "inspect", input_ids=[1], probe_token_ids=[2], top_k=1, ignored=True
            ),
        ]
        for value in cases:
            with self.subTest(operation=value["op"], fields=sorted(value)):
                with self.assertRaisesRegex(ProtocolError, "does not accept fields"):
                    prepare_core_request(value, context())

    def test_training_loss_mask_must_match_tokens(self) -> None:
        core, _ = prepare_core_request(
            request(
                "train", input_ids=[1, 2, 3], loss_mask=[0, 1, 1], reset_state=True
            ),
            context(),
        )
        self.assertEqual(core["loss_mask"], [0, 1, 1])
        with self.assertRaisesRegex(ProtocolError, "same length"):
            prepare_core_request(
                request("train", input_ids=[1, 2, 3], loss_mask=[1]), context()
            )

    def test_flush_is_a_payload_free_operation(self) -> None:
        core, input_ids = prepare_core_request(request("flush"), context())
        self.assertEqual(core, request("flush"))
        self.assertIsNone(input_ids)
        with self.assertRaisesRegex(ProtocolError, "does not accept fields"):
            prepare_core_request(request("flush", input_ids=[1, 2]), context())

    def test_logit_inspection_is_bounded_and_forwarded(self) -> None:
        core, input_ids = prepare_core_request(
            request(
                "inspect",
                input_ids=[1, 2, 3],
                probe_token_ids=[0, 7],
                top_k=4,
            ),
            context(),
        )
        self.assertEqual(input_ids, [1, 2, 3])
        self.assertEqual(core["probe_token_ids"], [0, 7])
        self.assertEqual(core["top_k"], 4)
        with self.assertRaisesRegex(ProtocolError, "must be unique"):
            prepare_core_request(
                request(
                    "inspect",
                    input_ids=[1, 2],
                    probe_token_ids=[7, 7],
                    top_k=2,
                ),
                context(),
            )

    def test_generate_response_is_correlated_decoded_and_preserves_metrics(
        self,
    ) -> None:
        def transact(core: Mapping[str, Any]) -> dict[str, Any]:
            self.assertEqual(core["sampling_vocabulary_size"], 256)
            return {
                "protocol": PROTOCOL,
                "request_id": core["request_id"],
                "ok": True,
                "output_ids": [79, 75],
                "finish_reason": "eos",
                "spike_metrics": {"spikes_per_token": 3.5},
                "latency_ms": 1.25,
            }

        response, stop = process_request(
            request("generate", input_ids=[72, 105], max_new_tokens=2, seed=99),
            context(),
            transact,
        )
        self.assertFalse(stop)
        self.assertEqual(response["text"], "OK")
        self.assertEqual(response["spike_metrics"]["spikes_per_token"], 3.5)
        self.assertEqual(response["latency_ms"], 1.25)
        self.assertEqual(response["tokenizer_fingerprint_sha256"], FINGERPRINT)

    def test_mismatched_core_response_is_rejected(self) -> None:
        def transact(_: Mapping[str, Any]) -> dict[str, Any]:
            return {"protocol": PROTOCOL, "request_id": "wrong", "ok": True}

        with self.assertRaisesRegex(ProtocolError, "does not match"):
            process_request(request("metadata"), context(), transact)

    def test_long_lived_stream_recovers_from_bad_json_then_shuts_down(self) -> None:
        input_stream = io.StringIO(
            "not-json\n"
            + json.dumps(request("metadata"))
            + "\n"
            + json.dumps(request("shutdown"))
            + "\n"
        )
        output_stream = io.StringIO()

        def transact(core: Mapping[str, Any]) -> dict[str, Any]:
            return {
                "protocol": PROTOCOL,
                "request_id": core["request_id"],
                "ok": True,
                "metadata": {"core": "fake"},
            }

        serve(input_stream, output_stream, context(), transact)
        responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
        self.assertEqual(len(responses), 3)
        self.assertEqual(responses[0]["error"]["code"], "invalid_json")
        self.assertTrue(responses[1]["ok"])
        self.assertTrue(responses[2]["ok"])

    def test_stream_rejects_duplicate_fields_at_all_object_levels(self) -> None:
        duplicate_top_level = (
            '{"protocol":"snnbase.chatbot.tokens/v1",'
            '"request_id":"first","request_id":"second","op":"metadata"}\n'
        )
        duplicate_nested = (
            '{"protocol":"snnbase.chatbot.tokens/v1",'
            '"request_id":"nested","op":"generate",'
            '"messages":[{"role":"user","content":"one",'
            '"content":"two"}],"max_new_tokens":1,"seed":0}\n'
        )
        nonstandard_number = (
            '{"protocol":"snnbase.chatbot.tokens/v1",'
            '"request_id":"nan","op":"generate","temperature":NaN,'
            '"text":"hello","max_new_tokens":1,"seed":0}\n'
        )
        input_stream = io.StringIO(
            duplicate_top_level + duplicate_nested + nonstandard_number
        )
        output_stream = io.StringIO()
        transactions: list[Mapping[str, Any]] = []

        def transact(core: Mapping[str, Any]) -> dict[str, Any]:
            transactions.append(core)
            return {
                "protocol": PROTOCOL,
                "request_id": core["request_id"],
                "ok": True,
            }

        serve(input_stream, output_stream, context(), transact)
        responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
        self.assertEqual(
            [item["error"]["code"] for item in responses], ["invalid_json"] * 3
        )
        self.assertIn("duplicate JSON", responses[0]["error"]["message"])
        self.assertIn("duplicate JSON", responses[1]["error"]["message"])
        self.assertEqual(transactions, [])

    def test_stream_line_limit_counts_payload_utf8_bytes_not_characters(self) -> None:
        serialized = json.dumps(
            request("metadata", note="é"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        payload_bytes = len(serialized.encode("utf-8"))
        output_stream = io.StringIO()

        def transact(core: Mapping[str, Any]) -> dict[str, Any]:
            return {
                "protocol": PROTOCOL,
                "request_id": core["request_id"],
                "ok": True,
            }

        # The exact byte-boundary line reaches contract validation (where the
        # deliberately unknown field is rejected), rather than size rejection.
        serve(
            io.StringIO(serialized + "\n"),
            output_stream,
            context(max_line_bytes=payload_bytes),
            transact,
        )
        exact = json.loads(output_stream.getvalue())
        self.assertEqual(exact["error"]["code"], "unexpected_field")

        output_stream = io.StringIO()
        serve(
            io.StringIO(serialized + "\n"),
            output_stream,
            context(max_line_bytes=payload_bytes - 1),
            transact,
        )
        oversized = json.loads(output_stream.getvalue())
        self.assertEqual(oversized["error"]["code"], "request_too_large")

    def test_stream_drains_one_oversized_physical_line_then_recovers(self) -> None:
        valid = json.dumps(request("metadata"), separators=(",", ":"))
        limit = len(valid.encode("utf-8"))
        input_stream = io.StringIO("x" * (limit + 20) + "\n" + valid + "\n")
        output_stream = io.StringIO()

        def transact(core: Mapping[str, Any]) -> dict[str, Any]:
            return {
                "protocol": PROTOCOL,
                "request_id": core["request_id"],
                "ok": True,
            }

        serve(
            input_stream,
            output_stream,
            context(max_line_bytes=limit),
            transact,
        )
        responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0]["error"]["code"], "request_too_large")
        self.assertTrue(responses[1]["ok"])

    def test_protocol_description_documents_multi_eos_and_metrics(self) -> None:
        description = protocol_description()
        self.assertIn("eos_token_ids", description["generate"]["optional"])
        self.assertIn("spike_metrics", description["optional_response_metrics"])
        self.assertIn("latency_ms", description["optional_response_metrics"])


if __name__ == "__main__":
    unittest.main()
