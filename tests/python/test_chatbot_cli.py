from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from typing import Any, Mapping

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools"))

from chatbot_cli import (  # noqa: E402
    ChatSession,
    GenerationOptions,
    build_parser,
    build_core_command,
    run_interactive,
)
from chatbot_token_protocol import CoreProcess, FrontendContext, Limits  # noqa: E402
from qwen_contract import verify_asset_manifest  # noqa: E402

ASSETS = REPOSITORY / "tests" / "fixtures" / "chatbot" / "fake_qwen" / "assets"
FINGERPRINT = "e2f448429ba76678d32ed7cf99e8d9adb031944a11a170771365a0f76f29a742"

FAKE_CORE = r"""
import json
import sys

generation_count = 0
for line in sys.stdin:
    request = json.loads(line)
    response = {
        "protocol": request["protocol"],
        "request_id": request["request_id"],
        "ok": True,
    }
    if request["op"] == "generate":
        generation_count += 1
        response.update(
            {
                "output_ids": list(f"reply-{generation_count}".encode("utf-8")) + [0],
                "finish_reason": "eos",
                "seen_generation": {
                    key: request[key]
                    for key in (
                        "max_new_tokens",
                        "seed",
                        "temperature",
                        "top_k",
                        "top_p",
                        "eos_token_ids",
                    )
                },
            }
        )
    elif request["op"] == "reset":
        generation_count = 0
        response["reset"] = True
    elif request["op"] == "shutdown":
        response["shutdown"] = True
    else:
        response = {
            "protocol": request["protocol"],
            "request_id": request["request_id"],
            "ok": False,
            "error": {"code": "unsupported_operation", "message": "fake core"},
        }
    print(json.dumps(response, separators=(",", ":")), flush=True)
    if request["op"] == "shutdown":
        break
"""


class FakeByteTokenizer:
    def __init__(self) -> None:
        self.template_calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    def __len__(self) -> int:
        return 256

    def encode(self, text: str, **_: object) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int], **_: object) -> str:
        return bytes(token_ids).decode("utf-8")

    def apply_chat_template(
        self,
        messages: list[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str | list[int]:
        copied = [dict(message) for message in messages]
        self.template_calls.append(
            (
                copied,
                {
                    "tokenize": tokenize,
                    "add_generation_prompt": add_generation_prompt,
                    "enable_thinking": enable_thinking,
                },
            )
        )
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>\n"
            for message in copied
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return self.encode(rendered) if tokenize else rendered


def context(tokenizer: FakeByteTokenizer, *, max_messages: int = 4) -> FrontendContext:
    assets = verify_asset_manifest(
        ASSETS,
        "tests/FakeQwen",
        "0123456789abcdef0123456789abcdef01234567",
        FINGERPRINT,
    )
    return FrontendContext(
        tokenizer,
        assets,
        Limits(
            max_line_bytes=65_536,
            max_input_tokens=512,
            max_new_tokens=32,
            max_text_bytes=4_096,
            max_messages=max_messages,
        ),
    )


def fake_core() -> CoreProcess:
    return CoreProcess([sys.executable, "-u", "-c", FAKE_CORE], 65_536)


class ChatbotCliTests(unittest.TestCase):
    def test_exact_qwen_geometry_flag_has_a_stable_destination(self) -> None:
        arguments = build_parser().parse_args(
            [
                "--assets-dir",
                "assets",
                "--core-executable",
                "core",
                "--qwen3-0.6b",
            ]
        )
        self.assertTrue(arguments.qwen3_06b)

    def test_multi_turn_history_template_flags_bounds_and_reset(self) -> None:
        tokenizer = FakeByteTokenizer()
        core = fake_core()
        generation = GenerationOptions(
            max_new_tokens=12,
            seed=123,
            temperature=0.75,
            top_k=7,
            top_p=0.8,
            eos_token_ids=(0,),
            enable_thinking=True,
        )
        session = ChatSession(
            context(tokenizer, max_messages=5),
            core.transact,
            generation,
            system_prompt="Be brief.",
        )
        try:
            first = session.ask("one")
            second = session.ask("two")
            third = session.ask("three")

            self.assertEqual(
                (first.text, second.text, third.text),
                (
                    "reply-1",
                    "reply-2",
                    "reply-3",
                ),
            )
            self.assertNotIn("\x00", third.text)
            self.assertEqual(
                third.response["seen_generation"],
                {
                    "max_new_tokens": 12,
                    "seed": 123,
                    "temperature": 0.75,
                    "top_k": 7,
                    "top_p": 0.8,
                    "eos_token_ids": [0],
                },
            )
            self.assertEqual(third.pruned_turns, 1)
            self.assertEqual(
                [(message["role"], message["content"]) for message in session.history],
                [
                    ("system", "Be brief."),
                    ("user", "two"),
                    ("assistant", "reply-2"),
                    ("user", "three"),
                    ("assistant", "reply-3"),
                ],
            )

            rendered_messages, template_options = tokenizer.template_calls[-1]
            self.assertEqual(
                [message["role"] for message in rendered_messages],
                ["system", "user", "assistant", "user"],
            )
            self.assertEqual(rendered_messages[2]["content"], "reply-2")
            self.assertTrue(template_options["add_generation_prompt"])
            self.assertTrue(template_options["enable_thinking"])

            session.reset()
            self.assertEqual(
                session.history,
                ({"role": "system", "content": "Be brief."},),
            )
            self.assertEqual(session.ask("fresh").text, "reply-1")
            session.shutdown()
            self.assertTrue(session.closed)
        finally:
            core.close()

    def test_interactive_reset_quit_and_decoded_output(self) -> None:
        tokenizer = FakeByteTokenizer()
        core = fake_core()
        session = ChatSession(
            context(tokenizer, max_messages=6),
            core.transact,
            GenerationOptions(max_new_tokens=12, eos_token_ids=(0,)),
        )
        output = io.StringIO()
        errors = io.StringIO()
        try:
            status = run_interactive(
                session,
                io.StringIO("hello\n/reset\nfresh\n/quit\n"),
                output,
                errors,
            )
            self.assertEqual(status, 0)
            self.assertEqual(output.getvalue().count("Assistant> reply-1"), 2)
            self.assertIn("Conversation reset.", output.getvalue())
            self.assertIn("Session closed.", output.getvalue())
            self.assertEqual(errors.getvalue(), "")
            self.assertTrue(session.closed)
        finally:
            core.close()

    def test_core_command_is_an_argv_list_not_a_shell_command(self) -> None:
        command = build_core_command(
            Path("bin/core; touch SHOULD_NOT_EXIST"),
            context_tokens=256,
            device="cpu",
            checkpoint=Path("weights/model; echo unsafe.pt"),
            qwen3_06b=True,
            variant="ann",
        )
        self.assertEqual(
            command,
            [
                "bin/core; touch SHOULD_NOT_EXIST",
                "serve",
                "--context-tokens",
                "256",
                "--device",
                "cpu",
                "--qwen3-0.6b",
                "--checkpoint",
                "weights/model; echo unsafe.pt",
                "--ann",
            ],
        )
        self.assertIsInstance(command, list)

        archive_command = build_core_command(
            Path("bin/chatbot_experiment"),
            context_tokens=512,
            device="cpu",
            checkpoint=None,
            qwen_archive=Path("weights/qwen-dense.snnq"),
            qwen_archive_sha256="a" * 64,
            variant="ann",
        )
        self.assertEqual(
            archive_command,
            [
                "bin/chatbot_experiment",
                "serve",
                "--context-tokens",
                "512",
                "--device",
                "cpu",
                "--qwen3-0.6b",
                "--qwen-archive",
                "weights/qwen-dense.snnq",
                "--qwen-archive-sha256",
                "a" * 64,
                "--ann",
            ],
        )


if __name__ == "__main__":
    unittest.main()
