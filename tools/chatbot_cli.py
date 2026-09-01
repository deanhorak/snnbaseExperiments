#!/usr/bin/env python3
"""Interactive client for the pinned Qwen tokenizer and chatbot token core.

The client keeps one verified tokenizer and one persistent C++ core process.
Every turn is rendered through the upstream tokenizer's chat template and then
validated by ``chatbot_token_protocol`` before integer token IDs cross the
JSONL boundary. Commands are ``/reset`` and ``/quit``.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, IO, Mapping, Sequence

from chatbot_token_protocol import (
    MAX_LINE_BYTES_DEFAULT,
    PROTOCOL,
    CoreProcess,
    FrontendContext,
    Limits,
    ProtocolError,
    prepare_core_request,
    process_request,
)
from qwen_contract import (
    ContractError,
    PHASE0_MODEL_ID,
    PHASE0_REVISION,
    PHASE0_TOKENIZER_FINGERPRINT,
    load_transformers_tokenizer,
    require_sha256,
    verify_asset_manifest,
)

DEFAULT_CONTEXT_TOKENS = 128
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_MAX_HISTORY_MESSAGES = 64
DEFAULT_MAX_HISTORY_BYTES = 262_144
DEFAULT_EOS_TOKEN_IDS = (151645, 151643)


@dataclass(frozen=True)
class GenerationOptions:
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    seed: int = 42
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    eos_token_ids: tuple[int, ...] = DEFAULT_EOS_TOKEN_IDS
    enable_thinking: bool = False


@dataclass(frozen=True)
class AssistantReply:
    text: str
    response: Mapping[str, Any]
    pruned_turns: int
    retained_in_history: bool


class ChatSession:
    """Bounded multi-turn state over one protocol transaction function."""

    def __init__(
        self,
        context: FrontendContext,
        transact: Callable[[Mapping[str, Any]], Any],
        generation: GenerationOptions,
        *,
        system_prompt: str | None = None,
    ):
        if context.limits.max_messages < (3 if system_prompt is not None else 2):
            raise ContractError(
                "max-history-messages must retain one complete user/assistant turn"
            )
        if system_prompt is not None and (
            not system_prompt
            or len(system_prompt.encode("utf-8")) > context.limits.max_text_bytes
        ):
            raise ContractError(
                "system prompt must be non-empty and within max-history-bytes"
            )
        self._context = context
        self._transact = transact
        self._generation = generation
        self._base_history: list[dict[str, str]] = (
            [{"role": "system", "content": system_prompt}]
            if system_prompt is not None
            else []
        )
        self._history = [dict(message) for message in self._base_history]
        self._request_number = 0
        self._closed = False

    @property
    def history(self) -> tuple[Mapping[str, str], ...]:
        return tuple(dict(message) for message in self._history)

    @property
    def closed(self) -> bool:
        return self._closed

    def _request_id(self) -> str:
        self._request_number += 1
        return f"interactive-{self._request_number:08d}"

    def _request(self, operation: str, **fields: Any) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "request_id": self._request_id(),
            "op": operation,
            **fields,
        }

    @staticmethod
    def _drop_oldest_complete_turn(messages: list[dict[str, str]]) -> bool:
        first_turn = 1 if messages and messages[0]["role"] == "system" else 0
        if len(messages) - first_turn < 2:
            return False
        if (
            messages[first_turn]["role"] != "user"
            or messages[first_turn + 1]["role"] != "assistant"
        ):
            raise ContractError("internal chat history lost turn ordering")
        del messages[first_turn : first_turn + 2]
        return True

    def _generation_request(
        self, request_id: str, messages: Sequence[Mapping[str, str]]
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "protocol": PROTOCOL,
            "request_id": request_id,
            "op": "generate",
            "messages": [dict(message) for message in messages],
            "max_new_tokens": self._generation.max_new_tokens,
            "seed": self._generation.seed,
            "temperature": self._generation.temperature,
            "top_k": self._generation.top_k,
            "top_p": self._generation.top_p,
            "eos_token_ids": list(self._generation.eos_token_ids),
            "enable_thinking": self._generation.enable_thinking,
        }
        return request

    def _fit_prompt(
        self, user_text: str
    ) -> tuple[dict[str, Any], list[dict[str, str]], int]:
        if not user_text or not user_text.strip():
            raise ProtocolError("invalid_input", "user message must be non-empty")
        candidate = [dict(message) for message in self._history]
        candidate.append({"role": "user", "content": user_text})
        pruned = 0

        # Reserve one stored-history slot for the assistant response. Context
        # and byte limits are then enforced by the shared protocol validator.
        while len(candidate) + 1 > self._context.limits.max_messages:
            if not self._drop_oldest_complete_turn(candidate):
                raise ProtocolError(
                    "history_too_large",
                    "the current user/system messages exceed the history bound",
                )
            pruned += 1

        request_id = self._request_id()
        while True:
            request = self._generation_request(request_id, candidate)
            try:
                prepare_core_request(request, self._context)
                return request, candidate, pruned
            except ProtocolError as error:
                if error.code not in {
                    "invalid_messages",
                    "invalid_tokens",
                    "request_too_large",
                } or not self._drop_oldest_complete_turn(candidate):
                    raise
                pruned += 1

    @staticmethod
    def _require_success(response: Mapping[str, Any]) -> None:
        if response.get("ok") is True:
            return
        error = response.get("error")
        if isinstance(error, Mapping):
            code = error.get("code")
            message = error.get("message")
            if isinstance(code, str) and isinstance(message, str):
                raise ProtocolError(code, message, response.get("request_id"))
        raise ProtocolError(
            "invalid_core_response",
            "core returned an unsuccessful response without a valid error",
            response.get("request_id"),
        )

    def _decoded_assistant_text(self, response: Mapping[str, Any]) -> str:
        output_ids = list(response["output_ids"])
        stop_tokens = set(self._generation.eos_token_ids)
        while output_ids and output_ids[-1] in stop_tokens:
            output_ids.pop()
        decoded = self._context.tokenizer.decode(
            output_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(decoded, str):
            raise ProtocolError(
                "tokenizer_failure",
                "tokenizer decode did not return assistant text",
                response.get("request_id"),
            )
        return decoded

    def _store_completed_turn(
        self, prompt: list[dict[str, str]], assistant_text: str
    ) -> tuple[int, bool]:
        completed = [*prompt, {"role": "assistant", "content": assistant_text}]
        pruned = 0
        while (
            len(completed) > self._context.limits.max_messages
            or sum(len(message["content"].encode("utf-8")) for message in completed)
            > self._context.limits.max_text_bytes
        ):
            if not self._drop_oldest_complete_turn(completed):
                self._history = [dict(message) for message in self._base_history]
                return pruned + 1, False
            pruned += 1
        self._history = completed
        current_retained = bool(
            len(completed) >= 2
            and completed[-2]["role"] == "user"
            and completed[-1]["role"] == "assistant"
        )
        return pruned, current_retained

    def ask(self, user_text: str) -> AssistantReply:
        if self._closed:
            raise ProtocolError("session_closed", "chat session is closed")
        request, prompt, pruned_before = self._fit_prompt(user_text)
        response, should_stop = process_request(request, self._context, self._transact)
        if should_stop:
            raise ProtocolError(
                "invalid_core_response", "generate unexpectedly stopped the session"
            )
        self._require_success(response)
        assistant_text = self._decoded_assistant_text(response)
        pruned_after, retained = self._store_completed_turn(prompt, assistant_text)
        return AssistantReply(
            text=assistant_text,
            response=response,
            pruned_turns=pruned_before + pruned_after,
            retained_in_history=retained,
        )

    def reset(self) -> None:
        if self._closed:
            raise ProtocolError("session_closed", "chat session is closed")
        response, _ = process_request(
            self._request("reset"), self._context, self._transact
        )
        self._require_success(response)
        self._history = [dict(message) for message in self._base_history]

    def shutdown(self) -> None:
        if self._closed:
            return
        try:
            response, should_stop = process_request(
                self._request("shutdown"), self._context, self._transact
            )
            self._require_success(response)
            if not should_stop:
                raise ProtocolError(
                    "invalid_core_response", "shutdown did not stop the session"
                )
        finally:
            self._closed = True


def run_interactive(
    session: ChatSession,
    input_stream: IO[str],
    output_stream: IO[str],
    error_stream: IO[str],
) -> int:
    output_stream.write("Interactive spiking chatbot. Commands: /reset, /quit\n")
    output_stream.flush()
    while True:
        output_stream.write("You> ")
        output_stream.flush()
        line = input_stream.readline()
        if line == "":
            try:
                session.shutdown()
                return 0
            except ProtocolError as error:
                error_stream.write(f"chatbot_cli: {error.code}: {error}\n")
                error_stream.flush()
                return 1
        user_text = line.rstrip("\r\n")
        command = user_text.strip().casefold()
        if command == "/quit":
            try:
                session.shutdown()
            except ProtocolError as error:
                error_stream.write(f"chatbot_cli: {error.code}: {error}\n")
                error_stream.flush()
                return 1
            output_stream.write("Session closed.\n")
            output_stream.flush()
            return 0
        if command == "/reset":
            try:
                session.reset()
            except ProtocolError as error:
                error_stream.write(f"chatbot_cli: {error.code}: {error}\n")
                error_stream.flush()
                continue
            output_stream.write("Conversation reset.\n")
            output_stream.flush()
            continue
        try:
            reply = session.ask(user_text)
        except ProtocolError as error:
            error_stream.write(f"chatbot_cli: {error.code}: {error}\n")
            error_stream.flush()
            continue
        output_stream.write(f"Assistant> {reply.text}\n")
        output_stream.flush()
        if reply.pruned_turns:
            error_stream.write(
                f"chatbot_cli: pruned {reply.pruned_turns} oldest turn(s) "
                "to stay within context limits\n"
            )
            error_stream.flush()
        if not reply.retained_in_history:
            error_stream.write(
                "chatbot_cli: current turn exceeded the history byte bound and "
                "was not retained\n"
            )
            error_stream.flush()


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _seed(value: str) -> int:
    parsed = _nonnegative_integer(value)
    if parsed > (1 << 64) - 1:
        raise argparse.ArgumentTypeError("seed must fit an unsigned 64-bit integer")
    return parsed


def _temperature(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("temperature must be numeric") from error
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 100.0:
        raise argparse.ArgumentTypeError("temperature must be in [0, 100]")
    return parsed


def _top_p(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("top-p must be numeric") from error
    if not math.isfinite(parsed) or not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("top-p must be in (0, 1]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets-dir",
        required=True,
        type=Path,
        help="verified local Qwen/Qwen3-0.6B-Base tokenizer snapshot",
    )
    parser.add_argument(
        "--core-executable",
        required=True,
        type=Path,
        help="chatbot_experiment executable; launched directly without a shell",
    )
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--checkpoint", type=Path)
    weights.add_argument(
        "--qwen-archive",
        type=Path,
        help="converted exact Qwen dense archive produced by qwen_convert.py",
    )
    parser.add_argument(
        "--qwen-archive-sha256",
        help="required whole-file SHA-256 from the conversion manifest",
    )
    parser.add_argument(
        "--qwen3-0.6b",
        dest="qwen3_06b",
        action="store_true",
        help="instantiate the exact Qwen3-0.6B geometry when loading a runner checkpoint",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    architecture = parser.add_mutually_exclusive_group()
    architecture.add_argument(
        "--ann", action="store_const", const="ann", dest="variant"
    )
    architecture.add_argument(
        "--snn", action="store_const", const="snn", dest="variant"
    )
    parser.add_argument("--system-prompt")
    parser.add_argument(
        "--context-tokens", type=_positive_integer, default=DEFAULT_CONTEXT_TOKENS
    )
    parser.add_argument(
        "--max-new-tokens", type=_positive_integer, default=DEFAULT_MAX_NEW_TOKENS
    )
    parser.add_argument("--seed", type=_seed, default=42)
    parser.add_argument("--temperature", type=_temperature, default=0.0)
    parser.add_argument("--top-k", type=_nonnegative_integer, default=0)
    parser.add_argument("--top-p", type=_top_p, default=1.0)
    parser.add_argument(
        "--eos-token-id",
        type=_nonnegative_integer,
        action="append",
        dest="eos_token_ids",
        help="repeat for each stop token; defaults to Qwen im_end and endoftext",
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--max-history-messages",
        type=_positive_integer,
        default=DEFAULT_MAX_HISTORY_MESSAGES,
    )
    parser.add_argument(
        "--max-history-bytes",
        type=_positive_integer,
        default=DEFAULT_MAX_HISTORY_BYTES,
    )
    parser.add_argument(
        "--max-line-bytes", type=_positive_integer, default=MAX_LINE_BYTES_DEFAULT
    )
    return parser


def build_core_command(
    core_executable: Path,
    *,
    context_tokens: int,
    device: str,
    checkpoint: Path | None,
    qwen_archive: Path | None = None,
    qwen_archive_sha256: str | None = None,
    qwen3_06b: bool = False,
    variant: str | None = None,
) -> list[str]:
    command = [
        str(core_executable),
        "serve",
        "--context-tokens",
        str(context_tokens),
        "--device",
        device,
    ]
    if qwen3_06b or qwen_archive is not None:
        command.append("--qwen3-0.6b")
    if checkpoint is not None:
        command.extend(("--checkpoint", str(checkpoint)))
    if qwen_archive is not None:
        command.extend(
            (
                "--qwen-archive",
                str(qwen_archive),
                "--qwen-archive-sha256",
                str(qwen_archive_sha256),
            )
        )
    if variant is not None:
        command.append(f"--{variant}")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.max_new_tokens >= arguments.context_tokens:
        parser.error("--max-new-tokens must be smaller than --context-tokens")
    if (arguments.qwen_archive is None) != (arguments.qwen_archive_sha256 is None):
        parser.error(
            "--qwen-archive and --qwen-archive-sha256 must be supplied together"
        )
    if arguments.qwen_archive_sha256 is not None:
        try:
            require_sha256(arguments.qwen_archive_sha256, "Qwen archive SHA-256")
        except ContractError as error:
            parser.error(str(error))
    if arguments.checkpoint is not None and arguments.variant is None:
        parser.error("--checkpoint requires exactly one of --ann or --snn")
    if arguments.qwen_archive is not None and arguments.variant is None:
        arguments.variant = "ann"
    required_history_messages = 3 if arguments.system_prompt is not None else 2
    if arguments.max_history_messages < required_history_messages:
        parser.error(
            f"--max-history-messages must be at least {required_history_messages}"
        )

    core: CoreProcess | None = None
    try:
        assets = verify_asset_manifest(
            arguments.assets_dir,
            PHASE0_MODEL_ID,
            PHASE0_REVISION,
            PHASE0_TOKENIZER_FINGERPRINT,
        )
        tokenizer = load_transformers_tokenizer(assets)
        limits = Limits(
            max_line_bytes=arguments.max_line_bytes,
            max_input_tokens=arguments.context_tokens - arguments.max_new_tokens,
            max_new_tokens=arguments.max_new_tokens,
            max_text_bytes=arguments.max_history_bytes,
            max_messages=arguments.max_history_messages,
        )
        generation = GenerationOptions(
            max_new_tokens=arguments.max_new_tokens,
            seed=arguments.seed,
            temperature=arguments.temperature,
            top_k=arguments.top_k,
            top_p=arguments.top_p,
            eos_token_ids=tuple(arguments.eos_token_ids or DEFAULT_EOS_TOKEN_IDS),
            enable_thinking=arguments.enable_thinking,
        )
        command = build_core_command(
            arguments.core_executable,
            context_tokens=arguments.context_tokens,
            device=arguments.device,
            checkpoint=arguments.checkpoint,
            qwen_archive=arguments.qwen_archive,
            qwen_archive_sha256=arguments.qwen_archive_sha256,
            qwen3_06b=arguments.qwen3_06b,
            variant=arguments.variant,
        )
        core = CoreProcess(command, limits.max_line_bytes)
        session = ChatSession(
            FrontendContext(tokenizer, assets, limits),
            core.transact,
            generation,
            system_prompt=arguments.system_prompt,
        )
        return run_interactive(session, sys.stdin, sys.stdout, sys.stderr)
    except KeyboardInterrupt:
        print("\nchatbot_cli: interrupted", file=sys.stderr)
        return 130
    except (ContractError, ProtocolError, OSError, ValueError) as error:
        print(f"chatbot_cli: {error}", file=sys.stderr)
        return 2
    finally:
        if core is not None:
            try:
                core.close()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
