#!/usr/bin/env python3
"""Long-lived JSONL tokenizer frontend for the spiking chatbot core.

Core protocol ``snnbase.chatbot.tokens/v1`` (one compact JSON object per line):

* common request fields: ``protocol``, ``request_id``, and ``op``;
* ``generate`` adds ``input_ids``, required ``max_new_tokens``, ``seed``, and
  ``sampling_vocabulary_size``, plus optional ``temperature``, ``top_k``,
  ``top_p``, and ``eos_token_ids``;
* ``train``/``evaluate`` add ``input_ids`` and optional equal-length binary
  ``loss_mask`` and ``reset_state`` (a core may return ``unsupported_operation``);
* ``metadata``, ``reset``, ``flush``, and ``shutdown`` have no
  operation-specific fields;
* success responses contain ``ok: true`` and operation results; errors contain
  ``ok: false`` and ``error: {code, message}``.  Generate responses use
  ``output_ids`` for newly generated tokens and ``finish_reason``.  A core may
  also return ``spike_metrics`` and ``latency_ms``.

The frontend accepts the same requests, but a generate request may replace
``input_ids`` with exactly one of ``text`` or ``messages``.  It tokenizes that
input, forwards only token protocol fields to one persistent child process, and
adds ``input_ids`` and decoded ``text`` to successful generate responses.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, IO, Iterable, Mapping, Sequence

from qwen_contract import (
    ContractError,
    PHASE0_MODEL_ID,
    PHASE0_REVISION,
    PHASE0_TOKENIZER_FINGERPRINT,
    VerifiedAssets,
    load_transformers_tokenizer,
    verify_asset_manifest,
)

PROTOCOL = "snnbase.chatbot.tokens/v1"
SUPPORTED_OPERATIONS = frozenset(
    {
        "metadata",
        "generate",
        "inspect",
        "train",
        "evaluate",
        "reset",
        "flush",
        "shutdown",
    }
)
MAX_LINE_BYTES_DEFAULT = 1_048_576
MAX_INPUT_TOKENS_DEFAULT = 32_768
MAX_NEW_TOKENS_DEFAULT = 4_096
MAX_TEXT_BYTES_DEFAULT = 1_048_576
MAX_MESSAGES_DEFAULT = 256
MAX_REQUEST_ID_BYTES = 128
# Backward-compatible import name; the limit has always been a transport bound
# and is intentionally measured in UTF-8 bytes.
MAX_REQUEST_ID_LENGTH = MAX_REQUEST_ID_BYTES
MAX_U64 = (1 << 64) - 1
COMMON_REQUEST_FIELDS = frozenset({"protocol", "request_id", "op"})


class _DuplicateJsonField(ValueError):
    pass


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonField(f"duplicate JSON object field: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _strict_json_loads(value: str) -> Any:
    return json.loads(
        value,
        object_pairs_hook=_reject_duplicate_fields,
        parse_constant=_reject_nonstandard_constant,
    )


class ProtocolError(RuntimeError):
    def __init__(self, code: str, message: str, request_id: str | None = None):
        super().__init__(message)
        self.code = code
        self.request_id = request_id


@dataclass(frozen=True)
class Limits:
    max_line_bytes: int = MAX_LINE_BYTES_DEFAULT
    max_input_tokens: int = MAX_INPUT_TOKENS_DEFAULT
    max_new_tokens: int = MAX_NEW_TOKENS_DEFAULT
    max_text_bytes: int = MAX_TEXT_BYTES_DEFAULT
    max_messages: int = MAX_MESSAGES_DEFAULT


@dataclass(frozen=True)
class FrontendContext:
    tokenizer: Any
    assets: VerifiedAssets
    limits: Limits


def _request_id(value: Any) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ProtocolError(
            "invalid_request_id",
            f"request_id must contain 1-{MAX_REQUEST_ID_BYTES} UTF-8 bytes without NUL",
        )
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ProtocolError(
            "invalid_request_id", "request_id must be valid UTF-8"
        ) from error
    if size > MAX_REQUEST_ID_BYTES:
        raise ProtocolError(
            "invalid_request_id",
            f"request_id must contain 1-{MAX_REQUEST_ID_BYTES} UTF-8 bytes without NUL",
        )
    return value


def _reject_unexpected_fields(
    request: Mapping[str, Any],
    allowed: frozenset[str] | set[str],
    operation: str,
    request_id: str,
) -> None:
    extras = set(request) - set(allowed)
    if extras:
        ordered = sorted(extras, key=repr)
        raise ProtocolError(
            "unexpected_field",
            f"{operation} does not accept fields: {ordered}",
            request_id,
        )


def _integer(
    value: Any, label: str, minimum: int, maximum: int, request_id: str
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise ProtocolError(
            "invalid_field",
            f"{label} must be an integer in [{minimum}, {maximum}]",
            request_id,
        )
    return value


def _number(
    value: Any,
    label: str,
    minimum: float,
    maximum: float,
    request_id: str,
    *,
    minimum_exclusive: bool = False,
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProtocolError("invalid_field", f"{label} must be numeric", request_id)
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError("invalid_field", f"{label} must be finite", request_id)
    lower_ok = result > minimum if minimum_exclusive else result >= minimum
    if not lower_ok or result > maximum:
        bracket = "(" if minimum_exclusive else "["
        raise ProtocolError(
            "invalid_field",
            f"{label} must be in {bracket}{minimum}, {maximum}]",
            request_id,
        )
    return result


def _token_ids(
    value: Any,
    label: str,
    request_id: str,
    *,
    vocabulary_size: int,
    maximum_count: int,
    allow_empty: bool = False,
) -> list[int]:
    if (
        not isinstance(value, list)
        or (not value and not allow_empty)
        or len(value) > maximum_count
    ):
        raise ProtocolError(
            "invalid_tokens",
            f"{label} must contain {'0' if allow_empty else '1'}-{maximum_count} token ids",
            request_id,
        )
    result: list[int] = []
    for index, token in enumerate(value):
        if (
            not isinstance(token, int)
            or isinstance(token, bool)
            or token < 0
            or token >= vocabulary_size
        ):
            raise ProtocolError(
                "invalid_tokens",
                f"{label}[{index}] is outside tokenizer range [0, {vocabulary_size})",
                request_id,
            )
        result.append(token)
    return result


def _normalize_encoded(value: Any, request_id: str) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise ProtocolError(
            "tokenizer_failure", "tokenizer returned invalid input_ids", request_id
        )
    return value


def _validated_messages(
    value: Any, request_id: str, limits: Limits
) -> list[Mapping[str, str]]:
    if not isinstance(value, list) or not value or len(value) > limits.max_messages:
        raise ProtocolError(
            "invalid_messages",
            f"messages must contain 1-{limits.max_messages} entries",
            request_id,
        )
    messages: list[Mapping[str, str]] = []
    total_bytes = 0
    for index, raw_message in enumerate(value):
        if not isinstance(raw_message, Mapping):
            raise ProtocolError(
                "invalid_messages", f"messages[{index}] must be an object", request_id
            )
        if set(raw_message) != {"role", "content"}:
            raise ProtocolError(
                "invalid_messages",
                f"messages[{index}] must contain exactly role and content",
                request_id,
            )
        role = raw_message.get("role")
        content = raw_message.get("content")
        if (
            role not in ("system", "user", "assistant")
            or not isinstance(content, str)
            or not content
        ):
            raise ProtocolError(
                "invalid_messages",
                f"messages[{index}] requires role system/user/assistant and non-empty string content",
                request_id,
            )
        if "\0" in content:
            raise ProtocolError(
                "invalid_messages",
                f"messages[{index}] content must not contain NUL",
                request_id,
            )
        try:
            total_bytes += len(content.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise ProtocolError(
                "invalid_messages",
                f"messages[{index}] content must be valid UTF-8",
                request_id,
            ) from error
        messages.append({"role": role, "content": content})
    if total_bytes > limits.max_text_bytes:
        raise ProtocolError(
            "request_too_large",
            "message text exceeds configured byte limit",
            request_id,
        )
    first_turn = 1 if messages[0]["role"] == "system" else 0
    if first_turn == len(messages):
        raise ProtocolError(
            "invalid_messages",
            "messages must contain a user turn after the optional system message",
            request_id,
        )
    for index in range(first_turn, len(messages)):
        expected = "user" if (index - first_turn) % 2 == 0 else "assistant"
        if messages[index]["role"] != expected:
            raise ProtocolError(
                "invalid_messages",
                "messages must optionally start with system and then strictly "
                "alternate user/assistant",
                request_id,
            )
    if messages[-1]["role"] != "user":
        raise ProtocolError(
            "invalid_messages",
            "generate messages must end with a user turn",
            request_id,
        )
    return messages


def _encode_frontend_input(
    request: Mapping[str, Any], context: FrontendContext, request_id: str
) -> list[int]:
    alternatives = [
        field for field in ("input_ids", "text", "messages") if field in request
    ]
    if len(alternatives) != 1:
        raise ProtocolError(
            "invalid_input",
            "generate requires exactly one of input_ids, text, or messages",
            request_id,
        )
    vocabulary_size = len(context.tokenizer)
    if alternatives[0] == "input_ids":
        return _token_ids(
            request["input_ids"],
            "input_ids",
            request_id,
            vocabulary_size=vocabulary_size,
            maximum_count=context.limits.max_input_tokens,
        )
    if alternatives[0] == "text":
        text = request["text"]
        if not isinstance(text, str) or not text or "\0" in text:
            raise ProtocolError(
                "invalid_input",
                "text must be non-empty valid UTF-8 without NUL and within the configured byte limit",
                request_id,
            )
        try:
            text_bytes = len(text.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise ProtocolError(
                "invalid_input", "text must be valid UTF-8", request_id
            ) from error
        if text_bytes > context.limits.max_text_bytes:
            raise ProtocolError(
                "invalid_input",
                "text must be non-empty valid UTF-8 without NUL and within the configured byte limit",
                request_id,
            )
        encoded = context.tokenizer.encode(text, add_special_tokens=False)
    else:
        messages = _validated_messages(request["messages"], request_id, context.limits)
        enable_thinking = request.get("enable_thinking", False)
        if not isinstance(enable_thinking, bool):
            raise ProtocolError(
                "invalid_field", "enable_thinking must be boolean", request_id
            )
        encoded = context.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    normalized = _normalize_encoded(encoded, request_id)
    return _token_ids(
        normalized,
        "encoded input_ids",
        request_id,
        vocabulary_size=vocabulary_size,
        maximum_count=context.limits.max_input_tokens,
    )


def _common_request(request: Any) -> tuple[Mapping[str, Any], str, str]:
    if not isinstance(request, Mapping):
        raise ProtocolError("invalid_request", "request must be a JSON object")
    request_id = _request_id(request.get("request_id"))
    if request.get("protocol") != PROTOCOL:
        raise ProtocolError(
            "unsupported_protocol", f"protocol must be {PROTOCOL}", request_id
        )
    operation = request.get("op")
    if operation not in SUPPORTED_OPERATIONS:
        raise ProtocolError(
            "unsupported_operation", f"unsupported op: {operation!r}", request_id
        )
    return request, request_id, operation


def prepare_core_request(
    request: Any, context: FrontendContext
) -> tuple[dict[str, Any], list[int] | None]:
    request, request_id, operation = _common_request(request)
    if operation == "generate":
        allowed = set(COMMON_REQUEST_FIELDS) | {
            "input_ids",
            "text",
            "messages",
            "max_new_tokens",
            "seed",
            "temperature",
            "top_k",
            "top_p",
            "eos_token_ids",
        }
        if "messages" in request:
            allowed.add("enable_thinking")
        _reject_unexpected_fields(request, allowed, operation, request_id)
    elif operation in ("train", "evaluate"):
        _reject_unexpected_fields(
            request,
            set(COMMON_REQUEST_FIELDS) | {"input_ids", "loss_mask", "reset_state"},
            operation,
            request_id,
        )
    elif operation == "inspect":
        _reject_unexpected_fields(
            request,
            set(COMMON_REQUEST_FIELDS) | {"input_ids", "probe_token_ids", "top_k"},
            operation,
            request_id,
        )
    else:
        _reject_unexpected_fields(request, COMMON_REQUEST_FIELDS, operation, request_id)
    core: dict[str, Any] = {
        "protocol": PROTOCOL,
        "request_id": request_id,
        "op": operation,
    }
    input_ids: list[int] | None = None

    if operation == "generate":
        input_ids = _encode_frontend_input(request, context, request_id)
        core["input_ids"] = input_ids
        core["max_new_tokens"] = _integer(
            request.get("max_new_tokens"),
            "max_new_tokens",
            1,
            context.limits.max_new_tokens,
            request_id,
        )
        core["sampling_vocabulary_size"] = len(context.tokenizer)
        core["seed"] = _integer(request.get("seed"), "seed", 0, MAX_U64, request_id)
        if "temperature" in request:
            core["temperature"] = _number(
                request["temperature"], "temperature", 0.0, 100.0, request_id
            )
        if "top_k" in request:
            core["top_k"] = _integer(
                request["top_k"], "top_k", 0, len(context.tokenizer), request_id
            )
        if "top_p" in request:
            core["top_p"] = _number(
                request["top_p"], "top_p", 0.0, 1.0, request_id, minimum_exclusive=True
            )
        if "eos_token_ids" in request:
            eos = _token_ids(
                request["eos_token_ids"],
                "eos_token_ids",
                request_id,
                vocabulary_size=len(context.tokenizer),
                maximum_count=64,
            )
            if len(set(eos)) != len(eos):
                raise ProtocolError(
                    "invalid_tokens", "eos_token_ids must be unique", request_id
                )
            core["eos_token_ids"] = eos
        return core, input_ids

    if operation in ("train", "evaluate"):
        input_ids = _token_ids(
            request.get("input_ids"),
            "input_ids",
            request_id,
            vocabulary_size=len(context.tokenizer),
            maximum_count=context.limits.max_input_tokens,
        )
        core["input_ids"] = input_ids
        if "loss_mask" in request:
            mask = request["loss_mask"]
            if (
                not isinstance(mask, list)
                or len(mask) != len(input_ids)
                or any(value not in (0, 1) or isinstance(value, bool) for value in mask)
            ):
                raise ProtocolError(
                    "invalid_loss_mask",
                    "loss_mask must be a binary array with the same length as input_ids",
                    request_id,
                )
            core["loss_mask"] = mask
        if "reset_state" in request:
            if not isinstance(request["reset_state"], bool):
                raise ProtocolError(
                    "invalid_field", "reset_state must be boolean", request_id
                )
            core["reset_state"] = request["reset_state"]
        return core, input_ids

    if operation == "inspect":
        input_ids = _token_ids(
            request.get("input_ids"),
            "input_ids",
            request_id,
            vocabulary_size=len(context.tokenizer),
            maximum_count=context.limits.max_input_tokens,
        )
        probes = _token_ids(
            request.get("probe_token_ids"),
            "probe_token_ids",
            request_id,
            vocabulary_size=len(context.tokenizer),
            maximum_count=64,
        )
        if len(set(probes)) != len(probes):
            raise ProtocolError(
                "invalid_tokens", "probe_token_ids must be unique", request_id
            )
        core["input_ids"] = input_ids
        core["probe_token_ids"] = probes
        core["top_k"] = _integer(
            request.get("top_k"), "top_k", 1, len(context.tokenizer), request_id
        )
        return core, input_ids
    return core, None


def _validate_core_response(
    response: Any, request_id: str, context: FrontendContext
) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        raise ProtocolError(
            "invalid_core_response", "core response must be a JSON object", request_id
        )
    if response.get("protocol") != PROTOCOL or response.get("request_id") != request_id:
        raise ProtocolError(
            "core_correlation_error",
            "core response protocol/request_id does not match",
            request_id,
        )
    if not isinstance(response.get("ok"), bool):
        raise ProtocolError(
            "invalid_core_response", "core response requires boolean ok", request_id
        )
    result = dict(response)
    if not result["ok"]:
        error = result.get("error")
        if (
            not isinstance(error, Mapping)
            or not isinstance(error.get("code"), str)
            or not isinstance(error.get("message"), str)
        ):
            raise ProtocolError(
                "invalid_core_response", "core error response is malformed", request_id
            )
        return result
    if "latency_ms" in result:
        _number(result["latency_ms"], "latency_ms", 0.0, 86_400_000.0, request_id)
    if "spike_metrics" in result and not isinstance(result["spike_metrics"], Mapping):
        raise ProtocolError(
            "invalid_core_response", "spike_metrics must be an object", request_id
        )
    return result


def process_request(
    request: Any,
    context: FrontendContext,
    transact: Callable[[Mapping[str, Any]], Any],
) -> tuple[dict[str, Any], bool]:
    core_request, input_ids = prepare_core_request(request, context)
    request_id = core_request["request_id"]
    response = _validate_core_response(transact(core_request), request_id, context)
    should_stop = core_request["op"] == "shutdown"
    if response["ok"] and core_request["op"] == "generate":
        output_ids = _token_ids(
            response.get("output_ids"),
            "output_ids",
            request_id,
            vocabulary_size=len(context.tokenizer),
            maximum_count=context.limits.max_new_tokens,
            allow_empty=True,
        )
        if response.get("finish_reason") not in ("eos", "length"):
            raise ProtocolError(
                "invalid_core_response",
                "generate finish_reason must be eos or length",
                request_id,
            )
        decoded = context.tokenizer.decode(
            output_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(decoded, str):
            raise ProtocolError(
                "tokenizer_failure", "decode did not return text", request_id
            )
        response["input_ids"] = input_ids
        response["text"] = decoded
        response["tokenizer_fingerprint_sha256"] = context.assets.tokenizer_fingerprint
    return response, should_stop


def error_response(error: ProtocolError) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "request_id": error.request_id,
        "ok": False,
        "error": {"code": error.code, "message": str(error)},
    }


def _bounded_text_lines(
    input_stream: IO[str], maximum_line_bytes: int
) -> Iterable[tuple[str, bool]]:
    """Yield bounded text chunks and flag lines known to exceed the limit.

    Text streams bound ``readline`` in characters, so the caller still checks
    exact UTF-8 bytes.  The character cap nevertheless limits allocation to at
    most four UTF-8 bytes per character and drains an oversized physical line
    before resuming the JSONL protocol.
    """

    while True:
        line = input_stream.readline(maximum_line_bytes + 2)
        if not line:
            return
        if line.endswith("\n") or len(line) < maximum_line_bytes + 2:
            yield line, False
            continue
        while line and not line.endswith("\n"):
            line = input_stream.readline(maximum_line_bytes + 2)
        yield "", True


class CoreProcess:
    def __init__(self, command: Sequence[str], max_line_bytes: int):
        if not command or any(
            not isinstance(argument, str) or not argument for argument in command
        ):
            raise ContractError("core command must contain non-empty argv entries")
        self._process = (
            subprocess.Popen(  # noqa: S603 - argv is explicit and shell is disabled.
                list(command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                encoding="utf-8",
                errors="strict",
                bufsize=1,
                shell=False,
            )
        )
        self._max_line_bytes = max_line_bytes

    def transact(self, request: Mapping[str, Any]) -> Any:
        if self._process.stdin is None or self._process.stdout is None:
            raise ProtocolError(
                "core_io_error", "core pipes are unavailable", request.get("request_id")
            )
        if self._process.poll() is not None:
            raise ProtocolError(
                "core_exited",
                f"core exited with status {self._process.returncode}",
                request.get("request_id"),
            )
        serialized = json.dumps(
            request, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if len(serialized.encode("utf-8")) > self._max_line_bytes:
            raise ProtocolError(
                "request_too_large",
                "core request exceeds line limit",
                request.get("request_id"),
            )
        try:
            self._process.stdin.write(serialized + "\n")
            self._process.stdin.flush()
            line = self._process.stdout.readline(self._max_line_bytes + 2)
        except (BrokenPipeError, OSError, UnicodeError) as error:
            raise ProtocolError(
                "core_io_error",
                f"core transport failed: {error}",
                request.get("request_id"),
            ) from error
        if not line:
            raise ProtocolError(
                "core_exited", "core closed stdout", request.get("request_id")
            )
        if (
            not line.endswith("\n")
            or len(line.encode("utf-8")) > self._max_line_bytes + 1
        ):
            raise ProtocolError(
                "core_response_too_large",
                "core response exceeds line limit",
                request.get("request_id"),
            )
        try:
            return _strict_json_loads(line)
        except (json.JSONDecodeError, _DuplicateJsonField, ValueError) as error:
            raise ProtocolError(
                "invalid_core_response",
                f"core emitted invalid JSON: {error}",
                request.get("request_id"),
            ) from error

    def close(self) -> None:
        if self._process.stdin is not None and not self._process.stdin.closed:
            try:
                self._process.stdin.close()
            except OSError:
                pass
        if self._process.poll() is None:
            self._process.terminate()
        try:
            self._process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5.0)
        if self._process.stdout is not None and not self._process.stdout.closed:
            self._process.stdout.close()


def serve(
    input_stream: IO[str],
    output_stream: IO[str],
    context: FrontendContext,
    transact: Callable[[Mapping[str, Any]], Any],
) -> None:
    for line, known_oversized in _bounded_text_lines(
        input_stream, context.limits.max_line_bytes
    ):
        should_stop = False
        if known_oversized:
            response = error_response(
                ProtocolError("request_too_large", "input line exceeds limit")
            )
            encoded_line = b""
        else:
            try:
                encoded_line = line.encode("utf-8")
            except UnicodeEncodeError as error:
                response = error_response(
                    ProtocolError("invalid_json", f"input is not valid UTF-8: {error}")
                )
                encoded_line = b""
        payload = encoded_line[:-1] if encoded_line.endswith(b"\n") else encoded_line
        if encoded_line and len(payload) > context.limits.max_line_bytes:
            response = error_response(
                ProtocolError("request_too_large", "input line exceeds limit")
            )
        elif encoded_line:
            try:
                request = _strict_json_loads(line)
                response, should_stop = process_request(request, context, transact)
            except (json.JSONDecodeError, _DuplicateJsonField, ValueError) as error:
                response = error_response(ProtocolError("invalid_json", str(error)))
            except ProtocolError as error:
                response = error_response(error)
        output_stream.write(
            json.dumps(response, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        )
        output_stream.flush()
        if should_stop:
            break


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
    describe = subparsers.add_parser(
        "describe", description="Print the protocol contract as JSON."
    )
    describe.set_defaults(describe=True)

    server = subparsers.add_parser(
        "serve", description="Start the persistent tokenizer/core bridge."
    )
    _contract_arguments(server)
    server.add_argument("--core-executable", required=True, type=Path)
    server.add_argument("--core-arg", action="append", default=[])
    server.add_argument("--max-line-bytes", type=int, default=MAX_LINE_BYTES_DEFAULT)
    server.add_argument(
        "--max-input-tokens", type=int, default=MAX_INPUT_TOKENS_DEFAULT
    )
    server.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS_DEFAULT)
    server.add_argument("--max-text-bytes", type=int, default=MAX_TEXT_BYTES_DEFAULT)
    server.add_argument("--max-messages", type=int, default=MAX_MESSAGES_DEFAULT)
    return parser


def protocol_description() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "transport": "utf-8-json-lines",
        "request_objects_reject_unknown_and_duplicate_fields": True,
        "maximum_request_id_utf8_bytes": MAX_REQUEST_ID_BYTES,
        "operations": sorted(SUPPORTED_OPERATIONS),
        "generate": {
            "required": [
                "input_ids",
                "max_new_tokens",
                "seed",
                "sampling_vocabulary_size",
            ],
            "optional": ["temperature", "top_k", "top_p", "eos_token_ids"],
            "output_ids_semantics": "newly generated tokens only",
            "finish_reasons": ["eos", "length"],
        },
        "frontend_generate": {
            "input": "exactly one of input_ids, text, or messages",
            "messages": "optional system, alternating user/assistant, ending user",
            "enable_thinking": "optional only with messages",
        },
        "train_evaluate": {
            "required": ["input_ids"],
            "optional": ["loss_mask", "reset_state"],
            "loss_mask_semantics": "binary causal-target mask, same length as input_ids",
        },
        "inspect": {
            "required": ["input_ids", "probe_token_ids", "top_k"],
            "response": "bounded final-position probe logits and sorted top-k",
        },
        "optional_response_metrics": ["spike_metrics", "latency_ms"],
    }


def _positive_limit(value: int, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"{label} must be positive")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "describe":
        print(json.dumps(protocol_description(), indent=2, sort_keys=True))
        return 0
    core: CoreProcess | None = None
    try:
        limits = Limits(
            max_line_bytes=_positive_limit(arguments.max_line_bytes, "max-line-bytes"),
            max_input_tokens=_positive_limit(
                arguments.max_input_tokens, "max-input-tokens"
            ),
            max_new_tokens=_positive_limit(arguments.max_new_tokens, "max-new-tokens"),
            max_text_bytes=_positive_limit(arguments.max_text_bytes, "max-text-bytes"),
            max_messages=_positive_limit(arguments.max_messages, "max-messages"),
        )
        assets = verify_asset_manifest(
            arguments.assets_dir,
            arguments.model_id,
            arguments.revision,
            arguments.expected_tokenizer_fingerprint,
        )
        tokenizer = load_transformers_tokenizer(assets)
        command = [str(arguments.core_executable), *arguments.core_arg]
        core = CoreProcess(command, limits.max_line_bytes)
        serve(
            sys.stdin,
            sys.stdout,
            FrontendContext(tokenizer, assets, limits),
            core.transact,
        )
        return 0
    except (ContractError, OSError, ValueError) as error:
        print(f"chatbot_token_protocol: {error}", file=sys.stderr)
        return 2
    finally:
        if core is not None:
            core.close()


if __name__ == "__main__":
    raise SystemExit(main())
