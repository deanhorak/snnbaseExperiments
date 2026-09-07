#!/usr/bin/env python3
"""Convert a pinned OASST2 ready-tree export to canonical conversations.

The converter intentionally accepts the exact recursive schema observed in the
2023-11-05 ``ready.trees.jsonl.gz`` export.  Variable label, emoji, and
detoxify maps are bounded and validated against explicit allow-lists.  It emits
at most one complete, assistant-ended path per ``message_tree_id``.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import errno
import gzip
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
import unicodedata
import uuid
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence
from urllib.parse import urlsplit

CONVERTER_VERSION = "1.1.0"
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_KIND = "snnbase.oasst2-conversion-manifest"
CANONICAL_SERIALIZATION = "utf8-json-sort-keys-compact-lf-v1"
CONVERSATIONS_FILENAME = "conversations.jsonl"
LINEAGE_FILENAME = "lineage.jsonl"
MANIFEST_FILENAME = "conversion-manifest.json"
PROFILE_CONSERVATIVE_ZERO = "conservative-zero"
PROFILE_QUALITY05 = "quality05"
PROFILES = (PROFILE_CONSERVATIVE_ZERO, PROFILE_QUALITY05)
SELECTION_ALGORITHM = "profile-ranked-assistant-endpoint-v1"
SPLIT_ALGORITHM = "sha256-seed-bucket-v1"
SPLIT_SEED = 42
SPLIT_BUCKET_COUNT = 10_000
DEFAULT_MAX_INPUT_TOKENS = 512
LEAKAGE_AUDIT_KIND = "snnbase.chatbot-leakage-audit"
LEAKAGE_EXCLUSIONS_KIND = "snnbase.chatbot-leakage-exclusions"
LEAKAGE_REMEDIATION_ALGORITHM = "collision-components-retain-highest-priority-split-v1"

DEFAULT_MAXIMUM_COMPRESSED_BYTES = 128 * 1024 * 1024
DEFAULT_MAXIMUM_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
DEFAULT_MAXIMUM_LINE_BYTES = 4 * 1024 * 1024
DEFAULT_MAXIMUM_TREES = 100_000
DEFAULT_MAXIMUM_MESSAGES = 2_000_000
DEFAULT_MAXIMUM_TREE_MESSAGES = 10_000
DEFAULT_MAXIMUM_DEPTH = 64
DEFAULT_MAXIMUM_REPLIES = 1_024
DEFAULT_MAXIMUM_MESSAGE_CONTENT_BYTES = 1024 * 1024
DEFAULT_MAXIMUM_CONVERSATION_CONTENT_BYTES = 32 * 1024
MAXIMUM_PROVENANCE_FIELD_BYTES = 4_096
MAXIMUM_METADATA_STRING_BYTES = 512
MAXIMUM_EMOJI_KEYS = 128

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
LANGUAGE_RE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})?$")

TOP_LEVEL_FIELDS = frozenset({"message_tree_id", "origin", "prompt", "tree_state"})
MESSAGE_FIELDS = frozenset(
    {
        "message_id",
        "parent_id",
        "user_id",
        "created_date",
        "text",
        "role",
        "lang",
        "review_count",
        "review_result",
        "deleted",
        "rank",
        "synthetic",
        "model_name",
        "emojis",
        "replies",
        "labels",
        "events",
        "detoxify",
        "message_tree_id",
        "tree_state",
    }
)
LABEL_FIELDS = frozenset(
    {
        "spam",
        "fails_task",
        "lang_mismatch",
        "pii",
        "not_appropriate",
        "hate_speech",
        "sexual_content",
        "quality",
        "toxicity",
        "humor",
        "helpfulness",
        "creativity",
        "violence",
        "moral_judgement",
        "political_content",
    }
)
DETOXIFY_FIELDS = frozenset(
    {
        "toxicity",
        "severe_toxicity",
        "obscene",
        "identity_attack",
        "insult",
        "threat",
        "sexual_explicit",
    }
)
COMMON_ADVERSE_LABELS = (
    "spam",
    "lang_mismatch",
    "not_appropriate",
    "hate_speech",
    "sexual_content",
    "toxicity",
    "violence",
)
ASSISTANT_ADVERSE_LABELS = ("fails_task",)
PII_LABEL = "pii"


class Oasst2ConversionError(RuntimeError):
    """Raised when input, policy, provenance, or output fails closed."""


class _DuplicateJsonField(ValueError):
    pass


@dataclass(frozen=True)
class ConversionLimits:
    maximum_compressed_bytes: int = DEFAULT_MAXIMUM_COMPRESSED_BYTES
    maximum_uncompressed_bytes: int = DEFAULT_MAXIMUM_UNCOMPRESSED_BYTES
    maximum_line_bytes: int = DEFAULT_MAXIMUM_LINE_BYTES
    maximum_trees: int = DEFAULT_MAXIMUM_TREES
    maximum_messages: int = DEFAULT_MAXIMUM_MESSAGES
    maximum_tree_messages: int = DEFAULT_MAXIMUM_TREE_MESSAGES
    maximum_depth: int = DEFAULT_MAXIMUM_DEPTH
    maximum_replies: int = DEFAULT_MAXIMUM_REPLIES
    maximum_message_content_bytes: int = DEFAULT_MAXIMUM_MESSAGE_CONTENT_BYTES


@dataclass(frozen=True)
class ConversionPolicy:
    profile: str
    maximum_conversation_content_bytes: int = DEFAULT_MAXIMUM_CONVERSATION_CONTENT_BYTES
    maximum_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS


@dataclass(frozen=True)
class TokenizerIdentity:
    model_id: str
    revision: str
    fingerprint_sha256: str


@dataclass(frozen=True)
class LeakageRemediation:
    excluded_tree_ids: frozenset[str]
    baseline_conversion_manifest_sha256: str
    baseline_conversations_sha256: str
    baseline_conversations_size_bytes: int
    baseline_record_count: int
    audit_sha256: str
    audit_size_bytes: int
    exclusions_sha256: str
    exclusions_size_bytes: int
    semantic_model_id: str
    semantic_model_revision: str
    semantic_model_snapshot_sha256: str


@dataclass
class _ConversionState:
    tree_ids: set[str]
    message_ids: set[str]
    source_trees: int = 0
    source_messages: int = 0
    source_leaves: int = 0
    eligible_paths: int = 0
    selected_records: int = 0
    selected_messages: int = 0
    selected_content_bytes: int = 0
    selected_maximum_token_count: int = 0
    pre_remediation_records: int = 0
    leakage_exclusions_applied: int = 0
    path_filter_reasons: Counter[str] | None = None
    tree_filter_reasons: Counter[str] | None = None
    split_counts: Counter[str] | None = None
    normalized_root_prompt_digests: set[bytes] | None = None

    def __post_init__(self) -> None:
        self.path_filter_reasons = Counter()
        self.tree_filter_reasons = Counter()
        self.split_counts = Counter()
        self.normalized_root_prompt_digests = set()


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonField(f"duplicate JSON field: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _load_json_line(payload: bytes, line_number: int) -> Any:
    try:
        text = payload.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_nonstandard_constant,
        )
    except (
        UnicodeError,
        json.JSONDecodeError,
        _DuplicateJsonField,
        ValueError,
        RecursionError,
    ) as error:
        raise Oasst2ConversionError(
            f"invalid UTF-8/JSON on source line {line_number}: {error}"
        ) from error


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise Oasst2ConversionError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Oasst2ConversionError(f"{label} must be a nonnegative integer")
    return value


def _finite_unit_interval(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise Oasst2ConversionError(f"{label} must be a finite number in [0, 1]")
    return float(value)


def _utf8_bytes(value: str, label: str) -> bytes:
    try:
        return value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise Oasst2ConversionError(f"{label} is not valid UTF-8") from error


def _reject_controls(value: str, label: str, *, allow_text_whitespace: bool) -> None:
    for character in value:
        codepoint = ord(character)
        if codepoint == 0:
            raise Oasst2ConversionError(f"{label} contains NUL")
        if codepoint < 32 and not (
            allow_text_whitespace and character in ("\n", "\r", "\t")
        ):
            raise Oasst2ConversionError(f"{label} contains a control character")
        if codepoint == 127:
            raise Oasst2ConversionError(f"{label} contains a control character")


def _bounded_string(
    value: Any,
    label: str,
    *,
    maximum_bytes: int,
    allow_text_whitespace: bool = False,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or (not value and not allow_empty):
        raise Oasst2ConversionError(f"{label} must be a non-empty string")
    _reject_controls(value, label, allow_text_whitespace=allow_text_whitespace)
    if len(_utf8_bytes(value, label)) > maximum_bytes:
        raise Oasst2ConversionError(f"{label} exceeds its UTF-8 byte limit")
    return value


def _bounded_message_text(value: Any, label: str, maximum_bytes: int) -> str:
    """Validate encoding/size while leaving content controls to path policy."""

    if not isinstance(value, str) or not value:
        raise Oasst2ConversionError(f"{label} must be a non-empty string")
    if len(_utf8_bytes(value, label)) > maximum_bytes:
        raise Oasst2ConversionError(f"{label} exceeds its UTF-8 byte limit")
    if not value.strip():
        raise Oasst2ConversionError(f"{label} must contain non-whitespace text")
    return value


def _contains_disallowed_control(value: str) -> bool:
    allowed = {"\t", "\n", "\r"}
    return any(
        character not in allowed and unicodedata.category(character) == "Cc"
        for character in value
    )


def _canonical_uuid(value: Any, label: str) -> str:
    text = _bounded_string(value, label, maximum_bytes=36, allow_text_whitespace=False)
    try:
        parsed = uuid.UUID(text)
    except (ValueError, AttributeError) as error:
        raise Oasst2ConversionError(f"{label} must be a UUID") from error
    if str(parsed) != text:
        raise Oasst2ConversionError(f"{label} must be a canonical lowercase UUID")
    return text


def _validate_created_date(value: Any, label: str) -> None:
    text = _bounded_string(value, label, maximum_bytes=MAXIMUM_METADATA_STRING_BYTES)
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise Oasst2ConversionError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Oasst2ConversionError(f"{label} must contain a timezone")


def _validate_labels(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise Oasst2ConversionError(f"{label} must be an object")
    unknown = set(value) - LABEL_FIELDS
    if unknown:
        raise Oasst2ConversionError(
            f"{label} contains unsupported fields: {sorted(unknown)}"
        )
    for name, raw_record in value.items():
        if type(raw_record) is not dict or set(raw_record) != {"value", "count"}:
            raise Oasst2ConversionError(
                f"{label}.{name} must contain exactly value and count"
            )
        _finite_unit_interval(raw_record["value"], f"{label}.{name}.value")
        _positive_integer(raw_record["count"], f"{label}.{name}.count")
    return value


def _validate_emojis(value: Any, label: str) -> None:
    if value is None:
        return
    if type(value) is not dict or len(value) > MAXIMUM_EMOJI_KEYS:
        raise Oasst2ConversionError(f"{label} must be a bounded object or null")
    for name, count in value.items():
        _bounded_string(name, f"{label} key", maximum_bytes=128)
        _nonnegative_integer(count, f"{label}.{name}")


def _validate_detoxify(value: Any, label: str) -> None:
    if value is None:
        return
    if type(value) is not dict or set(value) != DETOXIFY_FIELDS:
        raise Oasst2ConversionError(
            f"{label} must be null or contain the exact detoxify score fields"
        )
    for name, score in value.items():
        _finite_unit_interval(score, f"{label}.{name}")


def _validate_message_metadata(
    message: dict[str, Any], label: str, limits: ConversionLimits
) -> None:
    if set(message) != MESSAGE_FIELDS:
        missing = sorted(MESSAGE_FIELDS - set(message))
        extra = sorted(set(message) - MESSAGE_FIELDS)
        raise Oasst2ConversionError(
            f"{label} has the wrong fields; missing={missing}, extra={extra}"
        )
    _canonical_uuid(message["message_id"], f"{label}.message_id")
    _canonical_uuid(message["user_id"], f"{label}.user_id")
    _validate_created_date(message["created_date"], f"{label}.created_date")
    _bounded_message_text(
        message["text"],
        f"{label}.text",
        limits.maximum_message_content_bytes,
    )
    if message["role"] not in ("prompter", "assistant"):
        raise Oasst2ConversionError(f"{label}.role is unsupported")
    language = _bounded_string(message["lang"], f"{label}.lang", maximum_bytes=32)
    if LANGUAGE_RE.fullmatch(language) is None:
        raise Oasst2ConversionError(f"{label}.lang is malformed")
    _nonnegative_integer(message["review_count"], f"{label}.review_count")
    if message["review_result"] is not None and not isinstance(
        message["review_result"], bool
    ):
        raise Oasst2ConversionError(f"{label}.review_result must be boolean or null")
    if not isinstance(message["deleted"], bool):
        raise Oasst2ConversionError(f"{label}.deleted must be boolean")
    rank = message["rank"]
    if rank is not None:
        _nonnegative_integer(rank, f"{label}.rank")
    if message["role"] == "prompter" and rank is not None:
        raise Oasst2ConversionError(f"{label}.rank must be null for a prompter")
    if not isinstance(message["synthetic"], bool):
        raise Oasst2ConversionError(f"{label}.synthetic must be boolean")
    if message["model_name"] is not None:
        raise Oasst2ConversionError(f"{label}.model_name must be null")
    _validate_emojis(message["emojis"], f"{label}.emojis")
    if type(message["replies"]) is not list:
        raise Oasst2ConversionError(f"{label}.replies must be an array")
    if len(message["replies"]) > limits.maximum_replies:
        raise Oasst2ConversionError(f"{label}.replies exceeds the reply limit")
    _validate_labels(message["labels"], f"{label}.labels")
    if message["events"] is not None:
        raise Oasst2ConversionError(f"{label}.events must be null")
    _validate_detoxify(message["detoxify"], f"{label}.detoxify")
    if message["message_tree_id"] is not None or message["tree_state"] is not None:
        raise Oasst2ConversionError(
            f"{label} nested message_tree_id/tree_state must be null"
        )


def _validate_tree(
    value: Any,
    line_number: int,
    state: _ConversionState,
    limits: ConversionLimits,
) -> tuple[str, dict[str, Any], int, int]:
    if type(value) is not dict or set(value) != TOP_LEVEL_FIELDS:
        raise Oasst2ConversionError(
            f"source line {line_number} must contain exactly "
            "message_tree_id, origin, prompt, and tree_state"
        )
    tree_id = _canonical_uuid(
        value["message_tree_id"], f"source line {line_number}.message_tree_id"
    )
    if tree_id in state.tree_ids:
        raise Oasst2ConversionError(f"duplicate message_tree_id: {tree_id}")
    state.tree_ids.add(tree_id)
    if value["tree_state"] != "ready_for_export" or value["origin"] is not None:
        raise Oasst2ConversionError(
            f"tree {tree_id} must have tree_state=ready_for_export and origin=null"
        )
    prompt = value["prompt"]
    if type(prompt) is not dict:
        raise Oasst2ConversionError(f"tree {tree_id}.prompt must be an object")

    local_messages = 0
    local_leaves = 0
    stack: list[tuple[dict[str, Any], str | None, str, int]] = [
        (prompt, None, "prompter", 1)
    ]
    while stack:
        message, expected_parent, expected_role, depth = stack.pop()
        if depth > limits.maximum_depth:
            raise Oasst2ConversionError(f"tree {tree_id} exceeds maximum depth")
        if type(message) is not dict:
            raise Oasst2ConversionError(f"tree {tree_id} contains a non-object message")
        local_messages += 1
        state.source_messages += 1
        if local_messages > limits.maximum_tree_messages:
            raise Oasst2ConversionError(f"tree {tree_id} exceeds its message limit")
        if state.source_messages > limits.maximum_messages:
            raise Oasst2ConversionError("source exceeds maximum_messages")
        label = f"tree {tree_id} message {local_messages}"
        _validate_message_metadata(message, label, limits)
        message_id = message["message_id"]
        if message_id in state.message_ids:
            raise Oasst2ConversionError(f"duplicate message_id: {message_id}")
        state.message_ids.add(message_id)
        if message["parent_id"] != expected_parent:
            raise Oasst2ConversionError(
                f"message {message_id} does not point to its immediate parent"
            )
        if message["role"] != expected_role:
            raise Oasst2ConversionError(
                f"message {message_id} violates prompter/assistant alternation"
            )
        replies = message["replies"]
        if not replies:
            local_leaves += 1
            state.source_leaves += 1
        next_role = "assistant" if expected_role == "prompter" else "prompter"
        for child in reversed(replies):
            if type(child) is not dict:
                raise Oasst2ConversionError(
                    f"message {message_id} contains a non-object reply"
                )
            stack.append((child, message_id, next_role, depth + 1))

    if prompt["message_id"] != tree_id or prompt["parent_id"] is not None:
        raise Oasst2ConversionError(
            f"tree {tree_id} prompt identity/parent does not match the tree"
        )
    return tree_id, prompt, local_messages, local_leaves


def _conservative_zero_filter_reason(
    message: Mapping[str, Any], root_language: str
) -> str | None:
    if _contains_disallowed_control(message["text"]):
        return "disallowed_control"
    if message["deleted"]:
        return "message_deleted"
    if message["synthetic"]:
        return "message_synthetic"
    if message["lang"] != root_language:
        return "message_language_mismatch"
    if message["review_result"] is not True or message["review_count"] <= 0:
        return "message_not_reviewed"
    labels = message["labels"]
    if PII_LABEL not in labels:
        return "missing_label_pii"
    if float(labels[PII_LABEL]["value"]) > 0.0:
        return "adverse_label_pii"
    required = list(COMMON_ADVERSE_LABELS)
    if message["role"] == "assistant":
        required.extend(ASSISTANT_ADVERSE_LABELS)
    for name in required:
        if name not in labels:
            return f"missing_label_{name}"
        if float(labels[name]["value"]) > 0.0:
            return f"adverse_label_{name}"
    # A prompter does not require fails_task, but a present adverse vote still
    # fails closed rather than being silently ignored.
    if (
        message["role"] == "prompter"
        and "fails_task" in labels
        and float(labels["fails_task"]["value"]) > 0.0
    ):
        return "adverse_label_fails_task"
    return None


def _quality05_filter_reason(
    message: Mapping[str, Any], root_language: str
) -> str | None:
    if _contains_disallowed_control(message["text"]):
        return "disallowed_control"
    if message["deleted"]:
        return "message_deleted"
    if message["synthetic"]:
        return "message_synthetic"
    if message["lang"] != root_language:
        return "message_language_mismatch"
    if message["review_result"] is not True or message["review_count"] <= 0:
        return "message_not_reviewed"
    labels = message["labels"]
    for name in ("spam", "lang_mismatch", "pii"):
        if name in labels and float(labels[name]["value"]) >= 0.5:
            return f"adverse_label_{name}"
    if message["role"] == "assistant":
        for name in ("quality", "helpfulness"):
            if name in labels and float(labels[name]["value"]) < 0.5:
                return f"insufficient_label_{name}"
        if "fails_task" in labels and float(labels["fails_task"]["value"]) >= 0.5:
            return "adverse_label_fails_task"
    return None


def _message_filter_reason(
    message: Mapping[str, Any], root_language: str, profile: str
) -> str | None:
    if profile == PROFILE_CONSERVATIVE_ZERO:
        return _conservative_zero_filter_reason(message, root_language)
    if profile == PROFILE_QUALITY05:
        return _quality05_filter_reason(message, root_language)
    raise AssertionError(f"unsupported validated profile: {profile}")


def _edge_selection_key(message: Mapping[str, Any]) -> tuple[Any, ...]:
    if message["role"] == "prompter":
        return (False, 0, message["message_id"])
    rank = message["rank"]
    return (rank is None, rank if rank is not None else 0, message["message_id"])


def _path_selection_key(path: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
    return (
        -sum(message["role"] == "assistant" for message in path),
        tuple(_edge_selection_key(message) for message in path[1:]),
        tuple(message["message_id"] for message in path),
    )


def _token_ids(tokenizer: Any, path: Sequence[Mapping[str, Any]]) -> list[int]:
    messages = [
        {
            "role": "user" if message["role"] == "prompter" else "assistant",
            "content": message["text"],
        }
        for message in path
    ]
    try:
        value = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=True,
        )
    except Exception as error:
        raise Oasst2ConversionError(
            f"verified tokenizer could not render a candidate path: {error}"
        ) from error
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in value
        )
    ):
        raise Oasst2ConversionError(
            "verified tokenizer returned a non-flat/nonnegative token-id array"
        )
    return value


def _select_path(
    prompt: dict[str, Any],
    policy: ConversionPolicy,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]] | None, int, int, Counter[str]]:
    candidates: list[tuple[list[dict[str, Any]], int]] = []
    reasons: Counter[str] = Counter()
    root_language = prompt["lang"]
    stack: list[tuple[dict[str, Any], list[dict[str, Any]], str | None]] = [
        (prompt, [], None)
    ]
    while stack:
        message, prefix, prior_reason = stack.pop()
        path = [*prefix, message]
        reason = prior_reason or _message_filter_reason(
            message, root_language, policy.profile
        )
        replies = message["replies"]
        if message["role"] == "assistant":
            endpoint_reason = reason
            if endpoint_reason is None:
                content_bytes = sum(
                    len(_utf8_bytes(item["text"], "selected message text"))
                    for item in path
                )
                if content_bytes > policy.maximum_conversation_content_bytes:
                    endpoint_reason = "conversation_content_bytes_exceeded"
            token_count = 0
            if endpoint_reason is None:
                token_count = len(_token_ids(tokenizer, path))
                if token_count > policy.maximum_input_tokens:
                    endpoint_reason = "maximum_input_tokens_exceeded"
            if endpoint_reason is None:
                candidates.append((path, token_count))
            else:
                reasons[endpoint_reason] += 1
        for child in reversed(replies):
            stack.append((child, path, reason))
    if not candidates:
        return None, 0, 0, reasons
    candidates.sort(key=lambda candidate: _path_selection_key(candidate[0]))
    selected_path, selected_tokens = candidates[0]
    return selected_path, selected_tokens, len(candidates), reasons


def _split_name(identifier: str) -> str:
    payload = SPLIT_SEED.to_bytes(8, "big") + identifier.encode("utf-8")
    bucket = (
        int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % SPLIT_BUCKET_COUNT
    )
    if bucket <= 999:
        return "test"
    if bucket <= 1999:
        return "validation"
    return "train"


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_and_hash_source(
    path: Path, maximum_bytes: int
) -> tuple[Path, BinaryIO, os.stat_result, str]:
    """Open once, hash that descriptor, rewind it, and retain it for gzip."""

    source = path.absolute()
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise Oasst2ConversionError(
            f"could not securely open source: {error}"
        ) from error
    stream: BinaryIO | None = None
    try:
        snapshot = os.fstat(descriptor)
        if not stat.S_ISREG(snapshot.st_mode):
            raise Oasst2ConversionError("source must be a regular file")
        if snapshot.st_size <= 0 or snapshot.st_size > maximum_bytes:
            raise Oasst2ConversionError(
                "compressed source size is outside the configured limit: "
                f"{snapshot.st_size}"
            )
        stream = os.fdopen(descriptor, "rb", closefd=True)
        descriptor = -1
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        stream.seek(0)
        return source, stream, snapshot, digest.hexdigest()
    except BaseException:
        if stream is not None:
            stream.close()
        elif descriptor >= 0:
            os.close(descriptor)
        raise


def _verify_source_unchanged(
    source: Path, stream: BinaryIO, snapshot: os.stat_result
) -> None:
    try:
        descriptor_state = os.fstat(stream.fileno())
        path_state = os.stat(source, follow_symlinks=False)
    except OSError as error:
        raise Oasst2ConversionError(
            "source path changed or disappeared during conversion"
        ) from error
    expected = _stat_identity(snapshot)
    if (
        not stat.S_ISREG(path_state.st_mode)
        or _stat_identity(descriptor_state) != expected
        or _stat_identity(path_state) != expected
    ):
        raise Oasst2ConversionError("source changed during conversion")


def _hash_converter() -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with Path(__file__).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
    except OSError as error:
        raise Oasst2ConversionError(
            f"could not hash the converter implementation: {error}"
        ) from error
    if size <= 0:
        raise Oasst2ConversionError("converter implementation is empty")
    return digest.hexdigest(), size


def _validate_limits(limits: ConversionLimits) -> None:
    for name, value in vars(limits).items():
        _positive_integer(value, name)


def _validate_policy(policy: ConversionPolicy) -> None:
    if policy.profile not in PROFILES:
        raise Oasst2ConversionError(f"profile must be one of: {', '.join(PROFILES)}")
    _positive_integer(
        policy.maximum_conversation_content_bytes,
        "maximum_conversation_content_bytes",
    )
    _positive_integer(policy.maximum_input_tokens, "maximum_input_tokens")


def _validate_tokenizer_identity(identity: TokenizerIdentity) -> None:
    _bounded_string(identity.model_id, "tokenizer model_id", maximum_bytes=512)
    if REVISION_RE.fullmatch(identity.revision) is None:
        raise Oasst2ConversionError(
            "tokenizer revision must be an explicit lowercase commit SHA"
        )
    if SHA256_RE.fullmatch(identity.fingerprint_sha256) is None:
        raise Oasst2ConversionError(
            "tokenizer fingerprint must be an explicit lowercase SHA-256"
        )


def _provenance_string(value: str, label: str) -> str:
    return _bounded_string(value, label, maximum_bytes=MAXIMUM_PROVENANCE_FIELD_BYTES)


def _source_uri(value: str) -> str:
    uri = _provenance_string(value, "source_uri")
    try:
        parsed = urlsplit(uri)
    except ValueError as error:
        raise Oasst2ConversionError("source_uri is malformed") from error
    if not parsed.scheme or parsed.scheme.lower() == "file":
        raise Oasst2ConversionError(
            "source_uri must be an absolute, non-file provenance URI"
        )
    if parsed.scheme.lower() in ("http", "https") and not parsed.netloc:
        raise Oasst2ConversionError("HTTP source_uri must include an authority")
    return uri


def _immutable_source_version(value: str) -> str:
    version = _provenance_string(value, "source_version")
    if REVISION_RE.fullmatch(version) is None:
        raise Oasst2ConversionError(
            "source_version must be an immutable lowercase 40-character commit SHA"
        )
    return version


def _read_bounded_json_file(
    path: Path, label: str, maximum_bytes: int = MAXIMUM_PROVENANCE_FIELD_BYTES * 1024
) -> tuple[dict[str, Any], str, int]:
    source = path.absolute()
    try:
        metadata = source.stat(follow_symlinks=False)
    except OSError as error:
        raise Oasst2ConversionError(f"could not stat {label}: {error}") from error
    if (
        source.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size <= 0
        or metadata.st_size > maximum_bytes
    ):
        raise Oasst2ConversionError(f"{label} must be a bounded regular file")
    try:
        payload = source.read_bytes()
    except OSError as error:
        raise Oasst2ConversionError(f"could not read {label}: {error}") from error
    if len(payload) != metadata.st_size:
        raise Oasst2ConversionError(f"{label} changed while it was read")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_nonstandard_constant,
        )
    except (
        UnicodeError,
        json.JSONDecodeError,
        _DuplicateJsonField,
        ValueError,
        RecursionError,
    ) as error:
        raise Oasst2ConversionError(f"{label} is not strict JSON: {error}") from error
    if type(value) is not dict:
        raise Oasst2ConversionError(f"{label} must contain one JSON object")
    return value, hashlib.sha256(payload).hexdigest(), len(payload)


def load_leakage_remediation(
    audit_path: Path,
    exclusions_path: Path,
    expected_audit_sha256: str | None = None,
    expected_exclusions_sha256: str | None = None,
) -> LeakageRemediation:
    """Validate and bind a completed leakage audit and its exclusion decision."""

    audit, audit_sha256, audit_size = _read_bounded_json_file(
        audit_path, "leakage audit"
    )
    exclusions, exclusions_sha256, exclusions_size = _read_bounded_json_file(
        exclusions_path, "leakage exclusions"
    )
    if expected_audit_sha256 is not None:
        if (
            SHA256_RE.fullmatch(expected_audit_sha256) is None
            or audit_sha256 != expected_audit_sha256
        ):
            raise Oasst2ConversionError("leakage audit does not match expected SHA-256")
    if expected_exclusions_sha256 is not None:
        if (
            SHA256_RE.fullmatch(expected_exclusions_sha256) is None
            or exclusions_sha256 != expected_exclusions_sha256
        ):
            raise Oasst2ConversionError(
                "leakage exclusions do not match expected SHA-256"
            )
    expected_audit_fields = {
        "schema_version",
        "kind",
        "status",
        "audit_version",
        "implementation",
        "scope",
        "input",
        "split",
        "semantic_model",
        "thresholds",
        "results",
        "remediation",
        "limitations",
    }
    if set(audit) != expected_audit_fields:
        raise Oasst2ConversionError("leakage audit has unexpected top-level fields")
    if (
        audit["schema_version"] != 1
        or audit["kind"] != LEAKAGE_AUDIT_KIND
        or audit["status"] != "remediation_required"
        or audit["audit_version"] != "1.0.0"
    ):
        raise Oasst2ConversionError("leakage audit identity/status is unsupported")
    implementation = audit.get("implementation")
    if type(implementation) is not dict or set(implementation) != {
        "filename",
        "version",
        "sha256",
        "size_bytes",
    }:
        raise Oasst2ConversionError("leakage audit implementation is malformed")
    if (
        implementation.get("filename") != "chatbot_leakage_audit.py"
        or implementation.get("version") != audit["audit_version"]
        or SHA256_RE.fullmatch(str(implementation.get("sha256", ""))) is None
    ):
        raise Oasst2ConversionError("leakage audit implementation is invalid")
    _positive_integer(
        implementation.get("size_bytes"), "leakage audit implementation size_bytes"
    )
    if (
        set(exclusions)
        != {
            "schema_version",
            "kind",
            "algorithm",
            "conversion_manifest_sha256",
            "tree_ids",
        }
        or exclusions.get("schema_version") != 1
    ):
        raise Oasst2ConversionError("leakage exclusions have an unsupported schema")
    if (
        exclusions.get("kind") != LEAKAGE_EXCLUSIONS_KIND
        or exclusions.get("algorithm") != LEAKAGE_REMEDIATION_ALGORITHM
    ):
        raise Oasst2ConversionError(
            "leakage exclusion identity/algorithm is unsupported"
        )
    tree_ids = exclusions.get("tree_ids")
    if (
        type(tree_ids) is not list
        or not tree_ids
        or len(tree_ids) > DEFAULT_MAXIMUM_TREES
    ):
        raise Oasst2ConversionError("leakage exclusions tree_ids are invalid")
    validated_ids = [
        _canonical_uuid(value, f"leakage exclusions tree_ids[{index}]")
        for index, value in enumerate(tree_ids)
    ]
    if validated_ids != sorted(validated_ids) or len(set(validated_ids)) != len(
        validated_ids
    ):
        raise Oasst2ConversionError(
            "leakage exclusions tree_ids must be sorted and unique"
        )
    input_record = audit.get("input")
    results = audit.get("results")
    remediation = audit.get("remediation")
    semantic_model = audit.get("semantic_model")
    scope = audit.get("scope")
    if not all(
        type(value) is dict
        for value in (input_record, results, remediation, semantic_model, scope)
    ):
        raise Oasst2ConversionError("leakage audit contains malformed records")
    if (
        scope.get("cross_split_only") is not True
        or scope.get("raw_text_in_report") is not False
    ):
        raise Oasst2ConversionError("leakage audit scope is unsupported")
    manifest_input = input_record.get("conversion_manifest")
    conversations_input = input_record.get("canonical_conversations")
    if type(manifest_input) is not dict or type(conversations_input) is not dict:
        raise Oasst2ConversionError("leakage audit input binding is malformed")
    baseline_manifest_sha256 = str(manifest_input.get("sha256", ""))
    baseline_conversations_sha256 = str(conversations_input.get("sha256", ""))
    if (
        SHA256_RE.fullmatch(baseline_manifest_sha256) is None
        or SHA256_RE.fullmatch(baseline_conversations_sha256) is None
        or exclusions.get("conversion_manifest_sha256") != baseline_manifest_sha256
    ):
        raise Oasst2ConversionError("leakage audit input digests do not bind")
    baseline_size = _positive_integer(
        conversations_input.get("size_bytes"),
        "leakage audit canonical conversations size_bytes",
    )
    baseline_count = _positive_integer(
        conversations_input.get("record_count"),
        "leakage audit canonical conversations record_count",
    )
    if (
        results.get("recommended_exclusion_count") != len(validated_ids)
        or not isinstance(results.get("high_confidence_cross_split_pair_count"), int)
        or results["high_confidence_cross_split_pair_count"] <= 0
        or remediation.get("algorithm") != LEAKAGE_REMEDIATION_ALGORITHM
        or remediation.get("split_priority") != ["test", "validation", "train"]
    ):
        raise Oasst2ConversionError("leakage remediation counts/policy do not bind")
    ids_digest = hashlib.sha256(
        ("\n".join(validated_ids) + "\n").encode("utf-8")
    ).hexdigest()
    if remediation.get("exclusions_sha256") != ids_digest:
        raise Oasst2ConversionError("leakage exclusion IDs do not match the audit")
    model_id = _bounded_string(
        semantic_model.get("model_id"),
        "leakage semantic model_id",
        maximum_bytes=512,
    )
    model_revision = str(semantic_model.get("revision", ""))
    model_snapshot_sha256 = str(semantic_model.get("snapshot_sha256", ""))
    if (
        REVISION_RE.fullmatch(model_revision) is None
        or SHA256_RE.fullmatch(model_snapshot_sha256) is None
    ):
        raise Oasst2ConversionError("leakage semantic model binding is invalid")
    return LeakageRemediation(
        excluded_tree_ids=frozenset(validated_ids),
        baseline_conversion_manifest_sha256=baseline_manifest_sha256,
        baseline_conversations_sha256=baseline_conversations_sha256,
        baseline_conversations_size_bytes=baseline_size,
        baseline_record_count=baseline_count,
        audit_sha256=audit_sha256,
        audit_size_bytes=audit_size,
        exclusions_sha256=exclusions_sha256,
        exclusions_size_bytes=exclusions_size,
        semantic_model_id=model_id,
        semantic_model_revision=model_revision,
        semantic_model_snapshot_sha256=model_snapshot_sha256,
    )


def _normalized_prompt_digest(text: str, maximum_bytes: int) -> bytes:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    encoded = _utf8_bytes(normalized, "normalized root prompt")
    if not encoded or len(encoded) > maximum_bytes:
        raise Oasst2ConversionError(
            "normalized root prompt is empty or exceeds its configured byte bound"
        )
    return hashlib.sha256(encoded).digest()


def _prepare_output_directory(path: Path) -> tuple[Path, Path]:
    destination = path.absolute()
    if os.path.lexists(destination):
        raise Oasst2ConversionError(
            f"output directory already exists and is immutable: {path}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent = destination.parent.resolve(strict=True)
    except OSError as error:
        raise Oasst2ConversionError("could not resolve output parent") from error
    if not parent.is_dir():
        raise Oasst2ConversionError("output parent is not a directory")
    destination = parent / destination.name
    try:
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
    except OSError as error:
        raise Oasst2ConversionError(
            f"could not create output staging directory: {error}"
        ) from error
    return destination, staging


def _canonical_json_line(value: Mapping[str, Any]) -> bytes:
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


def _write_canonical_file(path: Path, value: Mapping[str, Any]) -> None:
    encoded = _canonical_json_line(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o444)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _cleanup_staging(staging: Path) -> None:
    if not staging.exists():
        return
    try:
        os.chmod(staging, 0o700)
    except OSError:
        pass
    for name in (CONVERSATIONS_FILENAME, LINEAGE_FILENAME, MANIFEST_FILENAME):
        try:
            (staging / name).unlink(missing_ok=True)
        except OSError:
            pass
    try:
        staging.rmdir()
    except OSError:
        pass


def _publish_directory_exclusive(staging: Path, destination: Path) -> None:
    """Atomically rename a staged directory without replacing any destination."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise Oasst2ConversionError(
            "atomic no-replace directory publication is unavailable"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(staging),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in (errno.EEXIST, errno.ENOTEMPTY):
            raise Oasst2ConversionError(
                "output directory already exists and is immutable"
            )
        raise Oasst2ConversionError(
            "could not atomically publish output directory: "
            f"{os.strerror(error_number)}"
        )
    parent_descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _limits_audit(limits: ConversionLimits) -> dict[str, int]:
    return dict(vars(limits))


def convert_oasst2(
    *,
    source_path: Path,
    output_directory: Path,
    expected_source_sha256: str,
    expected_source_bytes: int,
    source_uri: str,
    source_version: str,
    source_license: str,
    license_reviewed: bool,
    policy_reviewed: bool,
    tokenizer: Any,
    tokenizer_identity: TokenizerIdentity,
    policy: ConversionPolicy,
    leakage_remediation: LeakageRemediation | None = None,
    limits: ConversionLimits = ConversionLimits(),
) -> dict[str, Any]:
    """Convert a pinned gzip JSONL source into one immutable directory."""

    _validate_limits(limits)
    _validate_policy(policy)
    _validate_tokenizer_identity(tokenizer_identity)
    if tokenizer is None or not callable(
        getattr(tokenizer, "apply_chat_template", None)
    ):
        raise Oasst2ConversionError("a verified chat-template tokenizer is required")
    if not isinstance(license_reviewed, bool) or not isinstance(policy_reviewed, bool):
        raise Oasst2ConversionError("review provenance flags must be boolean")
    if not license_reviewed or not policy_reviewed:
        raise Oasst2ConversionError(
            "--license-reviewed and --policy-reviewed attestations are required"
        )
    if (
        not isinstance(expected_source_sha256, str)
        or SHA256_RE.fullmatch(expected_source_sha256) is None
    ):
        raise Oasst2ConversionError("expected_source_sha256 must be lowercase SHA-256")
    _positive_integer(expected_source_bytes, "expected_source_bytes")
    if expected_source_bytes > limits.maximum_compressed_bytes:
        raise Oasst2ConversionError(
            "expected_source_bytes exceeds maximum_compressed_bytes"
        )
    source_uri = _source_uri(source_uri)
    source_version = _immutable_source_version(source_version)
    source_license = _provenance_string(source_license, "source_license")
    destination, staging = _prepare_output_directory(output_directory)
    conversations_path = staging / CONVERSATIONS_FILENAME
    lineage_path = staging / LINEAGE_FILENAME
    manifest_path = staging / MANIFEST_FILENAME
    converter_sha256, converter_size = _hash_converter()
    output_digest = hashlib.sha256()
    pre_remediation_digest = hashlib.sha256()
    lineage_digest = hashlib.sha256()
    uncompressed_digest = hashlib.sha256()
    output_bytes = 0
    pre_remediation_bytes = 0
    lineage_bytes = 0
    uncompressed_bytes = 0
    state = _ConversionState(set(), set())
    applied_exclusion_ids: set[str] = set()
    compressed_stream: BinaryIO | None = None
    try:
        source, compressed_stream, source_snapshot, source_sha256 = (
            _open_and_hash_source(source_path, limits.maximum_compressed_bytes)
        )
        source_size = source_snapshot.st_size
        if (
            source_size != expected_source_bytes
            or source_sha256 != expected_source_sha256
        ):
            raise Oasst2ConversionError(
                "source does not match the explicit expected byte count/SHA-256"
            )
        output_descriptor = os.open(
            conversations_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        try:
            lineage_descriptor = os.open(
                lineage_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
        except BaseException:
            os.close(output_descriptor)
            raise
        with os.fdopen(
            output_descriptor, "wb", closefd=True
        ) as output_stream, os.fdopen(
            lineage_descriptor, "wb", closefd=True
        ) as lineage_stream:
            try:
                with gzip.GzipFile(
                    fileobj=compressed_stream, mode="rb"
                ) as decoded_stream:
                    while True:
                        encoded = decoded_stream.readline(limits.maximum_line_bytes + 2)
                        if not encoded:
                            break
                        uncompressed_bytes += len(encoded)
                        uncompressed_digest.update(encoded)
                        if uncompressed_bytes > limits.maximum_uncompressed_bytes:
                            raise Oasst2ConversionError(
                                "gzip content exceeds maximum_uncompressed_bytes"
                            )
                        line_number = state.source_trees + 1
                        if not encoded.endswith(b"\n"):
                            if len(encoded) > limits.maximum_line_bytes:
                                raise Oasst2ConversionError(
                                    f"source line {line_number} exceeds "
                                    "maximum_line_bytes"
                                )
                            raise Oasst2ConversionError(
                                f"source line {line_number} is not newline terminated"
                            )
                        payload = encoded[:-1]
                        if len(payload) > limits.maximum_line_bytes:
                            raise Oasst2ConversionError(
                                f"source line {line_number} exceeds maximum_line_bytes"
                            )
                        if payload.endswith(b"\r"):
                            payload = payload[:-1]
                        if not payload:
                            raise Oasst2ConversionError(
                                f"source line {line_number} is blank"
                            )
                        state.source_trees += 1
                        if state.source_trees > limits.maximum_trees:
                            raise Oasst2ConversionError("source exceeds maximum_trees")
                        raw_tree = _load_json_line(payload, line_number)
                        tree_id, prompt, _, _ = _validate_tree(
                            raw_tree, line_number, state, limits
                        )
                        path, token_count, eligible_count, local_reasons = _select_path(
                            prompt, policy, tokenizer
                        )
                        state.eligible_paths += eligible_count
                        assert state.path_filter_reasons is not None
                        state.path_filter_reasons.update(local_reasons)
                        if path is None:
                            assert state.tree_filter_reasons is not None
                            if local_reasons:
                                primary = min(
                                    local_reasons,
                                    key=lambda reason: (-local_reasons[reason], reason),
                                )
                            else:
                                primary = "no_eligible_assistant_endpoint"
                            state.tree_filter_reasons[primary] += 1
                            continue
                        prompt_digest = _normalized_prompt_digest(
                            prompt["text"], limits.maximum_message_content_bytes
                        )
                        assert state.normalized_root_prompt_digests is not None
                        if prompt_digest in state.normalized_root_prompt_digests:
                            assert state.tree_filter_reasons is not None
                            state.tree_filter_reasons[
                                "duplicate_normalized_root_prompt"
                            ] += 1
                            continue
                        state.normalized_root_prompt_digests.add(prompt_digest)
                        conversation = {
                            "id": tree_id,
                            "messages": [
                                {
                                    "role": (
                                        "user"
                                        if message["role"] == "prompter"
                                        else "assistant"
                                    ),
                                    "content": message["text"],
                                }
                                for message in path
                            ],
                        }
                        line = _canonical_json_line(conversation)
                        pre_remediation_digest.update(line)
                        pre_remediation_bytes += len(line)
                        state.pre_remediation_records += 1
                        if (
                            leakage_remediation is not None
                            and tree_id in leakage_remediation.excluded_tree_ids
                        ):
                            applied_exclusion_ids.add(tree_id)
                            state.leakage_exclusions_applied += 1
                            assert state.tree_filter_reasons is not None
                            state.tree_filter_reasons[
                                "cross_split_near_duplicate_leakage"
                            ] += 1
                            continue
                        output_stream.write(line)
                        output_digest.update(line)
                        output_bytes += len(line)
                        lineage_record = {
                            "message_ids": [message["message_id"] for message in path],
                            "tree_id": tree_id,
                        }
                        lineage_line = _canonical_json_line(lineage_record)
                        lineage_stream.write(lineage_line)
                        lineage_digest.update(lineage_line)
                        lineage_bytes += len(lineage_line)
                        state.selected_records += 1
                        state.selected_messages += len(path)
                        state.selected_content_bytes += sum(
                            len(_utf8_bytes(message["text"], "selected text"))
                            for message in path
                        )
                        state.selected_maximum_token_count = max(
                            state.selected_maximum_token_count, token_count
                        )
                        assert state.split_counts is not None
                        state.split_counts[_split_name(tree_id)] += 1
            except (gzip.BadGzipFile, EOFError, zlib.error, OSError) as error:
                raise Oasst2ConversionError(
                    f"could not decode gzip source: {error}"
                ) from error
            assert compressed_stream is not None
            _verify_source_unchanged(source, compressed_stream, source_snapshot)
            output_stream.flush()
            os.fsync(output_stream.fileno())
            lineage_stream.flush()
            os.fsync(lineage_stream.fileno())
        if compressed_stream is not None:
            compressed_stream.close()
            compressed_stream = None
        if leakage_remediation is not None:
            if applied_exclusion_ids != set(leakage_remediation.excluded_tree_ids):
                missing = sorted(
                    set(leakage_remediation.excluded_tree_ids) - applied_exclusion_ids
                )
                raise Oasst2ConversionError(
                    "leakage exclusions did not match the selected baseline IDs: "
                    f"{missing[:3]}"
                )
            if (
                pre_remediation_digest.hexdigest()
                != leakage_remediation.baseline_conversations_sha256
                or pre_remediation_bytes
                != leakage_remediation.baseline_conversations_size_bytes
                or state.pre_remediation_records
                != leakage_remediation.baseline_record_count
            ):
                raise Oasst2ConversionError(
                    "pre-remediation conversion does not match the audited baseline"
                )
        if state.selected_records == 0:
            raise Oasst2ConversionError("conversion selected no usable conversations")
        os.chmod(conversations_path, 0o444)
        os.chmod(lineage_path, 0o444)

        assert state.path_filter_reasons is not None
        assert state.tree_filter_reasons is not None
        manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "kind": MANIFEST_KIND,
            "status": "completed",
            "canonical_serialization": CANONICAL_SERIALIZATION,
            "converter": {
                "filename": Path(__file__).name,
                "version": CONVERTER_VERSION,
                "sha256": converter_sha256,
                "size_bytes": converter_size,
            },
            "tokenizer": {
                "model_id": tokenizer_identity.model_id,
                "revision": tokenizer_identity.revision,
                "fingerprint_sha256": tokenizer_identity.fingerprint_sha256,
                "template": {
                    "add_generation_prompt": False,
                    "enable_thinking": True,
                },
                "maximum_input_tokens": policy.maximum_input_tokens,
                "selected_maximum_token_count": (state.selected_maximum_token_count),
            },
            "source": {
                "filename": source.name,
                "source_uri": source_uri,
                "version": source_version,
                "license": source_license,
                "license_reviewed": license_reviewed,
                "policy_reviewed": policy_reviewed,
                "compression": "gzip",
                "compressed": {
                    "sha256": source_sha256,
                    "size_bytes": source_size,
                },
                "decompressed": {
                    "sha256": uncompressed_digest.hexdigest(),
                    "size_bytes": uncompressed_bytes,
                },
            },
            "output": {
                "path": CONVERSATIONS_FILENAME,
                "sha256": output_digest.hexdigest(),
                "size_bytes": output_bytes,
                "record_count": state.selected_records,
                "message_count": state.selected_messages,
                "content_bytes": state.selected_content_bytes,
                "fields": ["id", "messages"],
            },
            "lineage": {
                "path": LINEAGE_FILENAME,
                "sha256": lineage_digest.hexdigest(),
                "size_bytes": lineage_bytes,
                "record_count": state.selected_records,
                "fields": ["tree_id", "message_ids"],
                "contains_user_ids_or_text": False,
            },
            "counts": {
                "source_tree_count": state.source_trees,
                "source_message_count": state.source_messages,
                "source_original_leaf_count": state.source_leaves,
                "eligible_assistant_endpoint_count": state.eligible_paths,
                "selected_tree_count": state.selected_records,
                "filtered_tree_count": state.source_trees - state.selected_records,
                "duplicate_normalized_root_prompt_count": (
                    state.tree_filter_reasons["duplicate_normalized_root_prompt"]
                ),
                "pre_remediation_selected_tree_count": state.pre_remediation_records,
                "leakage_exclusion_count": state.leakage_exclusions_applied,
                "rejected_path_count": sum(state.path_filter_reasons.values()),
            },
            "filter_reasons": {
                "path_primary_reason_counts": dict(
                    sorted(state.path_filter_reasons.items())
                ),
                "filtered_tree_dominant_reason_counts": dict(
                    sorted(state.tree_filter_reasons.items())
                ),
                "path_reason_basis": (
                    "one reason per rejected assistant endpoint: the first "
                    "ineligible message in path order and fixed profile-check order"
                ),
                "tree_reason_basis": (
                    "most frequent rejected-path reason within a tree; lexical "
                    "tie-break"
                ),
            },
            "policy": {
                "profile": policy.profile,
                "selection_algorithm": SELECTION_ALGORITHM,
                "selection_tie_break": (
                    "after eligibility: descending assistant-turn count, then "
                    "prompter child message_id and assistant child "
                    "(rank_is_null, rank, message_id) per edge, then full IDs"
                ),
                "language": "exactly equal to each tree root prompt language",
                "require_ready_for_export": True,
                "require_review_result_true": True,
                "require_positive_review_count": True,
                "require_not_deleted": True,
                "require_not_synthetic": True,
                "assistant_endpoints_include_internal_nodes": True,
                "one_record_per_message_tree_id": True,
                "prompter_output_role": "user",
                "cross_tree_root_prompt_deduplication": {
                    "algorithm": "nfkc-casefold-unicode-whitespace-sha256-v1",
                    "normalization": "NFKC, casefold, split/join Unicode whitespace",
                    "retention": "first selected eligible tree in pinned source order",
                    "digest": "sha256",
                    "maximum_entries": limits.maximum_trees,
                    "near_duplicate_detection": (
                        "performed with bound audit"
                        if leakage_remediation is not None
                        else "not performed; non-exact semantic or fuzzy duplicates may remain"
                    ),
                },
                "cross_split_leakage_remediation": (
                    {
                        "active": True,
                        "algorithm": LEAKAGE_REMEDIATION_ALGORITHM,
                        "baseline_conversion_manifest_sha256": (
                            leakage_remediation.baseline_conversion_manifest_sha256
                        ),
                        "baseline_conversations_sha256": (
                            leakage_remediation.baseline_conversations_sha256
                        ),
                        "baseline_record_count": (
                            leakage_remediation.baseline_record_count
                        ),
                        "audit_sha256": leakage_remediation.audit_sha256,
                        "audit_size_bytes": leakage_remediation.audit_size_bytes,
                        "exclusions_sha256": leakage_remediation.exclusions_sha256,
                        "exclusions_size_bytes": leakage_remediation.exclusions_size_bytes,
                        "excluded_tree_count": len(
                            leakage_remediation.excluded_tree_ids
                        ),
                        "semantic_model_id": leakage_remediation.semantic_model_id,
                        "semantic_model_revision": (
                            leakage_remediation.semantic_model_revision
                        ),
                        "semantic_model_snapshot_sha256": (
                            leakage_remediation.semantic_model_snapshot_sha256
                        ),
                        "residual_limit": (
                            "thresholded heuristic audit; false negatives remain possible"
                        ),
                    }
                    if leakage_remediation is not None
                    else {"active": False}
                ),
                "conservative_zero": {
                    "active": policy.profile == PROFILE_CONSERVATIVE_ZERO,
                    "comparison": "every required label value == 0.0",
                    "common_required_zero_labels": [
                        PII_LABEL,
                        *COMMON_ADVERSE_LABELS,
                    ],
                    "assistant_required_zero_labels": list(ASSISTANT_ADVERSE_LABELS),
                },
                "quality05": {
                    "active": policy.profile == PROFILE_QUALITY05,
                    "missing_labels_allowed": True,
                    "threshold": 0.5,
                    "present_below_half_labels": [
                        "spam",
                        "lang_mismatch",
                        "pii",
                    ],
                    "assistant_present_at_least_half_labels": [
                        "quality",
                        "helpfulness",
                    ],
                    "assistant_present_below_half_labels": ["fails_task"],
                    "below_half_comparison": "value < 0.5",
                    "at_least_half_comparison": "value >= 0.5",
                    "topic_content_and_detoxify_thresholds": None,
                },
                "maximum_conversation_content_bytes": (
                    policy.maximum_conversation_content_bytes
                ),
                "overlimit_policy": "filter_without_truncation",
                "source_license_interpretation": (
                    "not performed; caller-provided provenance and review flag only"
                ),
                "limits": _limits_audit(limits),
            },
            "split": {
                "algorithm": SPLIT_ALGORITHM,
                "seed": SPLIT_SEED,
                "bucket_count": SPLIT_BUCKET_COUNT,
                "test_buckets_inclusive": [0, 999],
                "validation_buckets_inclusive": [1000, 1999],
                "train_buckets_inclusive": [2000, 9999],
                "record_counts": {
                    name: state.split_counts[name]
                    for name in ("train", "validation", "test")
                },
            },
        }
        _write_canonical_file(manifest_path, manifest)
        directory_descriptor = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        os.chmod(staging, 0o555)
        _publish_directory_exclusive(staging, destination)
        return manifest
    except BaseException:
        if compressed_stream is not None:
            compressed_stream.close()
        _cleanup_staging(staging)
        raise


def _positive_cli_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--assets-dir", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--expected-tokenizer-fingerprint", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument(
        "--expected-source-bytes", required=True, type=_positive_cli_integer
    )
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--source-license", required=True)
    parser.add_argument("--license-reviewed", action="store_true")
    parser.add_argument("--policy-reviewed", action="store_true")
    parser.add_argument("--leakage-audit", type=Path)
    parser.add_argument("--leakage-exclusions", type=Path)
    parser.add_argument("--expected-leakage-audit-sha256")
    parser.add_argument("--expected-leakage-exclusions-sha256")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--max-input-tokens",
        type=_positive_cli_integer,
        default=DEFAULT_MAX_INPUT_TOKENS,
    )
    parser.add_argument(
        "--maximum-conversation-content-bytes",
        type=_positive_cli_integer,
        default=DEFAULT_MAXIMUM_CONVERSATION_CONTENT_BYTES,
    )
    parser.add_argument(
        "--maximum-compressed-bytes",
        type=_positive_cli_integer,
        default=DEFAULT_MAXIMUM_COMPRESSED_BYTES,
    )
    parser.add_argument(
        "--maximum-uncompressed-bytes",
        type=_positive_cli_integer,
        default=DEFAULT_MAXIMUM_UNCOMPRESSED_BYTES,
    )
    parser.add_argument(
        "--maximum-line-bytes",
        type=_positive_cli_integer,
        default=DEFAULT_MAXIMUM_LINE_BYTES,
    )
    parser.add_argument(
        "--maximum-trees",
        type=_positive_cli_integer,
        default=DEFAULT_MAXIMUM_TREES,
    )
    parser.add_argument(
        "--maximum-messages",
        type=_positive_cli_integer,
        default=DEFAULT_MAXIMUM_MESSAGES,
    )
    parser.add_argument(
        "--maximum-tree-messages",
        type=_positive_cli_integer,
        default=DEFAULT_MAXIMUM_TREE_MESSAGES,
    )
    parser.add_argument(
        "--maximum-depth",
        type=_positive_cli_integer,
        default=DEFAULT_MAXIMUM_DEPTH,
    )
    parser.add_argument(
        "--maximum-replies",
        type=_positive_cli_integer,
        default=DEFAULT_MAXIMUM_REPLIES,
    )
    parser.add_argument(
        "--maximum-message-content-bytes",
        type=_positive_cli_integer,
        default=DEFAULT_MAXIMUM_MESSAGE_CONTENT_BYTES,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        from qwen_contract import (
            load_transformers_tokenizer,
            verify_asset_manifest,
        )

        assets = verify_asset_manifest(
            arguments.assets_dir,
            arguments.model_id,
            arguments.revision,
            arguments.expected_tokenizer_fingerprint,
        )
        tokenizer = load_transformers_tokenizer(assets)
        leakage_arguments = (
            arguments.leakage_audit,
            arguments.leakage_exclusions,
            arguments.expected_leakage_audit_sha256,
            arguments.expected_leakage_exclusions_sha256,
        )
        if any(value is not None for value in leakage_arguments) and not all(
            value is not None for value in leakage_arguments
        ):
            raise Oasst2ConversionError(
                "leakage audit/exclusions paths and expected SHA-256 values must all be provided together"
            )
        leakage_remediation = (
            load_leakage_remediation(
                arguments.leakage_audit,
                arguments.leakage_exclusions,
                arguments.expected_leakage_audit_sha256,
                arguments.expected_leakage_exclusions_sha256,
            )
            if arguments.leakage_audit is not None
            else None
        )
        manifest = convert_oasst2(
            source_path=arguments.input,
            output_directory=arguments.output_dir,
            expected_source_sha256=arguments.expected_source_sha256,
            expected_source_bytes=arguments.expected_source_bytes,
            source_uri=arguments.source_uri,
            source_version=arguments.source_version,
            source_license=arguments.source_license,
            license_reviewed=arguments.license_reviewed,
            policy_reviewed=arguments.policy_reviewed,
            tokenizer=tokenizer,
            tokenizer_identity=TokenizerIdentity(
                model_id=assets.model_id,
                revision=assets.revision,
                fingerprint_sha256=assets.tokenizer_fingerprint,
            ),
            policy=ConversionPolicy(
                profile=arguments.profile,
                maximum_conversation_content_bytes=(
                    arguments.maximum_conversation_content_bytes
                ),
                maximum_input_tokens=arguments.max_input_tokens,
            ),
            leakage_remediation=leakage_remediation,
            limits=ConversionLimits(
                maximum_compressed_bytes=arguments.maximum_compressed_bytes,
                maximum_uncompressed_bytes=arguments.maximum_uncompressed_bytes,
                maximum_line_bytes=arguments.maximum_line_bytes,
                maximum_trees=arguments.maximum_trees,
                maximum_messages=arguments.maximum_messages,
                maximum_tree_messages=arguments.maximum_tree_messages,
                maximum_depth=arguments.maximum_depth,
                maximum_replies=arguments.maximum_replies,
                maximum_message_content_bytes=(arguments.maximum_message_content_bytes),
            ),
        )
    except (ImportError, RuntimeError, OSError, UnicodeError, ValueError) as error:
        print(f"oasst2_convert: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "output_directory": str(arguments.output_dir.resolve()),
                "record_count": manifest["output"]["record_count"],
                "output_sha256": manifest["output"]["sha256"],
                "profile": arguments.profile,
                "tokenizer_fingerprint_sha256": assets.tokenizer_fingerprint,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
