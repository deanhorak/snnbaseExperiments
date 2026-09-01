#!/usr/bin/env python3
"""Convert canonical conversations to immutable, content-hashed token shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import IO, Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from chatbot_token_protocol import MAX_INPUT_TOKENS_DEFAULT, PROTOCOL
from qwen_contract import (
    ContractError,
    PHASE0_MODEL_ID,
    PHASE0_REVISION,
    PHASE0_TOKENIZER_FINGERPRINT,
    load_transformers_tokenizer,
    sha256_file,
    verify_asset_manifest,
    write_json_atomic,
)

DATASET_SCHEMA_VERSION = 1
DATASET_KIND = "snnbase.chatbot-token-dataset"
SPLIT_ALGORITHM = "sha256-seed-bucket-v1"
SPLIT_SEED = 42
SPLIT_BUCKET_COUNT = 10_000
MAX_SOURCE_RECORDS_DEFAULT = 1_000_000
MAX_SOURCE_LINE_BYTES_DEFAULT = 4 * 1024 * 1024
MAX_SOURCE_ID_BYTES_DEFAULT = 256
MAX_SOURCE_MESSAGES_PER_CONVERSATION_DEFAULT = 128
MAX_SOURCE_CONTENT_BYTES_PER_MESSAGE_DEFAULT = 1024 * 1024
MAX_PROVENANCE_FIELD_BYTES = 4096


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


def _load_source_json(line: str, line_number: int) -> Any:
    try:
        return json.loads(
            line,
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_nonstandard_constant,
        )
    except (json.JSONDecodeError, _DuplicateJsonField, ValueError) as error:
        raise ContractError(
            f"invalid JSON on input line {line_number}: {error}"
        ) from error


def _positive_limit(value: int, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"{label} must be positive")
    return value


def _utf8_size(value: str, label: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ContractError(f"{label} must be valid UTF-8") from error


def _line_payload(line: str | bytes, line_number: int) -> tuple[str, int]:
    if isinstance(line, bytes):
        encoded = line
        try:
            decoded = line.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ContractError(
                f"invalid UTF-8 on input line {line_number}: {error}"
            ) from error
    elif isinstance(line, str):
        decoded = line
        try:
            encoded = line.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ContractError(
                f"invalid UTF-8 on input line {line_number}: {error}"
            ) from error
    else:
        raise ContractError(f"input line {line_number} must be str or bytes")
    if encoded.endswith(b"\n"):
        encoded = encoded[:-1]
    return decoded, len(encoded)


def bounded_binary_lines(
    stream: IO[bytes], *, maximum_line_bytes: int
) -> Iterable[bytes]:
    """Yield binary JSONL lines without ever buffering an unbounded record."""

    maximum_line_bytes = _positive_limit(maximum_line_bytes, "maximum_line_bytes")
    line_number = 0
    while True:
        line = stream.readline(maximum_line_bytes + 2)
        if not line:
            return
        line_number += 1
        payload = line[:-1] if line.endswith(b"\n") else line
        if len(payload) > maximum_line_bytes:
            raise ContractError(
                f"input line {line_number} exceeds maximum_line_bytes "
                f"({maximum_line_bytes})"
            )
        yield line


def _flat_token_ids(value: Any, label: str) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(
        not isinstance(token, int) or isinstance(token, bool) or token < 0
        for token in value
    ):
        raise ContractError(
            f"{label} did not produce a flat nonnegative token-id array"
        )
    return value


def _messages(
    value: Any,
    identifier: str,
    *,
    maximum_messages_per_conversation: int,
    maximum_content_bytes_per_message: int,
) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ContractError(
            f"conversation {identifier!r} requires a non-empty messages array"
        )
    if len(value) > maximum_messages_per_conversation:
        raise ContractError(
            f"conversation {identifier!r} exceeds maximum message count "
            f"({maximum_messages_per_conversation})"
        )
    result: list[dict[str, str]] = []
    for index, raw_message in enumerate(value):
        if not isinstance(raw_message, Mapping):
            raise ContractError(
                f"conversation {identifier!r} message {index} must be an object"
            )
        if set(raw_message) != {"role", "content"}:
            raise ContractError(
                f"conversation {identifier!r} message {index} must contain "
                "exactly role and content"
            )
        role = raw_message.get("role")
        content = raw_message.get("content")
        if (
            role not in ("system", "user", "assistant")
            or not isinstance(content, str)
            or not content
        ):
            raise ContractError(
                f"conversation {identifier!r} message {index} requires a valid role and string content"
            )
        if "\0" in content:
            raise ContractError(
                f"conversation {identifier!r} message {index} content must not contain NUL"
            )
        content_bytes = _utf8_size(
            content, f"conversation {identifier!r} message {index} content"
        )
        if content_bytes > maximum_content_bytes_per_message:
            raise ContractError(
                f"conversation {identifier!r} message {index} content exceeds "
                "maximum_content_bytes_per_message "
                f"({maximum_content_bytes_per_message})"
            )
        result.append({"role": role, "content": content})
    roles = [message["role"] for message in result]
    if roles and roles[0] == "system":
        roles = roles[1:]
    if not roles or roles[0] != "user" or roles[-1] != "assistant":
        raise ContractError(
            f"conversation {identifier!r} must optionally start with system, then start user and end assistant"
        )
    expected = "user"
    for role in roles:
        if role != expected:
            raise ContractError(
                f"conversation {identifier!r} must strictly alternate user and assistant messages"
            )
        expected = "assistant" if expected == "user" else "user"
    return result


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for left_value, right_value in zip(left, right):
        if left_value != right_value:
            break
        count += 1
    return count


def _common_suffix_length(
    left: Sequence[int], right: Sequence[int], prefix: int
) -> int:
    maximum = min(len(left), len(right)) - prefix
    count = 0
    while (
        count < maximum and left[len(left) - 1 - count] == right[len(right) - 1 - count]
    ):
        count += 1
    return count


def tokenize_conversation(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    assistant_only_loss: bool,
) -> tuple[list[int], list[int] | None]:
    full_ids = _flat_token_ids(
        tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=True,
        ),
        "full conversation template",
    )
    if not assistant_only_loss:
        return full_ids, None

    loss_mask = [0] * len(full_ids)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        # Render the same complete role sequence in both variants. Qwen's chat
        # template changes earlier rendering based on which role ends the
        # conversation, so a truncated assistant prefix is not stable in a
        # multi-turn conversation. Replacing only this assistant's content
        # keeps every global template branch identical and isolates its token
        # span through the common prefix/suffix.
        empty_messages = [dict(item) for item in messages]
        empty_messages[index] = {"role": "assistant", "content": ""}
        empty_full_ids = _flat_token_ids(
            tokenizer.apply_chat_template(
                empty_messages,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=True,
            ),
            f"empty assistant conversation {index}",
        )
        content_start = _common_prefix_length(empty_full_ids, full_ids)
        common_suffix = _common_suffix_length(empty_full_ids, full_ids, content_start)
        content_end = len(full_ids) - common_suffix
        # Include the first shared suffix token (Qwen's <|im_end|>) as the
        # assistant's causal termination target, while excluding trailing layout.
        target_end = min(len(full_ids), content_end + (1 if common_suffix else 0))
        if target_end <= content_start:
            raise ContractError(
                "assistant target produced an empty or ambiguous token span"
            )
        loss_mask[content_start:target_end] = [1] * (target_end - content_start)
    if not any(loss_mask):
        raise ContractError("assistant-only masking produced no target tokens")
    return full_ids, loss_mask


def prepare_record(
    tokenizer: Any,
    value: Any,
    *,
    max_input_tokens: int,
    maximum_id_bytes: int = MAX_SOURCE_ID_BYTES_DEFAULT,
    maximum_messages_per_conversation: int = (
        MAX_SOURCE_MESSAGES_PER_CONVERSATION_DEFAULT
    ),
    maximum_content_bytes_per_message: int = (
        MAX_SOURCE_CONTENT_BYTES_PER_MESSAGE_DEFAULT
    ),
) -> dict[str, Any]:
    max_input_tokens = _positive_limit(max_input_tokens, "max_input_tokens")
    maximum_id_bytes = _positive_limit(maximum_id_bytes, "maximum_id_bytes")
    maximum_messages_per_conversation = _positive_limit(
        maximum_messages_per_conversation, "maximum_messages_per_conversation"
    )
    maximum_content_bytes_per_message = _positive_limit(
        maximum_content_bytes_per_message, "maximum_content_bytes_per_message"
    )
    if not isinstance(value, Mapping):
        raise ContractError("each JSONL record must be an object")
    if set(value) != {"id", "messages"}:
        raise ContractError(
            "each source record must contain exactly the top-level fields id and messages"
        )
    identifier = value.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ContractError("conversation id must be a non-empty string")
    if "\0" in identifier:
        raise ContractError("conversation id must not contain NUL")
    if any(ord(character) < 0x20 for character in identifier):
        raise ContractError("conversation id must not contain control characters")
    if _utf8_size(identifier, "conversation id") > maximum_id_bytes:
        raise ContractError(
            f"conversation id exceeds maximum_id_bytes ({maximum_id_bytes})"
        )
    messages = _messages(
        value.get("messages"),
        identifier,
        maximum_messages_per_conversation=maximum_messages_per_conversation,
        maximum_content_bytes_per_message=maximum_content_bytes_per_message,
    )
    input_ids, loss_mask = tokenize_conversation(
        tokenizer, messages, assistant_only_loss=True
    )
    if not input_ids or len(input_ids) > max_input_tokens:
        raise ContractError(
            f"conversation {identifier!r} encoded to {len(input_ids)} tokens; "
            f"the fail-closed limit is {max_input_tokens} and truncation is forbidden"
        )
    if loss_mask is None:
        raise ContractError("prepared datasets require an explicit loss_mask")
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "id": identifier,
        "input_ids": input_ids,
        "loss_mask": loss_mask,
    }


def split_bucket(identifier: str, seed: int = SPLIT_SEED) -> int:
    if not isinstance(identifier, str) or not identifier:
        raise ContractError("split id must be non-empty")
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed < 0
        or seed > (1 << 64) - 1
    ):
        raise ContractError("split seed must be an unsigned 64-bit integer")
    payload = seed.to_bytes(8, "big") + identifier.encode("utf-8")
    return (
        int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % SPLIT_BUCKET_COUNT
    )


def split_name(identifier: str) -> str:
    bucket = split_bucket(identifier)
    if bucket <= 999:
        return "test"
    if bucket <= 1999:
        return "validation"
    return "train"


def prepare_stream(
    tokenizer: Any,
    lines: Iterable[str | bytes],
    *,
    max_input_tokens: int,
    maximum_records: int = MAX_SOURCE_RECORDS_DEFAULT,
    maximum_line_bytes: int = MAX_SOURCE_LINE_BYTES_DEFAULT,
    maximum_id_bytes: int = MAX_SOURCE_ID_BYTES_DEFAULT,
    maximum_messages_per_conversation: int = (
        MAX_SOURCE_MESSAGES_PER_CONVERSATION_DEFAULT
    ),
    maximum_content_bytes_per_message: int = (
        MAX_SOURCE_CONTENT_BYTES_PER_MESSAGE_DEFAULT
    ),
) -> dict[str, list[dict[str, Any]]]:
    max_input_tokens = _positive_limit(max_input_tokens, "max_input_tokens")
    maximum_records = _positive_limit(maximum_records, "maximum_records")
    maximum_line_bytes = _positive_limit(maximum_line_bytes, "maximum_line_bytes")
    maximum_id_bytes = _positive_limit(maximum_id_bytes, "maximum_id_bytes")
    maximum_messages_per_conversation = _positive_limit(
        maximum_messages_per_conversation, "maximum_messages_per_conversation"
    )
    maximum_content_bytes_per_message = _positive_limit(
        maximum_content_bytes_per_message, "maximum_content_bytes_per_message"
    )
    records: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        decoded, payload_bytes = _line_payload(line, line_number)
        if payload_bytes > maximum_line_bytes:
            raise ContractError(
                f"input line {line_number} exceeds maximum_line_bytes "
                f"({maximum_line_bytes})"
            )
        if not decoded.strip():
            raise ContractError(f"blank input line {line_number} is not allowed")
        if len(seen) >= maximum_records:
            raise ContractError(
                f"input exceeds maximum_records ({maximum_records}) at line "
                f"{line_number}"
            )
        value = _load_source_json(decoded, line_number)
        record = prepare_record(
            tokenizer,
            value,
            max_input_tokens=max_input_tokens,
            maximum_id_bytes=maximum_id_bytes,
            maximum_messages_per_conversation=maximum_messages_per_conversation,
            maximum_content_bytes_per_message=maximum_content_bytes_per_message,
        )
        if record["id"] in seen:
            raise ContractError(f"duplicate conversation id: {record['id']}")
        seen.add(record["id"])
        records[split_name(record["id"])].append(record)
    if not any(records.values()):
        raise ContractError("input contains no conversation records")
    return records


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    record, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())


def _provenance_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ContractError(f"{label} must be a non-empty, trimmed string")
    if "\0" in value or any(ord(character) < 0x20 for character in value):
        raise ContractError(f"{label} must not contain control characters")
    if _utf8_size(value, label) > MAX_PROVENANCE_FIELD_BYTES:
        raise ContractError(f"{label} exceeds {MAX_PROVENANCE_FIELD_BYTES} UTF-8 bytes")
    return value


def _source_uri(value: Any) -> str:
    result = _provenance_value(value, "source_uri")
    try:
        parsed = urlsplit(result)
    except ValueError as error:
        raise ContractError(f"source_uri is invalid: {error}") from error
    if not parsed.scheme or not (parsed.netloc or parsed.path):
        raise ContractError("source_uri must be an absolute URI")
    return result


def _source_version(value: Any) -> str:
    result = _provenance_value(value, "source_version")
    mutable_names = {
        "current",
        "head",
        "latest",
        "main",
        "master",
        "stable",
        "tip",
        "trunk",
    }
    normalized = result.casefold()
    leaf = normalized.rstrip("/").rsplit("/", 1)[-1]
    if (
        normalized in mutable_names
        or leaf in mutable_names
        or normalized.startswith(("branch:", "refs/heads/"))
    ):
        raise ContractError(
            "source_version must identify an immutable snapshot, not a moving ref"
        )
    return result


def write_dataset(
    output_dir: Path,
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    source_path: Path,
    source_size_bytes: int,
    source_sha256: str,
    source_uri: str,
    source_version: str,
    source_license: str,
    model_id: str,
    revision: str,
    tokenizer_fingerprint: str,
) -> dict[str, Any]:
    source_uri = _source_uri(source_uri)
    source_version = _source_version(source_version)
    source_license = _provenance_value(source_license, "source_license")
    if source_path.is_symlink() or not source_path.is_file():
        raise ContractError(f"source_path is not a regular file: {source_path}")
    if (
        not isinstance(source_size_bytes, int)
        or isinstance(source_size_bytes, bool)
        or source_size_bytes < 0
        or not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise ContractError("consumed source size/SHA-256 provenance is invalid")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ContractError(
            "output-dir already exists; prepared datasets are immutable"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        shard_manifest: dict[str, Any] = {}
        for split in ("train", "validation", "test"):
            path = temporary / f"{split}.jsonl"
            split_records = list(records[split])
            _write_jsonl(path, split_records)
            shard_manifest[split] = {
                "path": path.name,
                "record_count": len(split_records),
                "input_token_count": sum(
                    len(record["input_ids"]) for record in split_records
                ),
                "target_token_count": sum(
                    sum(record["loss_mask"]) for record in split_records
                ),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        if (
            source_path.stat().st_size != source_size_bytes
            or sha256_file(source_path) != source_sha256
        ):
            raise ContractError("source changed after it was read for tokenization")
        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "kind": DATASET_KIND,
            "token_protocol": PROTOCOL,
            "source": {
                "filename": source_path.name,
                "size_bytes": source_size_bytes,
                "sha256": source_sha256,
                "record_schema": "exactly {id,messages}",
                "source_uri": source_uri,
                "version": source_version,
                "license": source_license,
            },
            "tokenizer": {
                "model_id": model_id,
                "revision": revision,
                "fingerprint_sha256": tokenizer_fingerprint,
            },
            "split": {
                "algorithm": SPLIT_ALGORITHM,
                "seed": SPLIT_SEED,
                "hash_payload": "seed u64 big-endian || id UTF-8",
                "bucket_count": SPLIT_BUCKET_COUNT,
                "test_buckets_inclusive": [0, 999],
                "validation_buckets_inclusive": [1000, 1999],
                "train_buckets_inclusive": [2000, 9999],
            },
            "shards": shard_manifest,
        }
        write_json_atomic(temporary / "dataset-manifest.json", manifest)
        temporary.replace(output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", required=True, type=Path)
    parser.add_argument("--model-id", required=True, help=f"Phase 0: {PHASE0_MODEL_ID}")
    parser.add_argument("--revision", required=True, help=f"Phase 0: {PHASE0_REVISION}")
    parser.add_argument(
        "--expected-tokenizer-fingerprint",
        required=True,
        help=f"Phase 0: {PHASE0_TOKENIZER_FINGERPRINT}",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--source-license", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--max-input-tokens", type=int, default=MAX_INPUT_TOKENS_DEFAULT
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.max_input_tokens <= 0:
            raise ContractError("max-input-tokens must be positive")
        assets = verify_asset_manifest(
            arguments.assets_dir,
            arguments.model_id,
            arguments.revision,
            arguments.expected_tokenizer_fingerprint,
        )
        tokenizer = load_transformers_tokenizer(assets)
        source_digest = hashlib.sha256()
        source_size_bytes = 0
        with arguments.input.open("rb") as stream:

            def consumed_lines() -> Iterable[bytes]:
                nonlocal source_size_bytes
                for line in bounded_binary_lines(
                    stream, maximum_line_bytes=MAX_SOURCE_LINE_BYTES_DEFAULT
                ):
                    source_digest.update(line)
                    source_size_bytes += len(line)
                    yield line

            records = prepare_stream(
                tokenizer,
                consumed_lines(),
                max_input_tokens=arguments.max_input_tokens,
            )
        manifest = write_dataset(
            arguments.output_dir,
            records,
            source_path=arguments.input,
            source_size_bytes=source_size_bytes,
            source_sha256=source_digest.hexdigest(),
            source_uri=arguments.source_uri,
            source_version=arguments.source_version,
            source_license=arguments.source_license,
            model_id=assets.model_id,
            revision=assets.revision,
            tokenizer_fingerprint=assets.tokenizer_fingerprint,
        )
    except (ContractError, OSError, UnicodeError, ValueError) as error:
        print(f"chatbot_prepare: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "records": sum(
                    shard["record_count"] for shard in manifest["shards"].values()
                ),
                "output": str(arguments.output_dir.resolve()),
                "tokenizer_fingerprint_sha256": assets.tokenizer_fingerprint,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
