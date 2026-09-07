#!/usr/bin/env python3
"""Audit canonical chatbot conversations for cross-split content leakage.

The audit is intentionally separate from model training.  It verifies the
immutable conversion bundle, derives the configured deterministic split from
each conversation ID, and compares three behavior-relevant views:

* the root user prompt;
* all user turns in the conversation; and
* all assistant targets in the conversation.

Reports contain identifiers and similarity measurements, never message text.
High-confidence collision components are accompanied by a deterministic
exclusion recommendation which retains test before validation before train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

AUDIT_SCHEMA_VERSION = 1
AUDIT_KIND = "snnbase.chatbot-leakage-audit"
AUDIT_VERSION = "1.0.0"
EXCLUSIONS_KIND = "snnbase.chatbot-leakage-exclusions"
CONVERSION_KIND = "snnbase.oasst2-conversion-manifest"
SPLIT_SEED = 42
SPLIT_BUCKET_COUNT = 10_000
MAXIMUM_MANIFEST_BYTES = 4 * 1024 * 1024
MAXIMUM_LINE_BYTES = 4 * 1024 * 1024
MAXIMUM_RECORDS = 100_000
MAXIMUM_TEXT_BYTES = 1024 * 1024
MAXIMUM_FLAGGED_PAIRS = 250_000
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class LeakageAuditError(RuntimeError):
    """Raised when an input or audit invariant fails closed."""


@dataclass(frozen=True)
class ViewPolicy:
    semantic_threshold: float
    review_threshold: float
    minimum_characters: int


VIEW_POLICIES: dict[str, ViewPolicy] = {
    "root_prompt": ViewPolicy(0.92, 0.86, 8),
    "user_context": ViewPolicy(0.94, 0.88, 16),
    "assistant_targets": ViewPolicy(0.97, 0.92, 32),
}
FUZZY_JACCARD_THRESHOLD = 0.85
FUZZY_CONTAINMENT_THRESHOLD = 0.95
FUZZY_MINIMUM_COMPACT_CHARACTERS = 20
NGRAM_WIDTH = 5
MAXIMUM_NGRAM_DOCUMENT_FREQUENCY = 128
BOTTOM_HASH_SKETCH_SIZE = 16
MAXIMUM_FUZZY_CANDIDATE_PAIRS = 5_000_000


@dataclass(frozen=True)
class Conversation:
    identifier: str
    split: str
    views: Mapping[str, str]


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LeakageAuditError(f"duplicate JSON field: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise LeakageAuditError(f"non-standard JSON constant: {value}")


def _strict_json(payload: bytes, label: str) -> Any:
    try:
        text = payload.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, LeakageAuditError) as error:
        raise LeakageAuditError(f"{label} is not strict JSON: {error}") from error


def _regular_file(path: Path, maximum_bytes: int, label: str) -> os.stat_result:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise LeakageAuditError(f"could not stat {label}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise LeakageAuditError(f"{label} must be a regular non-symlink file")
    if metadata.st_size <= 0 or metadata.st_size > maximum_bytes:
        raise LeakageAuditError(f"{label} has an invalid or excessive size")
    return metadata


def _file_digest(path: Path, maximum_bytes: int, label: str) -> tuple[str, int]:
    metadata = _regular_file(path, maximum_bytes, label)
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(block)
                if size > maximum_bytes:
                    raise LeakageAuditError(f"{label} exceeds its byte limit")
                digest.update(block)
    except OSError as error:
        raise LeakageAuditError(f"could not read {label}: {error}") from error
    if size != metadata.st_size:
        raise LeakageAuditError(f"{label} changed while it was read")
    return digest.hexdigest(), size


def _load_conversion_manifest(path: Path) -> tuple[dict[str, Any], str, int]:
    digest, size = _file_digest(path, MAXIMUM_MANIFEST_BYTES, "conversion manifest")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise LeakageAuditError(
            f"could not read conversion manifest: {error}"
        ) from error
    value = _strict_json(payload, "conversion manifest")
    if type(value) is not dict:
        raise LeakageAuditError("conversion manifest must be an object")
    if value.get("schema_version") != 1 or value.get("kind") != CONVERSION_KIND:
        raise LeakageAuditError("unsupported conversion manifest")
    if value.get("status") != "completed":
        raise LeakageAuditError("conversion manifest is not completed")
    output = value.get("output")
    split = value.get("split")
    if type(output) is not dict or type(split) is not dict:
        raise LeakageAuditError("conversion manifest lacks output/split records")
    if output.get("path") != "conversations.jsonl":
        raise LeakageAuditError("conversion manifest has a non-canonical output path")
    if not isinstance(output.get("record_count"), int) or not (
        0 < output["record_count"] <= MAXIMUM_RECORDS
    ):
        raise LeakageAuditError("conversion manifest record_count is invalid")
    if SHA256_RE.fullmatch(str(output.get("sha256", ""))) is None:
        raise LeakageAuditError("conversion manifest output SHA-256 is invalid")
    if not isinstance(output.get("size_bytes"), int) or output["size_bytes"] <= 0:
        raise LeakageAuditError("conversion manifest output size is invalid")
    if (
        split.get("algorithm") != "sha256-seed-bucket-v1"
        or split.get("seed") != SPLIT_SEED
        or split.get("bucket_count") != SPLIT_BUCKET_COUNT
    ):
        raise LeakageAuditError("conversion manifest split contract is unsupported")
    return value, digest, size


def _split_name(identifier: str) -> str:
    bucket = (
        int.from_bytes(
            hashlib.sha256(
                SPLIT_SEED.to_bytes(8, "big") + identifier.encode("utf-8")
            ).digest()[:8],
            "big",
        )
        % SPLIT_BUCKET_COUNT
    )
    if bucket <= 999:
        return "test"
    if bucket <= 1999:
        return "validation"
    return "train"


def _bounded_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LeakageAuditError(f"{label} must be a nonempty string")
    encoded = value.encode("utf-8", errors="strict")
    if len(encoded) > MAXIMUM_TEXT_BYTES or "\x00" in value:
        raise LeakageAuditError(f"{label} is invalid or exceeds its byte limit")
    return value


def _conversation(value: Any, line_number: int) -> Conversation:
    if type(value) is not dict or set(value) != {"id", "messages"}:
        raise LeakageAuditError(
            f"conversation line {line_number} must contain exactly id and messages"
        )
    identifier = _bounded_text(value["id"], f"line {line_number} id")
    messages = value["messages"]
    if type(messages) is not list or not messages or len(messages) > 128:
        raise LeakageAuditError(f"line {line_number} messages is invalid")
    users: list[str] = []
    assistants: list[str] = []
    expected = "user"
    for index, message in enumerate(messages):
        if type(message) is not dict or set(message) != {"role", "content"}:
            raise LeakageAuditError(
                f"line {line_number} message {index} has invalid fields"
            )
        if message["role"] != expected:
            raise LeakageAuditError(
                f"line {line_number} message {index} violates role alternation"
            )
        content = _bounded_text(
            message["content"], f"line {line_number} message {index} content"
        )
        if expected == "user":
            users.append(content)
            expected = "assistant"
        else:
            assistants.append(content)
            expected = "user"
    if expected != "user" or not users or not assistants:
        raise LeakageAuditError(f"line {line_number} must end with an assistant")
    return Conversation(
        identifier=identifier,
        split=_split_name(identifier),
        views={
            "root_prompt": users[0],
            "user_context": "\n".join(users),
            "assistant_targets": "\n".join(assistants),
        },
    )


def load_conversations(
    manifest_path: Path,
) -> tuple[list[Conversation], dict[str, Any], dict[str, Any]]:
    manifest, manifest_sha256, manifest_bytes = _load_conversion_manifest(manifest_path)
    output_path = manifest_path.parent / "conversations.jsonl"
    expected = manifest["output"]
    digest, size = _file_digest(
        output_path,
        max(int(expected["size_bytes"]), 1),
        "canonical conversations",
    )
    if digest != expected["sha256"] or size != expected["size_bytes"]:
        raise LeakageAuditError("canonical conversations do not match the manifest")
    conversations: list[Conversation] = []
    identifiers: set[str] = set()
    split_counts: dict[str, int] = defaultdict(int)
    try:
        with output_path.open("rb") as stream:
            while True:
                encoded = stream.readline(MAXIMUM_LINE_BYTES + 2)
                if not encoded:
                    break
                if len(encoded) > MAXIMUM_LINE_BYTES or not encoded.endswith(b"\n"):
                    raise LeakageAuditError("canonical conversation line is overlong")
                line_number = len(conversations) + 1
                value = _strict_json(encoded[:-1], f"conversation line {line_number}")
                record = _conversation(value, line_number)
                if record.identifier in identifiers:
                    raise LeakageAuditError(
                        "canonical conversations contain duplicate IDs"
                    )
                identifiers.add(record.identifier)
                conversations.append(record)
                split_counts[record.split] += 1
                if len(conversations) > MAXIMUM_RECORDS:
                    raise LeakageAuditError(
                        "canonical conversations exceed record limit"
                    )
    except OSError as error:
        raise LeakageAuditError(
            f"could not stream canonical conversations: {error}"
        ) from error
    if len(conversations) != expected["record_count"]:
        raise LeakageAuditError("canonical conversation count does not match manifest")
    declared_counts = manifest["split"].get("record_counts")
    actual_counts = {
        name: split_counts[name] for name in ("train", "validation", "test")
    }
    if declared_counts != actual_counts or any(
        value == 0 for value in actual_counts.values()
    ):
        raise LeakageAuditError("recomputed split counts do not match the manifest")
    provenance = {
        "conversion_manifest": {
            "path": manifest_path.name,
            "sha256": manifest_sha256,
            "size_bytes": manifest_bytes,
        },
        "canonical_conversations": {
            "path": output_path.name,
            "sha256": digest,
            "size_bytes": size,
            "record_count": len(conversations),
        },
    }
    return conversations, manifest, provenance


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _compact(text: str) -> str:
    return "".join(character for character in _normalize(text) if character.isalnum())


def _ngrams(text: str) -> frozenset[str]:
    compact = _compact(text)
    if len(compact) < NGRAM_WIDTH:
        return frozenset({compact}) if compact else frozenset()
    return frozenset(
        compact[index : index + NGRAM_WIDTH]
        for index in range(len(compact) - NGRAM_WIDTH + 1)
    )


def _pair_key(left: Conversation, right: Conversation) -> tuple[str, str]:
    return tuple(sorted((left.identifier, right.identifier)))  # type: ignore[return-value]


def _pair_record(
    left: Conversation,
    right: Conversation,
    view: str,
    *,
    exact: bool = False,
    jaccard: float | None = None,
    containment: float | None = None,
    semantic: float | None = None,
) -> dict[str, Any]:
    first, second = sorted((left, right), key=lambda item: item.identifier)
    value: dict[str, Any] = {
        "ids": [first.identifier, second.identifier],
        "splits": [first.split, second.split],
        "views": [view],
        "reasons": [],
    }
    if exact:
        value["reasons"].append("exact_normalized")
    if jaccard is not None:
        value["maximum_ngram_jaccard"] = round(jaccard, 8)
        if jaccard >= FUZZY_JACCARD_THRESHOLD:
            value["reasons"].append("fuzzy_ngram_jaccard")
    if containment is not None:
        value["maximum_ngram_containment"] = round(containment, 8)
        if containment >= FUZZY_CONTAINMENT_THRESHOLD:
            value["reasons"].append("fuzzy_ngram_containment")
    if semantic is not None:
        value["maximum_semantic_cosine"] = round(semantic, 8)
        value["reasons"].append("semantic_cosine")
    return value


def _merge_pair(target: dict[str, Any], addition: Mapping[str, Any]) -> None:
    target["views"] = sorted(set(target["views"]) | set(addition["views"]))
    target["reasons"] = sorted(set(target["reasons"]) | set(addition["reasons"]))
    for score in (
        "maximum_ngram_jaccard",
        "maximum_ngram_containment",
        "maximum_semantic_cosine",
    ):
        if score in addition:
            target[score] = max(float(target.get(score, -1.0)), float(addition[score]))


def lexical_collisions(
    conversations: Sequence[Conversation],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Return deterministic exact/fuzzy cross-split collision records."""

    collisions: dict[tuple[str, str], dict[str, Any]] = {}
    for view, policy in VIEW_POLICIES.items():
        exact_index: dict[str, list[int]] = defaultdict(list)
        grams: list[frozenset[str]] = []
        compacts: list[str] = []
        sketch_index: dict[bytes, list[int]] = defaultdict(list)
        for index, conversation in enumerate(conversations):
            normalized = _normalize(conversation.views[view])
            exact_index[normalized].append(index)
            compacts.append(_compact(conversation.views[view]))
            current = _ngrams(conversation.views[view])
            grams.append(current)
            bottom_hashes = sorted(
                hashlib.sha256(gram.encode("utf-8")).digest() for gram in current
            )[:BOTTOM_HASH_SKETCH_SIZE]
            for gram_hash in bottom_hashes:
                sketch_index[gram_hash].append(index)
        for indices in exact_index.values():
            if len(indices) < 2:
                continue
            for position, left_index in enumerate(indices):
                left = conversations[left_index]
                if len(compacts[left_index]) < policy.minimum_characters:
                    continue
                for right_index in indices[position + 1 :]:
                    right = conversations[right_index]
                    if left.split == right.split:
                        continue
                    key = _pair_key(left, right)
                    addition = _pair_record(left, right, view, exact=True)
                    if key in collisions:
                        _merge_pair(collisions[key], addition)
                    else:
                        collisions[key] = addition
        candidates: set[tuple[int, int]] = set()
        for right_index, current in enumerate(grams):
            for gram in current:
                gram_hash = hashlib.sha256(gram.encode("utf-8")).digest()
                indices = sketch_index.get(gram_hash, ())
                if len(indices) > MAXIMUM_NGRAM_DOCUMENT_FREQUENCY:
                    continue
                for left_index in indices:
                    if left_index >= right_index:
                        continue
                    if (
                        conversations[left_index].split
                        != conversations[right_index].split
                    ):
                        candidates.add((left_index, right_index))
            if len(candidates) > MAXIMUM_FUZZY_CANDIDATE_PAIRS:
                raise LeakageAuditError("fuzzy candidates exceed the configured bound")
        for left_index, right_index in sorted(candidates):
            left_grams = grams[left_index]
            right_grams = grams[right_index]
            minimum_size = min(len(left_grams), len(right_grams))
            if minimum_size == 0:
                continue
            shared_count = len(left_grams & right_grams)
            if shared_count < 3:
                continue
            union_size = len(left_grams) + len(right_grams) - shared_count
            jaccard = shared_count / union_size
            containment = shared_count / minimum_size
            left = conversations[left_index]
            right = conversations[right_index]
            if min(len(compacts[left_index]), len(compacts[right_index])) < (
                FUZZY_MINIMUM_COMPACT_CHARACTERS
            ):
                continue
            if not (
                jaccard >= FUZZY_JACCARD_THRESHOLD
                or containment >= FUZZY_CONTAINMENT_THRESHOLD
            ):
                continue
            key = _pair_key(left, right)
            addition = _pair_record(
                left,
                right,
                view,
                jaccard=jaccard,
                containment=containment,
            )
            if key in collisions:
                _merge_pair(collisions[key], addition)
            else:
                collisions[key] = addition
            if len(collisions) > MAXIMUM_FLAGGED_PAIRS:
                raise LeakageAuditError("lexical collisions exceed the report bound")
    return collisions


def _semantic_pairs(
    conversations: Sequence[Conversation],
    embeddings: Any,
    view: str,
    device: str,
) -> tuple[list[tuple[int, int, float]], int]:
    try:
        import torch
    except ImportError as error:
        raise LeakageAuditError("semantic audit requires PyTorch") from error
    matrix = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    if matrix.ndim != 2 or matrix.shape[0] != len(conversations):
        raise LeakageAuditError(f"semantic encoder returned invalid {view} embeddings")
    norms = torch.linalg.vector_norm(matrix, dim=1)
    if not bool(torch.all(torch.isfinite(matrix))) or bool(torch.any(norms <= 0)):
        raise LeakageAuditError(
            f"semantic encoder returned non-finite/zero {view} embeddings"
        )
    matrix = matrix / norms[:, None]
    threshold = VIEW_POLICIES[view].semantic_threshold
    review_threshold = VIEW_POLICIES[view].review_threshold
    split_indices = {
        name: [index for index, item in enumerate(conversations) if item.split == name]
        for name in ("train", "validation", "test")
    }
    flagged: dict[tuple[int, int], float] = {}
    review_count = 0
    for left_name, right_name in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        left_indices = split_indices[left_name]
        right_indices = split_indices[right_name]
        right_tensor = matrix[right_indices]
        for offset in range(0, len(left_indices), 512):
            block_indices = left_indices[offset : offset + 512]
            similarities = matrix[block_indices] @ right_tensor.T
            review_count += int(torch.count_nonzero(similarities >= review_threshold))
            positions = torch.nonzero(similarities >= threshold, as_tuple=False)
            for row, column in positions.detach().cpu().tolist():
                left_index = block_indices[row]
                right_index = right_indices[column]
                score = float(similarities[row, column])
                key = (min(left_index, right_index), max(left_index, right_index))
                flagged[key] = max(flagged.get(key, -1.0), score)
            if len(flagged) > MAXIMUM_FLAGGED_PAIRS:
                raise LeakageAuditError("semantic collisions exceed the report bound")
    return [
        (left, right, score) for (left, right), score in sorted(flagged.items())
    ], review_count


def semantic_collisions(
    conversations: Sequence[Conversation],
    model: Any,
    device: str,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, int]]:
    collisions: dict[tuple[str, str], dict[str, Any]] = {}
    review_counts: dict[str, int] = {}
    for view in VIEW_POLICIES:
        texts = [conversation.views[view] for conversation in conversations]
        try:
            embeddings = model.encode(
                texts,
                batch_size=128,
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=False,
                device=device,
            )
        except Exception as error:
            raise LeakageAuditError(
                f"semantic encoder failed for {view}: {error}"
            ) from error
        pairs, review_count = _semantic_pairs(conversations, embeddings, view, device)
        review_counts[view] = review_count
        for left_index, right_index, score in pairs:
            left = conversations[left_index]
            right = conversations[right_index]
            if min(
                len(_compact(left.views[view])), len(_compact(right.views[view]))
            ) < (VIEW_POLICIES[view].minimum_characters):
                continue
            key = _pair_key(left, right)
            addition = _pair_record(left, right, view, semantic=score)
            if key in collisions:
                _merge_pair(collisions[key], addition)
            else:
                collisions[key] = addition
            if len(collisions) > MAXIMUM_FLAGGED_PAIRS:
                raise LeakageAuditError("semantic collisions exceed the report bound")
    return collisions, review_counts


def merge_collisions(
    *collections: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for collection in collections:
        for key, value in collection.items():
            if key in merged:
                _merge_pair(merged[key], value)
            else:
                merged[key] = dict(value)
    return [merged[key] for key in sorted(merged)]


def recommended_exclusions(
    conversations: Sequence[Conversation], collisions: Sequence[Mapping[str, Any]]
) -> list[str]:
    parent: dict[str, str] = {
        item.identifier: item.identifier for item in conversations
    }

    def find(identifier: str) -> str:
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    involved: set[str] = set()
    for collision in collisions:
        left, right = collision["ids"]
        union(left, right)
        involved.update((left, right))
    components: dict[str, list[str]] = defaultdict(list)
    for identifier in involved:
        components[find(identifier)].append(identifier)
    by_id = {item.identifier: item for item in conversations}
    priority = {"test": 0, "validation": 1, "train": 2}
    exclusions: list[str] = []
    for identifiers in components.values():
        retained_priority = min(priority[by_id[value].split] for value in identifiers)
        exclusions.extend(
            value
            for value in identifiers
            if priority[by_id[value].split] > retained_priority
        )
    return sorted(exclusions)


def _model_files(model_directory: Path) -> tuple[list[dict[str, Any]], str]:
    if not model_directory.is_dir() or model_directory.is_symlink():
        raise LeakageAuditError(
            "semantic model directory must be a non-symlink directory"
        )
    files: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in sorted(model_directory.rglob("*")):
        if ".cache" in path.relative_to(model_directory).parts:
            continue
        if path.is_dir():
            continue
        relative = path.relative_to(model_directory).as_posix()
        digest, size = _file_digest(path, 1024 * 1024 * 1024, f"model file {relative}")
        record = {"path": relative, "sha256": digest, "size_bytes": size}
        files.append(record)
        aggregate.update(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        aggregate.update(b"\n")
    if not files:
        raise LeakageAuditError("semantic model directory contains no files")
    return files, aggregate.hexdigest()


def _implementation_record() -> dict[str, Any]:
    path = Path(__file__).resolve()
    digest, size = _file_digest(path, 4 * 1024 * 1024, "audit implementation")
    return {
        "filename": path.name,
        "version": AUDIT_VERSION,
        "sha256": digest,
        "size_bytes": size,
    }


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def run_audit(
    *,
    conversion_manifest: Path,
    model_directory: Path,
    model_id: str,
    model_revision: str,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if REVISION_RE.fullmatch(model_revision) is None:
        raise LeakageAuditError(
            "semantic model revision must be a lowercase commit SHA"
        )
    conversations, conversion, provenance = load_conversations(conversion_manifest)
    files, snapshot_sha256 = _model_files(model_directory)
    if device == "cuda":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        import sentence_transformers
        import torch
        import transformers
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise LeakageAuditError(
            "sentence-transformers and PyTorch are required for semantic audit"
        ) from error
    torch.manual_seed(0)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise LeakageAuditError("CUDA was requested but is unavailable")
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    try:
        model = SentenceTransformer(str(model_directory), device=device)
    except Exception as error:
        raise LeakageAuditError(f"could not load semantic model: {error}") from error
    lexical = lexical_collisions(conversations)
    semantic, review_counts = semantic_collisions(conversations, model, device)
    collisions = merge_collisions(lexical, semantic)
    exclusions = recommended_exclusions(conversations, collisions)
    split_counts = conversion["split"]["record_counts"]
    exclusion_counts = {name: 0 for name in ("train", "validation", "test")}
    by_id = {item.identifier: item for item in conversations}
    for identifier in exclusions:
        exclusion_counts[by_id[identifier].split] += 1
    report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "kind": AUDIT_KIND,
        "status": "pass" if not collisions else "remediation_required",
        "audit_version": AUDIT_VERSION,
        "implementation": _implementation_record(),
        "scope": {
            "cross_split_only": True,
            "views": list(VIEW_POLICIES),
            "raw_text_in_report": False,
            "within_split_duplicates": "out_of_scope",
        },
        "input": provenance,
        "split": {
            "algorithm": "sha256-seed-bucket-v1",
            "seed": SPLIT_SEED,
            "record_counts": split_counts,
        },
        "semantic_model": {
            "model_id": model_id,
            "revision": model_revision,
            "license_declared": "Apache-2.0",
            "snapshot_sha256": snapshot_sha256,
            "files": files,
            "runtime": {
                "device": device,
                "seed": 0,
                "deterministic_algorithms": True,
                "cublas_workspace_config": (
                    os.environ.get("CUBLAS_WORKSPACE_CONFIG")
                    if device == "cuda"
                    else None
                ),
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "sentence_transformers_version": sentence_transformers.__version__,
            },
        },
        "thresholds": {
            "views": {
                name: {
                    "semantic_cosine": policy.semantic_threshold,
                    "review_band_cosine": policy.review_threshold,
                    "minimum_compact_characters": policy.minimum_characters,
                }
                for name, policy in VIEW_POLICIES.items()
            },
            "fuzzy_ngram_width": NGRAM_WIDTH,
            "fuzzy_ngram_jaccard": FUZZY_JACCARD_THRESHOLD,
            "fuzzy_ngram_containment": FUZZY_CONTAINMENT_THRESHOLD,
            "fuzzy_minimum_compact_characters": FUZZY_MINIMUM_COMPACT_CHARACTERS,
            "semantic_candidate_policy": "all_cross_split_pairs_at_or_above_threshold",
        },
        "results": {
            "high_confidence_cross_split_pair_count": len(collisions),
            "records_in_high_confidence_components": len(
                {identifier for item in collisions for identifier in item["ids"]}
            ),
            "recommended_exclusion_count": len(exclusions),
            "recommended_exclusion_counts_by_split": exclusion_counts,
            "post_exclusion_record_counts": {
                name: int(split_counts[name]) - exclusion_counts[name]
                for name in ("train", "validation", "test")
            },
            "review_band_cross_split_pair_counts": review_counts,
            "pairs": collisions,
        },
        "remediation": {
            "algorithm": "collision-components-retain-highest-priority-split-v1",
            "split_priority": ["test", "validation", "train"],
            "same_split_retention": "retain all records in the highest-priority split",
            "exclusions_sha256": hashlib.sha256(
                ("\n".join(exclusions) + ("\n" if exclusions else "")).encode("utf-8")
            ).hexdigest(),
        },
        "limitations": [
            "Similarity thresholds are conservative heuristics, not proof that every semantic duplicate is found.",
            "The embedding model can produce false positives and false negatives, especially across languages.",
            "The audit does not determine privacy, contributor rights, safety, or legal fitness.",
            "Within-split duplicates are not a train/evaluation leakage condition and are not removed by this audit.",
        ],
    }
    exclusions_document = {
        "schema_version": 1,
        "kind": EXCLUSIONS_KIND,
        "algorithm": report["remediation"]["algorithm"],
        "conversion_manifest_sha256": provenance["conversion_manifest"]["sha256"],
        "tree_ids": exclusions,
    }
    return report, exclusions_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion-manifest", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclusions-output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        report, exclusions = run_audit(
            conversion_manifest=arguments.conversion_manifest,
            model_directory=arguments.model_dir,
            model_id=arguments.model_id,
            model_revision=arguments.model_revision,
            device=arguments.device,
        )
        _write_exclusive_json(arguments.output, report)
        _write_exclusive_json(arguments.exclusions_output, exclusions)
    except (LeakageAuditError, OSError) as error:
        print(f"chatbot leakage audit failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "high_confidence_cross_split_pair_count": report["results"][
                    "high_confidence_cross_split_pair_count"
                ],
                "recommended_exclusion_count": report["results"][
                    "recommended_exclusion_count"
                ],
                "output": str(arguments.output),
                "exclusions_output": str(arguments.exclusions_output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
