#!/usr/bin/env python3
"""Generate tokenizer and causal-logit golden fixtures from verified Qwen assets."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from qwen_contract import (
    ContractError,
    PHASE0_MODEL_ID,
    PHASE0_REVISION,
    PHASE0_TOKENIZER_FINGERPRINT,
    VerifiedAssets,
    load_json,
    load_transformers_tokenizer,
    package_versions,
    sha256_bytes,
    verify_asset_manifest,
    write_json_atomic,
)

TOKENIZER_GOLDEN_KIND = "snnbase.qwen-tokenizer-golden"
REFERENCE_LOGITS_KIND = "snnbase.qwen-reference-logits"
REFERENCE_SCHEMA_VERSION = 1


def _add_contract_arguments(
    parser: argparse.ArgumentParser, *, require_weights: bool
) -> None:
    parser.add_argument("--assets-dir", required=True, type=Path)
    parser.add_argument(
        "--model-id",
        required=True,
        help=f"explicit repository id (Phase 0: {PHASE0_MODEL_ID})",
    )
    parser.add_argument(
        "--revision",
        required=True,
        help=f"explicit commit SHA (Phase 0: {PHASE0_REVISION})",
    )
    parser.add_argument(
        "--expected-tokenizer-fingerprint",
        required=True,
        help=f"required fingerprint (Phase 0: {PHASE0_TOKENIZER_FINGERPRINT})",
    )
    parser.set_defaults(require_weights=require_weights)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    tokenizer = subparsers.add_parser(
        "tokenizer-golden",
        description="Encode, decode, and render chat-template cases.",
    )
    _add_contract_arguments(tokenizer, require_weights=False)
    tokenizer.add_argument("--cases", required=True, type=Path)
    tokenizer.add_argument("--output", required=True, type=Path)

    logits = subparsers.add_parser(
        "reference-logits",
        description="Generate deterministic final-position logit probes.",
    )
    _add_contract_arguments(logits, require_weights=True)
    logits.add_argument("--cases", required=True, type=Path)
    logits.add_argument("--output", required=True, type=Path)
    logits.add_argument("--top-k", type=int, default=16)
    logits.add_argument("--absolute-tolerance", type=float, default=1.0e-5)
    logits.add_argument("--relative-tolerance", type=float, default=1.0e-5)
    return parser


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be a JSON object")
    return value


def _require_cases(document: Any, field: str) -> list[Mapping[str, Any]]:
    root = _require_mapping(document, "case document")
    if root.get("schema_version") != 1:
        raise ContractError("case document schema_version must be 1")
    raw_cases = root.get(field)
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ContractError(f"case document field {field!r} must be a non-empty array")
    result: list[Mapping[str, Any]] = []
    identifiers: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        case = _require_mapping(raw_case, f"{field}[{index}]")
        identifier = case.get("id")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
        ):
            raise ContractError(
                f"{field}[{index}].id must be a unique non-empty string"
            )
        identifiers.add(identifier)
        result.append(case)
    return result


def _normalize_token_ids(value: Any, label: str) -> list[int]:
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
            f"{label} did not produce a flat array of nonnegative token ids"
        )
    return value


def _tokenizer_metadata(tokenizer: Any) -> dict[str, Any]:
    special_ids: dict[str, int | None] = {}
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        value = getattr(tokenizer, name, None)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise ContractError(f"tokenizer {name} is not an integer or null")
        special_ids[name] = value
    vocabulary_size = getattr(tokenizer, "vocab_size", None)
    total_size = len(tokenizer)
    if not isinstance(vocabulary_size, int) or vocabulary_size <= 0 or total_size <= 0:
        raise ContractError("tokenizer reported invalid vocabulary sizes")
    return {
        "class": type(tokenizer).__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "vocab_size": vocabulary_size,
        "size_with_added_tokens": total_size,
        "special_token_ids": special_ids,
    }


def generate_tokenizer_golden(
    tokenizer: Any,
    assets: VerifiedAssets,
    case_document: Any,
) -> dict[str, Any]:
    encode_cases = _require_cases(case_document, "encode_cases")
    chat_cases_raw = _require_mapping(case_document, "case document").get(
        "chat_cases", []
    )
    if not isinstance(chat_cases_raw, list):
        raise ContractError("case document field 'chat_cases' must be an array")

    encoded_results: list[dict[str, Any]] = []
    for case in encode_cases:
        text = case.get("text")
        add_special_tokens = case.get("add_special_tokens", False)
        if not isinstance(text, str) or not isinstance(add_special_tokens, bool):
            raise ContractError(f"invalid encode case {case['id']!r}")
        token_ids = _normalize_token_ids(
            tokenizer.encode(text, add_special_tokens=add_special_tokens),
            f"encode case {case['id']}",
        )
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(decoded, str):
            raise ContractError(f"decode case {case['id']} did not return text")
        encoded_results.append(
            {
                "id": case["id"],
                "text": text,
                "add_special_tokens": add_special_tokens,
                "token_ids": token_ids,
                "decoded": decoded,
            }
        )

    chat_results: list[dict[str, Any]] = []
    chat_identifiers: set[str] = set()
    for index, raw_case in enumerate(chat_cases_raw):
        case = _require_mapping(raw_case, f"chat_cases[{index}]")
        identifier = case.get("id")
        messages = case.get("messages")
        add_generation_prompt = case.get("add_generation_prompt", False)
        enable_thinking = case.get("enable_thinking", True)
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in chat_identifiers
            or not isinstance(messages, list)
            or not isinstance(add_generation_prompt, bool)
            or not isinstance(enable_thinking, bool)
        ):
            raise ContractError(f"invalid chat case at index {index}")
        chat_identifiers.add(identifier)
        for message_index, message_value in enumerate(messages):
            message = _require_mapping(
                message_value, f"chat case {identifier} message {message_index}"
            )
            if message.get("role") not in (
                "system",
                "user",
                "assistant",
            ) or not isinstance(message.get("content"), str):
                raise ContractError(f"invalid message in chat case {identifier}")

        keyword_arguments = {
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
        }
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, **keyword_arguments
        )
        tokenized = tokenizer.apply_chat_template(
            messages, tokenize=True, **keyword_arguments
        )
        token_ids = _normalize_token_ids(tokenized, f"chat case {identifier}")
        if not isinstance(rendered, str):
            raise ContractError(f"chat case {identifier} did not render text")
        rendered_token_ids = _normalize_token_ids(
            tokenizer.encode(rendered, add_special_tokens=False),
            f"rendered chat case {identifier}",
        )
        if token_ids != rendered_token_ids:
            raise ContractError(
                f"chat case {identifier} differs between tokenized and rendered-template encoding"
            )
        chat_results.append(
            {
                "id": identifier,
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
                "rendered": rendered,
                "token_ids": token_ids,
            }
        )

    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "kind": TOKENIZER_GOLDEN_KIND,
        "oracle": {
            "model_id": assets.model_id,
            "revision": assets.revision,
            "tokenizer_fingerprint_sha256": assets.tokenizer_fingerprint,
        },
        "implementation": package_versions(
            ("transformers", "tokenizers", "huggingface-hub")
        ),
        "tokenizer": _tokenizer_metadata(tokenizer),
        "encode_cases": encoded_results,
        "chat_cases": chat_results,
    }


def _canonical_tensor_bytes(tensor: Any) -> bytes:
    try:
        import torch
    except Exception as error:
        raise ContractError(
            "PyTorch is required to generate reference logits"
        ) from error
    contiguous = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if sys.byteorder != "little":
        raise ContractError(
            "reference-logit hashing currently requires a little-endian host"
        )
    return contiguous.view(torch.uint8).numpy().tobytes(order="C")


def _load_reference_model(assets: VerifiedAssets) -> Any:
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except Exception as error:
        raise ContractError(
            "could not import the locked PyTorch/Transformers model stack"
        ) from error
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(assets.root),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float32,
            device_map=None,
            attn_implementation="eager",
        )
    except Exception as error:
        raise ContractError(
            f"could not load the verified local model weights: {error}"
        ) from error
    model.to("cpu")
    model.eval()
    return model


def generate_reference_logits(
    tokenizer: Any,
    model: Any,
    assets: VerifiedAssets,
    case_document: Any,
    *,
    top_k: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ContractError("top-k must be positive")
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ContractError("absolute tolerance must be finite and nonnegative")
    if not math.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise ContractError("relative tolerance must be finite and nonnegative")
    cases = _require_cases(case_document, "logit_cases")

    try:
        import torch
    except Exception as error:
        raise ContractError(
            "PyTorch is required to generate reference logits"
        ) from error
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for case in cases:
            text = case.get("text")
            probes = case.get("probe_token_ids", [])
            if not isinstance(text, str) or not text:
                raise ContractError(f"logit case {case['id']} requires non-empty text")
            if not isinstance(probes, list) or any(
                not isinstance(token, int) or isinstance(token, bool) or token < 0
                for token in probes
            ):
                raise ContractError(
                    f"invalid probe_token_ids for logit case {case['id']}"
                )
            inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")
            if "input_ids" not in inputs or inputs["input_ids"].shape[-1] == 0:
                raise ContractError(
                    f"logit case {case['id']} encoded to an empty prompt"
                )
            outputs = model(**inputs, use_cache=False)
            final_logits = (
                outputs.logits[0, -1].detach().to(torch.float32).cpu().contiguous()
            )
            vocabulary_size = int(final_logits.numel())
            if any(token >= vocabulary_size for token in probes):
                raise ContractError(
                    f"probe token outside vocabulary in logit case {case['id']}"
                )
            selected_k = min(top_k, vocabulary_size)
            values, indices = torch.topk(
                final_logits, selected_k, largest=True, sorted=True
            )
            top = [
                {
                    "token_id": int(token_id),
                    "logit": float(logit),
                    "decoded": tokenizer.decode(
                        [int(token_id)],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                }
                for token_id, logit in zip(
                    indices.tolist(), values.tolist(), strict=True
                )
            ]
            results.append(
                {
                    "id": case["id"],
                    "text": text,
                    "input_ids": _normalize_token_ids(
                        inputs["input_ids"], f"logit case {case['id']}"
                    ),
                    "final_logits": {
                        "shape": [vocabulary_size],
                        "canonical_dtype": "float32",
                        "byte_order": "little",
                        "sha256": sha256_bytes(_canonical_tensor_bytes(final_logits)),
                        "top_k": top,
                        "probes": [
                            {
                                "token_id": token,
                                "logit": float(final_logits[token].item()),
                            }
                            for token in probes
                        ],
                    },
                }
            )

    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "kind": REFERENCE_LOGITS_KIND,
        "oracle": {
            "model_id": assets.model_id,
            "revision": assets.revision,
            "tokenizer_fingerprint_sha256": assets.tokenizer_fingerprint,
            "checkpoint_files": assets.manifest["weight_files"],
        },
        "implementation": package_versions(
            ("torch", "transformers", "tokenizers", "safetensors", "huggingface-hub")
        ),
        "execution": {
            "device": "cpu",
            "model_dtype": "float32",
            "attention_implementation": "eager",
            "deterministic_algorithms": True,
            "seed": 0,
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
        },
        "cases": results,
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    assets = verify_asset_manifest(
        arguments.assets_dir,
        arguments.model_id,
        arguments.revision,
        arguments.expected_tokenizer_fingerprint,
        require_weights=arguments.require_weights,
    )
    case_document = load_json(arguments.cases)
    tokenizer = load_transformers_tokenizer(assets)
    if arguments.command == "tokenizer-golden":
        result = generate_tokenizer_golden(tokenizer, assets, case_document)
    else:
        model = _load_reference_model(assets)
        result = generate_reference_logits(
            tokenizer,
            model,
            assets,
            case_document,
            top_k=arguments.top_k,
            absolute_tolerance=arguments.absolute_tolerance,
            relative_tolerance=arguments.relative_tolerance,
        )
    write_json_atomic(arguments.output, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        result = run(parser.parse_args(argv))
    except (ContractError, OSError, ValueError) as error:
        print(f"qwen_reference: {error}", file=sys.stderr)
        return 2
    print(
        f"wrote {result['kind']} schema {result['schema_version']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
