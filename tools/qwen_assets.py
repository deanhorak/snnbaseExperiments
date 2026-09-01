#!/usr/bin/env python3
"""Create or verify a content-addressed local Qwen asset snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from qwen_contract import (
    ASSET_MANIFEST_NAME,
    ContractError,
    PHASE0_MODEL_ID,
    PHASE0_REVISION,
    PHASE0_TOKENIZER_FINGERPRINT,
    create_asset_manifest,
    verify_asset_manifest,
    write_json_atomic,
)


def _add_contract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--assets-dir", required=True, type=Path)
    parser.add_argument(
        "--model-id",
        required=True,
        help=f"explicit repository id (Phase 0: {PHASE0_MODEL_ID})",
    )
    parser.add_argument(
        "--revision",
        required=True,
        help=f"explicit 40-character commit (Phase 0: {PHASE0_REVISION})",
    )
    parser.add_argument(
        "--expected-tokenizer-fingerprint",
        required=True,
        help=f"canonical tokenizer fingerprint (Phase 0: {PHASE0_TOKENIZER_FINGERPRINT})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or verify a fail-closed Qwen local asset manifest."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser(
        "create",
        description="Hash a snapshot downloaded by `hf download --revision <commit>`.",
    )
    _add_contract_arguments(create)
    create.add_argument(
        "--output",
        type=Path,
        help=f"manifest path (default: ASSETS_DIR/{ASSET_MANIFEST_NAME})",
    )

    verify = subparsers.add_parser(
        "verify", description="Verify every declared local asset."
    )
    _add_contract_arguments(verify)
    verify.add_argument("--manifest", type=Path)
    verify.add_argument("--require-weights", action="store_true")
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.command == "create":
        manifest = create_asset_manifest(
            arguments.assets_dir,
            arguments.model_id,
            arguments.revision,
            arguments.expected_tokenizer_fingerprint,
        )
        output = arguments.output or arguments.assets_dir / ASSET_MANIFEST_NAME
        write_json_atomic(output, manifest)
        return {
            "ok": True,
            "command": "create",
            "manifest": str(output.resolve()),
            "model_id": manifest["model_id"],
            "revision": manifest["revision"],
            "tokenizer_fingerprint": manifest["tokenizer"]["fingerprint_sha256"],
            "weights_present": manifest["weights_present"],
        }

    verified = verify_asset_manifest(
        arguments.assets_dir,
        arguments.model_id,
        arguments.revision,
        arguments.expected_tokenizer_fingerprint,
        require_weights=arguments.require_weights,
        manifest_path=arguments.manifest,
    )
    return {
        "ok": True,
        "command": "verify",
        "manifest": str(verified.manifest_path),
        "model_id": verified.model_id,
        "revision": verified.revision,
        "tokenizer_fingerprint": verified.tokenizer_fingerprint,
        "weights_present": bool(verified.manifest["weights_present"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        result = run(parser.parse_args(argv))
    except (ContractError, OSError, ValueError) as error:
        print(f"qwen_assets: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
