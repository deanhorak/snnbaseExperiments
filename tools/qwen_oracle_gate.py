#!/usr/bin/env python3
"""Gate imported ANN logits against the exact pinned Qwen reference fixture."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from chatbot_token_protocol import PROTOCOL
from qwen_contract import (
    ContractError,
    PHASE0_MODEL_ID,
    PHASE0_REVISION,
    PHASE0_TOKENIZER_FINGERPRINT,
    load_json,
    require_sha256,
    sha256_file,
    write_json_atomic,
)

GATE_KIND = "snnbase.qwen-import-oracle-gate"
GATE_SCHEMA_VERSION = 1
MAX_RESPONSE_BYTES = 64 * 1024


class OracleCore:
    def __init__(self, command: Sequence[str]):
        self.command = list(command)
        self.process = subprocess.Popen(  # noqa: S603 - explicit argv, no shell.
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            errors="strict",
            bufsize=1,
            shell=False,
        )

    def transact(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.process.stdin is None or self.process.stdout is None:
            raise ContractError("oracle core pipes are unavailable")
        encoded = json.dumps(request, sort_keys=True, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise ContractError("oracle inspect request exceeds its byte limit")
        try:
            self.process.stdin.write(encoded + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline(MAX_RESPONSE_BYTES + 2)
        except (BrokenPipeError, OSError, UnicodeError) as error:
            raise ContractError(f"oracle core transport failed: {error}") from error
        if (
            not line
            or not line.endswith("\n")
            or len(line.encode()) > MAX_RESPONSE_BYTES + 1
        ):
            raise ContractError(
                "oracle core response is missing, unterminated, or oversized"
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractError(f"oracle core emitted invalid JSON: {error}") from error
        if (
            not isinstance(response, Mapping)
            or response.get("protocol") != PROTOCOL
            or response.get("request_id") != request["request_id"]
            or response.get("ok") is not True
        ):
            raise ContractError(
                f"oracle core rejected/miscorrelated {request['request_id']}"
            )
        return response

    def shutdown(self) -> None:
        self.transact(
            {"protocol": PROTOCOL, "request_id": "oracle-shutdown", "op": "shutdown"}
        )
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            status = self.process.wait(timeout=30.0)
        except subprocess.TimeoutExpired as error:
            self.process.terminate()
            self.process.wait(timeout=5.0)
            raise ContractError("oracle core did not exit after shutdown") from error
        if self.process.stdout is not None:
            self.process.stdout.close()
        if status != 0:
            raise ContractError(f"oracle core exited with status {status}")

    def abort(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)
        if self.process.stdout is not None:
            self.process.stdout.close()


def _reference(path: Path) -> Mapping[str, Any]:
    document = load_json(path)
    if not isinstance(document, Mapping):
        raise ContractError("reference logits fixture must be an object")
    oracle = document.get("oracle")
    execution = document.get("execution")
    cases = document.get("cases")
    if (
        document.get("schema_version") != 1
        or document.get("kind") != "snnbase.qwen-reference-logits"
        or not isinstance(oracle, Mapping)
        or oracle.get("model_id") != PHASE0_MODEL_ID
        or oracle.get("revision") != PHASE0_REVISION
        or oracle.get("tokenizer_fingerprint_sha256") != PHASE0_TOKENIZER_FINGERPRINT
        or not isinstance(execution, Mapping)
        or execution.get("device") != "cpu"
        or execution.get("model_dtype") != "float32"
        or not isinstance(cases, list)
        or not cases
    ):
        raise ContractError("reference logits fixture provenance/contract mismatch")
    for tolerance in ("absolute_tolerance", "relative_tolerance"):
        value = execution.get(tolerance)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0 < value < 1
        ):
            raise ContractError(f"reference {tolerance} is invalid")
    return document


def _value_map(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{label} must be a non-empty array")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if (
            not isinstance(item, Mapping)
            or set(item) != {"token_id", "logit"}
            or not isinstance(item["token_id"], int)
            or isinstance(item["token_id"], bool)
            or not isinstance(item["logit"], (int, float))
            or isinstance(item["logit"], bool)
            or not math.isfinite(float(item["logit"]))
        ):
            raise ContractError(f"{label}[{index}] is malformed")
        result.append(item)
    return result


def run_gate(
    runner: Path,
    archive: Path,
    archive_sha256: str,
    reference_path: Path,
) -> dict[str, Any]:
    archive_sha256 = require_sha256(archive_sha256, "converted archive SHA-256")
    runner = runner.resolve(strict=True)
    archive = archive.resolve(strict=True)
    reference_path = reference_path.resolve(strict=True)
    if sha256_file(archive) != archive_sha256:
        raise ContractError("converted archive does not match --archive-sha256")
    reference = _reference(reference_path)
    command = [
        str(runner),
        "serve",
        "--qwen3-0.6b",
        "--qwen-archive",
        str(archive),
        "--qwen-archive-sha256",
        archive_sha256,
        "--ann",
        "--device",
        "cpu",
    ]
    absolute = float(reference["execution"]["absolute_tolerance"])
    relative = float(reference["execution"]["relative_tolerance"])
    core = OracleCore(command)
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    try:
        for case_index, expected_case in enumerate(reference["cases"]):
            if not isinstance(expected_case, Mapping):
                raise ContractError("reference case must be an object")
            expected_final = expected_case.get("final_logits")
            input_ids = expected_case.get("input_ids")
            if not isinstance(expected_final, Mapping) or not isinstance(
                input_ids, list
            ):
                raise ContractError("reference case final_logits/input_ids is invalid")
            expected_probes = expected_final.get("probes")
            expected_top = expected_final.get("top_k")
            if not isinstance(expected_probes, list) or not isinstance(
                expected_top, list
            ):
                raise ContractError("reference probes/top_k must be arrays")
            response = core.transact(
                {
                    "protocol": PROTOCOL,
                    "request_id": f"oracle-{case_index:04d}",
                    "op": "inspect",
                    "input_ids": input_ids,
                    "probe_token_ids": [item["token_id"] for item in expected_probes],
                    "top_k": len(expected_top),
                }
            )
            final = response.get("final_position")
            if not isinstance(final, Mapping) or set(final) != {"probes", "top_k"}:
                raise ContractError(
                    "inspect response must contain only bounded probes/top_k"
                )
            actual_probes = _value_map(final["probes"], "actual probes")
            actual_top = _value_map(final["top_k"], "actual top_k")
            case_failures: list[str] = []
            maximum_absolute_error = 0.0
            maximum_relative_error = 0.0
            for label, expected_values, actual_values in (
                ("probe", expected_probes, actual_probes),
                ("top_k", expected_top, actual_top),
            ):
                expected_ids = [item["token_id"] for item in expected_values]
                actual_ids = [item["token_id"] for item in actual_values]
                if actual_ids != expected_ids:
                    case_failures.append(
                        f"{label} token ids differ: expected {expected_ids}, found {actual_ids}"
                    )
                    continue
                for expected_value, actual_value in zip(expected_values, actual_values):
                    expected_logit = float(expected_value["logit"])
                    actual_logit = float(actual_value["logit"])
                    absolute_error = abs(actual_logit - expected_logit)
                    relative_error = absolute_error / max(abs(expected_logit), 1.0e-30)
                    maximum_absolute_error = max(maximum_absolute_error, absolute_error)
                    maximum_relative_error = max(maximum_relative_error, relative_error)
                    if not math.isclose(
                        actual_logit, expected_logit, rel_tol=relative, abs_tol=absolute
                    ):
                        case_failures.append(
                            f"{label} token {expected_value['token_id']} logit differs: "
                            f"expected {expected_logit:.17g}, found {actual_logit:.17g}"
                        )
            identifier = expected_case.get("id")
            failures.extend(f"{identifier}: {message}" for message in case_failures)
            results.append(
                {
                    "id": identifier,
                    "passed": not case_failures,
                    "maximum_absolute_error": maximum_absolute_error,
                    "maximum_relative_error": maximum_relative_error,
                    "actual": final,
                }
            )
        core.shutdown()
    except BaseException:
        core.abort()
        raise
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "kind": GATE_KIND,
        "status": "passed" if not failures else "failed",
        "oracle": reference["oracle"],
        "execution_contract": {
            "architecture": "ann",
            "device": "cpu",
            "dtype": "float32",
            "absolute_tolerance": absolute,
            "relative_tolerance": relative,
            "bounded_output": "final-position probes and top-k only",
        },
        "inputs": {
            "runner": {
                "path": str(runner),
                "sha256": sha256_file(runner),
            },
            "archive": {
                "path": str(archive),
                "sha256": archive_sha256,
            },
            "reference": {
                "path": str(reference_path),
                "sha256": sha256_file(reference_path),
            },
            "command": command,
        },
        "cases": results,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.output.exists():
            raise ContractError("output already exists; gate reports are immutable")
        report = run_gate(
            arguments.runner,
            arguments.archive,
            arguments.archive_sha256,
            arguments.reference,
        )
        write_json_atomic(arguments.output, report)
    except (ContractError, OSError, UnicodeError, ValueError) as error:
        print(f"qwen_oracle_gate: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ok": report["status"] == "passed",
                "status": report["status"],
                "failure_count": len(report["failures"]),
                "report": str(arguments.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
