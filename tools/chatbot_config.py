#!/usr/bin/env python3
"""Validate chatbot config v1 and emit its verified C++ core argv.

This module intentionally uses only the Python standard library.  Structural
validation is performed against ``schemas/chatbot-config-v1.schema.json`` and
the semantic pass then proves that the stored runner arguments resolve to the
same architecture, geometry, initialization, and training values as the JSON
fields.  Unknown fields, duplicate JSON keys, duplicate CLI options, and
unrepresentable configuration values fail closed.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

CONFIG_SCHEMA_VERSION = "snnbase-chatbot-config-v1"
COMPACT_TRACK = "compact_trainable_ann_snn"
EXACT_TRACK = "exact_qwen_ann_parity_and_hybrid_snn_conversion"

PHASE0_MODEL_ID = "Qwen/Qwen3-0.6B-Base"
PHASE0_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
PHASE0_TOKENIZER_FINGERPRINT = (
    "6a4dac166a643066a1af821907cfd1c998c8a195e6458a90979953e848b6e237"
)
PHASE0_ARCHIVE_SHA256 = (
    "333c8be1b3fde5013ee8d1dcb96f23051dd934f8e1f359aff329e7e5cf8ccac8"
)

MAX_CONFIG_BYTES = 2 * 1024 * 1024
MAX_U64 = (1 << 64) - 1
SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "schemas" / "chatbot-config-v1.schema.json"
)


class ConfigError(ValueError):
    """A fail-closed chatbot configuration contract violation."""


@dataclass(frozen=True)
class ValidatedConfig:
    """A structurally and semantically validated chatbot configuration."""

    document: Mapping[str, Any]
    source: str
    core_argv: tuple[str, ...]

    @property
    def experiment_id(self) -> str:
        return str(self.document["experiment_id"])

    @property
    def track(self) -> str:
        return str(self.document["track"])

    @property
    def architecture(self) -> str:
        return str(self.document["architecture"])


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path, *, maximum_bytes: int = MAX_CONFIG_BYTES) -> Any:
    try:
        if not path.is_file() or path.is_symlink():
            raise ConfigError(f"not a regular non-symlink file: {path}")
        payload = path.read_bytes()
    except OSError as error:
        raise ConfigError(f"cannot read {path}: {error}") from error
    if len(payload) > maximum_bytes:
        raise ConfigError(f"JSON file exceeds {maximum_bytes} bytes: {path}")
    try:
        text = payload.decode("utf-8", errors="strict")
        return json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
    except UnicodeDecodeError as error:
        raise ConfigError(f"invalid UTF-8 in {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"invalid JSON in {path}: {error}") from error


def _json_type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return type(value) is dict
    if expected == "array":
        return type(value) is list
    if expected == "string":
        return type(value) is str
    if expected == "boolean":
        return type(value) is bool
    if expected == "integer":
        return type(value) is int
    if expected == "number":
        return type(value) in (int, float) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    raise ConfigError(f"schema uses unsupported JSON type: {expected}")


def _json_equal(left: Any, right: Any) -> bool:
    """JSON Schema equality, keeping booleans distinct from JSON numbers."""

    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is bool and type(right) is bool and left == right
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    if type(left) is not type(right):
        return False
    if type(left) is list:
        return len(left) == len(right) and all(
            _json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    if type(left) is dict:
        return set(left) == set(right) and all(
            _json_equal(left[key], right[key]) for key in left
        )
    return left == right


def _resolve_local_ref(root: Mapping[str, Any], reference: str) -> Any:
    if not reference.startswith("#/"):
        raise ConfigError(f"schema uses unsupported non-local reference: {reference}")
    value: Any = root
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if type(value) is not dict or part not in value:
            raise ConfigError(f"schema contains unresolved reference: {reference}")
        value = value[part]
    return value


def _schema_validate(
    value: Any,
    schema: Any,
    root: Mapping[str, Any],
    path: str,
) -> None:
    if schema is True:
        return
    if schema is False:
        raise ConfigError(f"{path}: field is forbidden for this config variant")
    if type(schema) is not dict:
        raise ConfigError(f"schema node at {path} is not an object or boolean")

    reference = schema.get("$ref")
    if reference is not None:
        if type(reference) is not str:
            raise ConfigError(f"schema reference at {path} is not a string")
        _schema_validate(value, _resolve_local_ref(root, reference), root, path)

    for child in schema.get("allOf", []):
        _schema_validate(value, child, root, path)

    alternatives = schema.get("oneOf")
    if alternatives is not None:
        matches = 0
        failures: list[str] = []
        for child in alternatives:
            try:
                _schema_validate(value, child, root, path)
            except ConfigError as error:
                failures.append(str(error))
            else:
                matches += 1
        if matches != 1:
            detail = failures[0] if failures else "multiple alternatives matched"
            raise ConfigError(
                f"{path}: expected exactly one config variant; "
                f"{matches} matched ({detail})"
            )

    expected_type = schema.get("type")
    if expected_type is not None:
        types = [expected_type] if type(expected_type) is str else expected_type
        if type(types) is not list or not all(type(item) is str for item in types):
            raise ConfigError(f"schema type at {path} is invalid")
        if not any(_json_type_matches(value, item) for item in types):
            raise ConfigError(f"{path}: expected {' or '.join(types)}")

    if "const" in schema and not _json_equal(value, schema["const"]):
        raise ConfigError(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and not any(
        _json_equal(value, candidate) for candidate in schema["enum"]
    ):
        raise ConfigError(f"{path}: value is outside the permitted enum")

    if type(value) is dict:
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise ConfigError(f"{path}: missing required field {key!r}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, child_value in value.items():
            child_path = f"{path}.{key}"
            if key in properties:
                _schema_validate(child_value, properties[key], root, child_path)
            elif additional is False:
                raise ConfigError(f"{path}: unexpected field {key!r}")
            elif additional is not True:
                _schema_validate(child_value, additional, root, child_path)

    if type(value) is list:
        if len(value) < schema.get("minItems", 0):
            raise ConfigError(f"{path}: array is shorter than minItems")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ConfigError(f"{path}: array is longer than maxItems")
        if schema.get("uniqueItems", False):
            encoded = [json.dumps(item, sort_keys=True) for item in value]
            if len(encoded) != len(set(encoded)):
                raise ConfigError(f"{path}: array items must be unique")
        if "items" in schema:
            for index, child_value in enumerate(value):
                _schema_validate(child_value, schema["items"], root, f"{path}[{index}]")

    if type(value) is str:
        if len(value) < schema.get("minLength", 0):
            raise ConfigError(f"{path}: string is shorter than minLength")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ConfigError(f"{path}: string is longer than maxLength")
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, value) is None:
            raise ConfigError(f"{path}: string does not match the required pattern")

    if type(value) in (int, float) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise ConfigError(f"{path}: number must be finite")
        if "minimum" in schema and number < float(schema["minimum"]):
            raise ConfigError(f"{path}: number is below minimum")
        if "maximum" in schema and number > float(schema["maximum"]):
            raise ConfigError(f"{path}: number is above maximum")
        if "exclusiveMinimum" in schema and number <= float(schema["exclusiveMinimum"]):
            raise ConfigError(f"{path}: number is not above exclusiveMinimum")
        if "exclusiveMaximum" in schema and number >= float(schema["exclusiveMaximum"]):
            raise ConfigError(f"{path}: number is not below exclusiveMaximum")


@functools.lru_cache(maxsize=1)
def _config_schema() -> Mapping[str, Any]:
    schema = _load_json(SCHEMA_PATH)
    if type(schema) is not dict:
        raise ConfigError("chatbot config schema root must be an object")
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ConfigError("chatbot config schema must use JSON Schema draft 2020-12")
    return schema


RUNNER_FLAGS = {
    "--ann",
    "--snn",
    "--qwen3-0.6b",
    "--qk-norm",
    "--no-qk-norm",
}
RUNNER_VALUE_OPTIONS = {
    "--checkpoint",
    "--save-checkpoint",
    "--qwen-archive",
    "--qwen-archive-sha256",
    "--device",
    "--vocabulary-size",
    "--context-tokens",
    "--model-dimension",
    "--layers",
    "--query-heads",
    "--kv-heads",
    "--head-dimension",
    "--ffn-dimension",
    "--rms-epsilon",
    "--rope-base",
    "--simulation-steps",
    "--threshold",
    "--leak",
    "--surrogate-slope",
    "--learning-rate",
    "--weight-decay",
    "--gradient-clip",
    "--gradient-accumulation",
    "--total-optimizer-steps",
    "--warmup-optimizer-steps",
    "--minimum-lr-ratio",
    "--spike-rate-target",
    "--spike-rate-penalty",
    "--seed",
}


@dataclass(frozen=True)
class ParsedRunnerArguments:
    flags: frozenset[str]
    values: Mapping[str, str]
    order: tuple[str, ...]


def _parse_runner_arguments(arguments: Sequence[str]) -> ParsedRunnerArguments:
    flags: set[str] = set()
    values: dict[str, str] = {}
    order: list[str] = []
    index = 0
    while index < len(arguments):
        option = arguments[index]
        if not option or "\0" in option:
            raise ConfigError("runner_binding.arguments contains an empty/NUL value")
        if option in RUNNER_FLAGS:
            if option in flags:
                raise ConfigError(f"runner argument is duplicated: {option}")
            flags.add(option)
            order.append(option)
            index += 1
            continue
        if option not in RUNNER_VALUE_OPTIONS:
            raise ConfigError(f"unknown or positional runner argument: {option}")
        if option in values:
            raise ConfigError(f"runner argument is duplicated: {option}")
        if index + 1 >= len(arguments):
            raise ConfigError(f"runner argument requires a value: {option}")
        raw_value = arguments[index + 1]
        if not raw_value or "\0" in raw_value:
            raise ConfigError(f"runner argument has an empty/NUL value: {option}")
        values[option] = raw_value
        order.append(option)
        index += 2
    return ParsedRunnerArguments(frozenset(flags), values, tuple(order))


def _relative_path(value: str, label: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or "\0" in value
        or "\\" in value
        or str(path) != value
        or value == "."
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ConfigError(f"{label} must be a normalized repository-relative path")


def _reject_nul_strings(value: Any, path: str = "$") -> None:
    if type(value) is str:
        if "\0" in value:
            raise ConfigError(f"{path}: strings must not contain NUL")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _reject_nul_strings(child, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, child in value.items():
            if "\0" in key:
                raise ConfigError(f"{path}: field names must not contain NUL")
            _reject_nul_strings(child, f"{path}.{key}")


def _runner_integer(raw: str, option: str) -> int:
    if re.fullmatch(r"[0-9]+", raw) is None:
        raise ConfigError(f"{option} must be an unsigned decimal integer")
    value = int(raw, 10)
    if value > MAX_U64:
        raise ConfigError(f"{option} exceeds unsigned 64-bit range")
    return value


def _runner_number(raw: str, option: str) -> float:
    try:
        value = float(raw)
    except ValueError as error:
        raise ConfigError(f"{option} must be numeric") from error
    if not math.isfinite(value):
        raise ConfigError(f"{option} must be finite")
    return value


def _same_number(actual: str, expected: Any, option: str) -> None:
    if _runner_number(actual, option) != float(expected):
        raise ConfigError(
            f"runner {option}={actual!r} does not match resolved value {expected!r}"
        )


def _same_integer(actual: str, expected: Any, option: str) -> None:
    if _runner_integer(actual, option) != expected:
        raise ConfigError(
            f"runner {option}={actual!r} does not match resolved value {expected!r}"
        )


def _require_argument_shape(
    parsed: ParsedRunnerArguments,
    *,
    flags: set[str],
    order: Sequence[str],
    value_options: set[str],
) -> None:
    if parsed.flags != flags:
        raise ConfigError(
            "runner architecture/preset flags do not match the resolved config: "
            f"expected {sorted(flags)}, found {sorted(parsed.flags)}"
        )
    if set(parsed.values) != value_options:
        missing = sorted(value_options - set(parsed.values))
        extra = sorted(set(parsed.values) - value_options)
        raise ConfigError(
            f"runner value-option set mismatch; missing={missing}, extra={extra}"
        )
    if parsed.order != tuple(order):
        raise ConfigError(
            "runner arguments are not in canonical config-v1 order: "
            f"expected {list(order)}, found {list(parsed.order)}"
        )


def _validate_common_semantics(config: Mapping[str, Any]) -> None:
    seed = config["seed"]
    if seed > MAX_U64:
        raise ConfigError("$.seed exceeds unsigned 64-bit range")
    reference = config["reference"]
    if reference["repository"] != PHASE0_MODEL_ID:
        raise ConfigError("reference repository is not the immutable Phase 0 model")
    if reference["revision"] != PHASE0_REVISION:
        raise ConfigError("reference revision is not the immutable Phase 0 commit")
    if reference["tokenizer_fingerprint_sha256"] != PHASE0_TOKENIZER_FINGERPRINT:
        raise ConfigError("reference tokenizer fingerprint is not the Phase 0 pin")


def _validate_data_and_training_semantics(config: Mapping[str, Any]) -> None:
    architecture = config["architecture"]
    model = config["model"]
    training = config["training"]
    split = config["dataset"]["split"]
    tokenization = config["tokenization"]
    _relative_path(config["dataset"]["path"], "dataset.path")
    if split["seed"] != config["seed"]:
        raise ConfigError("dataset split seed does not match the config seed")
    if split["validation_basis_points"] + split["test_basis_points"] >= 10_000:
        raise ConfigError("validation and test basis points leave no train split")
    if tokenization["maximum_sequence_tokens"] != model["maximum_sequence_length"]:
        raise ConfigError("tokenization and model sequence limits differ")
    if tokenization["assistant_eom_id"] != tokenization["im_end_id"]:
        raise ConfigError("assistant EOM must be the pinned im_end token")
    if model["spiking"] != (architecture == "snn"):
        raise ConfigError("model.spiking does not match root architecture")
    if model["query_head_count"] % model["key_value_head_count"] != 0:
        raise ConfigError("query heads must be divisible by key/value heads")
    if (
        tokenization["maximum_sequence_tokens"]
        <= config["evaluation"]["generation"]["maximum_new_tokens"]
    ):
        raise ConfigError("generation completion must be smaller than context")
    if training["gradient_accumulation_steps"] < 1:
        raise ConfigError("gradient accumulation must be positive")


def _validate_compact(config: Mapping[str, Any]) -> None:
    architecture = config["architecture"]
    model = config["model"]
    training = config["training"]
    binding = config["runner_binding"]
    _validate_data_and_training_semantics(config)
    if model["model_dimension"] % model["query_head_count"] != 0:
        raise ConfigError("compact model width must be divisible by query heads")
    resolved_head = model["model_dimension"] // model["query_head_count"]
    if model["head_dimension"] != resolved_head:
        raise ConfigError(
            "compact head_dimension must equal model_dimension/query_head_count "
            "because config-v1 uses the runner's implicit compact head width"
        )
    arguments = binding["arguments"]
    parsed = _parse_runner_arguments(arguments)
    base_order = [
        f"--{architecture}",
        "--device",
        "--context-tokens",
        "--model-dimension",
        "--layers",
        "--query-heads",
        "--kv-heads",
        "--ffn-dimension",
        "--rms-epsilon",
        "--rope-base",
        "--simulation-steps",
    ]
    if architecture == "snn":
        base_order.extend(["--threshold", "--leak", "--surrogate-slope"])
    base_order.extend(
        [
            "--learning-rate",
            "--weight-decay",
            "--gradient-clip",
            "--gradient-accumulation",
        ]
    )
    if architecture == "snn":
        base_order.extend(["--spike-rate-target", "--spike-rate-penalty"])
    base_order.append("--seed")
    value_options = set(base_order) - {f"--{architecture}"}
    _require_argument_shape(
        parsed,
        flags={f"--{architecture}"},
        order=base_order,
        value_options=value_options,
    )

    values = parsed.values
    if values["--device"] != binding["device"]:
        raise ConfigError("runner --device does not match runner_binding.device")
    integer_fields = {
        "--context-tokens": model["maximum_sequence_length"],
        "--model-dimension": model["model_dimension"],
        "--layers": model["layer_count"],
        "--query-heads": model["query_head_count"],
        "--kv-heads": model["key_value_head_count"],
        "--ffn-dimension": model["feed_forward_dimension"],
        "--simulation-steps": model["simulation_steps"],
        "--gradient-accumulation": training["gradient_accumulation_steps"],
        "--seed": config["seed"],
    }
    for option, expected in integer_fields.items():
        _same_integer(values[option], expected, option)
    number_fields = {
        "--rms-epsilon": model["rms_epsilon"],
        "--rope-base": model["rope_base"],
        "--learning-rate": training["learning_rate"],
        "--weight-decay": training["weight_decay"],
        "--gradient-clip": training["gradient_clip_norm"],
    }
    if architecture == "snn":
        spiking = config["spiking"]
        if spiking["time_steps"] != model["simulation_steps"]:
            raise ConfigError(
                "spiking time_steps does not match model simulation_steps"
            )
        number_fields.update(
            {
                "--threshold": spiking["threshold"],
                "--leak": spiking["leak"],
                "--surrogate-slope": spiking["surrogate_slope"],
                "--spike-rate-target": spiking["spike_rate_target"],
                "--spike-rate-penalty": spiking["spike_rate_penalty"],
            }
        )
    for option, expected in number_fields.items():
        _same_number(values[option], expected, option)


def _validate_exact(config: Mapping[str, Any]) -> None:
    architecture = config["architecture"]
    model = config["model"]
    archive = config["archive"]
    binding = config["runner_binding"]
    training = config["training"]
    _validate_data_and_training_semantics(config)
    _relative_path(archive["path"], "archive.path")
    if archive["sha256"] != PHASE0_ARCHIVE_SHA256:
        raise ConfigError("archive SHA-256 is not the immutable Phase 0 conversion")
    if model["spiking"] != (architecture == "snn"):
        raise ConfigError("model.spiking does not match root architecture")
    if model["query_projection_dimension"] != (
        model["query_head_count"] * model["head_dimension"]
    ):
        raise ConfigError("query projection width does not match heads × head width")
    if model["key_value_projection_dimension"] != (
        model["key_value_head_count"] * model["head_dimension"]
    ):
        raise ConfigError(
            "key/value projection width does not match heads times head width"
        )

    parsed = _parse_runner_arguments(binding["arguments"])
    order = [
        "--qwen3-0.6b",
        "--qwen-archive",
        "--qwen-archive-sha256",
        f"--{architecture}",
        "--device",
        "--simulation-steps",
    ]
    if architecture == "snn":
        order.extend(["--threshold", "--leak", "--surrogate-slope"])
    order.extend(
        [
            "--learning-rate",
            "--weight-decay",
            "--gradient-clip",
            "--gradient-accumulation",
        ]
    )
    if architecture == "snn":
        order.extend(["--spike-rate-target", "--spike-rate-penalty"])
    order.append("--seed")
    value_options = set(order) - {"--qwen3-0.6b", f"--{architecture}"}
    _require_argument_shape(
        parsed,
        flags={"--qwen3-0.6b", f"--{architecture}"},
        order=order,
        value_options=value_options,
    )
    values = parsed.values
    if values["--qwen-archive"] != archive["path"]:
        raise ConfigError("runner archive path does not match archive.path")
    if values["--qwen-archive-sha256"] != archive["sha256"]:
        raise ConfigError("runner archive hash does not match archive.sha256")
    if values["--device"] != binding["device"]:
        raise ConfigError("runner --device does not match runner_binding.device")
    _same_integer(
        values["--simulation-steps"], model["simulation_steps"], "--simulation-steps"
    )
    _same_integer(
        values["--gradient-accumulation"],
        training["gradient_accumulation_steps"],
        "--gradient-accumulation",
    )
    _same_integer(values["--seed"], config["seed"], "--seed")
    for option, expected in {
        "--learning-rate": training["learning_rate"],
        "--weight-decay": training["weight_decay"],
        "--gradient-clip": training["gradient_clip_norm"],
    }.items():
        _same_number(values[option], expected, option)
    if architecture == "ann":
        if binding["device"] != config["oracle"]["device"]:
            raise ConfigError("runner device does not match the recorded oracle device")
        _relative_path(config["oracle"]["fixture"], "oracle.fixture")
        if (
            config["oracle"]["maximum_absolute_error"]
            > config["oracle"]["absolute_tolerance"]
        ):
            raise ConfigError(
                "recorded maximum absolute oracle error exceeds tolerance"
            )
        if (
            config["oracle"]["maximum_relative_error"]
            > config["oracle"]["relative_tolerance"]
        ):
            raise ConfigError(
                "recorded maximum relative oracle error exceeds tolerance"
            )
    else:
        spiking = config["spiking"]
        if spiking["time_steps"] != model["simulation_steps"]:
            raise ConfigError(
                "exact SNN time_steps do not match model simulation_steps"
            )
        if spiking["threshold_initialization"] <= 0.0:
            raise ConfigError("exact SNN threshold initialization must be positive")
        for option, expected in {
            "--threshold": spiking["threshold_initialization"],
            "--leak": spiking["leak_initialization"],
            "--surrogate-slope": spiking["surrogate_slope"],
            "--spike-rate-target": spiking["spike_rate_target"],
            "--spike-rate-penalty": spiking["spike_rate_penalty"],
        }.items():
            _same_number(values[option], expected, option)


def validate_config(
    document: Mapping[str, Any], *, source: str = "<memory>"
) -> ValidatedConfig:
    """Validate an in-memory config and return its safe core argv."""

    schema = _config_schema()
    _schema_validate(document, schema, schema, "$")
    _reject_nul_strings(document)
    _validate_common_semantics(document)
    if document["track"] == COMPACT_TRACK:
        _validate_compact(document)
    elif document["track"] == EXACT_TRACK:
        _validate_exact(document)
    else:  # The schema should have rejected this first.
        raise ConfigError(f"unsupported chatbot config track: {document['track']!r}")
    binding = document["runner_binding"]
    if binding["entrypoint"] != "chatbot_experiment serve":
        raise ConfigError("runner entrypoint is not chatbot_experiment serve")
    return ValidatedConfig(
        document=document,
        source=source,
        core_argv=("serve", *binding["arguments"]),
    )


def load_config(path: str | os.PathLike[str]) -> ValidatedConfig:
    """Load and validate one config from a regular UTF-8 JSON file."""

    selected = Path(path)
    document = _load_json(selected)
    if type(document) is not dict:
        raise ConfigError(f"config root must be an object: {selected}")
    return validate_config(document, source=str(selected))


def load_core_argv(path: str | os.PathLike[str]) -> list[str]:
    """Load/validate one config and return canonical argv beginning with serve."""

    return list(load_config(path).core_argv)


def core_argv(config: ValidatedConfig, executable: str | None = None) -> list[str]:
    """Return a new argv list, optionally prefixed by a concrete executable."""

    prefix = [executable] if executable is not None else []
    if executable is not None and (not executable or "\0" in executable):
        raise ConfigError("executable must be a non-empty string without NUL")
    return [*prefix, *config.core_argv]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser(
        "validate", help="validate one or more config files"
    )
    validate.add_argument("configs", nargs="+", type=Path)
    emit = subparsers.add_parser(
        "argv", help="emit a validated core argv as a JSON string array"
    )
    emit.add_argument("config", type=Path)
    emit.add_argument(
        "--executable",
        help="optional concrete chatbot_experiment path to prefix to the argv",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "validate":
            for path in arguments.configs:
                config = load_config(path)
                print(
                    json.dumps(
                        {
                            "architecture": config.architecture,
                            "config": config.source,
                            "experiment_id": config.experiment_id,
                            "ok": True,
                            "track": config.track,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            return 0
        config = load_config(arguments.config)
        print(
            json.dumps(
                core_argv(config, arguments.executable),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        return 0
    except (ConfigError, OSError) as error:
        print(f"chatbot_config: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
