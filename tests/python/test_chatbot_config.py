from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools"))

from chatbot_config import (  # noqa: E402
    COMPACT_TRACK,
    EXACT_TRACK,
    ConfigError,
    core_argv,
    load_config,
    load_core_argv,
    main,
    validate_config,
)

CONFIGS = {
    "ann": REPOSITORY / "configs" / "chatbot" / "ann-baseline.json",
    "snn": REPOSITORY / "configs" / "chatbot" / "snn-baseline.json",
    "qwen-ann": (REPOSITORY / "configs" / "chatbot" / "qwen3-0.6b-ann-import.json"),
    "qwen-snn": (REPOSITORY / "configs" / "chatbot" / "qwen3-0.6b-hybrid-snn.json"),
}


def document(name: str) -> dict[str, object]:
    return json.loads(CONFIGS[name].read_text(encoding="utf-8"))


def replace_argument(config: dict[str, object], option: str, value: str) -> None:
    binding = config["runner_binding"]
    assert isinstance(binding, dict)
    arguments = binding["arguments"]
    assert isinstance(arguments, list)
    index = arguments.index(option)
    arguments[index + 1] = value


class ChatbotConfigTests(unittest.TestCase):
    def test_all_four_checked_configs_validate_and_emit_core_argv(self) -> None:
        expected = {
            "ann": (COMPACT_TRACK, "ann", "cuda"),
            "snn": (COMPACT_TRACK, "snn", "cuda"),
            "qwen-ann": (EXACT_TRACK, "ann", "cpu"),
            "qwen-snn": (EXACT_TRACK, "snn", "cuda"),
        }
        for name, path in CONFIGS.items():
            with self.subTest(config=name):
                validated = load_config(path)
                track, architecture, device = expected[name]
                self.assertEqual(validated.track, track)
                self.assertEqual(validated.architecture, architecture)
                self.assertEqual(validated.core_argv[0], "serve")
                self.assertEqual(load_core_argv(path), list(validated.core_argv))
                device_index = validated.core_argv.index("--device")
                self.assertEqual(validated.core_argv[device_index + 1], device)
                emitted = core_argv(validated, "/opt/bin/chatbot_experiment")
                self.assertEqual(emitted[0], "/opt/bin/chatbot_experiment")
                self.assertEqual(emitted[1:], list(validated.core_argv))

    def test_cli_emits_unambiguous_json_argv(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(
                [
                    "argv",
                    str(CONFIGS["qwen-ann"]),
                    "--executable",
                    "./chatbot_experiment",
                ]
            )
        self.assertEqual(status, 0)
        argv = json.loads(output.getvalue())
        self.assertEqual(argv[:3], ["./chatbot_experiment", "serve", "--qwen3-0.6b"])
        self.assertIn("--qwen-archive-sha256", argv)
        device_index = argv.index("--device")
        self.assertEqual(argv[device_index + 1], "cpu")
        self.assertEqual(argv[-2:], ["--seed", "42"])

    def test_unknown_fields_and_wrong_json_types_fail_closed(self) -> None:
        extra = document("ann")
        model = extra["model"]
        assert isinstance(model, dict)
        model["ignored_future_field"] = 1
        with self.assertRaisesRegex(ConfigError, "unexpected field"):
            validate_config(extra)

        wrong_type = document("ann")
        wrong_type["seed"] = True
        with self.assertRaisesRegex(ConfigError, "expected integer"):
            validate_config(wrong_type)

        boolean_as_number = document("ann")
        status = boolean_as_number["implementation_status"]
        assert isinstance(status, dict)
        status["config_consumed_automatically"] = 1
        with self.assertRaisesRegex(ConfigError, "expected constant True"):
            validate_config(boolean_as_number)

    def test_duplicate_json_keys_are_rejected_before_validation(self) -> None:
        original = CONFIGS["ann"].read_text(encoding="utf-8")
        duplicated = original.replace(
            '  "seed": 42,', '  "seed": 42,\n  "seed": 42,', 1
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(duplicated, encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "duplicate JSON key: seed"):
                load_config(path)

    def test_architecture_geometry_and_training_argv_drift_is_rejected(self) -> None:
        geometry = document("ann")
        replace_argument(geometry, "--model-dimension", "128")
        with self.assertRaisesRegex(ConfigError, "--model-dimension.*does not match"):
            validate_config(geometry)

        training = document("ann")
        replace_argument(training, "--learning-rate", "0.001")
        with self.assertRaisesRegex(ConfigError, "--learning-rate.*does not match"):
            validate_config(training)

        mode = document("snn")
        binding = mode["runner_binding"]
        assert isinstance(binding, dict)
        arguments = binding["arguments"]
        assert isinstance(arguments, list)
        arguments[arguments.index("--snn")] = "--ann"
        with self.assertRaisesRegex(ConfigError, "architecture/preset flags"):
            validate_config(mode)

    def test_spiking_fields_must_match_runner_values(self) -> None:
        compact = document("snn")
        spiking = compact["spiking"]
        assert isinstance(spiking, dict)
        spiking["threshold"] = 1.25
        with self.assertRaisesRegex(ConfigError, "--threshold.*does not match"):
            validate_config(compact)

        exact = document("qwen-snn")
        replace_argument(exact, "--simulation-steps", "8")
        with self.assertRaisesRegex(ConfigError, "--simulation-steps.*does not match"):
            validate_config(exact)

    def test_exact_training_contract_binds_optimizer_and_publication_fields(
        self,
    ) -> None:
        for name in ("qwen-ann", "qwen-snn"):
            with self.subTest(config=name):
                value = document(name)
                training = value["training"]
                dataset = value["dataset"]
                publication = value["publication_gate"]
                model = value["model"]
                assert isinstance(training, dict)
                assert isinstance(dataset, dict)
                assert isinstance(publication, dict)
                assert isinstance(model, dict)
                self.assertEqual(training["optimizer"], "adamw")
                self.assertEqual(training["epochs"], 1)
                self.assertEqual(dataset["format"], "snnbase-chatbot-jsonl-v1")
                self.assertEqual(
                    publication["manifest_schema"], "snnbase-chatbot-run-v1"
                )
                self.assertEqual(model["maximum_sequence_length"], 512)
                self.assertEqual(model["simulation_steps"], 4)

        mismatch = document("qwen-ann")
        replace_argument(mismatch, "--weight-decay", "0.02")
        with self.assertRaisesRegex(ConfigError, "--weight-decay.*does not match"):
            validate_config(mismatch)

    def test_exact_archive_path_hash_and_qwen_preset_are_bound(self) -> None:
        wrong_path = document("qwen-ann")
        replace_argument(wrong_path, "--qwen-archive", "artifacts/other.snnq")
        with self.assertRaisesRegex(ConfigError, "archive path"):
            validate_config(wrong_path)

        wrong_hash = document("qwen-ann")
        replace_argument(wrong_hash, "--qwen-archive-sha256", "0" * 64)
        with self.assertRaisesRegex(ConfigError, "archive hash"):
            validate_config(wrong_hash)

        shape_override = document("qwen-ann")
        binding = shape_override["runner_binding"]
        assert isinstance(binding, dict)
        arguments = binding["arguments"]
        assert isinstance(arguments, list)
        arguments.extend(["--layers", "28"])
        with self.assertRaisesRegex(ConfigError, "value-option set mismatch"):
            validate_config(shape_override)

    def test_device_is_a_resolved_field_and_matches_oracle(self) -> None:
        mismatched = document("qwen-ann")
        binding = mismatched["runner_binding"]
        assert isinstance(binding, dict)
        binding["device"] = "cuda"
        with self.assertRaisesRegex(ConfigError, "--device"):
            validate_config(mismatched)

        invalid = document("ann")
        invalid_binding = invalid["runner_binding"]
        assert isinstance(invalid_binding, dict)
        invalid_binding["device"] = "tpu"
        with self.assertRaisesRegex(ConfigError, "permitted enum"):
            validate_config(invalid)

    def test_runner_options_must_be_unique_and_canonically_ordered(self) -> None:
        duplicate = document("ann")
        binding = duplicate["runner_binding"]
        assert isinstance(binding, dict)
        arguments = binding["arguments"]
        assert isinstance(arguments, list)
        arguments.extend(["--seed", "42"])
        with self.assertRaisesRegex(ConfigError, "duplicated: --seed"):
            validate_config(duplicate)

        reordered = document("ann")
        reordered_binding = reordered["runner_binding"]
        assert isinstance(reordered_binding, dict)
        reordered_args = reordered_binding["arguments"]
        assert isinstance(reordered_args, list)
        reordered_args[0:3] = ["--device", "cuda", "--ann"]
        with self.assertRaisesRegex(ConfigError, "canonical config-v1 order"):
            validate_config(reordered)

    def test_repository_relative_artifact_paths_cannot_escape(self) -> None:
        escaped = document("qwen-ann")
        archive = escaped["archive"]
        assert isinstance(archive, dict)
        archive["path"] = "../qwen-dense.snnq"
        replace_argument(escaped, "--qwen-archive", "../qwen-dense.snnq")
        with self.assertRaisesRegex(ConfigError, "repository-relative"):
            validate_config(escaped)

    def test_validation_does_not_mutate_the_document(self) -> None:
        value = document("ann")
        before = copy.deepcopy(value)
        validate_config(value)
        self.assertEqual(value, before)


if __name__ == "__main__":
    unittest.main()
