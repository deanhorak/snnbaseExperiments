from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import temporal_llm as llm


class ByteTokenizer:
    def encode(self, text, **kwargs):
        if not text.isascii():
            raise ValueError("test tokenizer accepts ASCII only")
        return SimpleNamespace(ids=list(text.encode()), offsets=[(i, i + 1) for i in range(len(text))])

    def decode(self, ids, **kwargs):
        return bytes(ids).decode()

    def get_vocab_size(self):
        return 256

    def token_to_id(self, name):
        return None


class TemporalLlmTests(unittest.TestCase):
    def arguments(self, *argv):
        args = llm.build_parser().parse_args(["--profile", "tiny", *argv])
        llm.resolve_configuration(args)
        return args

    def test_qwen_parameter_estimate_matches_real_archive_inventory(self):
        profile = llm.load_profile("qwen3-0.6b")
        self.assertEqual(profile["calibration_headroom"], 16.0)
        plan = llm.resource_plan(profile)
        self.assertEqual(plan["dense_parameter_count"], 596049920)
        self.assertEqual(plan["fp32_weight_bytes"], 2384199680)
        self.assertEqual(plan["bf16_weight_bytes_if_supported_by_backend"], 1192099840)
        self.assertEqual(plan["last_position_logits_fp32_bytes_batch_one"] * 512,
                         plan["full_sequence_logits_fp32_bytes_batch_one"])

    def test_temporal_diagnostics_are_aggregated_by_element_count(self):
        calls = 0

        def request(op, **fields):
            nonlocal calls
            self.assertEqual(op, "diagnose")
            calls += 1
            count = 2 if calls == 1 else 6
            site = {"element_count": count, "saturated_count": calls - 1,
                    "absolute_error_sum": float(calls),
                    "absolute_input_sum": float(10 * calls),
                    "maximum_scale_ratio": 0.5 * calls,
                    "silent_nonzero_count": calls}
            return {"temporal_diagnostics": {"layers": [
                {"layer": 0, "attention": dict(site),
                 "feed_forward": dict(site)}]}}

        result = llm.diagnose_records(request, [
            {"input_ids": [1, 2]}, {"input_ids": [3, 4, 5]}])
        summary = result["summary"]["attention"]
        self.assertEqual(summary["element_count"], 8)
        self.assertEqual(summary["saturated_count"], 1)
        self.assertEqual(summary["maximum_scale_ratio"], 1.0)
        self.assertAlmostEqual(summary["mean_absolute_error"], 3 / 8)
        self.assertAlmostEqual(summary["relative_absolute_error"], 3 / 30)

    def test_plan_never_loads_tokenizer_or_starts_process(self):
        with patch.object(llm, "load_tokenizer", side_effect=AssertionError("tokenizer")), \
             patch.object(llm, "Session", side_effect=AssertionError("process")), \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(llm.main(["--profile", "qwen3-0.6b", "--plan"]), 0)
        self.assertEqual(json.loads(output.getvalue())["profile"], "qwen3-0.6b")

    def test_arbitrary_larger_untied_profile_is_plannable(self):
        profile = llm.load_profile("tiny")
        profile["model"].update(model_dimension=4096, layer_count=32, feed_forward_dimension=14336,
                                query_head_count=32, key_value_head_count=8, head_dimension=128)
        tied = llm.resource_plan(llm.validate_profile(profile))["dense_parameter_count"]
        profile["model"]["tied_embeddings"] = False
        untied = llm.resource_plan(llm.validate_profile(profile))["dense_parameter_count"]
        self.assertEqual(untied - tied, 151936 * 4096)
        args = self.arguments()
        command = llm.build_core_command(profile, args)
        self.assertIn("--untied-embeddings", command)
        self.assertNotIn("--qwen3-0.6b", command)

    def test_invalid_geometry_and_pinned_shape_drift_rejected(self):
        for key, value in (("layer_count", True), ("head_dimension", 7),
                           ("simulation_steps", 25),
                           ("temporal_decay", float("nan")), ("temporal_decay", 1.0)):
            profile = llm.load_profile("tiny")
            profile["model"][key] = value
            with self.assertRaises(llm.ContractError):
                llm.validate_profile(profile)
        profile = llm.load_profile("qwen3-0.6b")
        profile["model"]["model_dimension"] = 2048
        with self.assertRaisesRegex(llm.ContractError, "exact dense geometry"):
            llm.validate_profile(profile)
        profile = llm.load_profile("qwen3-0.6b")
        profile["calibration_headroom"] = float("inf")
        with self.assertRaisesRegex(llm.ContractError, "calibration_headroom"):
            llm.validate_profile(profile)

    def test_import_and_ann_commands_keep_temporal_mode_explicit(self):
        args = self.arguments()
        profile = llm.load_profile("qwen3-0.6b")
        command = llm.build_core_command(profile, args, ann=True)
        self.assertIn("--qwen3-0.6b", command)
        self.assertIn("--qwen-archive-sha256", command)
        self.assertLess(command.index("--temporal-spiking"), command.index("--ann"))
        self.assertNotIn("--model-dimension", command)

    def test_response_mask_and_corpus_limit(self):
        record = llm.encoded_record(ByteTokenizer(), {"prompt": "abc", "response": "de"}, 8)
        self.assertEqual(record["loss_mask"], [0, 0, 0, 1, 1])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.jsonl"
            path.write_text('{"text":"abc"}\n{"text":"def"}\nnot json\n')
            records = llm.load_corpus(path, ByteTokenizer(), 8, 2)
            self.assertEqual(len(records), 2)
        with self.assertRaisesRegex(llm.ContractError, "2..2"):
            llm.encoded_record(ByteTokenizer(), "abc", 2)

    def test_identical_evaluation_data_is_rejected_before_model_load(self):
        row = llm.encoded_record(ByteTokenizer(), "abc", 10)
        with self.assertRaisesRegex(llm.ContractError, "calibration"):
            llm.check_eval_disjoint({"calibration": [row], "evaluation": [row]})
        llm.check_eval_disjoint({"calibration": [row], "training": [row]})

    def test_evaluation_aggregates_token_weighted_loss(self):
        rows = [llm.encoded_record(ByteTokenizer(), text, 10) for text in ("abc", "defg")]

        def request(op, **fields):
            if op == "inspect":
                return {"final_position": {"top_k": [{"token_id": fields["probe_token_ids"][0]}]}}
            count = len(fields["input_ids"]) - 1
            return {"metrics": {"token_count": count, "loss": float(count),
                                "correct_token_count": 1, "mean_spike_rate": 0.25}}

        result = llm.evaluate_records(request, rows)
        self.assertEqual(result["predicted_tokens"], 5)
        self.assertAlmostEqual(result["loss"], 13 / 5)
        self.assertEqual(result["final_position_predictions"], [99, 103])

    def test_generation_sets_decodable_vocabulary_and_rejects_padded_rows(self):
        args = self.arguments("--max-new-tokens", "2")
        seen = []

        def request(op, **fields):
            seen.append(fields)
            return {"output_ids": [65, 66], "finish_reason": "length"}

        result = llm.generate(request, ByteTokenizer(), "abc", llm.load_profile("tiny"), args)
        self.assertEqual(result["text"], "AB")
        self.assertEqual(seen[0]["sampling_vocabulary_size"], 256)
        self.assertEqual(seen[0]["prefill_chunk_size"], 128)
        with self.assertRaisesRegex(llm.ContractError, "decodable"):
            llm.generate(lambda *a, **k: {"output_ids": [256]}, ByteTokenizer(), "abc", llm.load_profile("tiny"), args)

    def test_benchmark_reuses_session_retains_warmups_and_excludes_them_from_medians(self):
        calls, starts = [], []

        class FakeSession:
            def __init__(self, command):
                starts.append(command)

            def request(self, op, **fields):
                if op == "metadata":
                    return {"metadata": {"device": "cpu"}}
                self_case.assertEqual(op, "generate")
                index = len(calls)
                calls.append(fields)
                latency = [1000.0, 800.0, 20.0, 10.0, 30.0][index]
                return {"output_ids": [65 + index], "finish_reason": "length",
                        "metrics": {"generated_tokens": 1, "latency_ms": latency,
                                    "time_to_first_token_ms": latency / 2,
                                    "generated_tokens_per_second": 1000 / latency}}

            def close(self):
                calls.append("closed")

        self_case = self
        args = self.arguments("--prompt", "abc", "--max-new-tokens", "1",
                              "--benchmark-repeats", "3")
        with patch.object(llm, "Session", FakeSession), \
             patch.object(llm, "load_tokenizer", return_value=ByteTokenizer()), \
             patch.object(llm.time, "perf_counter", side_effect=range(12)), \
             patch("sys.stdout", new_callable=io.StringIO):
            result = llm.run(args)
        bench = result["benchmark"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(calls[-1], "closed")
        self.assertEqual(len(calls[:-1]), 5)
        self.assertTrue(all(fields == calls[0] for fields in calls[:-1]))
        self.assertEqual([row["output_ids"] for row in bench["warmups"]], [[65], [66]])
        self.assertEqual([row["output_ids"] for row in bench["samples"]], [[67], [68], [69]])
        self.assertEqual(bench["summary"]["median_latency_ms"], 20)
        self.assertEqual(bench["summary"]["median_time_to_first_token_ms"], 10)
        self.assertEqual(bench["summary"]["median_generated_tokens_per_second"], 50)
        self.assertEqual(bench["summary"]["median_wall_ms"], 1000)
        self.assertEqual(bench["summary"]["sum_sample_wall_ms"], 3000)
        self.assertEqual(bench["total_wall_ms"], 11000)
        self.assertFalse(bench["summary"]["all_output_ids_identical"])
        self.assertTrue(bench["warmups_excluded_from_summary"])
        self.assertEqual(result["generation"]["output_ids"], [67])

    def test_benchmark_argument_bounds_rejected_before_loading(self):
        for argv in (
            ["--prompt", "abc", "--benchmark-repeats", "-1"],
            ["--prompt", "abc", "--benchmark-repeats", "101"],
            ["--prompt", "abc", "--benchmark-repeats", "2", "--benchmark-warmups", "-1"],
            ["--prompt", "abc", "--benchmark-repeats", "2", "--benchmark-warmups", "21"],
            ["--prompt", "abc", "--benchmark-warmups", "0"],
            ["--benchmark-repeats", "2"],
            ["--chat", "--benchmark-repeats", "2"],
            ["--plan", "--prompt", "abc", "--benchmark-repeats", "2"],
        ):
            with self.subTest(argv=argv), \
                 patch.object(llm, "load_tokenizer", side_effect=AssertionError("tokenizer")), \
                 patch.object(llm, "Session", side_effect=AssertionError("process")), \
                 self.assertRaises(llm.ContractError):
                llm.run(llm.build_parser().parse_args(argv))

    def test_benchmark_zero_warmups_and_invalid_core_metrics(self):
        args = self.arguments("--prompt", "abc", "--max-new-tokens", "1",
                              "--benchmark-repeats", "1", "--benchmark-warmups", "0")
        response = {"output_ids": [65], "finish_reason": "length",
                    "metrics": {"generated_tokens": 1, "latency_ms": 10.0,
                                "time_to_first_token_ms": 8.0, "generated_tokens_per_second": 100.0}}
        result = llm.benchmark_generation(lambda *a, **k: copy.deepcopy(response),
                                         ByteTokenizer(), llm.load_profile("tiny"), args)
        self.assertEqual(result["warmups"], [])
        self.assertEqual(len(result["samples"]), 1)
        self.assertTrue(result["summary"]["all_output_ids_identical"])
        for name, value in (("latency_ms", float("nan")), ("latency_ms", True),
                            ("time_to_first_token_ms", 11), ("generated_tokens", 2),
                            ("generated_tokens_per_second", -1)):
            invalid = copy.deepcopy(response)
            invalid["metrics"][name] = value
            with self.subTest(name=name, value=value), self.assertRaises(llm.ContractError):
                llm.benchmark_generation(lambda *a, **k: invalid,
                                         ByteTokenizer(), llm.load_profile("tiny"), args)

    def test_checkpoint_resume_restores_settings_and_checks_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            checkpoint.write_bytes(b"test weights")
            profile = llm.load_profile("tiny")
            profile["model"]["simulation_steps"] = 8
            sidecar = {"kind": "snnbase.temporal-llm-checkpoint", "schema_version": 1,
                       "profile": profile, "checkpoint_sha256": llm.sha256_file(checkpoint),
                       "runner": {"ann": False, "seed": 7, "learning_rate": 0.002, "gradient_accumulation": 2}}
            llm.write_report(llm.checkpoint_sidecar(checkpoint), sidecar)
            args = llm.build_parser().parse_args(["--checkpoint", str(checkpoint)])
            restored, _ = llm.resolve_configuration(args)
            self.assertEqual(restored["model"]["simulation_steps"], 8)
            self.assertEqual(args.learning_rate, 0.002)
            args = llm.build_parser().parse_args(["--checkpoint", str(checkpoint), "--seed", "8"])
            with self.assertRaisesRegex(llm.ContractError, "must match"):
                llm.resolve_configuration(args)
            inference = llm.build_parser().parse_args(["--checkpoint", str(checkpoint),
                "--weights-only", "--device", "cuda", "--seed", "8", "--learning-rate", "0.0001"])
            inference_profile, _ = llm.resolve_configuration(inference)
            self.assertEqual(inference.seed, 8)
            self.assertIn("--weights-only", llm.build_core_command(inference_profile, inference))
            invalid = llm.build_parser().parse_args(["--checkpoint", str(checkpoint),
                "--weights-only", "--train-corpus", "train.txt"])
            with self.assertRaisesRegex(llm.ContractError, "inference/evaluation only"):
                llm.resolve_configuration(invalid)
            checkpoint.write_bytes(b"wrong weights")
            with self.assertRaisesRegex(llm.ContractError, "content does not match"):
                llm.resolve_configuration(llm.build_parser().parse_args(["--checkpoint", str(checkpoint)]))

    def test_custom_archive_identity_is_expanded_and_tokenizer_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "identity.json"
            profile = llm.load_profile("tiny")
            identity = {"model_id": "tests/custom", "revision": "1" * 40,
                        "source_checkpoint_sha256": "2" * 64, "config_sha256": "3" * 64,
                        "tokenizer_fingerprint_sha256": profile["tokenizer"]["fingerprint_sha256"]}
            path.write_text(json.dumps(identity))
            profile["initialization"] = {"type": "qwen_archive", "path": "test.snnq", "sha256": "4" * 64,
                                         "identity_path": str(path)}
            args = self.arguments()
            command = llm.build_core_command(llm.validate_profile(profile), args)
            self.assertIn("--qwen-model-id", command)
            self.assertNotIn("--qwen3-0.6b", command)
            identity["tokenizer_fingerprint_sha256"] = "5" * 64
            path.write_text(json.dumps(identity))
            with self.assertRaisesRegex(llm.ContractError, "tokenizer disagree"):
                llm.build_core_command(profile, args)

    def test_calibration_training_and_resume_flow_with_a_protocol_double(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            train, calibration, evaluation = (base / name for name in ("train.txt", "calibration.txt", "eval.txt"))
            train.write_text("abcd\nefgh\n")
            calibration.write_text("ijkl\n")
            evaluation.write_text("mnop\n")
            checkpoint = base / "model.pt"
            calls = []

            class FakeSession:
                def __init__(self, command):
                    self.command = command

                def request(self, op, **fields):
                    calls.append((op, fields))
                    if op == "metadata":
                        return {"metadata": {"device": "cpu"}}
                    if op == "calibrate":
                        return {"calibrated_tokens": len(fields["input_ids"])}
                    if op == "flush":
                        checkpoint.write_bytes(b"checkpoint fixture")
                        return {"training_state": {"optimizer_step_count": 3}}
                    if op in ("train", "evaluate"):
                        return {"metrics": {"token_count": 3, "correct_token_count": 1,
                                            "loss": 2.0, "mean_spike_rate": 0.1}}
                    if op == "inspect":
                        return {"final_position": {"top_k": [{"token_id": 65}]}}
                    raise AssertionError(op)

                def close(self):
                    calls.append(("closed", {}))

            args = self.arguments("--calibration-corpus", str(calibration), "--train-corpus", str(train),
                                  "--eval-corpus", str(evaluation), "--train-steps", "3",
                                  "--save-checkpoint", str(checkpoint))
            with patch.object(llm, "Session", FakeSession), patch.object(llm, "load_tokenizer", return_value=ByteTokenizer()):
                report = llm.run(args)
                self.assertEqual(len(report["training"]), 3)
                self.assertEqual([data["reset_state"] for op, data in calls if op == "calibrate"], [True])
                self.assertEqual([data["calibration_headroom"] for op, data in calls if op == "calibrate"], [4.0])
                restored = llm.read_json(llm.checkpoint_sidecar(checkpoint))
                self.assertEqual(len(restored["learning_record_sha256"]), 3)
                resume = llm.build_parser().parse_args(["--checkpoint", str(checkpoint), "--eval-corpus", str(train)])
                with self.assertRaisesRegex(llm.ContractError, "resumed checkpoint learning"):
                    llm.run(resume)


if __name__ == "__main__":
    unittest.main()
