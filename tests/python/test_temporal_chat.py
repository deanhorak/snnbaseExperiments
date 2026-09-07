#!/usr/bin/env python3
"""Protocol/causality tests; verified real-tokenizer smoke is optional offline."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import temporal_chat as chat
from qwen_contract import ContractError

CORE = Path(os.environ.get("TEMPORAL_CHAT_CORE", ROOT / "build-temporal-chat/temporal_chat"))
ASSETS = Path(os.environ.get("TEMPORAL_CHAT_TOKENIZER_DIR", ROOT / "artifacts/chatbot/qwen3-0.6b-base"))

class FakeTokenizer:
    @staticmethod
    def encode(text, add_special_tokens=False):
        return type("Encoding", (), {"ids": list(text.encode("ascii"))})()

class ValidationTests(unittest.TestCase):
    def test_invalid_ids_are_rejected(self):
        for value in [[], [True], [-1], [999], [1.0], "1", [1, 2, 3]]:
            with self.assertRaises(ContractError): chat.validate_ids(value, 2, {1, 2})

    def test_duplicate_tokenized_prompts_are_rejected(self):
        rows = [{"split": "train", "prompt": "hi", "response": "ok"},
                {"split": "held_out", "prompt": "hi", "response": "ok"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ContractError, "duplicate|overlapping"):
                chat.prepare_corpus(path, FakeTokenizer(), set(range(256)) | {chat.STOP})

    def test_empty_and_oversize_prompts_are_rejected(self):
        for prompt in ["", " ", "x" * 8193]:
            with self.assertRaises(ContractError): chat.encode_prompt(FakeTokenizer(), prompt, set(range(256)))

@unittest.skipUnless(CORE.is_file(), "build temporal_chat or set TEMPORAL_CHAT_CORE")
class ProtocolTests(unittest.TestCase):
    def run_core(self, payload):
        return subprocess.run([str(CORE)], input=payload, text=True, capture_output=True, timeout=120)

    def test_bad_requests_fail_closed(self):
        for payload in ["WRONG\n", "SNNBASE_TEMPORAL_CHAT_V1 -1 1 99 1 1 0 0\n",
                        "SNNBASE_TEMPORAL_CHAT_V1 42 301 99 1 1 0 0\n", "x" * 65537 + "\n"]:
            with self.subTest(payload=payload[:60]):
                result = self.run_core(payload)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(result.stdout)

    def test_overlap_and_embedded_stop_are_rejected(self):
        for train, held in [
            ("PAIR 1 2 1 10 99", "PAIR 1 2 1 10 99"),
            ("PAIR 1 3 1 99 10 99", "PAIR 1 2 2 10 99"),
        ]:
            result = self.run_core(f"SNNBASE_TEMPORAL_CHAT_V1 42 1 99 1 1 0 0\n{train}\n{held}\n")
            self.assertEqual(result.returncode, 2)

    def test_training_and_generation_protocol(self):
        request = "\n".join([
            "SNNBASE_TEMPORAL_CHAT_V1 42 120 99 4 2 1 0",
            "PAIR 3 3 1 2 7 10 12 99", "PAIR 3 3 1 3 7 11 12 99",
            "PAIR 3 3 4 2 7 10 12 99", "PAIR 3 3 4 3 7 11 12 99",
            "PAIR 3 3 5 2 7 10 12 99", "PAIR 3 3 5 3 7 11 12 99",
            "GENERATE 8 3 1 2 7", "GENERATE 8 3 1 3 7", "QUIT", ""])
        result = self.run_core(request)
        self.assertEqual(result.returncode, 0, result.stderr)
        trained, left, right = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(trained["event"], "trained")
        self.assertGreater(trained["stdp_absolute_weight_change"], 0)
        self.assertLess(trained["trained_held_out"]["loss"], trained["untrained_held_out"]["loss"])
        self.assertEqual(left["ids"], [10, 12, 99])
        self.assertEqual(right["ids"], [11, 12, 99])
        self.assertGreater(left["spikes"], 0)

@unittest.skipUnless(CORE.is_file() and (ASSETS / "tokenizer.json").is_file()
                     and importlib.util.find_spec("tokenizers") is not None,
                     "verified local Qwen tokenizer and core required; no downloads in tests")
class RealTokenizerTests(unittest.TestCase):
    def test_real_tokenizer_training_to_text(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "run.json"
            result = subprocess.run([sys.executable, str(ROOT / "tools/temporal_chat.py"),
                "--core", str(CORE), "--tokenizer-dir", str(ASSETS), "--report", str(report_path),
                "--prompt", "What is the color of grass?"], capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(report_path.read_text())
            metrics = report["metrics"]
            self.assertFalse(report["tokenizer"]["model_weights_used"])
            self.assertGreater(metrics["stdp_absolute_weight_change"], 0)
            self.assertLess(metrics["trained_held_out"]["loss"], metrics["untrained_held_out"]["loss"])
            self.assertLess(metrics["trained_held_out"]["loss"], metrics["unigram_held_out"]["loss"])
            self.assertGreaterEqual(metrics["held_out_exact_match"], 0.5)
            self.assertEqual(result.stdout.strip(), "green.")

if __name__ == "__main__": unittest.main()
