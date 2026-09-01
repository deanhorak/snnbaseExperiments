from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools"))

from qwen_contract import (  # noqa: E402
    PHASE0_MODEL_ID,
    PHASE0_REVISION,
    PHASE0_TOKENIZER_FINGERPRINT,
    sha256_file,
)
from qwen_oracle_gate import run_gate  # noqa: E402

FAKE_RUNNER = r"""#!/usr/bin/env python3
import json, sys
protocol = "snnbase.chatbot.tokens/v1"
for line in sys.stdin:
    request = json.loads(line)
    response = {"protocol": protocol, "request_id": request["request_id"], "ok": True}
    if request["op"] == "inspect":
        base = sum(request["input_ids"])
        response["final_position"] = {
            "probes": [
                {"token_id": token, "logit": base + token * 0.01}
                for token in request["probe_token_ids"]
            ],
            "top_k": [
                {"token_id": 99 - index, "logit": 20.0 - index}
                for index in range(request["top_k"])
            ],
        }
    else:
        response["shutdown"] = True
    print(json.dumps(response, sort_keys=True, separators=(",", ":")), flush=True)
    if request["op"] == "shutdown":
        break
"""


def _reference(path: Path, probe_delta: float = 0.0) -> None:
    document = {
        "schema_version": 1,
        "kind": "snnbase.qwen-reference-logits",
        "oracle": {
            "model_id": PHASE0_MODEL_ID,
            "revision": PHASE0_REVISION,
            "tokenizer_fingerprint_sha256": PHASE0_TOKENIZER_FINGERPRINT,
            "checkpoint_files": [
                {
                    "path": "model.safetensors",
                    "size_bytes": 1_192_135_096,
                    "sha256": "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
                }
            ],
        },
        "execution": {
            "device": "cpu",
            "model_dtype": "float32",
            "absolute_tolerance": 1.0e-5,
            "relative_tolerance": 1.0e-5,
        },
        "cases": [
            {
                "id": "fake-case",
                "input_ids": [1, 2, 3],
                "final_logits": {
                    "probes": [
                        {"token_id": 4, "logit": 6.04 + probe_delta},
                        {"token_id": 7, "logit": 6.07},
                    ],
                    "top_k": [
                        {"token_id": 99, "logit": 20.0, "decoded": "x"},
                        {"token_id": 98, "logit": 19.0, "decoded": "y"},
                    ],
                },
            }
        ],
    }
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")


class QwenOracleGateTests(unittest.TestCase):
    def test_bounded_probe_and_top_k_gate_passes_and_reports_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            runner = temporary / "fake-runner"
            runner.write_text(FAKE_RUNNER, encoding="utf-8")
            os.chmod(runner, 0o700)
            archive = temporary / "fake.snnq"
            archive.write_bytes(b"bounded-fake-archive")
            reference = temporary / "reference.json"
            _reference(reference)
            report = run_gate(runner, archive, sha256_file(archive), reference)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["failures"], [])
            self.assertEqual(set(report["cases"][0]["actual"]), {"probes", "top_k"})

            bad_reference = temporary / "bad-reference.json"
            _reference(bad_reference, probe_delta=0.25)
            failed = run_gate(runner, archive, sha256_file(archive), bad_reference)
            self.assertEqual(failed["status"], "failed")
            self.assertRegex(failed["failures"][0], "token 4 logit differs")


if __name__ == "__main__":
    unittest.main()
