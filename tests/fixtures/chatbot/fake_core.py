#!/usr/bin/env python3
"""Standard-library-only deterministic token-protocol core for offline tests."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROTOCOL = "snnbase.chatbot.tokens/v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("serve",))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--save-checkpoint", type=Path)
    parser.add_argument("--ignore-checkpoint-state", action="store_true")
    parser.add_argument("--perturb-reloaded-loss", action="store_true")
    arguments = parser.parse_args()
    operations: list[str] = []
    micro_batches = 0
    optimizer_steps = 0
    if arguments.checkpoint is not None and not arguments.ignore_checkpoint_state:
        restored = json.loads(arguments.checkpoint.read_text(encoding="utf-8"))
        micro_batches = restored["micro_batches"]
        optimizer_steps = restored["optimizer_steps"]

    def write_checkpoint() -> None:
        if arguments.save_checkpoint is None:
            return
        arguments.save_checkpoint.write_text(
            json.dumps(
                {
                    "micro_batches": micro_batches,
                    "optimizer_steps": optimizer_steps,
                    "operations": operations,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )

    for line in sys.stdin:
        request = json.loads(line)
        request_id = request.get("request_id")
        operation = request.get("op")
        operations.append(operation)
        response: dict[str, object] = {
            "protocol": PROTOCOL,
            "request_id": request_id,
            "ok": True,
        }
        if request.get("protocol") != PROTOCOL:
            response = {
                **response,
                "ok": False,
                "error": {"code": "unsupported_protocol", "message": "bad protocol"},
            }
        elif operation == "metadata":
            response["metadata"] = {
                "device": "fake",
                "parameter_count": 123,
                "vocabulary_size": 100,
                "maximum_sequence_length": 64,
                "model_dimension": 8,
                "layer_count": 1,
                "simulation_steps": 2,
                "head_dimension": 4,
                "query_key_normalization": True,
                "spiking": True,
                "training_state": {
                    "micro_batch_count": micro_batches,
                    "optimizer_step_count": optimizer_steps,
                    "pending_accumulation_steps": 0,
                    "learning_rate": 0.001,
                },
            }
        elif operation in ("train", "evaluate"):
            token_count = sum(request["loss_mask"][1:])
            if operation == "train":
                micro_batches += 1
                optimizer_steps += 1
            loss = 2.0 / (1.0 + optimizer_steps)
            if arguments.checkpoint is not None and arguments.perturb_reloaded_loss:
                loss += 0.25
            response["metrics"] = {
                "token_count": token_count,
                "correct_token_count": min(token_count, optimizer_steps),
                "loss": loss,
                "objective_loss": loss,
                "perplexity": math.exp(loss),
                "token_accuracy": min(token_count, optimizer_steps) / token_count,
                "mean_spike_rate": 0.125,
                "optimizer_updated": operation == "train",
                "micro_batch_count": micro_batches,
                "optimizer_step_count": optimizer_steps,
                "learning_rate": 0.001,
            }
        elif operation == "flush":
            response["training_state"] = {
                "micro_batch_count": micro_batches,
                "optimizer_step_count": optimizer_steps,
                "pending_accumulation_steps": 0,
                "learning_rate": 0.001,
            }
            write_checkpoint()
        elif operation == "shutdown":
            response["shutdown"] = True
        else:
            response = {
                **response,
                "ok": False,
                "error": {"code": "unsupported_operation", "message": str(operation)},
            }
        print(json.dumps(response, sort_keys=True, separators=(",", ":")), flush=True)
        if operation == "shutdown":
            break
    write_checkpoint()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
