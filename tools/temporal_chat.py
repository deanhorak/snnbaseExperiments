#!/usr/bin/env python3
"""CPU temporal-SNN token experiment. No pretrained language-model weights."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time
from typing import Any

from qwen_contract import (
    ContractError, PHASE0_MODEL_ID, PHASE0_REVISION,
    PHASE0_TOKENIZER_FINGERPRINT, PHASE0_TOKENIZER_JSON_SHA256,
    PHASE0_TOKENIZER_CONFIG_SHA256, load_json, package_versions,
    sha256_file, tokenizer_fingerprint, write_json_atomic,
)

ROOT = Path(__file__).resolve().parents[1]
MAX_LINE = 1_048_576
STOP = 151645

def load_tokenizer(root: Path) -> tuple[Any, dict[str, Any]]:
    """Reuse the existing contract's hash logic, checking tokenizer files only.

    The full qwen-assets manifest can include >1 GB of model weights. They are
    intentionally neither opened nor required by this separate experiment.
    """
    fingerprint, files = tokenizer_fingerprint(root)
    if fingerprint != PHASE0_TOKENIZER_FINGERPRINT:
        raise ContractError("tokenizer fingerprint does not match the pinned Qwen3 contract")
    actual = {record["path"]: record["sha256"] for record in files}
    if actual.get("tokenizer.json") != PHASE0_TOKENIZER_JSON_SHA256 or actual.get("tokenizer_config.json") != PHASE0_TOKENIZER_CONFIG_SHA256:
        raise ContractError("pinned tokenizer hashes disagree")
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    except (ImportError, OSError, ValueError) as error:
        raise ContractError(f"install tokenizers and provide the pinned local snapshot: {error}") from error
    if tokenizer.token_to_id("<|im_end|>") != STOP:
        raise ContractError("unexpected upstream stop token")
    return tokenizer, {"model_id": PHASE0_MODEL_ID, "revision": PHASE0_REVISION,
                       "fingerprint_sha256": fingerprint, "files": files,
                       "model_weights_used": False, "packages": package_versions(["tokenizers"])}

def validate_ids(ids: Any, maximum: int, valid_ids: set[int]) -> list[int]:
    if not isinstance(ids, list) or not 1 <= len(ids) <= maximum:
        raise ContractError(f"token array must contain 1..{maximum} entries")
    if any(type(token) is not int or token not in valid_ids for token in ids):
        raise ContractError("token array contains an invalid upstream token ID")
    return ids

def encode_prompt(tokenizer: Any, text: str, valid_ids: set[int]) -> list[int]:
    if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 8192:
        raise ContractError("prompt must be non-empty and at most 8192 UTF-8 bytes")
    # This is our toy training format, not an assertion of Qwen chat-template parity.
    ids = tokenizer.encode(f"User: {text}\nAssistant:", add_special_tokens=False).ids
    if any(token in {151643, 151644, 151645} for token in ids):
        raise ContractError("prompt may not contain reserved conversation tokens")
    return validate_ids(ids, 128, valid_ids)

def prepare_corpus(path: Path, tokenizer: Any, valid_ids: set[int]) -> tuple[list[Any], list[Any]]:
    if path.stat().st_size > 1_048_576:
        raise ContractError("corpus exceeds 1 MiB")
    rows = load_json(path)
    if not isinstance(rows, list) or not 2 <= len(rows) <= 160:
        raise ContractError("corpus must have 2..160 rows")
    train, held, seen = [], [], set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"split", "prompt", "response"}:
            raise ContractError("corpus rows require exactly split, prompt, response")
        if not isinstance(row["split"], str) or row["split"] not in {"train", "held_out"}:
            raise ContractError("unknown corpus split")
        prompt = encode_prompt(tokenizer, row["prompt"], valid_ids)
        if tuple(prompt) in seen:
            raise ContractError("duplicate or overlapping tokenized corpus prompts")
        seen.add(tuple(prompt))
        response = row["response"]
        if not isinstance(response, str) or not response.strip() or len(response.encode("utf-8")) > 2048:
            raise ContractError("invalid corpus response")
        answer = tokenizer.encode(response, add_special_tokens=False).ids
        if any(token in {151643, 151644, 151645} for token in answer):
            raise ContractError("corpus response contains reserved conversation tokens")
        answer = validate_ids(answer + [STOP], 32, valid_ids)
        (train if row["split"] == "train" else held).append((prompt, answer, row))
    if not 1 <= len(train) <= 128 or not 1 <= len(held) <= 32:
        raise ContractError("requires 1..128 training and 1..32 held-out rows")
    output_ids = {token for _, answer, _ in train for token in answer}
    if not 2 <= len(output_ids) <= 512:
        raise ContractError("training output vocabulary must have 2..512 tokens")
    if any(token not in output_ids for _, answer, _ in held for token in answer):
        raise ContractError("held-out response uses a token absent from training outputs")
    return train, held

def pair_line(pair: Any) -> str:
    prompt, answer, _ = pair
    return "PAIR " + " ".join(map(str, [len(prompt), len(answer), *prompt, *answer]))

class Core:
    def __init__(self, executable: Path):
        self.stderr = tempfile.TemporaryFile()
        self.process = subprocess.Popen([str(executable.resolve())], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.stderr)
        self.buffer = b""

    def send(self, value: str) -> None:
        data = (value + "\n").encode("ascii")
        if len(data) > 65537:
            raise ContractError("outbound protocol line too long")
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(data); self.process.stdin.flush()
        except BrokenPipeError as error:
            raise ContractError(self.error()) from error

    def error(self) -> str:
        self.stderr.seek(0)
        return self.stderr.read(4096).decode("utf-8", errors="replace") or "temporal chat core exited"

    def receive(self) -> dict[str, Any]:
        assert self.process.stdout is not None
        deadline = time.monotonic() + 120
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise ContractError("temporal chat core response timed out")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise ContractError(self.error())
            self.buffer += chunk
            if len(self.buffer) > MAX_LINE:
                raise ContractError("core response exceeds 1 MiB")
        raw, self.buffer = self.buffer.split(b"\n", 1)
        try:
            result = json.loads(raw)
        except (UnicodeError, ValueError) as error:
            raise ContractError("core emitted invalid JSON") from error
        if not isinstance(result, dict) or result.get("ok") is not True:
            raise ContractError("core returned unsuccessful response")
        return result

    def close(self) -> None:
        if self.process.stdin:
            try: self.process.stdin.close()
            except BrokenPipeError: pass
        try: self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill(); self.process.wait()
        if self.process.stdout: self.process.stdout.close()
        self.stderr.close()

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, default=ROOT / "build-temporal-chat/temporal_chat")
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "artifacts/chatbot/qwen3-0.6b-base")
    parser.add_argument("--corpus", type=Path, default=ROOT / "experiments/temporal_chat/corpus.json")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="What is the color of grass?")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--chat", action="store_true", help="independent single-turn prompts; /quit exits")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--no-stdp", action="store_true", help="skip unsupervised training-only input-synapse warmup")
    parser.add_argument("--lesion-reservoir", action="store_true", help="zero reservoir spike features; bias-only control")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    core = None
    try:
        if not 1 <= args.epochs <= 300 or not 0 <= args.seed <= 2**32 - 1 or not 1 <= args.max_new_tokens <= 64:
            raise ContractError("epochs, seed, or generation length outside limits")
        tokenizer, provenance = load_tokenizer(args.tokenizer_dir)
        valid_ids = set(tokenizer.get_vocab().values())
        train, held = prepare_corpus(args.corpus, tokenizer, valid_ids)
        core = Core(args.core)
        core.send(f"SNNBASE_TEMPORAL_CHAT_V1 {args.seed} {args.epochs} {STOP} {len(train)} {len(held)} {int(not args.no_stdp)} {int(args.lesion_reservoir)}")
        for pair in train + held: core.send(pair_line(pair))
        metrics = core.receive()
        if metrics.get("event") != "trained" or metrics.get("protocol") != "snnbase-temporal-chat-v1":
            raise ContractError("unexpected training response")
        output_vocabulary = validate_ids(metrics.get("output_vocabulary"), 512, valid_ids)
        expected_vocabulary = sorted({token for _, answer, _ in train for token in answer})
        if output_vocabulary != expected_vocabulary:
            raise ContractError("core output vocabulary differs from training targets")
        output_ids = set(output_vocabulary)
        for row, (_, expected, source) in zip(metrics["held_out_generations"], held, strict=True):
            generated = validate_ids(row["ids"], 32, output_ids)
            if row.get("expected_ids") != expected or STOP in generated[:-1]:
                raise ContractError("core held-out generation/target contract mismatch")
            row.update(prompt=source["prompt"], expected=source["response"],
                       text=tokenizer.decode(row["ids"], skip_special_tokens=True))
        report = {"schema": "snnbase-temporal-chat-run-v1", "tokenizer": provenance,
                  "corpus_sha256": sha256_file(args.corpus), "core_sha256": sha256_file(args.core),
                  "metrics": metrics, "generations": [],
                  "scope": "authored six-intent toy; reservoir STDP plus supervised categorical readout; no pretrained model weights"}
        print(f"Held-out token loss: {metrics['trained_held_out']['loss']:.4f} "
              f"(untrained {metrics['untrained_held_out']['loss']:.4f}, unigram {metrics['unigram_held_out']['loss']:.4f}); "
              f"exact replies: {metrics['held_out_exact_match']:.1%}", file=sys.stderr)

        def generate(text: str) -> None:
            prompt = encode_prompt(tokenizer, text, valid_ids)
            core.send("GENERATE " + " ".join(map(str, [args.max_new_tokens, len(prompt), *prompt])))
            result = core.receive()
            if result.get("event") != "generated": raise ContractError("unexpected generation response")
            generated = validate_ids(result.get("ids"), args.max_new_tokens, output_ids)
            if STOP in generated[:-1] or result.get("stopped") is not (generated[-1] == STOP):
                raise ContractError("core generation stop contract mismatch")
            result.update(prompt=text, text=tokenizer.decode(result["ids"], skip_special_tokens=True))
            report["generations"].append(result); print(result["text"])

        if args.chat:
            print("Toy six-intent chat: ask colors of tomato/sky/grass or sounds of cat/dog/bird. /quit exits.", file=sys.stderr)
            while True:
                try: prompt = input("You: ")
                except EOFError: break
                if prompt.strip() == "/quit": break
                try: generate(prompt)
                except ContractError as error: print(str(error), file=sys.stderr)
        elif not args.evaluate_only: generate(args.prompt)
        if args.report: write_json_atomic(args.report, report)
        core.send("QUIT")
        return 0
    except (ContractError, OSError, ValueError, KeyError) as error:
        print(f"temporal_chat: {error}", file=sys.stderr); return 2
    finally:
        if core is not None: core.close()

if __name__ == "__main__":
    raise SystemExit(main())
