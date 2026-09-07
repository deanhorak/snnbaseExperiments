#!/usr/bin/env python3
"""Offline, profile-driven snnbase temporal spiking language-model experiment.

--prompt is raw base-model continuation. --chat supplies a simple conversation
prefix; Qwen Base weights have not been instruction tuned by this experiment.
--plan reads configuration only and never loads a tokenizer, weights or Torch.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

from chatbot_token_protocol import CoreProcess, PROTOCOL, ProtocolError
from qwen_contract import ContractError, require_sha256, sha256_file, tokenizer_fingerprint

ROOT = Path(__file__).resolve().parents[1]
MAX_LINE_BYTES = 4 * 1024 * 1024
MAX_CORPUS_LINE_BYTES = 1024 * 1024
GEOMETRY = {
    "vocabulary_size": "--vocabulary-size",
    "maximum_sequence_length": "--context-tokens",
    "model_dimension": "--model-dimension",
    "layer_count": "--layers",
    "query_head_count": "--query-heads",
    "key_value_head_count": "--kv-heads",
    "head_dimension": "--head-dimension",
    "feed_forward_dimension": "--ffn-dimension",
}
FLOATS = {"rms_epsilon", "rope_base", "temporal_decay"}
MODEL_FIELDS = set(GEOMETRY) | FLOATS | {
    "simulation_steps", "query_key_normalization", "tied_embeddings"
}
IDENTITY_FLAGS = {
    "model_id": "--qwen-model-id",
    "revision": "--qwen-revision",
    "source_checkpoint_sha256": "--qwen-source-sha256",
    "config_sha256": "--qwen-config-sha256",
    "tokenizer_fingerprint_sha256": "--qwen-tokenizer-sha256",
}


def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    if path.stat().st_size > MAX_LINE_BYTES:
        raise ContractError(f"JSON exceeds {MAX_LINE_BYTES} bytes: {path}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_profile(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version", "id", "description", "tokenizer", "model", "initialization"
    } or type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ContractError("profile requires the temporal LLM schema version 1 fields")
    if not all(isinstance(raw[name], str) and raw[name] for name in ("id", "description")):
        raise ContractError("profile id and description must be nonempty strings")
    model = raw["model"]
    if not isinstance(model, dict) or set(model) != MODEL_FIELDS:
        raise ContractError("profile model must specify every supported geometry/neuron field")
    for name in set(GEOMETRY) | {"simulation_steps"}:
        if type(model[name]) is not int or not 0 < model[name] <= (1 << 31) - 1:
            raise ContractError(f"model {name} must be a positive int32")
    if model["simulation_steps"] > 24:
        raise ContractError("temporal simulation_steps must be in 1..24")
    if model["query_head_count"] % model["key_value_head_count"] or model["head_dimension"] % 2:
        raise ContractError("query heads must divide into KV groups and head width must be even")
    for name in FLOATS:
        if type(model[name]) not in (float, int) or not math.isfinite(model[name]):
            raise ContractError(f"model {name} must be finite")
    if model["rms_epsilon"] <= 0 or model["rope_base"] <= 1 or not 0 <= model["temporal_decay"] < 1:
        raise ContractError("invalid RMS epsilon, RoPE base or temporal decay")
    for name in ("query_key_normalization", "tied_embeddings"):
        if type(model[name]) is not bool:
            raise ContractError(f"model {name} must be boolean")
    tokenizer = raw["tokenizer"]
    if not isinstance(tokenizer, dict) or set(tokenizer) != {"assets_path", "fingerprint_sha256"}:
        raise ContractError("profile tokenizer requires assets_path and fingerprint_sha256")
    if not isinstance(tokenizer["assets_path"], str) or not tokenizer["assets_path"]:
        raise ContractError("tokenizer assets_path must be nonempty")
    require_sha256(tokenizer["fingerprint_sha256"], "tokenizer fingerprint")
    init = raw["initialization"]
    if not isinstance(init, dict) or init.get("type") not in {"random", "qwen_archive"}:
        raise ContractError("initialization type must be random or qwen_archive")
    if init["type"] == "random":
        if set(init) != {"type"}:
            raise ContractError("random initialization does not accept archive fields")
    else:
        if set(init) not in ({"type", "path", "sha256", "preset"}, {"type", "path", "sha256", "identity_path"}):
            raise ContractError("archive initialization requires path, sha256 and preset or identity_path")
        require_sha256(init["sha256"], "archive SHA-256")
        for name in set(init) - {"type", "sha256"}:
            if not isinstance(init[name], str) or not init[name]:
                raise ContractError(f"archive {name} must be nonempty")
        if "preset" in init:
            if init["preset"] != "qwen3-0.6b":
                raise ContractError("the only pinned import preset is qwen3-0.6b")
            expected = {"vocabulary_size": 151936, "model_dimension": 1024,
                        "layer_count": 28, "query_head_count": 16,
                        "key_value_head_count": 8, "head_dimension": 128,
                        "feed_forward_dimension": 3072, "query_key_normalization": True,
                        "tied_embeddings": True, "rms_epsilon": 1e-6, "rope_base": 1e6}
            if any(model[key] != value for key, value in expected.items()):
                raise ContractError("pinned Qwen archive requires its exact dense geometry")
            if model["maximum_sequence_length"] > 32768:
                raise ContractError("Qwen3-0.6B context exceeds its source configuration")
    return copy.deepcopy(raw)


def load_profile(value: str) -> dict[str, Any]:
    path = ROOT / "configs" / "temporal_llm" / f"{value}.json"
    if not path.is_file():
        path = Path(value)
    return validate_profile(read_json(path))


def resource_plan(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Explicit lower bounds; does not pretend these equal peak process memory."""
    m = profile["model"]
    width, ffn, layers = m["model_dimension"], m["feed_forward_dimension"], m["layer_count"]
    q, kv, head = m["query_head_count"], m["key_value_head_count"], m["head_dimension"]
    dense_per_layer = 2 * width * q * head + 2 * width * kv * head + 3 * width * ffn + 2 * width
    if m["query_key_normalization"]:
        dense_per_layer += 2 * head
    dense = m["vocabulary_size"] * width * (1 if m["tied_embeddings"] else 2) + layers * dense_per_layer + width
    context = m["maximum_sequence_length"]
    return {
        "profile": profile["id"], "description": profile["description"], "model": dict(m),
        "dense_parameter_count": dense,
        "fp32_weight_bytes": dense * 4,
        "bf16_weight_bytes_if_supported_by_backend": dense * 2,
        "fp32_adamw_weights_gradients_moments_lower_bound_bytes": dense * 16,
        "kv_cache_fp32_bytes_batch_one": 2 * layers * kv * head * context * 4,
        "full_sequence_logits_fp32_bytes_batch_one": context * m["vocabulary_size"] * 4,
        "last_position_logits_fp32_bytes_batch_one": m["vocabulary_size"] * 4,
        "dense_attention_scores_fp32_bytes_per_layer_batch_one": q * context * context * 4,
        "initialization": dict(profile["initialization"]),
        "limitations": [
            "Arithmetic estimates exclude activation graphs, allocator/workspace overhead and temporal neuron state.",
            "Current runner uses FP32; the BF16 estimate is a planning comparison, not a selectable runtime mode.",
            "Larger geometry is supported; pretrained compatibility still requires matching tensor inventory, tokenizer and archive identity.",
            "The temporal decoder retains dense attention, normalization, projections and vocabulary readout; sparse-kernel speedups are not established.",
        ],
    }


def load_tokenizer(profile: Mapping[str, Any], assets_dir: Path | None = None) -> Any:
    root = assets_dir or repo_path(profile["tokenizer"]["assets_path"])
    fingerprint, _ = tokenizer_fingerprint(root)
    if fingerprint != profile["tokenizer"]["fingerprint_sha256"]:
        raise ContractError("tokenizer fingerprint differs from the model profile")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    ids = set(tokenizer.get_vocab().values())
    if ids != set(range(len(ids))) or len(ids) > profile["model"]["vocabulary_size"]:
        raise ContractError("tokenizer vocabulary must be contiguous and fit the model vocabulary")
    return tokenizer


def encoded_record(tokenizer: Any, row: Any, context_tokens: int) -> dict[str, Any]:
    if isinstance(row, str):
        text, boundary = row, 0
    elif isinstance(row, dict) and set(row) == {"text"}:
        text, boundary = row["text"], 0
    elif isinstance(row, dict) and set(row) == {"prompt", "response"}:
        if not all(isinstance(row[key], str) and row[key] for key in row):
            raise ContractError("prompt/response records require two nonempty strings")
        text, boundary = row["prompt"] + row["response"], len(row["prompt"])
    else:
        raise ContractError("corpus rows must be plain text, {text}, or {prompt,response}")
    if not isinstance(text, str) or not text.strip() or "\0" in text:
        raise ContractError("corpus text must be nonempty and contain no NUL")
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if not 2 <= len(encoded.ids) <= context_tokens:
        raise ContractError(f"corpus record requires 2..{context_tokens} tokens; split long documents explicitly")
    mask = [int(end > boundary) for _, end in encoded.offsets]
    mask[0] = 0
    if not any(mask):
        raise ContractError("corpus record has no predictable target tokens")
    return {"input_ids": encoded.ids, "loss_mask": mask,
            "sha256": hashlib.sha256(json.dumps(encoded.ids).encode()).hexdigest()}


def load_corpus(path: Path, tokenizer: Any, context_tokens: int, limit: int) -> list[dict[str, Any]]:
    records = []
    scanned_bytes = 0
    with path.open("rb") as stream:
        while len(records) < limit:
            raw = stream.readline(MAX_CORPUS_LINE_BYTES + 1)
            if not raw:
                break
            scanned_bytes += len(raw)
            if scanned_bytes > 64 * 1024 * 1024:
                raise ContractError("bounded corpus selection exceeded 64 MiB before reaching its record limit")
            if len(raw) > MAX_CORPUS_LINE_BYTES:
                raise ContractError("corpus line exceeds 1 MiB")
            text = raw.decode("utf-8").strip()
            if not text:
                continue
            row = json.loads(text, object_pairs_hook=reject_duplicates) if path.suffix.lower() == ".jsonl" else text
            records.append(encoded_record(tokenizer, row, context_tokens))
    if not records:
        raise ContractError(f"corpus contains no usable records: {path}")
    return records


def check_eval_disjoint(corpora: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    held = {row["sha256"] for row in corpora.get("evaluation", [])}
    for name in ("calibration", "training"):
        if held & {row["sha256"] for row in corpora.get(name, [])}:
            raise ContractError(f"evaluation contains token-identical records from {name}")


def build_core_command(profile: Mapping[str, Any], args: argparse.Namespace, *, ann: bool = False) -> list[str]:
    m, init = profile["model"], profile["initialization"]
    command = [str(args.core_executable.resolve()), "serve", "--device", args.device]
    importing = not args.checkpoint and init["type"] == "qwen_archive"
    pinned = importing and init.get("preset") == "qwen3-0.6b" and args.qwen_identity is None
    if pinned:
        command.extend(["--qwen3-0.6b", "--context-tokens", str(m["maximum_sequence_length"])])
    else:
        for name, option in GEOMETRY.items():
            command.extend([option, str(m[name])])
        command.extend(["--rms-epsilon", str(m["rms_epsilon"]), "--rope-base", str(m["rope_base"])])
        command.append("--qk-norm" if m["query_key_normalization"] else "--no-qk-norm")
        if not m["tied_embeddings"]:
            command.append("--untied-embeddings")
    # Keep the same neuron configuration for ANN/temporal checkpoints; --ann
    # selects the explicit control after the temporal encoding mode is chosen.
    command.extend(["--temporal-spiking", "--temporal-decay", str(m["temporal_decay"]),
                    "--simulation-steps", str(m["simulation_steps"]), "--seed", str(args.seed),
                    "--learning-rate", str(args.learning_rate),
                    "--gradient-accumulation", str(args.gradient_accumulation)])
    if ann or args.ann:
        command.append("--ann")
    if args.checkpoint:
        command.extend(["--checkpoint", str(args.checkpoint.resolve())])
        if args.weights_only:
            command.append("--weights-only")
    elif importing:
        command.extend(["--qwen-archive", str(repo_path(init["path"])), "--qwen-archive-sha256", init["sha256"]])
        if not pinned:
            identity_path = args.qwen_identity or repo_path(init.get("identity_path", ""))
            identity = read_json(identity_path)
            if not isinstance(identity, dict):
                raise ContractError("Qwen identity must be a JSON object")
            # A flat identity object is deliberately explicit; source manifests
            # contain nested structures that should not be guessed here.
            for field, flag in IDENTITY_FLAGS.items():
                value = identity.get(field)
                if not isinstance(value, str) or not value:
                    raise ContractError(f"Qwen identity requires {field}")
                if field.endswith("sha256"):
                    require_sha256(value, field)
                command.extend([flag, value])
            if identity["tokenizer_fingerprint_sha256"] != profile["tokenizer"]["fingerprint_sha256"]:
                raise ContractError("archive identity and profile tokenizer disagree")
    if args.save_checkpoint and not ann:
        command.extend(["--save-checkpoint", str(args.save_checkpoint.resolve())])
    return command


def runtime_command(command: list[str], executable: Path, threads: int) -> list[str]:
    wrapper = ROOT / "scripts" / "run_with_chatbot_libtorch.sh"
    cache = executable.resolve().parent / "CMakeCache.txt"
    if cache.is_file() and wrapper.is_file():
        command = [str(wrapper), "--build-dir", str(cache.parent), "--", *command]
    return ["env", f"OMP_NUM_THREADS={threads}", f"MKL_NUM_THREADS={threads}", *command]


class Session:
    def __init__(self, command: Sequence[str]):
        self.core = CoreProcess(command, MAX_LINE_BYTES)
        self.counter = 0

    def request(self, op: str, **fields: Any) -> Mapping[str, Any]:
        self.counter += 1
        identifier = f"temporal-llm-{self.counter}"
        response = self.core.transact({"protocol": PROTOCOL, "request_id": identifier, "op": op, **fields})
        if not isinstance(response, dict) or response.get("request_id") != identifier or response.get("protocol") != PROTOCOL:
            raise ContractError("core response correlation mismatch")
        if response.get("ok") is not True:
            raise ContractError(f"core {op} failed: {response.get('error')}")
        return response

    def close(self) -> None:
        try:
            self.request("shutdown")
        finally:
            self.core.close()


def evaluate_records(request: Callable[..., Mapping[str, Any]], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tokens, loss_sum, correct, spike_sum = 0, 0.0, 0, 0.0
    final_predictions = []
    for record in records:
        metrics = request("evaluate", input_ids=record["input_ids"], loss_mask=record["loss_mask"], reset_state=True)["metrics"]
        count = metrics["token_count"]
        if type(count) is not int or count <= 0 or not math.isfinite(metrics["loss"]):
            raise ContractError("core returned invalid evaluation metrics")
        tokens += count
        loss_sum += count * metrics["loss"]
        correct += metrics["correct_token_count"]
        spike_sum += count * metrics["mean_spike_rate"]
        inspection = request("inspect", input_ids=record["input_ids"][:-1],
                             probe_token_ids=[record["input_ids"][-1]], top_k=1)
        final_predictions.append(inspection["final_position"]["top_k"][0]["token_id"])
    loss = loss_sum / tokens
    return {"records": len(records), "predicted_tokens": tokens, "loss": loss,
            "perplexity": math.exp(loss) if loss < 700 else None,
            "token_accuracy": correct / tokens, "mean_spike_rate": spike_sum / tokens,
            "final_position_predictions": final_predictions}


def generate(request: Callable[..., Mapping[str, Any]], tokenizer: Any, text: str,
             profile: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not text.strip() or "\0" in text or len(text.encode("utf-8")) > MAX_CORPUS_LINE_BYTES:
        raise ContractError("prompt must be nonempty, NUL-free and at most 1 MiB")
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if not ids or len(ids) + args.max_new_tokens > profile["model"]["maximum_sequence_length"]:
        raise ContractError("prompt plus requested completion exceeds profile context")
    stops = [tokenizer.token_to_id(name) for name in ("<|endoftext|>", "<|im_end|>")]
    response = dict(request("generate", input_ids=ids, max_new_tokens=args.max_new_tokens,
                            sampling_vocabulary_size=tokenizer.get_vocab_size(), seed=args.seed,
                            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                            prefill_chunk_size=args.prefill_chunk_size,
                            eos_token_ids=[token for token in stops if token is not None]))
    output = response.get("output_ids")
    if not isinstance(output, list) or len(output) > args.max_new_tokens or any(
        type(token) is not int or not 0 <= token < tokenizer.get_vocab_size() for token in output
    ):
        raise ContractError("core generated a token outside the decodable vocabulary")
    response["text"] = tokenizer.decode(output, skip_special_tokens=True)
    response["prompt_tokens"] = len(ids)
    return response


def checkpoint_sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".temporal.json")


def write_report(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", help="qwen3-0.6b, tiny, or custom profile JSON; default qwen3-0.6b")
    p.add_argument("--plan", "--inspect", action="store_true", help="estimate resources without allocating a model")
    p.add_argument("--assets-dir", type=Path)
    p.add_argument("--core-executable", type=Path, default=ROOT / "build-temporal-llm" / "chatbot_experiment")
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--context-tokens", type=int)
    p.add_argument("--simulation-steps", type=int)
    p.add_argument("--temporal-decay", type=float)
    p.add_argument("--ann", action="store_true", default=None, help="dense ANN control")
    p.add_argument("--checkpoint", type=Path, help="resume weights, optimizer and calibration with the .temporal.json sidecar")
    p.add_argument("--weights-only", action="store_true", help="load model/calibration for inference on another device; skip optimizer/RNG resume")
    p.add_argument("--save-checkpoint", type=Path)
    p.add_argument("--qwen-identity", type=Path, help="flat verified identity JSON for a custom import")
    p.add_argument("--calibration-corpus", type=Path)
    p.add_argument("--train-corpus", type=Path)
    p.add_argument("--eval-corpus", type=Path)
    p.add_argument("--compare-ann", action="store_true", help="compare against the same source initialization in a second process")
    p.add_argument("--max-records", type=int, default=8, help="maximum records read per corpus (default 8)")
    p.add_argument("--train-steps", type=int, default=8, help="bounded microbatch updates, cycling selected training records")
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--gradient-accumulation", type=int)
    p.add_argument("--seed", type=int)
    action = p.add_mutually_exclusive_group()
    action.add_argument("--chat", action="store_true")
    action.add_argument("--prompt")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--prefill-chunk-size", type=int, default=128, help="prompt tokens per cached prefill chunk; 0 processes the whole prompt")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--report", type=Path, help="write measured metrics and profile provenance as JSON")
    return p


def resolve_configuration(args: argparse.Namespace) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    if args.weights_only and (not args.checkpoint or args.train_corpus or args.calibration_corpus or args.save_checkpoint):
        raise ContractError("--weights-only requires a checkpoint and permits inference/evaluation only")
    restored = None
    if args.checkpoint:
        restored = read_json(checkpoint_sidecar(args.checkpoint))
        if not isinstance(restored, dict) or restored.get("kind") != "snnbase.temporal-llm-checkpoint" or restored.get("schema_version") != 1:
            raise ContractError("checkpoint requires its matching temporal LLM sidecar")
        require_sha256(restored.get("checkpoint_sha256"), "checkpoint SHA-256")
        # Planning is metadata-only, including when resuming a large checkpoint.
        if not args.plan and sha256_file(args.checkpoint) != restored["checkpoint_sha256"]:
            raise ContractError("checkpoint content does not match its temporal LLM sidecar")
    profile = load_profile(args.profile) if args.profile else (
        validate_profile(restored["profile"]) if restored else load_profile("qwen3-0.6b"))
    for argument, field in (("context_tokens", "maximum_sequence_length"),
                            ("simulation_steps", "simulation_steps"), ("temporal_decay", "temporal_decay")):
        if getattr(args, argument) is not None:
            profile["model"][field] = getattr(args, argument)
    validate_profile(profile)
    defaults = {"ann": False, "seed": 42, "learning_rate": 1e-5, "gradient_accumulation": 1}
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, restored["runner"][name] if restored else default)
    if restored and (profile != restored["profile"] or (not args.weights_only and any(
        getattr(args, name) != restored["runner"][name] for name in defaults
    )) or (args.weights_only and args.ann != restored["runner"]["ann"])):
        raise ContractError("checkpoint geometry, neuron settings and optimizer configuration must match its sidecar")
    return profile, restored


def run(args: argparse.Namespace) -> dict[str, Any]:
    profile, restored = resolve_configuration(args)
    if args.plan:
        return resource_plan(profile)
    for name in ("threads", "max_records", "train_steps", "gradient_accumulation", "max_new_tokens"):
        if getattr(args, name) <= 0:
            raise ContractError(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.seed < 1 << 64 or args.top_k < 0 or args.prefill_chunk_size < 0:
        raise ContractError("seed must be uint64; top-k and prefill-chunk-size must be nonnegative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or not math.isfinite(args.temperature) or args.temperature < 0 or not 0 < args.top_p <= 1:
        raise ContractError("invalid learning rate, temperature or top-p")
    if args.compare_ann and (not args.eval_corpus or args.ann or args.checkpoint or args.train_corpus):
        raise ContractError("--compare-ann requires evaluation from source weights with no checkpoint/training or --ann")
    if args.calibration_corpus and args.ann:
        raise ContractError("calibration requires the temporal spiking variant")
    if (args.calibration_corpus or args.train_corpus) and not args.save_checkpoint:
        raise ContractError("calibration/training requires --save-checkpoint to retain learned state")
    if not any((args.prompt, args.chat, args.calibration_corpus, args.train_corpus, args.eval_corpus)):
        raise ContractError("select --plan, --prompt, --chat, or a calibration/training/evaluation corpus")
    if args.save_checkpoint:
        if args.save_checkpoint.exists() and (not args.checkpoint or args.save_checkpoint.resolve() != args.checkpoint.resolve()):
            raise ContractError("save-checkpoint already exists; choose a new destination or explicitly resume it")
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(profile, args.assets_dir)
    corpora = {name: load_corpus(path, tokenizer, profile["model"]["maximum_sequence_length"], args.max_records)
               for name, path in (("calibration", args.calibration_corpus), ("training", args.train_corpus), ("evaluation", args.eval_corpus)) if path}
    check_eval_disjoint(corpora)
    learning_records = set(restored.get("learning_record_sha256", [])) if restored else set()
    for name in ("calibration", "training"):
        learning_records.update(row["sha256"] for row in corpora.get(name, []))
    if learning_records & {row["sha256"] for row in corpora.get("evaluation", [])}:
        raise ContractError("evaluation overlaps a current or resumed checkpoint learning record")
    report: dict[str, Any] = {"kind": "snnbase.temporal-llm-run", "schema_version": 1,
        "profile": profile, "variant": "ann" if args.ann else "abstract_temporal_spiking",
        "checkpoint_load_mode": ("weights_only" if args.weights_only else
                                 "full_resume" if args.checkpoint else "source_initialization"),
        "corpora": {name: {"records": len(rows), "record_sha256": [row["sha256"] for row in rows]}
                    for name, rows in corpora.items()},
        "quality_scope": "Bounded experiment; measurements do not establish general instruction following or larger-model performance."}
    command = build_core_command(profile, args)
    report["core_command"] = command
    session = Session(runtime_command(command, args.core_executable, args.threads))
    try:
        report["metadata"] = session.request("metadata")["metadata"]
        if corpora.get("calibration"):
            report["calibration"] = [dict(session.request("calibrate", input_ids=row["input_ids"], reset_state=(index == 0)))
                                     for index, row in enumerate(corpora["calibration"])]
        if corpora.get("training"):
            report["training"] = []
            rows = corpora["training"]
            for index in range(args.train_steps):
                row = rows[index % len(rows)]
                metrics = session.request("train", input_ids=row["input_ids"], loss_mask=row["loss_mask"], reset_state=True)["metrics"]
                report["training"].append(dict(metrics))
                print(f"train {index + 1}/{args.train_steps}: loss={metrics['loss']:.6f}", file=sys.stderr)
        if args.save_checkpoint:
            report["training_state"] = session.request("flush")["training_state"]
            if not args.save_checkpoint.is_file():
                raise ContractError("core flush did not produce the requested checkpoint")
            write_report(checkpoint_sidecar(args.save_checkpoint), {
                "kind": "snnbase.temporal-llm-checkpoint", "schema_version": 1, "profile": profile,
                "checkpoint_sha256": sha256_file(args.save_checkpoint),
                "runner": {name: getattr(args, name) for name in ("ann", "seed", "learning_rate", "gradient_accumulation")},
                "source_checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "calibration_applied": bool(corpora.get("calibration")) or bool(restored and restored.get("calibration_applied")),
                "learning_record_sha256": sorted(learning_records),
                "corpora": report["corpora"], "metadata": report["metadata"],
            })
        if corpora.get("evaluation"):
            report["evaluation"] = evaluate_records(session.request, corpora["evaluation"])
        if args.prompt:
            report["generation"] = generate(session.request, tokenizer, args.prompt, profile, args)
            print(report["generation"]["text"])
        if args.chat:
            print("Base-model conversation experiment. /reset clears history; /quit exits.", file=sys.stderr)
            turns: list[tuple[str, str]] = []
            while True:
                try:
                    text = input("You: ")
                except EOFError:
                    break
                if text.strip() == "/quit":
                    break
                if text.strip() == "/reset":
                    turns.clear()
                    session.request("reset")
                    continue
                if not text.strip():
                    continue
                prefix = "".join(f"User: {user}\nAssistant: {reply}\n" for user, reply in turns)
                prompt = prefix + f"User: {text}\nAssistant:"
                while turns and len(tokenizer.encode(prompt).ids) + args.max_new_tokens > profile["model"]["maximum_sequence_length"]:
                    turns.pop(0)
                    prompt = "".join(f"User: {user}\nAssistant: {reply}\n" for user, reply in turns) + f"User: {text}\nAssistant:"
                try:
                    reply = generate(session.request, tokenizer, prompt, profile, args)
                except ContractError as error:
                    print(f"temporal_llm: {error}", file=sys.stderr)
                    continue
                print(f"Assistant: {reply['text']}")
                turns.append((text, reply["text"]))
    finally:
        session.close()
    if args.compare_ann:
        control = Session(runtime_command(build_core_command(profile, args, ann=True), args.core_executable, args.threads))
        try:
            report["ann_evaluation"] = evaluate_records(control.request, corpora["evaluation"])
        finally:
            control.close()
        spikes = report["evaluation"]["final_position_predictions"]
        dense = report["ann_evaluation"]["final_position_predictions"]
        report["ann_comparison"] = {
            "sampled_positions": len(spikes),
            "final_position_next_token_agreement": sum(a == b for a, b in zip(spikes, dense)) / len(spikes),
            "loss_delta_temporal_minus_ann": report["evaluation"]["loss"] - report["ann_evaluation"]["loss"],
            "scope": "One final next-token prediction per selected evaluation record; NLL covers all selected targets.",
        }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
        if args.report:
            write_report(args.report, report)
        if args.plan or not (args.prompt or args.chat):
            print(json.dumps(report, indent=2, allow_nan=False))
        elif args.eval_corpus:
            print(json.dumps(report.get("evaluation"), indent=2), file=sys.stderr)
        return 0
    except KeyboardInterrupt:
        print("temporal_llm: interrupted", file=sys.stderr)
        return 130
    except (ContractError, ProtocolError, OSError, ValueError, ImportError, KeyError, TypeError) as error:
        print(f"temporal_llm: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
