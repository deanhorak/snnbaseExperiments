# OASST2 dataset approval record

## Decision

The pinned OASST2 candidate is **approved for internal research training and
evaluation** in the ANN/SNN chatbot experiment, subject to every restriction in
this record. This is a project data-governance decision, not legal advice and
not approval to redistribute the raw or derived conversations, deploy a public
chatbot, or make safety claims.

Approval date: 2026-09-01
Approval basis: project-owner instruction to complete dataset approval and
leakage review, plus the mechanical and policy evidence below.
Re-review trigger: any source revision, converter, tokenizer, selection profile,
sequence limit, split algorithm, leakage model, threshold, or exclusion change.

## Immutable source and declared license

- Dataset: OpenAssistant OASST2 ready trees, `2023-11-05` snapshot.
- Upstream revision: `179dd21fc55192153d94adb0e0ce8f69e222bf75`.
- Revision-pinned source: [Hugging Face archive](https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/179dd21fc55192153d94adb0e0ce8f69e222bf75/2023-11-05_oasst2_ready.trees.jsonl.gz).
- Compressed bytes: `54,370,156`.
- Compressed SHA-256:
  `7a886a16ccfc1173c4f00a6897523e3c95b2785a86ee44a18a98f4f2807ee29b`.
- Decompressed SHA-256:
  `93092c0f05695da0306d7ea78bfc10a25b00396a7b4625a2d18a1377fb0aae04`.
- The revision-pinned [dataset card](https://huggingface.co/datasets/OpenAssistant/oasst2/blob/179dd21fc55192153d94adb0e0ce8f69e222bf75/README.md)
  declares `Apache-2.0`.
- The authoritative [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt)
  requires retention of the license and applicable notices and identification
  of modified files when distributing covered material. It disclaims title and
  non-infringement warranties; the license declaration alone does not prove
  contributor rights or privacy clearance for every conversation.

The repository does not commit the source or derived conversations. Anyone who
redistributes them must conduct a separate legal review, preserve the required
license/notices, mark modifications, and assess privacy and contributor-rights
risk. No upstream `NOTICE` file has been represented here as absent or
inapplicable; a distributor must inspect the exact upstream revision.

## Selection and safety boundary

The approved candidate uses the `quality05` policy and the pinned
`Qwen/Qwen3-0.6B-Base` tokenizer at revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Before leakage remediation it
contains 12,427 conversations (9,990 train, 1,165 validation, 1,272 test), all
at most 512 rendered tokens without truncation. Exact normalized root-prompt
deduplication already removed 96 records.

`quality05` requires reviewed, non-deleted, non-synthetic, same-language path
messages with positive review counts. Present `spam`, `lang_mismatch`, and
`pii` labels must be below 0.5; assistant quality/helpfulness and task-failure
thresholds also apply. Missing labels are allowed. Topic/content labels and
Detoxify scores are not used as a universal safety gate. Consequently:

- this is not a PII-free, toxicity-free, or safety-certified corpus;
- raw prompt text must not be placed in logs, Git, published manifests, or
  leakage reports;
- generated model outputs require a separate safety evaluation before public
  use; and
- internal access must follow the machine/project's existing data controls.

## Cross-split leakage review

The reproducible audit is implemented by
[`chatbot_leakage_audit.py`](../tools/chatbot_leakage_audit.py). It verifies the
immutable conversion bundle and examines only cross-split pairs over three
views: root user prompt, combined user context, and combined assistant targets.
It performs NFKC/casefold/whitespace exact matching, deterministic character
5-gram fuzzy matching, and multilingual semantic comparison.

Semantic model:
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` at immutable
revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, whose model card declares
Apache-2.0. The report fingerprints every locally consumed model file; their
canonical aggregate SHA-256 is
`055ced0078369cff173434038d764b30ac416fad17a20f72346e225de8ebe3da`. Fixed
high-confidence semantic cosine thresholds are 0.92 for roots, 0.94 for the
combined user context, and 0.97 for assistant targets. Review-band thresholds
are 0.86/0.88/0.92. Fuzzy flags require 5-gram Jaccard at least 0.85 or
containment at least 0.95, with minimum-length controls.
The exact audit implementation is `chatbot_leakage_audit.py` version `1.0.0`,
34,300 bytes, SHA-256
`4f14a41e084f0e1601aa93add10186df408d4e44d37c9ffbd65ef1c524437359`.
The recorded runtime is CUDA with seed 0, deterministic PyTorch algorithms,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, PyTorch `2.5.1+cu121`, Transformers
`4.51.3`, and SentenceTransformers `4.1.0`.

Baseline audit result:

- audit SHA-256:
  `a5fc18c713e6e35e6c553b63e975de024a5b1500c98dc1294fe2b0f40c5f9d87`;
- exclusion decision SHA-256:
  `e8a297c92b3c5af0712b61f83a78d261433e68afca03020028665791efc154f0`;
- 168 high-confidence cross-split pairs involving 224 records;
- 160 root-prompt, 39 combined-user-context, and 7 assistant-target view flags
  (a pair can appear in more than one view);
- 155 semantic flags, 21 fuzzy-containment flags, and 14 fuzzy-Jaccard flags
  (also non-exclusive); and
- 144 deterministic exclusions: 137 train, 7 validation, and 0 test.

The exclusion rule forms connected components and retains every member of the
highest-priority represented split: test, then validation, then train. It does
not move IDs between hash-derived splits. The converter proves its
pre-remediation byte stream is exactly the audited 12,427-record baseline,
applies every declared UUID exactly once, and records the audit, exclusion, and
embedding-model fingerprints in the new conversion manifest.

Reviewed output (filled by the real regeneration gate):

- converter `oasst2_convert.py` version `1.1.0`, 72,316 bytes, SHA-256
  `ee4d6c7472e3f5a02020da96abbaef83451f3280faf67a2b991018d3fa7aea7d`;
- records: `12,283` (`9,853` train, `1,158` validation, `1,272` test);
- canonical conversations SHA-256:
  `d947b8089d0d2a4dba4c3cadb73a7b8ecd2fecaf121e6e5c01829063be1cf819`;
- text-free lineage SHA-256:
  `1ccb4810236477b2629e4926dc430eae55249a7e0fb79f72957857f327871b33`;
- conversion manifest SHA-256:
  `9fcb2e48090c6c93ca84f8c90eeaf9c43bb8a69e12a94d9243fa48107dff974e`;
- prepared dataset manifest SHA-256:
  `6e272e6b5200f3d1c4f16f86ddea2e8e6804986ec46a0e5fd7950615e479785c`;
- exhaustive post-remediation audit SHA-256:
  `b0c4bcbbf9934a16914024f3d60c399f49a2a6a49163b09618dd710269196c8b`
  (`status=pass`, zero high-confidence cross-split pairs); and
- empty post-remediation exclusion decision SHA-256:
  `fea8c81d686caeb9e9132ba3a37eb2776ed35660d9fa2cff293066ff1123b608`.

The prepared shards contain 2,975,598/351,834/386,040 input tokens and
2,428,987/289,641/316,544 assistant-target tokens for train/validation/test,
respectively. The training loader revalidated the manifest, shards, copied
conversion manifest, and text-free lineage.

## Residual risks and permitted claim

Embedding and fuzzy thresholds are heuristics. They can miss paraphrases and
cross-language equivalents and can flag unrelated material. Within-split
duplicates were not treated as train/evaluation leakage. The correct claim is:

The post-remediation audit still places 324 root-prompt, 71 combined-user, and
39 assistant-target cross-split pairs in its lower review bands. These are not
high-confidence matches under the recorded policy and were not exclusions;
they are retained as an explicit residual risk rather than represented as
proof of zero semantic similarity.

> The approved internal-research dataset has no retained cross-split pair that
> was flagged by the documented exact, fuzzy, or pinned-model semantic audit at
> its fixed high-confidence thresholds.

Do not claim that semantic leakage is impossible or that the test set is
universally deduplicated. A changed audit model or threshold requires a new
versioned review rather than silently changing this record.

## Approval checklist

- [x] Immutable source revision, sizes, and hashes recorded.
- [x] Upstream license declaration and Apache-2.0 obligations reviewed.
- [x] Source and derived data excluded from Git.
- [x] Selection, token, lineage, and split contracts validated.
- [x] Exact normalized root duplicates removed before splitting.
- [x] Cross-split fuzzy and semantic leakage audit completed.
- [x] High-confidence collisions deterministically removed from lower-priority
  splits without changing test membership.
- [x] Audit limitations and safety/privacy boundaries documented.
- [x] Approved scope limited to internal ANN/SNN research.
- [ ] Separate counsel/privacy approval for redistribution, external service,
  commercial use, or claims beyond this internal experiment. This is explicitly
  outside the present approval.
