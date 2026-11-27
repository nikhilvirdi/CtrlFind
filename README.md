# CtrlFind

CtrlFind takes a programming problem written in English and searches for the Python code most likely to solve it. It starts with semantic search, then runs the strongest candidates against an example from the problem statement to check whether their output matches.

CtrlFind uses an embedding model and local execution to retrieve and rank existing code. The published benchmark and the training experiments use separate query sets.

## Quickstart

Use Python 3.11, the version used by the Docker image. Create a virtual environment with `python -m venv .venv`. Activate it with `.\.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` in a POSIX shell.

```sh
python -m pip install -r requirements.txt
python demo.py
```

Open [the local demo](http://127.0.0.1:7860). On first launch, it downloads the dataset and model from Hugging Face and builds an embedding cache. The demo uses the benchmark's held-out problem list and relevance mappings. Use the training-only research scripts for development measurements. The repository includes three synthetic versions that you can search individually or together.

Use a separate checkout or the container for interactive use when preserving a frozen baseline. The demo and index utilities can refresh the checkout's default embedding cache.

Keep the demo local and use a disposable environment for execution checks. Leave **Run candidates on the example** unchecked for similarity-only search. The Docker image also runs the demo locally when you bind its port to loopback.

```sh
docker build -t ctrlfind .
docker run --rm -p 127.0.0.1:7860:7860 ctrlfind
```

## The problem

Searching code by keywords works poorly when the question and the implementation use different words. A problem might describe a game, a route, or a sequence of operations, while the solution is mostly loops, arithmetic, and data structures. The right code can look less similar to the description than an unrelated snippet.

The task is to take a natural-language query and rank a large collection of code snippets so that the relevant ones appear first. A useful retrieval system also has to keep working as code changes.

CtrlFind supports three related searches.

- **Code retrieval.** Rank Python snippets by relevance to a problem statement.
- **Version-scoped retrieval.** Search a particular version and reuse embeddings for unchanged snippets.
- **Evolutionary retrieval.** Search several versions together and distinguish very similar implementations, including working code and versions with small bugs.

## The approach

A similarity score is useful for finding candidates, but it does not tell us whether a program solves the problem. Two programs can look almost identical and behave differently because of one operator. Two correct programs can also look very different.

CtrlFind combines two signals. An embedding model first finds code that appears related to the question. Then, where the problem includes a usable input-output example, CtrlFind executes those candidates and checks the result. A candidate that passes the example receives priority, while semantic similarity still determines the order within each group.

Passing one example is evidence, not proof of correctness. An unrelated program can print the expected answer by accident, and a correct program can produce a different valid output. The verifier is deliberately conservative about what counts as a match.

## How CtrlFind works

### Semantic retrieval

CtrlFind uses E5-base-v2 to turn the query and Python snippets into vector embeddings. Cosine similarity gives an initial ranking. By default, execution verification checks at most the first 50 candidates (K50). The live ADAPT option considers the first 200 candidates and can return a different number of results. [E5-base-v2 model card](https://huggingface.co/intfloat/e5-base-v2)

The code embeddings are reused across queries. This first pass is important because execution can only help a correct snippet that made it into the candidate set.

### Execution verification

When the problem statement includes a supported example format, CtrlFind extracts the input and expected output. Each shortlisted program runs in a separate Python subprocess with a temporary working directory and a one-second timeout.

A program counts as passing only if it finishes with exit code 0 and produces the expected output. The verifier normally uses exact comparison. It can ignore trailing spaces and tabs on individual lines for narrowly identified, space-separated integer output; fixed-width and ambiguous layouts keep strict comparison.

Programs that crash, time out, or produce different output do not receive a verification promotion. The current default checks the first usable published example, not every possible input.

### Reranking

Candidates that pass verification move above candidates that do not. Within both groups, CtrlFind preserves their original semantic-similarity order.

With K50, the verifier checks all 50 selected candidates when an example and enough candidates are available. Opt-in ADAPT checks ranks 1–50 first and checks ranks 51–200 only if the first batch produces no pass. ADAPT uses a separate checkpoint and reports an infrastructure failure without recording a verdict for that query. The demo and version retrieval keep their existing fixed-depth policy.

## System architecture

![CtrlFind architecture](docs/architecture.svg)

The main path is query → embedding search → example-based verification → ranked snippets. The versioned snippet store supplies code for version-scoped and cross-version search. Embedding and verification caches reuse compatible results. The MTEB integration exposes the optional ADAPT policy in live evaluation.

## Retrieval across versions

The CoIR APPS corpus is a snapshot, so CtrlFind uses a generated three-version history to exercise version-aware retrieval. These are synthetic changes, not an imported Git history from a real software repository.

| Version | Changes |
|---|---|
| `v1` | Original snippets. |
| `v2` | Some snippets are reformatted; some receive a small injected bug. |
| `v3` | Some bugs are fixed, others introduced, with additional formatting changes. |

### Content-addressed storage

Each distinct snippet is stored under a hash of its content. A version maps document IDs to those stored contents. If identical code appears in several versions, it needs only one stored copy and one embedding.

This also gives the cache a useful identity: a changed snippet has a different content hash, so its old embedding or execution verdict should not be silently reused for the new code.

### Incremental indexing

When a new version is created, unchanged snippets reuse existing embeddings. Only new or modified content needs to be encoded. The amount of re-indexing therefore depends on what changed, rather than automatically processing the whole collection again.

The repository includes simulated-change and version-comparison tooling. Historical indexing timings appear under **Performance**; those measurements used a T4 GPU and should not be mistaken for CPU-only indexing results.

## Evolutionary retrieval

CtrlFind can search the combined pool of versions. Identical snippets are deduplicated, so they are embedded and verified once even if several versions reference them.

The difficult cases are snippets that differ by a single token. An embedding may give a working version and a buggy version nearly the same similarity score. If the example exposes the bug, execution separates them. If both happen to pass, their similarity order is preserved; a sample that misses the bug cannot establish which version is correct.

This version search has been tested on the generated history. The number of contested cases remains small, so the results are evidence for that experiment, not a general claim about real repository histories.

## Evaluation and results

### Dataset and evaluation metrics

The main benchmark is [CoIR AppsRetrieval](https://huggingface.co/datasets/CoIR-Retrieval/apps), built from programming problems and Python solutions in APPS. Its evaluated test split contains **3,765 queries** and a corpus of **8,765 snippets**. CtrlFind uses [MTEB](https://docs.mteb.org/) for scoring.

- **NDCG@10** rewards placing relevant results near the top of the first ten results.
- **MRR@10** measures how early the first relevant result appears within the first ten.
- **Recall@K** tells us whether the relevant snippet was retrieved among the first K candidates, before verification has a chance to help.

The published test result has been frozen. Subsequent verifier experiments use training queries; their scores are not presented as updated test-set results.

### Baseline and comparisons

The following numbers are for AppsRetrieval and use NDCG@10 on a 0–100 scale. The reference-model rows come from the [CoIR paper](https://aclanthology.org/2025.acl-long.1072/); the CtrlFind row is the repository's frozen result.

| Model | Size | NDCG@10 |
|---|---:|---:|
| BM25 | — | 0.95 |
| UniXcoder | 123M | 1.36 |
| E5-base | 110M | 11.52 |
| **E5-base + execution verification (CtrlFind)** | **110M** | **19.96** |
| E5-Mistral | 7B | 21.33 |
| Voyage-Code-002 | API | 26.52 |

These are benchmark comparisons, not claims of equal compute, model access, or inference cost. The E5-base retrieval baseline matches the paper's reported 11.52.

### Retrieval results

The historical E5 Recall@50 of **31.29%** used **3,765 candidate snippets**. The frozen MTEB NDCG@10 and MRR@10 figures below used the full **8,765-snippet** candidate corpus.

| Metric | E5-base retrieval | CtrlFind frozen baseline |
|---|---:|---:|
| NDCG@10 | 11.52 | **19.96** |
| MRR@10 | 9.88 | **19.05** |

In the original 3,765-query evaluation, verification improved the ranking for **420** queries, worsened it for **10**, and left **3,335** unchanged.

The frozen result is saved in [`artifacts/baseline_frozen.json`](artifacts/baseline_frozen.json). It predates the later execution-correctness changes, so **19.96 / 19.05 are not claimed as final scores for the current working implementation**.

### Version retrieval results

An earlier experiment searched **10,819 distinct snippet contents** across **191 problems** with usable examples. In the 22 cases with a buggy version competing in the top 50, similarity placed an example-passing sibling first in 16 cases and K50 did so in 22.

A larger training-only experiment used **326 Codeforces queries** and **5,899 logical-document/body entries** across its synthetic versions. Every analyzed gold family had an example-passing original and an example-failing blind mutation. The example-passing sibling ranked above all failing siblings on **122/326** queries with similarity alone and **173/326** with fixed K50. The paired gain was **15.64 points** (95% CI **11.66–19.63**); NDCG@10 rose from **5.19 to 24.52** using example-passing gold siblings as relevant. [Per-query version results](artifacts/step4_version_run/bonus.json)

## Performance

Execution is the expensive part of the current pipeline. Embedding lookup and cosine ranking are much cheaper than starting and running dozens of Python programs.

| Measurement | Observed result | Conditions |
|---|---|---|
| Execute 50 candidates | About 3.6 s per query | Earlier run, two CPU cores |
| Execute 50 candidates | 2.94 s median; 4.14 s p95 | Later 40-query training replay |
| Full index build | 128 s | Historical T4 GPU measurement |
| Incremental build, `v1` → `v2` | 20 s for 1,353 changed snippets | Historical T4 GPU measurement |
| Incremental build, `v2` → `v3` | 11 s for 701 changed snippets | Historical T4 GPU measurement |

The execution measurements come from separate runs under different conditions. The Docker demo uses CPU inference. The T4 figures describe historical indexing experiments.

## Research and experiments

The project keeps its experiment history in [`docs/research-log.md`](docs/research-log.md), including approaches that were tested and rejected.

Several findings have already changed the verifier:

- **Exit status matters.** A program can print the expected answer and then crash. Matching stdout alone was giving those programs credit; verification now requires a successful exit.
- **Output formatting needs care.** Narrow trailing-whitespace normalization recovered some valid outputs. Broad normalization was rejected because it can accept incorrectly formatted fixed-width output.
- **More examples are not automatically better.** Requiring two examples removed false promotions but also removed too many correct ones, reducing ranking accuracy on a fixed training sample.
- **More parsing is not automatically safer.** Expanded example extraction found additional correct matches, but it also created additional incorrect promotions. That experimental parser was not retained.

In a paired training audit, the retained verifier passed **393/511 (76.91%)** gold snippets with parseable examples, up from 390/511 under the original rule. Passes among the 2,555 sampled unrelated snippets fell from three to two. Repeat counts varied slightly with nondeterministic outputs and timeouts. These figures describe the sampled training candidates. [Paired verifier audit](docs/research-log.md#2026-11-24-verifier-audit-and-execution-performance-reference)

The verification-depth study used **477 Codeforces training queries** and **8,765 candidate snippets**. ADAPT reached **25.74 NDCG@10** versus **14.36** for K50, a gain of **11.38 points** (paired 95% CI **8.65–14.23**). It used **79,650** logical attempts versus **94,800** for K200. The study derived all policies from shared K200 batches.

A separate standalone study used **120 Codeforces training queries** from the same candidate corpus. ADAPT was faster on average than K200, with higher P95 in both repeats. It left two queries without results after Windows output-capture errors. On the 118 complete queries it gained **10.17 NDCG@10** over K50 (paired 95% CI **5.08–16.10**). It failed standalone validation and remains an opt-in research policy. Independent follow-up examples confirmed nine consequential non-gold passes were incorrect. [Depth study](docs/research-log.md#registration-v3-results) · [Standalone results](docs/research-log.md#standalone-measurements)

## Tech stack

| Area | Technology |
|---|---|
| Language | Python |
| Embedding model | E5-base-v2 |
| Embedding inference | Sentence Transformers, PyTorch |
| Dataset handling | Hugging Face Datasets |
| Vector ranking | NumPy, cosine similarity |
| Retrieval evaluation | MTEB |
| Execution checking | Python subprocesses, temporary directories |
| Interactive interface | Gradio |
| Versioning and caching | Content hashes, local cached artifacts |
| Packaging | Docker |

## Testing and reproducibility

```sh
python -m unittest discover -s tests -v
```

Windows process-tree tests run on Windows. Evaluation commands write new reports under `artifacts/evaluation_runs/` by default. Use `--output` with an unused JSON path for another run. The `cached` mode replays the original verdict ledger. See the [research log](docs/research-log.md#2026-11-25-e5-retrieval-and-live-verification-funnel) for the cache-only procedure and its required inputs.

The normal evaluator targets the held-out benchmark. If you are authorized to run it, use `python evaluate.py --mode full --output path/to/new-report.json` for live K50 and add `--adapt` to opt in. For training-only ADAPT validation, run the following from the repository root with the pinned dataset cached locally. Omit `--benchmark` to replay archived rankings without launching candidates.

```sh
python -B -X utf8 artifacts/step5_adapt_validation.py artifacts/step5_adapt_repro --benchmark
```

The frozen benchmark result is kept separately from experimental training runs. Research scripts and audit artifacts record the candidate samples, settings, and observed verification outcomes, while [`docs/research-log.md`](docs/research-log.md) explains why changes were retained or reverted.

The archived verification-depth replay reproduced the saved rankings, bootstrap draws, and noise ledger and matched the five frozen input hashes. K50 remains the default. [Replay receipt](artifacts/verification_depth_v3_post_stop_audit.json)

## Limitations

- **Verification depends on first-stage recall.** The historical Recall@50 of **31.29%** used **3,765 candidates**. The full MTEB benchmark used **8,765 candidates**. Verification can only promote snippets in the retrieved set.
- **Example coverage is limited.** The retained parser extracted a usable example for **511/5,000 training problems (10.22%)** in the audit. Earlier notebook figures used a different split and parsing method, so they are not interchangeable.
- **One example is weak evidence.** Incorrect programs can pass by coincidence, and correct programs can emit different valid outputs. General-purpose checking cannot resolve every problem-specific output rule safely.
- **Execution can be unstable.** CPU contention, timeouts, and randomized programs can change a verdict between runs. Some transient timeout failures can remain cached. The archived depth study recorded a Windows process cleanup error. A later cleanup fix passed 41 regression tests and matched saved K50 verdicts on a training sample. That sample excluded the original incident. [Windows validation](artifacts/windows_cleanup_k50_validation_train.json)
- **Execution requires isolation.** Windows Job Objects contain the candidate process tree. Retrieved Python retains the verifier's filesystem, environment, network and privilege access. The non-Windows runner times out the direct child, and output capture has no explicit byte limit. A standard Docker run gives candidates access to anything available to the container. Keep the demo local, keep credentials and sensitive mounts out of the environment, and use similarity-only mode when you cannot isolate candidate code.
- **Version evaluation is synthetic.** The current history is generated from a static dataset. Real Git histories and a larger contested-version sample have not yet been validated.
- **Generalization is unproven.** The current system targets Python programming-problem solutions; its performance on large, mixed-language software repositories has not been established.

## Licensing and data

The project has no repository-wide `LICENSE` file. The original APPS repository publishes an MIT license and E5-base-v2 declares MIT on its model card. The CoIR dataset card does not establish redistribution rights for each underlying problem statement or solution. Review provenance and applicable notices before redistributing dataset-derived bundles, including the version-store assets included in the demo image. [APPS license](https://github.com/hendrycks/apps/blob/main/LICENSE) · [CoIR APPS dataset card](https://huggingface.co/datasets/CoIR-Retrieval/apps) · [E5-base-v2 model card](https://huggingface.co/intfloat/e5-base-v2)

## References

### Research papers

| Paper | Link |
|---|---|
| CoIR: A Comprehensive Benchmark for Code Information Retrieval Models (ACL 2025) | [Paper](https://aclanthology.org/2025.acl-long.1072/) |
| Measuring Coding Challenge Competence With APPS (2021) | [Paper](https://arxiv.org/abs/2105.09938) |
| MTEB: Massive Text Embedding Benchmark | [Paper](https://arxiv.org/abs/2210.07316) |
| Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5) | [Paper](https://arxiv.org/abs/2212.03533) |
| CodeSearchNet Challenge: Evaluating the State of Semantic Code Search | [Paper](https://arxiv.org/abs/1909.09436) |

### Datasets, models and documentation

| Resource | Link |
|---|---|
| CoIR benchmark | [GitHub](https://github.com/CoIR-team/coir) |
| CoIR AppsRetrieval dataset | [Hugging Face](https://huggingface.co/datasets/CoIR-Retrieval/apps) |
| Original APPS dataset | [GitHub](https://github.com/hendrycks/apps) |
| E5-base-v2 model | [Hugging Face](https://huggingface.co/intfloat/e5-base-v2) |
| CodeSearchNet dataset and tools | [GitHub](https://github.com/github/CodeSearchNet) |
| MTEB | [Documentation](https://docs.mteb.org/) |
| Sentence Transformers | [Documentation](https://www.sbert.net/) |
