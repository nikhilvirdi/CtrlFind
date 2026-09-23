# Execution-verified code retrieval

Given a programming problem in English, find the Python code that solves it. Built for the CoIR APPS benchmark: 3,765 test problems searched against 8,765 solutions, on CPU.

Most retrieval systems guess which code matches a question by measuring how similar the two look. This one also checks. Contest problems ship an example input and its expected output, so the top candidates are run on that example, and any that print the right answer move to the top.

## Results

AppsRetrieval test split, scored through MTEB.

| Model | Size | NDCG@10 |
|---|---|---|
| BM25 | | 0.95 |
| UniXcoder | 123M | 1.36 |
| E5-base | 110M | 11.52 |
| **E5-base + execution verification (this repo)** | **110M** | **19.96** |
| E5-Mistral | 7B | 21.33 |
| Voyage-Code-002 | API | 26.52 |

Reference rows are from the CoIR paper (ACL 2025, Table 3). Our E5-base baseline reproduces its 11.52 exactly.

| | Baseline | With verification |
|---|---|---|
| NDCG@10 | 11.52 | **19.96** |
| MRR@10 | 9.88 | **19.05** |

Across the 3,765 test queries, verification improved 420, made 10 worse, and left 3,335 unchanged.

## How it works

**Stage 1, meaning.** E5-base-v2 embeds the problem and every snippet, and cosine similarity picks the 50 closest. Keyword search is useless here: a problem about accordions and the code that solves it share almost no words, which is why BM25 scores 0.95.

**Stage 2, verification.** The example input and expected output are parsed out of the problem. Each of the 50 candidates runs in its own Python process, in a throwaway directory, with a one-second limit. Candidates that print the expected output move above the rest, and both groups keep their stage 1 order.

Collecting every passing candidate matters. Similar problems often share the same input format and a small integer answer, so a wrong snippet sometimes passes too. Stopping at the first pass picked the wrong one 7 times out of 15 in an early test. Ranking all passers by similarity puts the right one first or second instead.

**Why stage 2 has limits.** It only helps when the right answer is already in the top 50, which happens for 31% of queries, and when the problem includes a parseable example, which 78% do. Everything else keeps its stage 1 rank. A stronger stage 1 model is the biggest remaining lever.

## Speed

Measured on Colab's free tier, 2 CPU cores for execution.

| Step | Time |
|---|---|
| Encode one query, CPU | a fraction of a second |
| Search 8,765 snippets | a few milliseconds |
| Run 50 candidates on the example | about 3.6 s |
| Build the full index, T4 GPU | 134 s |

Verification cost grows linearly with the number of candidates, at roughly 60 ms each, almost all of it Python interpreter startup. More threads than CPU cores makes it slower, not faster: starved processes hit the timeout even when they only needed 60 ms of work.

## Retrieval across versions (P1)

CoIR APPS is a single snapshot, so `versions.py build` generates a history:

| Version | What changed |
|---|---|
| v1 | the original snippets |
| v2 | some snippets reformatted, behaviour unchanged; some given a one-token bug (`<` to `<=`, `+` to `-`, `n` to `n + 1`) |
| v3 | some bugs fixed, some new ones introduced, more reformatting |

Versions are stored the way git stores files. Each distinct snippet text is saved once, keyed by its hash, and a version is a list of doc ids pointing at those hashes. Embeddings and execution results are cached by the same hashes, so moving to a new version only encodes and runs the snippets that changed. Everything unchanged is reused.

```
python versions.py build          generate v1, v2, v3 and index each one
python versions.py list
python versions.py diff v1 v2
python index.py --simulate-edits 100   time an incremental rebuild
```

## Searching all versions at once (bonus)

All versions share one pool. A snippet that did not change between versions is one entry, embedded once and run once.

The hard part is ranking. A working version and a buggy version of the same snippet differ by a single token, so their embeddings are almost identical and similarity cannot tell them apart. Running them can: the buggy version prints the wrong answer on the problem's example. So the same verification step that lifts P0 also separates good versions from broken ones.

When a buggy version happens to pass the example too, the two stay in similarity order, so verification never ranks worse than similarity alone.

`evaluate_versions.py` measures this against two baselines: similarity alone, and similarity with the newest version preferred. For the tracked problems, whether the newest version is the buggy one is a coin flip, so "always pick the newest" cannot game the test.

```
python evaluate_versions.py       writes artifacts/versions_report.json
```

## Running it

Python 3.10 or newer.

```
pip install -r requirements.txt
```

Download the release artifacts into `artifacts/`:

| File | What it is |
|---|---|
| `emb_cache.npz` | precomputed embeddings, so nothing needs encoding on first run |
| `exec_results.jsonl` | stage 2 verdicts for all 3,765 test queries |
| `doc_ids.npy` | document order used by `exec_results.jsonl` |
| `appsretrieval_results.json` | the submitted MTEB results |

Without `emb_cache.npz` everything still works, but the first run encodes all 8,765 snippets, which is slow on CPU.

### Demo

```
python demo.py
```

Open http://127.0.0.1:7860. Paste a problem or load one from the test set. The page shows each result, whether it produced the expected output, how far it moved, and where the known answer landed.

With a version history built, a selector switches between v1, v2, v3 and all versions at once. Snippets carrying an injected bug are labelled, so you can watch them drop below the working versions.

### Reproduce the score

```
python evaluate.py --mode cached     # stage 2 verdicts from exec_results.jsonl
python evaluate.py --mode full       # run stage 2 live, resumable, hours on CPU
python evaluate.py --mode baseline   # stage 1 only, should print 11.52
```

Each writes `artifacts/appsretrieval_results.json`.

### Docker

```
docker build -t code-retrieval .
docker run -p 7860:7860 -v "$(pwd)/artifacts:/app/artifacts" code-retrieval
```

Then open http://localhost:7860.

## Layout

```
src/
  config.py          paths and constants
  data.py            loads CoIR APPS from HuggingFace
  cache.py           content-addressed embedding cache
  encoder.py         stage 1, E5-base-v2
  execute.py         stage 2, example parsing and sandboxed runs
  rerank.py          merges stage 1 order with stage 2 verdicts
  mteb_wrappers.py   plugs the two stages into MTEB
  versions.py        content-addressed version store
  history.py         generates the v1, v2, v3 history
  evolution.py       one index across versions, and the three rankings
demo.py              web demo
evaluate.py          produces the MTEB results file
evaluate_versions.py measures P1 and the bonus
versions.py          builds and inspects versions
index.py             builds and refreshes the index
```

## Integrating with MTEB

MTEB calls an encoder and computes similarity itself, so a reranker cannot sit inside the encoder. It sits one level up. MTEB uses any model that already satisfies `SearchProtocol` without wrapping it, so `src/mteb_wrappers.py` subclasses `SearchEncoderWrapper` and reorders results after the embedding search.

Two things that cost time to find, in MTEB 2.21:

- the model needs a `ModelMeta`, including `loader` and `memory_usage_mb`, or evaluation finishes and then crashes while caching the result
- the results contain datetimes, so saving them needs `json.dump(..., default=str)`

## Limitations

- Only the first example in a problem is used. Problems with several examples could be checked more strictly.
- 27% of known answers fail their own example in a sampled check. The causes are not yet diagnosed; likely candidates are output formatting (trailing spaces, float precision) and multi-case inputs. Each one fixed converts directly into score.
- Snippets run as plain subprocesses with a timeout and a temporary working directory. Docker adds container isolation, but this is not a hardened sandbox for untrusted code.
