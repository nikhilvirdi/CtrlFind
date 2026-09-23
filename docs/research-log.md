# Text-to-Code-Retrieval-CoIR-APPS

Given a programming problem written in English, find the Python code that solves it. Ranking 3,765 candidate solutions, CPU-only.

This repo is the working record: every measurement, every source, every idea kept or dropped. Not a showcase. The shipped solution lives elsewhere.

Full problem statement: [Problem Statement.pdf](./Problem%20Statement.pdf)

## The problem

A query comes in as English prose. A corpus of Python snippets sits in an index. The job is to rank the snippets so the correct one lands as high as possible.

Retrieval only. No answer generation, no explanation of results. Anything after the ranking is out of scope.

Two further requirements beyond raw accuracy:

- **P1, retrieval across versions.** Code changes. Indexes and caches must rebuild for a new version in reasonable time.
- **Bonus, evolutionary retrieval.** Search across all versions at once. Hard because versions of the same snippet look nearly identical, so ranking between them is delicate.

Scored by NDCG@10 and MRR, through MTEB, on the test split.

## The dataset

`CoIR-Retrieval/apps` on HuggingFace. Built from the APPS dataset, which scrapes Codeforces, CodeWars, CodeChef, LeetCode and AtCoder.

**Queries are full contest problem statements.** Not developer questions. A typical one runs 1,300 characters and describes accordions or sofas before stating input and output format. The hackathon doc's example ("How is the input preprocessed before the main function?") does not resemble the real data.

**Corpus is accepted Python solutions.** Median 21 lines.

| | Value |
|---|---|
| Corpus | 8,765 total, 3,765 test |
| Queries | 8,765 total, 3,765 test |
| Gold answers per query | exactly 1 |
| Relevance scores | all 1, binary |
| Language | 100% Python |
| Title field | empty, unusable |

MTEB indexes the full 8,765-document corpus when evaluating, including the train partition, while scoring only test queries. So any number measured against 3,765 documents reads higher than the same method scored through MTEB.

### Measurements

**Qrels are strictly one-to-one.** `q5001 → d5001`, always, no exceptions. No query shares a problem with another. No false negatives. With one binary-relevant document per query, NDCG@10 and MRR become functions of the gold document's rank alone, so the two metrics move together and Recall@k is binary.

**Lengths run opposite to expectation.**

| | Median | p95 | Max |
|---|---|---|---|
| Query, chars | 1,316 | 2,903 | 13,955 |
| Code, chars | 332 | 1,560 | 289,048 |

Queries are roughly four times longer than the code. At a 512-token cap, 27% of queries truncate. At 256 tokens, 81% do. Model choice decides how bad this gets.

**Vocabulary overlap carries almost no signal.**

| Comparison | Shared words |
|---|---|
| Query vs its gold code | 2.48 |
| Query vs a random code | 2.18 |

The shared words are `print` (98% of docs), `input` (93%), `split` (77%). Every document has them.

**The corpus is homogeneous.** 97% use `print(`, 92% use `input()`. Only 34% define a function, 4% define a class. Snippets read alike.

**Signal sits in the narrative, not the spec.** Measuring vocabulary similarity between two random queries:

| Section | Similarity |
|---|---|
| Story and description | 0.063 |
| Input/Output spec | 0.166 |

The I/O sections are boilerplate. Compressing a query down to its spec would discard the distinguishing half.

**Example blocks parse at 78%.** 2,936 of 3,765 test queries yield a clean input and expected output pair. Format inside the block is `-----Examples-----` then `Input`, newline, data, then `Output`, newline, data.

**Metadata leaks.** `meta_information` holds the source URL for both queries and corpus, and gold pairs match on URL 100% of the time. MTEB passes only `text` to the encoder, so it isn't exploitable, and using it would be cheating. Noted so nobody builds on it by accident.

## What the field already found

From the CoIR paper (ACL 2025), Table 3. NDCG@10:

| Model | APPS | Avg across all 10 CoIR datasets |
|---|---|---|
| BM25 | 0.95 | 29.79 |
| UniXcoder (123M) | 1.36 | 37.33 |
| GTE-Base (110M) | 3.24 | 36.75 |
| BGE-Base (110M) | 4.05 | 42.77 |
| Contriever (110M) | 5.14 | 36.40 |
| BGE-M3 (567M) | 7.37 | 39.31 |
| OpenAI-Ada-002 | 8.70 | 45.59 |
| E5-base (110M) | 11.52 | 50.90 |
| E5-Mistral (7B) | 21.33 | 55.18 |
| Voyage-Code-002 | 26.52 | 56.26 |

APPS is the hardest of the ten by a wide margin. Models scoring 50 to 70 elsewhere drop to single digits here. The paper names it as such.

Three things worth carrying forward:

1. **BM25 scores 0.95.** Keyword matching is dead, not weak. Matches the overlap measurement above.
2. **UniXcoder, a code-specific model, scores 1.36.** Worse than generic text models. Code embedders train on docstring-to-function pairs; contest problems are a different distribution. Reaching for a code model here is wrong.
3. **Scale wins.** E5-Mistral at 7B reaches 21.33 where 110M models sit between 3 and 11. Not available to us on CPU.

Long context barely helps. Table 5 in the same paper: GTE moving from 512 to 4,096 tokens lifts APPS from 3.24 to 5.08.

Sources:
- CoIR paper: https://aclanthology.org/2025.acl-long.1072.pdf
- CoIR repo: https://github.com/CoIR-team/coir
- Dataset: https://huggingface.co/datasets/CoIR-Retrieval/apps

## The approach

Two stages.

**Stage 1, embedding retrieval.** E5-base-v2 with `query:` and `passage:` prefixes, 512 token cap, cosine similarity over normalised vectors. Pulls the top K candidates. Cheap, and it sets the ceiling for everything after it.

**Stage 2, execution verification.** Parse the example input and expected output out of the query. Run each of the K candidates against that input in a subprocess with a timeout. Every snippet producing the expected output moves above the rest, keeping embedding order within each group.

Stage 2 is not similarity. A snippet that produces the right answer on the problem's own test case is almost certainly the right snippet. That sidesteps the semantic gap entirely, and it is engineering rather than modelling.

Queries with no parseable example keep Stage 1's ordering untouched.

### Why the reranker is not an encoder

MTEB calls `encode()`, gets vectors, and computes similarity itself. Execution reranking has to happen after similarity, and the encoder interface has no hook there.

The hook is one level up. `mteb/abstasks/retrieval.py` picks the search model by type, and anything already satisfying `SearchProtocol` gets used as-is rather than being wrapped:

```python
if isinstance(model, EncoderProtocol) and not isinstance(model, SearchProtocol):
    search_model = SearchEncoderWrapper(model)
...
elif isinstance(model, SearchProtocol):
    search_model = model
```

So a subclass of `SearchEncoderWrapper` passes straight through. It inherits `index()` and `search()`, and `search()` returns a plain dict of query ID to document ID and score, which is exactly the level reranking needs. No monkey-patching.

Two traps worth recording:

- `search()` sets `self.task_corpus = None` before returning, so a subclass has to keep its own copy of the corpus during `index()`.
- `ModelMeta` is required. Leaving `mteb_model_meta` as `None` lets the evaluation run and then crashes in the result cache, which builds its folder path from the model name. In MTEB 2.21 the meta also requires `loader` and `memory_usage_mb`.

## Results

Measured on the CoIR APPS test split.

| Setup | Corpus | NDCG@10 | MRR |
|---|---|---|---|
| E5-base-v2, hand-rolled eval | 3,765 | 13.11 | 12.10 |
| E5-base-v2, through MTEB | 8,765 | **11.52** | 9.88 |
| CoIR paper, E5-base | 8,765 | 11.52 | — |
| Execution rerank, 150-query sample | 3,765 | 30.74 | 30.28 |
| **Execution rerank, full test split** | **8,765** | **19.96** | **19.43** |

The MTEB baseline reproduces the published figure exactly, which validates the integration.

**The final number is 19.96**, up from 11.52. Across all 3,765 test queries, 420 improved, 10 got worse, and 3,335 were unchanged.

Against the CoIR table:

| Model | Size | NDCG@10 |
|---|---|---|
| E5-base | 110M | 11.52 |
| **E5-base + execution rerank** | **110M** | **19.96** |
| E5-Mistral | 7B | 21.33 |
| Voyage-Code-002 | API | 26.52 |

Close to a model 65 times larger, on CPU.

The 150-query sample overstated the gain. It used the smaller 3,765-document corpus and included only queries with a parseable example. Over the full split, the 22% with no example and every query whose gold sits outside the top 50 get no lift at all. Stage 1 recall caps the result.

**Recall for Stage 1** (E5-base-v2, 3,765-document corpus):

| K | Recall |
|---|---|
| 1 | 8.50% |
| 5 | 14.85% |
| 10 | 19.04% |
| 50 | 31.29% |
| 100 | 38.57% |
| 200 | 48.07% |
| 500 | 62.82% |

Median gold rank is 228. Recall gates everything Stage 2 can do, so this is the number to improve.

**Ceiling if execution worked perfectly**, promoting gold to rank 1 whenever it sits inside the top K and the query has an example:

| K | Ceiling NDCG@10 | Fires on |
|---|---|---|
| 50 | 26.28 | 21.3% |
| 100 | 31.86 | 26.9% |
| 200 | 39.43 | 34.4% |
| 500 | 51.33 | 46.3% |

**Execution behaviour:**

| | |
|---|---|
| Gold passes its own example | 22/30 (73%) |
| Wrong snippets passing, sampled at random | 0.3 per 100 |
| Wrong snippet passing before gold, top-200 early exit | 7/15 (47%) |

That last row killed early exit. Among top-ranked candidates the false positive rate is roughly 150x the random rate, because semantically similar problems share input format and output shape, so coincidental matches are common. Random snippets crash instead. Running the full candidate set and collecting every passer avoids the problem: gold lands at rank 2 or 3 rather than 1, which still beats rank 25.

**Full execution pass** over all 3,765 test queries at K=50:

| | |
|---|---|
| Queries where something passed | 924 (24.5%) |
| Gold among the passers | 555 (60% of those) |
| Gold was the only passer | 454 (49% of those) |
| Total runtime | 143 minutes, 2 cores |
| Queries improved / hurt / unchanged | 420 / 10 / 3,335 |

**Timing**, Colab free tier, 2 vCPUs:

| Stage | Cost |
|---|---|
| Encode 8,765 documents, T4 | 134s |
| Encode 3,765 queries, T4 | 116s |
| Execution rerank, K=50 | 3.6s per query |
| Execution rerank, K=100 | 6.1s per query |
| Execution rerank, K=200 | 12s per query |

Cost is linear in K at roughly 60ms per candidate, which matches the 57ms median snippet runtime. The bottleneck is Python interpreter startup serialised across two cores, so no parameter fixes it. Raising thread count past the core count actively hurts: with 32 threads on 2 cores, healthy snippets get starved enough that wall-clock exceeds the timeout and they are killed despite needing only 57ms of CPU.

## Ideas

### Alive

**Two-stage retrieval.** Confirmed working. Stage 1 recall is the binding constraint.

**Execution-based reranking.** Confirmed working, with all passers collected rather than stopping at the first.

**Better Stage 1 model.** The largest remaining lever, now confirmed by the full run. Recall@50 at 31% means the reranker only ever gets a chance on about a third of queries.

**Fixing the 27% of gold snippets that fail their own example.** Likely whitespace, float formatting, or multi-case example blocks. Each fix converts directly into score.

**Query preprocessing.** Dropping example blocks before embedding buys room under the token cap. Cheap, no model needed. Untested.

**Code-to-English at index time.** Generate a description per snippet, embed that alongside the code. Costs nothing at query time. Risk: descriptions may flatten an already homogeneous corpus. Untested.

### Dropped

**BM25 or any lexical method as a primary signal.** Measured near-random on this data, confirmed by the paper's 0.95.

**Code-specific embedding models.** UniXcoder at 1.36. Evidence is unambiguous.

**HyDE, generating code from the query and searching with it.** Expensive, needs a generative model on the query path, and execution verification does the same job with a stronger guarantee.

**LLM query rewriting.** Considered for the truncation problem. Truncation turned out to be second-order at 512 tokens, and the cost lands on the query path where speed is judged.

**Early exit on first passing snippet.** Fails 47% of the time among top-ranked candidates. Collect all passers instead.

**More worker threads than cores.** Oversubscription causes timeouts rather than speedups.

## Open questions

- **Which small model performs best here now?** The CoIR numbers are from 2024, and Stage 1 recall is the binding constraint.
- **What breaks the 27% of gold snippets that fail their own example?** Diagnosis not yet done.
- **How does the pipeline handle P1 and the bonus goal?** Untouched so far.
- **Does the JavaScript/Python conflict between the overview deck and the theme PDF matter?** Unresolved. Overview says JavaScript, sample codebase, precision and recall. Theme PDF says Python, CoIR, NDCG and MRR.

## Experiment log

Append-only. Date, what ran, what came back, what it means.

**Notebook 01, baseline recall and execution validation.**
E5-base-v2 over the 3,765-document test corpus scores 13.11 NDCG@10. Recall@50 is 31.29%, recall@200 is 48.07%, median gold rank 228. Gold snippets pass their own example 22 times out of 30. Random wrong snippets pass 0.3 times per 100, but among top-200 candidates a wrong snippet passes before gold 47% of the time, which rules out early exit. Collecting all passers and keeping embedding order within groups lifts a 150-query sample from 12.78 to 30.74, improving 34 and hurting 1.

**Notebook 02, MTEB integration.**
`SearchProtocol` accepts a `SearchEncoderWrapper` subclass directly, so reranking needs no monkey-patching. The plain encoder through MTEB scores 11.52 NDCG@10 over the full 8,765-document corpus, matching the CoIR paper exactly. A 20-query smoke test of the reranker improved 2 and hurt 0.

**Full execution pass.**
All 3,765 test queries at K=50, 143 minutes on 2 cores. 924 queries had at least one passing snippet. Gold was among the passers in 555 cases and was the only passer in 454 of those.

**Final scoring.**
Applied the execution results to the embedding ranking over the full 8,765-document corpus. Baseline reproduced at 11.52, confirming the ranking matched the one the execution pass used. With reranking, NDCG@10 rose to 19.96 and MRR to 19.43. 420 queries improved, 10 got worse, 3,335 unchanged. Lower than the 30.74 sample projected, because the sample excluded queries without examples and used a smaller corpus.
