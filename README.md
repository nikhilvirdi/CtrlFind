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

## Ideas

### Alive

**Two-stage retrieval.** A cheap first pass pulls ~50 candidates. An expensive second pass reorders them. The first pass sets the ceiling, so its recall gates everything.

**Execution-based reranking.** Parse the example input and expected output from the query. Run each candidate against it in a sandbox with a timeout. A snippet producing the correct output is almost certainly the answer. This is verification, not similarity, and it sidesteps the semantic gap. Blocked on two unknowns: whether recall@50 is high enough to feed it, and whether gold snippets actually run correctly on their own examples.

**Query preprocessing.** Dropping example blocks before embedding buys room under the token cap. Cheap, no model needed.

**Code-to-English at index time.** Generate a description per snippet, embed that alongside the code. Costs nothing at query time. Risk: descriptions may flatten an already homogeneous corpus.

### Dropped

**BM25 or any lexical method as a primary signal.** Measured near-random on this data, confirmed by the paper's 0.95.

**Code-specific embedding models.** UniXcoder at 1.36. Evidence is unambiguous.

**HyDE, generating code from the query and searching with it.** Expensive, needs a generative model on the query path, and execution verification does the same job with a stronger guarantee.

**LLM query rewriting.** Considered for the truncation problem. Truncation turned out to be second-order at 512 tokens, and the cost lands on the query path where speed is judged.

## Open questions

- **Recall@50 for a small model on the test split.** The number that decides whether execution reranking is viable. Nobody publishes it.
- **Do gold snippets run correctly on their own example input?** If stdin format or parsing breaks, execution reranking dies regardless of recall.
- **How long does executing N candidates take?** Most wrong programs should fail fast, but that is an assumption, not a measurement.
- **Which small model performs best here now?** The CoIR numbers are from 2024.
- **Does the JavaScript/Python conflict between the overview deck and the theme PDF matter?** Unresolved. Overview says JavaScript, sample codebase, precision and recall. Theme PDF says Python, CoIR, NDCG and MRR.

## Experiment log

Append-only. Date, what ran, what came back, what it means.

_(empty)_
