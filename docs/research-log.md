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

**Historical test-split example coverage was 78%.** The baseline notebook counted 2,936 regex matches across 3,765 test queries. Its pattern required a dashed example heading and bare `Input` and `Output` labels. See the later [training comparison](#2026-11-24-colon-labeled-example-extraction) for the split and measurement differences.

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
| **Execution rerank, full test split** | **8,765** | **19.96** | **19.05** |

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

**Diagnosing gold snippets that fail their own example.** The [training audit](#2026-11-24-training-only-verification-audit) found that valid alternative outputs were the largest category among runnable examples. Example-label coverage was the largest measured opportunity overall.

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
- **How much can verification improve with better example coverage and output checks?** The [training audit](#2026-11-24-training-only-verification-audit) records the initial diagnosis and a parser probe.
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
Applied the execution results to the embedding ranking over the full 8,765-document corpus. Baseline reproduced at 11.52, confirming the ranking matched the one the execution pass used. With reranking, NDCG@10 rose to 19.96 and MRR to 19.05. 420 queries improved, 10 got worse, 3,335 unchanged. Lower than the 30.74 sample projected, because the sample excluded queries without examples and used a smaller corpus.

### 2026-11-24 Training-only verification audit

The current [verifier](../src/execute.py) looks for the first dashed `Example` or `Examples` section with bare `Input` and `Output` labels. It strips the two captures, appends one newline to stdin, and executes the paired code as a Python script in a temporary directory. It compares stripped stdout to the expected string exactly. Interior spacing, numeric formatting, and alternate valid answers therefore affect the verdict. The verdict ignores process exit status and stderr.

**Method.** Used the cached CoIR APPS snapshot at revision `f22508f96b7a36c2415181ed8bb76f76e04ae2d5`. Read only the train qrels and filtered both queries and snippets to `partition=train` before materializing their text. Shuffled the 5,000 lexicographically sorted training query IDs with `random.Random(20261124)` and scored every pair the existing parser accepted. The failure audit covers the first 50 failing gold pairs in that order. For each accepted query, sampled five distinct training document IDs uniformly from the sorted eligible IDs using a separate `random.Random(20261125)`, excluding the gold ID and byte-identical copies of its code. Each eligible query went through the current verifier with two workers and the default one-second per-candidate timeout on Windows with Python 3.13.13. A research wrapper captured subprocess evidence without altering the verdict. No test-split evaluation ran.

| Measurement | Count | Rate |
|---|---:|---:|
| Training queries accepted by the parser | 511 / 5,000 | 10.22% |
| Gold passes given a parsed example | 392 / 511 | 76.71% |
| Training queries whose gold received a pass, including skipped queries in the denominator | 392 / 5,000 | 7.84% |
| Unrelated training pairs that passed | 3 / 2,555 | 0.117% |

The repeat run gave 391 gold passes and the same three unrelated passes. In the saved run, 97 gold pairs exited successfully with mismatching stdout and 22 exited with an error. None of the gold trials timed out. The unrelated pass rate describes uniformly sampled documents. It does not estimate the pass rate for difficult negatives from the retriever's top results.

| Primary cause in the first 50 failing gold pairs | Count | Evidence |
|---|---:|---|
| Valid alternative output | 27 | Replayed or checked each program's output against the statement's constraints. For example, `q2142` prints `1 3 2` instead of `1 2 1`, and the labels give the required smaller-label counts in every subtree. |
| Trailing whitespace on intermediate lines | 11 | Per-line trimming of spaces and tabs preserves blank lines and makes the output match. `q2380` is representative. The same check found 16 recoverable gold mismatches across all 511 parsed queries. |
| Source syntax error | 7 | Five snippets contain a module-level `return` and two contain a `nonlocal` without an enclosing binding. Wrapping each source in a function in memory yielded six exact passes and one valid alternative. The saved run had 21 compile failures overall. |
| Floating-point formatting | 2 | `q2125` prints the same number with one fewer trailing zero. Both values printed for `q2041` are within the statement's tolerance of 10⁻⁶. |
| Explanation included in expected output | 1 | In `q1289`, stdout matches the answer before the plain `Explanation` heading. |
| Placeholder parsed as an example | 1 | `q1696` explicitly says it has no sample input and output, but the parser supplies this text to the verifier. |
| Incorrect behavior in the paired gold code | 1 | `q2387` prints a set of items and loses the required permutation order in cases 2 and 5. The training qrel still labels the snippet as gold. |
| Multiple cases, input-parsing failures, other runtime errors, timeouts | 0 | These were absent as primary causes in the first 50. Outside the sample, `q1694` feeds a literal truncation marker to an integer parser and raises `ValueError`. |

Two additional results matter. The unrelated `d1437` matched `q2111` by printing `2`, then crashed with `EOFError` and exit code 1. The current comparison counted it as a pass. Separately, 306 skipped training statements matched a narrow probe for dashed example blocks with a colon on a separate `Input` or `Output` label. A sample of 50 of those statements, drawn with seed `20261126`, yielded **38 exact gold passes** using the current executor and comparison on the extracted pairs. This probe measures additional example coverage. It did not measure negative passes for the new format.

**Fix order by observed opportunity**

1. **Handle colon-suffixed example labels first.** The probe recovered 38 exact passes from 50 currently skipped examples, the largest measured opportunity. Keep example boundaries explicit and use `q1289` and `q1696` as cases for a follow-up boundary and placeholder check.
2. **Use problem-specific output checks for tasks that allow multiple valid answers.** This accounts for 27 of the first 50 failures. Prefer authoritative checkers where available. Use the statement's acceptance rules before treating a different answer as wrong.
3. **Trace source normalization for snippets that fail to compile.** There were 21 such cases in the parsed training population. The wrapping probe shows that restoring function scope is worth investigating at ingestion.
4. **Ignore trailing horizontal whitespace per line.** This would recover 16 observed mismatches while preserving the number and order of lines.
5. **Honor explicit numerical tolerances on numeric outputs.** The two diagnosed examples have statement-defined tolerance. Preserve exact comparison for fields governed by exact rules.

Require successful process exit before crediting a match when changing comparison. This would have rejected the crashing false pass in this run. Recheck negative pass rates when broadening parsing or output matching.

The [audit artifact](../artifacts/train_verification_audit.json) records all 50 diagnoses, raw or excerpted outputs, diagnostic checks, negative matches, and the trial ledger. From the repository root, with the project environment active, the [research runner](../artifacts/verification_research.py) replays the baseline and parser probe using the cached snapshot.

```text
python -B -X utf8 artifacts/verification_research.py train_trials.jsonl
python -B -X utf8 artifacts/verification_research.py colon_probe.json --colon-probe
```

### 2026-11-24 Colon-labeled example extraction

The earlier 78.0% figure comes from the saved output of cell 13 in the [baseline notebook](../notebook/baseline_recall_and_execution_validation.ipynb). It measured regex matches on 3,765 **test** queries from CoIR APPS. The notebook loaded the dataset without an explicit revision. The training audit pinned revision `f22508f96b7a36c2415181ed8bb76f76e04ae2d5` and required both extracted fields to be nonempty after trimming. On the 5,000 **training** queries, the old regex matched exactly 511 and all 511 passed the nonempty check. The observed gap is between splits. There is no evidence here of a dataset revision effect.

The training format census explains why this parser covers so little of that split. Of 5,000 queries, 4,138 lack a dashed example heading, including 2,515 Codewars, 739 LeetCode, and 725 CodeChef statements. The old parser accepted 474 Codeforces and 37 CodeChef examples. The audit's narrower probe counted 306 additional CodeChef examples with a colon on a standalone label.

The parser now accepts bare, colon-suffixed, and mixed labels inside the same dashed section, including data written immediately after a colon. It stops output at another example, a dashed section, or an explicit note or explanation heading. It rejects empty fields and explicit no-sample or truncation text. Adding optional colons to the old regex alone would still accept the prose in `q1289` and the placeholders in `q1694` and `q1696`. Numbered labels and examples without the dashed header were left outside this change. Output comparison and subprocess behavior remain as in the baseline.

**Paired measurement.** Loaded the same pinned training snapshot and replayed the old parser from commit `0586e97` alongside the changed parser. Shuffled sorted training IDs with seed `20261124`. Using seed `20261125` and the audit's eligibility rules, assigned the same five distinct unrelated training document IDs per query to both arms. Assigned IDs to the original 511 queries first, in shuffled order, and verified that all 2,555 IDs reproduced the saved audit ledger. Continued the RNG through the other queries in shuffled order to fix the remaining pairs. Ran the existing verifier on the union of accepted queries with two workers and a one-second per-candidate timeout under Windows Python 3.13.13. Reused the exact same execution result for both arms whenever the extracted input and expected output were identical. A skipped example contributes zero passes to the fixed-denominator rates. Executed-only rates are shown with their respective denominators. No test queries were loaded or evaluated for this comparison.

| Measurement | Before | After |
|---|---:|---:|
| Parsed queries, fixed denominator | 511 / 5,000 (10.22%) | 845 / 5,000 (16.90%) |
| Gold passes, fixed denominator | 391 / 5,000 (7.82%) | 667 / 5,000 (13.34%) |
| Wrong passes, fixed denominator | 3 / 25,000 (0.012%) | 5 / 25,000 (0.020%) |
| Gold passes among executed gold pairs | 391 / 511 (76.52%) | 667 / 845 (78.93%) |
| Wrong passes among executed wrong pairs | 3 / 2,555 (0.1174%) | 5 / 4,225 (0.1183%) |

The parser extracted 336 new pairs with 275 gold passes. It corrected the expected output for `q1289`, which added one gold pass, and skipped the two previously accepted non-runnable examples `q1694` and `q1696`. The other 508 originally accepted pairs were identical. The two removed pairs had zero passes in either candidate class. No previously passing gold or wrong pair lost its pass. The paired baseline reproduced the previous audit's 391-pass repeat run. The initial audit run recorded 392.

**Validation.** All 12 parser regression tests passed with `python -B -m unittest discover -s tests -v`. The no-increase requirement failed. Both new wrong passes were reproduced twice and both candidates exited successfully.

- `q949 / d814` expected `1` for the square-counting example `4 4`. The unrelated snippet treats the missing array line as empty and prints `1`. The gold snippet also passes this example.
- `q1156 / d920` expected `2` for a rectangle problem. The unrelated sequence-editing snippet processes the first rectangle row as an array, ignores the remaining rows, and prints `2`. The paired gold source is just `print` and produces no stdout.

These are exact-output collisions on correctly extracted pairs. Rejecting scalar answers would discard valid examples, including `q949`. The coverage fix alone cannot meet the observed false-positive requirement with the current verification rules.

### 2026-11-24 Follow-up on the two new wrong passes

Replayed `q949 / d814` and `q1156 / d920` with the original gold snippets and the unchanged executor. Used the same cached CoIR APPS training revision, two workers, and a one-second timeout. The [saved audit](../artifacts/train_verification_audit.json) lists neither query in the legacy execution ledger. Its colon-label probe independently records `d1156` exiting successfully with empty stdout. The old parser returned no pair for either statement. The current parser extracted both published pairs exactly. Each replay below was run twice with identical stdout, empty stderr, exit code 0, and no timeout for both candidates. Displayed inputs omit the final newline that the executor appends.

| Training statement | Published stdin | Published expected stdout | Gold stdout | Wrong stdout |
|---|---|---|---|---|
| `q949` (CHNGFUNC) | `4 4` | `1` | `d949: "1\n"` | `d814: "1\n"` |
| `q1156` (NOPC2) | `1\n4\n0 4 0 4\n1 3 1 2\n2 3 1 3\n3 4 1 3` | `2` | `d1156: ""` | `d920: "2\n"` |

**`q949 / d814` was a coincidental match on a valid example.** The problem counts pairs for which `x*x + y` is a square. Direct enumeration confirms that `(1, 3)` is the only such pair for `4 4`. The unrelated snippet solves an array minimization problem. It interprets the first line as `n=4, k=4` and reads an empty array at EOF. Both costs over the empty array are zero, so its tie branch prints `1`. The statement contains one runnable example. Its explanation repeats the same pair.

**`q1156 / d920` matched a published answer that conflicts with the cached specification.** The unrelated snippet solves an even-sequence editing problem. It reads the first rectangle row `0 4 0 4` as its entire array, leaves the other three rows unread, and prints `4 - 2`. The gold-labeled source contains only the expression `print` and emits nothing. In the published sample, the first rectangle contains corners of the others, contrary to the input guarantee. All four rectangles include `(3, 1)`, so the maximum number with no common point is one under the stated closed-rectangle coordinates. Exhaustive enumeration confirms this. The cached statement has one published pair and no additional valid example. The intended upstream judging convention and the source of the discrepancy were outside this training-only replay.

Two controls constructed from the training statements show what additional evidence would do. For `q949`, input `1 2` has independently enumerated answer `0`. Gold printed `0` and `d814` still printed `1`. For `q1156`, input `1\n2\n0 1 0 1\n3 4 3 4` describes two separated rectangles and has answer `2`. The independent check returned `2`, `d920` printed `0`, and the stored gold snippet still printed nothing. These are diagnostic controls. Neither was added to the training measurements.

| Wrong-pass comparison | Before | After |
|---|---:|---:|
| Same full panel, with skipped pairs receiving no pass | 3 / 25,000 (0.012%) | 5 / 25,000 (0.020%) |
| Same union of 847 queries accepted by either parser | 3 / 4,235 (0.0708%) | 5 / 4,235 (0.1181%) |
| Same 508 queries with identical extracted pairs | 3 / 2,540 (0.1181%) | 3 / 2,540 (0.1181%) |

The two passes came entirely from the 336 newly parsed queries, with two matches in 1,680 wrong trials (0.1190%). For context, the executed-only rates were 0.1174% before and 0.1183% after, with different executed denominators. The absolute increase on the fixed panel is real. It reflects broader exposure to the existing single-example rule, and the replay found no worse verdict on previously evaluated pairs. Five matches from uniformly sampled negatives are too few to estimate whether the change affects false passes among retrieved candidates.

**Recommended verification rule.** Promote candidates only when they pass a sufficient set of independently checked cases. Run all available trustworthy examples and supplement a lone weak example with a vetted reference or official checker when one exists. Validate the expected answers before treating a mismatch as failure. Use an inconclusive verdict and preserve the candidate's existing rank when the evidence is insufficient or inconsistent. This keeps valid candidates eligible and would withhold both promotions studied here. A successful-exit requirement is still useful for crashes and would leave these two clean-exit matches in place. Generic scalar-output filters or checks for how much input a process reads cannot establish correctness. For arbitrary snippets, finite examples always leave some uncertainty.

### 2026-11-24 Multiple-example promotion experiment

Used the same pinned CoIR APPS training snapshot as the [audit](../artifacts/train_verification_audit.json), with 5,000 training queries, 5,000 original training snippets, and one qrel per query. Counted distinct, nonempty stdin/stdout pairs within the current dashed-example grammar. A research-only scan split at later input labels or new example headings and reused the current parser for each pair. It required the first extracted pair to equal the current parser's result and excluded repeated pairs. No conflicting outputs for an identical stdin were found by this scan. An extractable pair has enough structure to run through the current stdin harness. Its publication alone does not establish that the input satisfies the problem's constraints or that the expected output is correct. A batch inside one stdin block counts as one pair.

| Extractable published pairs per training query | Queries |
|---|---:|
| 0 | 4,155 |
| 1 | 602 |
| 2 | 156 |
| 3 | 72 |
| 4 | 15 |

There were 1,190 pairs in all. The 243 multi-example problems were all from Codeforces. Four had identical expected stdout across every extracted pair. The 4,155 without a pair may have examples in other formats, including function calls, which require a different execution harness.

**Independent evidence.** The local training snapshot provides query and snippet text, one relevance label per query, and `starter_code` and `url` metadata. It supplies zero machine-readable independent fixture suites or checkers. The stricter experiment used 10 ranking-sample problems whose 22 published pairs were independently validated before scoring. The checks were derived from the training statements and did not run or inspect retrieved code or the paired gold output as an oracle.

| Problem | Independent check |
|---|---|
| `q1608`, `q2006` | Enumerate letter triples and equally spaced binary triples within each interval, respectively. |
| `q2012`, `q2049` | Enumerate digit-sum preimages and simulate coin transfers, respectively. |
| `q2108`, `q2188` | Enumerate bracket wildcard substitutions and find the maximum matching among apple pairs with a common divisor greater than one. |
| `q2191`, `q2202` | Simulate simultaneous neighbor removals and minimize the maximum signed interval sum at rational line intersections. |
| `q2253`, `q2315` | Enumerate word pairs by character parity and reduce every query subsegment by adjacent XOR. |

The number of independently validated problems outside this reviewed subset is unknown. The earlier `q949` control shows that a single published example can sometimes be supplemented by an independently checked case. This control and the later `q2150` diagnostic were kept outside the verification arms.

**Gold-snippet census.** Ran each of the 845 eligible gold snippets on the current first example. For a first pass, ran subsequent examples until a mismatch or exhaustion. Used the existing exact comparison, Windows Python 3.13.13, one worker for this per-snippet census, and one second per execution. This yielded 662 first passes (78.34%), compared with 667 in the earlier six-candidate paired run. Within this census both arms used the very same first verdict.

| Gold outcome | Count |
|---|---:|
| First-example passes with only one published pair, therefore inconclusive under the gate | 472 / 662 (71.30%) |
| First-example passes with two or more published pairs | 190 |
| Of those, matched every published pair | 184 |
| Of those, failed on a later pair | 6 |

Even trusting every publication and keeping the comparison exact, the gate would withhold 478 of 662 gold promotions. All six additional gold mismatches happened on the second pair. On replay, every one was a valid alternative under the problem statement. The checks covered a tied divisor (`q1607`), a different index order (`q2097`), two valid jewelry colors (`q2106`), a valid cycle cover (`q2154`), a valid permutation (`q2354`), and a different graph with the required shortest-path count (`q2361`). Eight of the 10 independently vetted problems had gold snippets matching every exact string. The others printed a valid reversed apple pair (`q2188`) and a value within the stated tolerance (`q2202`) on the first case.

**Paired ranking replay.** Drew 80 queries from each of the zero, one, and multiple-published-pair strata with a single seeded generator (`20261127`), using sorted query IDs within each stratum and shuffling the concatenated sample with the same generator. The ordered query IDs joined by newlines have SHA-256 `fc97bff1ea4306892a5b14a27bc5bf9328df1ac7bf0f2b8ba40d043daebc05ae`. All arms used the same 240 queries and the same E5-base-v2 cosine ranking over the 5,000 training snippets. Encoded the queries offline with the cached model revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`, default prefixes, L2 normalization, and the 512-token cap. Reused all 5,000 training document vectors from the embedding cache and sorted cosine ties by document ID. Gold appeared in 111 top-50 sets.

Compared cosine order, current first-example verification, a gate requiring two published cases, an all-published gate with at least two, and a strict gate requiring independently validated cases. In every execution arm, candidates that pass move ahead of the others in their original cosine order. A missing case set is inconclusive and keeps cosine order. Used the current verifier with four workers and the same one-second timeout for all arms. Reused each first verdict and ran a later case only on candidates that had passed the preceding cases. An accepted gold or non-gold candidate is counted as promoted even if already at the front. Non-gold is defined by the qrels and may include another valid solution. NDCG and MRR use the same 240 queries, truncated at 10 and scaled to 0–100. Stratum-weighted results estimate the training mix using population weights of 4,155, 602, and 243 out of 5,000. Relevance labels were used for scoring only.

| Rule | NDCG@10 | MRR@10 | Weighted NDCG@10 | Weighted MRR@10 | Gold promotions / 111 | Non-gold promotions / 11,889 | Inconclusive queries / 240 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cosine only | 26.24 | 23.79 | 47.36 | 43.98 | 0 | 0 | 240 |
| Current first example | **35.60** | **33.70** | **49.89** | **46.64** | 40 | 13 | 80 |
| At least two published, use first two | 30.34 | 28.25 | 47.96 | 44.63 | 17 | 0 | 160 |
| At least two published, use all | 30.34 | 28.25 | 47.96 | 44.63 | 17 | 0 | 160 |
| Independently validated cases only | 27.48 | 25.27 | 47.54 | 44.20 | 4 | 0 | 230 |

On the fixed top-50 panel, the first-example rule promoted 36.04% of the 111 gold candidates and 0.109% of the 11,889 non-gold candidates. Those rates were 15.32% and 0% with the published gate and 3.60% and 0% with the strict gate. Among gold candidates actually tried, the counts were 40/49, 17/23, and 4/5, respectively. The inconclusive query counts correspond to 4,000, 8,000, and 11,500 top-50 candidate verdicts. Of the 23 lost gold promotions with the published gate, 22 lacked a second example and one was the valid alternative for `q2154`. Nine of the 13 non-gold matches were on single-example queries and four no longer matched the added case. The gold rank improved on one query, worsened on 15, and stayed the same on 224 compared with the current rule. With the strict gate those counts were one, 23, and 216.

The published gate lowered weighted NDCG@10 by 1.93 points and MRR@10 by 2.01. The 95% paired stratified bootstrap intervals from 10,000 query resamples were [−2.93, −1.06] and [−3.05, −1.11] points, respectively. The strict gate lowered weighted NDCG@10 by 2.35 and MRR@10 by 2.45 points. Its independently checked subset is a convenience sample. The measurements compare the available fixture coverage and fixed verdicts in this replay.

**Execution cost.** The first-example rule ran 8,000 candidate subprocesses, the two-published gate 4,022, the all-published gate 4,027, and the strict gate 506. The savings came from abstaining. On the same 80 multi-example queries the all-published gate added 27 executions over 4,000 first attempts, or 0.68%, by testing only survivors. Median per-query execution time was 1.43 seconds in either arm on that stratum. Windows emitted subprocess decoding warnings during the replay. The comparisons used the current behavior and shared those verdicts.

**Collisions can survive a second example.** In a separate replay of the five documented random-panel wrong matches, three queries had no second published pair. The second example caught the crashing `q2111 / d1437` match. The bracket-counting snippet `d2108` still passed both examples for the unrelated graph problem `q2150`. Both published answers are correctly `2` by independent graph traversal. A third valid input `2 1\n1` requires `1` and distinguishes the gold snippet from `d2108`, which still prints `2`. The document `d2108` was outside the ranking sample's top 50 for `q2150`. These targeted diagnostics were excluded from the measured ranking arms.

**Fixture availability and rejected alternatives.** The local data has no fixture column. The upstream APPS release documents a separate training JSONL with an `input_output` field and an optional function name. Its documented first training record repeats the published sample, so an upstream join alone does not guarantee new evidence. The train-only file was not cached and the workspace network probe could not resolve its host, so coverage after joining was not measured. A fixture-assisted follow-up would need to match training rows by source URL and confirm statement identity, deduplicate published inputs, and use only verified expected outputs. Using matched solution code, relevance labels, or URL equality to rank candidates would leak the target. Using unavailable judge fixtures would also change the evidence available at deployment. Neither splitting arbitrary multi-case stdin blocks nor deriving expected outputs from a retrieved or gold-labeled program provides independent evidence. [APPS data fields](https://huggingface.co/datasets/codeparrot/apps#data-fields) · [Upstream training-file listing](https://huggingface.co/datasets/codeparrot/apps/tree/main)

**Decision.** Reject a blanket multiple-example promotion requirement for the current data. It eliminated 13 measured non-gold promotions while losing 23 gold promotions under the generous published gate and reducing both ranking metrics. Keep the current first-example rule pending a separate evaluation of independently checked fixtures and problem-specific output checking. Diverse cases can be useful where trustworthy evidence exists, and twice matching a weak output is still weak evidence. No test-split data was evaluated, and the experiment left production behavior and the saved audit artifacts intact.

### 2026-11-24 Successful process exit and trailing-whitespace probe

Restored the original parser from `0586e97` and removed the experimental parser tests. Both arms of this comparison use the original parser, which accepts 511 of the 5,000 cached CoIR APPS training queries. The only execution change is to return stdout after a successful process exit. Successful runs still use the same exact comparison after outer trimming. A diagnostic written to stderr on a successful run is allowed. Saved execution verdicts are historical artifacts and were left intact.

The verifier tests cover matching and mismatching successful stdout, a successful run that writes to stderr, explicit nonzero exit after matching stdout, an exception after matching stdout, a timeout, and a subprocess launch error. All five tests pass. Running the same tests against the old verifier fails exactly the two nonzero-exit checks.

**Paired trial.** Used the same training dataset revision as the [saved audit](../artifacts/train_verification_audit.json). Reconstructed its seeded candidate panel and checked every query, gold ID, and sampled non-gold ID against the saved ledger. This gives 511 gold pairs and 2,555 unrelated pairs. Each before/after verdict reused the same observed subprocess result under Windows Python 3.13.13, two workers, and a one-second timeout. The old rule accepts matching stdout even when the process exits nonzero. The new rule adds the zero-exit condition. All new-rule predictions were checked against the actual patched verifier.

| Audit panel | Old verifier | Exit code 0 required | Exit check plus offline trailing trim |
|---|---:|---:|---:|
| Gold passes | 392 / 511 (76.71%) | 392 / 511 (76.71%) | 408 / 511 (79.84%) |
| Wrong passes | 3 / 2,555 (0.1174%) | 2 / 2,555 (0.0783%) | 2 / 2,555 (0.0783%) |

The exit check removed exactly one pass. For `q2111`, `d1437` printed `2\n`, raised `EOFError`, and exited with code 1. The gold snippet still passed. The two coincidental matches that exit successfully, `q2150 / d2108` and `q69 / d1491`, still passed. The restored parser leaves the two previously investigated colon-only examples outside this comparison.

**Ranking replay.** Drew 100 queries from the 511 originally parseable training queries using `random.Random(20261129).sample(sorted(query_ids), 100)`. The selected IDs in replay order, joined with newlines, hash to `e102a68ad6fdfd7960fba8f7b4111ece20f22d4035588531b054d5e8a75f80ff` with SHA-256. Used E5-base-v2 with the existing prefixes, 512-token limit, L2 normalization, and cached document vectors for all 5,000 training snippets. The fixed top 50 used cosine similarity and document ID tie order. Each rule reused the same executions and moved passing candidates ahead of the others in cosine order. NDCG@10 and MRR@10 use all 100 queries, including those whose gold document was outside the top 50. Scores are scaled to 0–100. This ranking sample is conditional on the parser accepting an example.

| Rule | NDCG@10 | MRR@10 | Gold passes in top 50 | Non-gold passes in top 50 |
|---|---:|---:|---:|---:|
| Cosine only | 11.43 | 9.73 | — | — |
| Old verifier | 28.63 | 28.50 | 28 / 32 (87.50%) | 4 / 4,968 (0.0805%) |
| Exit code 0 required | 28.63 | 28.50 | 28 / 32 (87.50%) | 1 / 4,968 (0.0201%) |
| Exit check plus offline trailing trim | 29.63 | 29.50 | 29 / 32 (90.63%) | 1 / 4,968 (0.0201%) |

The exit fix withheld three additional non-gold promotions in the retrieval sample. `d1027` and `d921` printed `1` for `q2256`, and `d576` printed `2` for `q2192`. Each then raised `EOFError` and exited 1. Both gold documents were outside the top 50, so these improvements did not move any gold ranks. The fix requires no additional subprocesses. The replay encountered Windows subprocess decoding warnings, and the affected executions used the same captured result in all comparisons.

**Separate whitespace probe.** After applying the exit check, compared the same observed outputs offline by trimming only trailing ASCII spaces and tabs on each line, on both stdout and the expected string. Kept all newline positions and all interior leading spaces and token separators. This tests output normalization independently of the process-exit fix. Global whitespace tokenization was excluded because it erases row and token-layout details. Synthetic fixed-width output such as `"X  \nYZ"` and `"X\nYZ"` collapses under the narrower per-line rule as well. The training corpus does not supply a checker that can establish where this distinction is significant.

All 11 diagnosed failures reproduced and matched under the offline comparison. There were 23 differing lines, all with ASCII spaces. Ten examples had padding on the published side and one had it on the snippet side. The statements describe line-oriented numbers, tokens, or constructions, and every token and row boundary matched, including intentional blank rows in `q1` and `q2392`. The probe recovered the same 16 gold IDs as the previous full audit and added 3.13 percentage points on the gold panel. This supports ignoring trailing padding for those cases. It does not establish that trailing padding can be ignored for every output contract.

The offline trim added no non-gold matches in either panel. In the retrieval replay it moved `q2392 / d2392` from rank 18 to rank 1 and increased each ranking metric by 1.00 point. Keep this as a separate, promising comparison experiment. A later change should validate the output contracts where trailing padding is insignificant before enabling it broadly.

**Decision.** Keep the exit-code fix. Gold pass rate on the audit panel remained above the 73% threshold, no gold passes or gold ranks regressed, and non-gold passes decreased in both panels. Trailing-whitespace behavior, retrieval, embeddings, ranking rules, and frozen artifacts remain unchanged. No test-split evaluation, commit, or push was performed.

### 2026-11-24 Scoped trailing-padding comparison

Enabled per-line trimming only for examples whose dashed output specification explicitly calls for space-separated integers or numbers, with an expected output containing only integer rows and blank lines. Layout-sensitive language, including fixed width, alignment, ASCII art, and required end spaces, keeps exact comparison. Unknown or mixed output also stays exact. The normalization strips ASCII spaces and tabs from each line's right edge. It preserves interior indentation, internal separators, line breaks, row order, content, and non-ASCII whitespace. The pre-existing outer trim and successful-exit requirement remain in effect.

The distinction matters in the training corpus. The statement for `q2598` requires space-padding values to a specified width, and `q4838` specifies left justification and 15-character fields. The ASCII-art tasks `q3155` and `q4408` require layout-sensitive output. These tasks do not opt in. Synthetic verification tests cover both fixed-width numeric rows and mixed graphic output, using examples in which unconditional trimming would accept a different layout. The policy is intentionally conservative because the cached dataset provides no checker metadata. It can miss valid output contracts expressed in other words.

**Paired measurement.** Reused the training-only setup from the exit-code study. The audit arm uses the same 511 gold pairs and 2,555 unrelated pairs, checked by ID against the saved ledger. The ranking arm uses the same seeded 100 parseable queries and fixed top-50 cosine candidates over 5,000 training documents. E5-base-v2 prefixes, embeddings, tie order, and reranking procedure are unchanged. Each comparison uses the same captured process result under Windows Python 3.13.13, two workers, and a one-second timeout. Exact matching with the exit check is the baseline. The scoped verdict is computed by the production verifier against the captured stdout, with no second execution. An unconditional per-line trim was measured offline as a separate control.

| Audit panel | Exact with exit check | Scoped trim | Unconditional trim, offline |
|---|---:|---:|---:|
| Gold passes | 390 / 511 (76.32%) | 393 / 511 (76.91%) | 406 / 511 (79.45%) |
| Wrong passes | 2 / 2,555 (0.0783%) | 2 / 2,555 (0.0783%) | 2 / 2,555 (0.0783%) |

Sixteen of the 511 statements qualified. Scoped trimming recovered `q118`, `q1276`, and `q2176`, including two of the 11 audit cases. Broader trimming recovered all 16 known whitespace-only cases, but was rejected because it can erase meaningful padding. The baseline has two fewer gold passes than the previous run. A diagnostic replay reproduced a different valid construction on `q2010` and a timeout on `q50`. Those variations are shared within this paired comparison.

| Ranking cohort and measurement | Exact with exit check | Scoped trim |
|---|---:|---:|
| Fixed 100 queries, NDCG@10 | 28.63 | 28.63 |
| Fixed 100 queries, MRR@10 | 28.50 | 28.50 |
| Fixed 100 queries, top-50 gold passes | 28 / 32 (87.50%) | 28 / 32 (87.50%) |
| Fixed 100 queries, top-50 non-gold passes | 1 / 4,968 (0.0201%) | 1 / 4,968 (0.0201%) |
| All 16 eligible queries, NDCG@10 | 10.19 | 16.44 |
| All 16 eligible queries, MRR@10 | 9.38 | 15.63 |
| All 16 eligible queries, top-50 gold passes | 2 / 3 (66.67%) | 3 / 3 (100%) |
| All 16 eligible queries, top-50 non-gold passes | 3 / 797 (0.3764%) | 3 / 797 (0.3764%) |

The fixed sample contained two eligible queries and neither changed a promotion. The unconditional control moved `q2392` to rank 1, but the scoped rule kept exact comparison for that statement because it does not explicitly declare separated numeric output. The separate cohort of all 16 eligible queries used the same model, corpus, K, timeout, and paired replay procedure. It moved `q118 / d118` from rank 17 to rank 1. Only three gold snippets were in its top-50 sets, so the 6.25-point gains describe this cohort only.

**Validation and decision.** Retain scoped trimming. The audit gained three gold passes, both ranking scores held or improved, and none of the panels gained a non-gold pass. No gold rank worsened. All 12 regression tests passed, covering multiline output, spaces and tabs, blank rows, indentation, internal spacing, changed content or order, fixed width, ASCII art, and exit and error behavior. Cache checks confirmed that exact and scoped verdicts remain separate for the same example. A synthetic live MTEB checkpoint test also confirmed that historical exact entries do not supply a scoped verdict. The original parser, candidate execution, and frozen benchmark artifacts were preserved. No test-split evaluation, commit, or push was performed.

### 2026-11-24 Scoped comparison replay and layout safeguards

Compared the prior exact-output counts against the saved audit ledger. The drop from 392 to 390 involved precisely `q2010 / d2010` and `q50 / d50`. The old audit recorded both as passes in 0.0661 s and 0.7978 s, respectively. The per-process traces from the 390-pass run were temporary, so its two statuses are supported by the recorded ID comparison and separate diagnostic runs.

- `d2010` calls `random.shuffle`. In 12 one-second diagnostic runs it printed the published `1 1 -1` eight times and `-1 1 1` four times. Every process exited 0, and both constructions meet the statement's vector-norm bound. Exact comparison accepts only the published string. The fresh full replay captured the published string again for this pair.
- `d50` builds and sorts a large enumeration before handling the sample input. All four one-second diagnostic attempts timed out before producing stdout. With a five-second diagnostic budget, the identical source and input produced the exact expected output and exited 0 in 1.4661 s. The fresh full run captured another timeout at 1.0412 s. The benchmark budget remained one second in all measured arms.

The parser, sample IDs, sources, expected answers, interpreter, worker count, and nominal timeout were the same. The exact arm still uses the exit-0 check and exact equality. The discrepancy is explained by an unseeded construction and a snippet near the wall-clock cutoff, with host load varying between launches. No demo, history, or MTEB verdict cache supplied a decision in these experiments.

The replay used the pinned training revision and matched all 511 gold IDs and 2,555 unrelated IDs to the archived ledger. For ranking, it rebuilt the original 100-query cohort from seed `20261129` and included a separate census of all 16 eligible problems. Both arms received the same candidate order from E5-base-v2 cosine scores over the original 5,000 training documents, with document-ID tie order. The existing document embedding cache was read-only. Both arms reused one fresh process observation per pair, ran with two workers and a one-second timeout, and used the same stable promotion rule and NDCG@10 and MRR@10 denominators.

**Layout and exit checks.** Known fixed-width and ASCII-art training statements `q2598`, `q4838`, `q3155`, and `q4408` do not opt in. Synthetic inputs confirmed exact comparison for numeric fixed-width, ASCII art, mixed output, and undeclared formats. Two independent controls exposed a gap in the prior guard. A declaration that required a blank after the final integer or described a visual pattern could still opt in and accept a changed layout. Extended the existing layout exclusion to cover those phrasings. Both new subcases fail under the prior guard and pass under the tightened guard. Eligibility is identical before and after this hardening on all 511 parseable training queries, so the same captured results apply. The guard remains textual and does not replace a problem-specific checker. The known `q2111 / d1437` crash printed the exact answer before exiting 1 and was rejected in both modes. All 12 regression tests pass, including the new subcases.

**Worktree inventory.** At the start of the review, `git diff --stat` showed seven modified tracked files, with 305 insertions and 35 deletions relative to `HEAD`. The index was empty. `src/execute.py` contains the prior exit check and scoped comparison. `demo.py` and `src/mteb_wrappers.py` carry the comparison choice into live verification and cache keys. `src/evolution.py` distinguishes version-history verdicts by comparison mode. `evaluate_versions.py` and `versions.py` pass the same choice to version evaluation and bug observability checks. `docs/research-log.md` contains the accumulated experiments.

The untracked `artifacts/train_verification_audit.json` is the saved audit, `artifacts/verification_research.py` is its replay runner, and `tests/test_execute.py` contains the exit, output, and cache regressions. The pre-existing demo work consists of `server.py` for the FastAPI entry point, `src/service.py` for its search implementation, and `web/app.js`, `web/index.html`, and `web/style.css` for client behavior, markup, and styling. Only the verification call and cache in `src/service.py` relate to the previous comparison change. The demo's unrelated work has been preserved.

| Fresh replay | Exact with exit check | Scoped trim with exit check |
|---|---:|---:|
| Audit gold passes | 389 / 511 (76.13%) | 392 / 511 (76.71%) |
| Audit wrong passes | 2 / 2,555 (0.0783%) | 2 / 2,555 (0.0783%) |
| Fixed 100-query top-50 gold passes | 28 / 32 (87.50%) | 28 / 32 (87.50%) |
| Fixed 100-query top-50 non-gold passes | 1 / 4,968 (0.0201%) | 1 / 4,968 (0.0201%) |
| Fixed 100-query NDCG@10 / MRR@10 | 28.63 / 28.50 | 28.63 / 28.50 |
| All 16 eligible queries, top-50 gold passes | 2 / 3 (66.67%) | 3 / 3 (100%) |
| All 16 eligible queries, top-50 non-gold passes | 3 / 797 (0.3764%) | 3 / 797 (0.3764%) |
| All 16 eligible queries, NDCG@10 / MRR@10 | 10.19 / 9.38 | 16.44 / 15.63 |

The same three gold pairs gained passes as in the earlier 390-to-393 trial, and `q118 / d118` again moved from rank 17 to rank 1 in the eligible cohort. No non-gold pass or gold rank worsened. The exact baseline's 389 passes were three below the saved audit. The missing IDs were `q50`, `q2280`, and `q2346`, while `q2010` passed. Both additional snippets perform large precomputations and passed six isolated attempts each. In a separate diagnostic after the scored replay, competing CPU work caused all three one-second attempts for each snippet to time out without stdout. The full-panel traces for those two misses were not captured, so that diagnostic demonstrates timing sensitivity without assigning a definite cause to the individual misses.

Retain the scoped change with the tightened layout guard. The gain reproduced on identical paired observations, both ranking cohorts reproduced the earlier scores, and wrong-pass counts were unchanged. The 16-query score change comes from a single gold promotion. The live checkpoint smoke test also passed with exact and scoped cache entries kept separate. Apart from the guard, its test cases, and this log, the worktree files and cached artifacts were preserved. No test-split run, commit, or push was performed.

### 2026-11-24 Verification completion and integration

**Setup.** The baseline was captured before changing the verifier. The [replay script](../artifacts/verification_completion.py) uses the cached CoIR APPS train partition at `f22508f96b7a36c2415181ed8bb76f76e04ae2d5`. It materializes only training queries, snippets, and relevance labels. The audit uses all 511 parseable queries, the exact gold IDs and five unrelated candidates per query from the saved ledger, and seeds 20261124 and 20261125. The ranking arm uses the first 40 draws of `random.Random(20261129).sample(sorted(parseable_ids), 40)`. The ordered query ID SHA-256 is `4786e3867cede3566ca454e12ae67d6830cc989f2743551b03742ed603c65b2d`. Rank over all 5,000 training documents using E5-base-v2 on CPU, four embedding threads, batches of eight queries, the existing prefixes and 512-token limit, normalized vectors, and stable document-ID tie order. Keep the fixed top 50 and preserve cosine order within the promoted and unpromoted groups. The rank cohort includes 27 queries with gold outside the top 50. The ranking scores average over all 40 queries on a 0–100 scale.

Both execution arms used Windows Python 3.13.13 in UTF-8 mode, one second per candidate and two workers. Candidate verdict caches were cold and bypassed in both captures. Each exact versus scoped comparison uses its capture's same process observations. The initial and final full captures are fresh independent executions, so compare their changed pair IDs and process outcomes as well as their rates. Digests of the ordered query and candidate lists are `25e562c6f34a52befce120e747ff5bbd16f6943183070e9386e528f902ef9f67` for the audit and `698507b8dc25e36e8198fcfe196f6f8eac1e6bd574aa41890da13f0126721710` for ranking. The inherited collector keyed traces by source and coalesced evidence for the same pair of byte-identical document IDs in six ranking queries. The actual exact verdicts were independent. A follow-up with the per-candidate collector reran both documents on all six queries and confirmed the same phases and zero promotions. The new script preserves separate evidence for such candidates.

| Measurement | Initial accepted verifier | Final cold run |
|---|---:|---:|
| Parsed training queries | 511 / 5,000 (10.22%) | 511 / 5,000 (10.22%) |
| Audit gold passes, exact | 391 / 511 (76.52%) | 389 / 511 (76.13%) |
| Audit gold passes, scoped | 394 / 511 (77.10%) | 392 / 511 (76.71%) |
| Audit gold passes among all training queries, scoped | 394 / 5,000 (7.88%) | 392 / 5,000 (7.84%) |
| Audit wrong passes, scoped | 2 / 2,555 (0.0783%) | 2 / 2,555 (0.0783%) |
| Top-50 gold passes, scoped | 12 / 13 (92.31%) | 12 / 13 (92.31%) |
| Top-50 non-gold passes, scoped | 1 / 1,987 (0.0503%) | 1 / 1,987 (0.0503%) |
| Top-50 NDCG@10 and MRR@10 | 30.00 / 30.00 | 30.00 / 30.00 |
| Audit candidate attempts | 3,066 | 3,066 |
| Audit nonzero exits, timeouts | 979, 11 | 979, 8 |
| Top-50 candidate attempts | 2,000 | 2,000 |
| Top-50 nonzero exits, timeouts | 1,354, 25 | 1,355, 24 |
| Audit query median / p95 wall time | 0.315 s / 0.690 s | 0.296 s / 0.598 s |
| Top-50 query median / p95 wall time | 2.880 s / 4.178 s | 2.938 s / 4.135 s |
| Audit full captured wall time | 4,240.16 s | 174.07 s |
| Top-50 full captured wall time | 127.24 s | 120.20 s |

Cosine-only NDCG@10 and MRR@10 were 13.38 and 11.29. Exact and scoped output produced identical top-50 gold ranks and promoted the same non-gold pair, `q2153 / d2083`, in both runs. The audit's two wrong pairs were again `q2150 / d2108` and `q69 / d1491`. Scoped output recovered the same three audit gold pairs, `q118`, `q1276`, and `q2176`. There were 22 nonzero gold exits in each audit run and no observed allocation or launch failure. Two baseline query timings, `q48` and `q2205`, contain apparent wall-clock stalls of 2,278 s and 1,773 s. Those rows and their timeouts remain in the totals. Re-running both batches took 0.313 s and 0.553 s. The initial stall source was unobserved, and the full totals cannot establish a throughput improvement.

The new script independently reproduced both fixed candidate digests and the final 392 audit gold passes, two random wrong passes, 12 top-50 gold passes, one top-50 non-gold pass, and both 30.00 ranking scores. This run had nine audit timeouts and 24 ranking timeouts. Median / p95 wall times per query were 0.273 s / 0.456 s for the audit and 2.902 s / 3.695 s for ranking. Its two gold differences against the initial verifier were timeouts in `q2280` (1.0225 s) and `q2312` (1.0172 s). The aggregate was stable across these two final runs even though the successful pairs differed.

The ranking collision is a valid first example with an unrelated solution. `q2153` asks how to move pieces of chocolate and expects `2` for `3\n1 0 1`. The biscuit-game snippet `d2083` also prints `2`, has empty stderr and exits 0. On the second published input `1\n1` it prints `0` and exits 0, while the correct answer is `-1` because one piece cannot yield any positive common divisor above one. Gold-labeled `d2153` fails to compile due to a top-level `return`. It ranked 664 by cosine. Requiring two examples removes this particular non-gold promotion and changes neither ranking metric. The earlier training-wide study still does not support requiring a second example by default.

**Gold-count discrepancy and execution stability.** All four initial/final verdict changes are accounted for. `q2346` and `q2312` exited successfully near the limit initially (0.9802 s and 0.9366 s), then timed out at 1.0481 s and 1.0135 s. `q2312` had already printed the complete expected sample. `q50` initially timed out at 1.0224 s after printing the expected answer, then exited successfully in 0.9980 s. `q2010` exited successfully both times with `1 1 -1` and `-1 1 1`, respectively. Both satisfy the statement. Earlier exact counts of 392, 390 and 389 and the fresh exact 391 are separate observations with the same sample. The new exact 389 includes two gold timeouts and the non-canonical construction.

To isolate the cleanup change, ran the original cleanup and the new cleanup eight times each per changed query, interleaved using seed 20261130, one subprocess at a time and the same one-second limit. For `q2346` and `q2312`, both implementations went 8/8. For `q2010`, the original and new implementations matched 5/8 and 4/8 and both produced the same two legal outputs. For `q50`, the original matched 0/8 and the new matched 2/8, with the other attempts timing out. These small trials show wall-time sensitivity and stochastic output. They do not establish that cleanup makes execution faster. On a separate twelve-candidate print-only control, median batch wall time over three trials was 0.741 s, 0.402 s, and 0.245 s at one, two, and four workers. That measures interpreter startup and scheduling under light load. Earlier competing CPU runs reproduced timeouts in `q2280` and `q2346`. Worker defaults were left unchanged.

**Timeout retry experiment.** Executed every timeout from the original audit and top-50 captures once more, in the original query order, with a one-second and separately a two-second retry budget. Eligibility depended only on the captured timeout. No blind increase or earlier-success replacement was used.

| Retry arm | Audit additional matches | Audit extra wall time | Top-50 additional matches | Top-50 extra wall time |
|---|---:|---:|---:|---:|
| 1 s, 11 audit and 25 ranking candidates | 0 | 7.79 s | 0 | 22.86 s |
| 2 s, 11 audit and 25 ranking candidates | 1 gold, 0 wrong | 11.16 s | 0 | 30.70 s |

The 1 s audit retry would keep 394 / 511 gold and 2 / 2,555 wrong. The 2 s retry would reach 395 / 511 (77.30%) with the same wrong count because `q50` passed. Top-50 promotions and both ranking scores stay unchanged in both arms. At 2 s the audit statuses were nine zero exits, one nonzero exit, and one repeat timeout. Ranking statuses were one zero exit, 21 nonzero exits, and three repeat timeouts. The benefit is one timing-sensitive gold pair in this audit and no measured ranking gain at material cost. Reject both retries as the default. These selected retries do not measure a blanket timeout increase.

**Independently validated changes.**

1. **Candidate temp-directory lifecycle.** Injecting a temporary-directory allocation failure previously raised from the worker and aborted all candidate verdicts. The new cleanup gives `[False, True]` for a run with the first allocation failing and the second candidate printing the expected output. A snippet that writes additional files previously left its working directory behind. A managed temporary directory removes those files on success and timeout. The same subprocess command, stdin, exit-code requirement, timeout, comparison, and worker scheduling are used. No extra launches. The full replay and interleaved checks above support retention.
2. **Live JSONL checkpoint.** On fresh training `q2111 / d1437`, the candidate printed `2\n`, raised `EOFError` and exited 1. A legacy checkpoint keyed only by query ID and the padding flag nevertheless boosted it. A truncated JSONL line separately stopped search before any verification. The live key now includes query ID, parsed input and expected stdout, the padding policy, timeout, worker count, verifier source digest, Python and encoding context, and the ordered candidate IDs and source. Unknown and malformed records are ignored. The first new write delimits an interrupted tail, and verified records resume on a later search. Existing checkpoints and frozen results were left in place. On a replay of all 40 fixed candidate lists using captured verdicts, cold and warm ranks and scores were identical over five repetitions. Each cold replay had 40 misses covering 2,000 candidates. Each warm replay had 40 hits covering the same candidates and invoked no verifier. Median checkpoint/replay overhead was 39.73 ms cold and 29.05 ms warm for all 40 queries, excluding embeddings and subprocesses. A separate real subprocess regression confirms the legacy crash cannot promote a candidate and that changing source, input, expected output, timeout, policy, or K invalidates the older key.
3. **Version-history example key.** The input–output pairs `("a\\x00b", "c")` and `("a", "b\\x00c")` serialized identically with the old delimiter. The fresh verdicts were pass and fail, but the old cache returned pass for the second. Keeping the parsed fields separate corrected that mismatch. The version fixture reuses an identical source and example across versions and misses when either source or example changes. This is a tested keying edge case with no NUL pairs in the measured APPS panel.
4. **Gradio source identity.** Replacing source in the same slot formerly reused that slot's verdict. The smoke test verifies exact versus scoped comparison with the same example and confirms that a changed source printing the expected output before a crash fails instead of acquiring the old pass. A subsequent identical search reuses both slot verdicts and yields the same rank order. Stable candidate sets carry no extra subprocesses. The pre-existing FastAPI service loads its pool at startup, has its own process-local index cache, and was explicitly protected from edits. An in-process reload without clearing it can still reuse a slot verdict. A process restart clears it.
5. **Evaluation modes.** The archived replay records do not have successful-exit provenance. The evaluation CLI now requires an explicit mode and labels a historical replay. MTEB's default result cache uses the model metadata and can return old aggregate scores before calling live search. Full mode explicitly disables that outer result cache and retains the live candidate checkpoint. A mocked CLI integration intercepted the evaluator before any dataset load and verified the live cache setting. A no-mode invocation stops in argument parsing. No evaluation command loaded the test split.

The live integration test exercises the real MTEB wrapper's indexing and search with a two-document in-memory corpus and deterministic encoder. A separate training smoke test ran `q2111` through the protected web service with injected training rows. Both cold and warm requests ranked `d2111` first and marked `d1437` failed. The second request reused both verdicts. The version-history smoke test used the same training rows and an added crash at the end of the gold source. Only the unmodified gold version passed and repeated checks reused all three verdicts.

**Limits of stronger comparisons.** The first-50 audit attributed 27 cases to valid alternative outputs, 11 to trailing padding, seven to invalid gold source, two to floating-point formatting, one to expected-output prose (`q1289`), one to a no-sample placeholder (`q1696`), and one to a mismatched reference (`q2387`). These are counts in a failure sample, and 21 gold sources in the wider saved run already had syntax errors. Existing parser boundaries also pass a truncation marker to `q1694`. A Windows subprocess decoding warning repeated in the replay. `q1696`, the no-sample placeholder, was the only parsed training pair with non-ASCII sample I/O. There was no evidence to justify changing child output encoding, fabricating missing fixtures, or applying generic numeric tolerance or construction heuristics. Numeric formatting, mixed output, fixed widths, ASCII art, ambiguous declaration text and meaningful internal line layout keep the existing exact policy. The original parser and executor already strip whitespace at the outer sample/stdout boundary. Exact comparison preserves the remaining characters. The lexical padding rule cannot substitute for a problem checker, and no tests have asserted support for semantically significant outer padding. The previous parser expansion and multiple-example gate remain rejected for the measured false-positive increase and lost correct promotions, respectively. Coincidental single-example matches remain possible even when runtime and output are valid.

**Reproduction.** Run the following with disposable paths. A fresh replay can vary around the wall-time limit and for randomized output.

```powershell
.venv/Scripts/python.exe -B -X utf8 artifacts/verification_completion.py plan PATH_TO_NEW_PLAN
.venv/Scripts/python.exe -B -X utf8 artifacts/verification_completion.py capture PATH_TO_PLAN NEW_AUDIT_JSONL NEW_RANK_JSONL
.venv/Scripts/python.exe -B -X utf8 artifacts/verification_completion.py summary AUDIT_JSONL RANK_JSONL
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

All 21 tests passed. The first temp-directory test run exposed an error in the test double's handling of the allocator's positional arguments. Correcting the double resolved it without another production change. Coverage includes successful exit, stdout before nonzero exit or timeout, launch and allocation errors, empty stdout, scoped integer rows, trailing spaces and tabs, leading and interior spacing, blank lines, floats, mixed and layout-sensitive output, checkpoint recovery, cache invalidation, and version/UI parity. The test suite does not start the demo or load assessment data. Retain the execution lifecycle and cache fixes. Gold remained above 73%, measured wrong counts and ranking scores did not regress, and the changed gold verdicts trace to the subprocess observations. Timing instabilities and memoized transient results remain limitations. No dependency, embedding, retrieval, or ranking changes were made during this validation.

### 2026-11-24 Final verification review

Gradio skips the verifier call when every verdict is already cached. The regression test checks that a fully warm search returns the same order without calling the verifier. Version history constructs each candidate's existing cache key once per verification request. The replay script uses the observation dictionaries already supplied by its collector. The subprocess runner, example parser, output comparison, and ranking rules were untouched in this cleanup. The review found no unused Python imports in the verification files. The earlier colon-label probe is part of the research record.

All 21 regression tests passed. The fresh replay used the same pinned training revision, sampling, and process settings as the completion benchmark. Parsing covered 511 of 5,000 training queries (10.22%). The cold audit attempted 3,066 candidates and passed 394 of 511 gold snippets (77.10%) with scoped padding and 391 (76.52%) with exact output. It passed the same two wrong candidates out of 2,555 (0.0783%). The previous 392-pass independent replay timed out on `q2280` and `q2312`. Both produced the expected output and exited successfully this time. There were 978 nonzero exits and eight timeouts in the fresh audit. The Windows decoding warning recurred.

Both ordered candidate digests matched the completion benchmark. On the fixed 40-query ranking panel, 12 of 13 gold candidates (92.31%) and one of 1,987 non-gold candidates (0.0503%) passed. NDCG@10 and MRR@10 each remained 30.00. The 2,000 ranking attempts included 1,356 nonzero exits and 23 timeouts. Median query wall times were 0.268 s for the audit and 2.805 s for the ranking panel.

Replayed the same captured verdicts through the live checkpoint. A fresh checkpoint called the verifier for 40 queries covering 2,000 candidates. Reloading it called the verifier zero times and returned the same scores and candidate order for every query. The separate two-candidate integration test exercised cold and warm paths with real subprocess execution. Scratch replay files were removed after inspection.

The checkpoints and process-local verdict caches store pass flags. A candidate that times out can therefore remain a cached non-pass even if it would finish under lighter load. Fresh execution can also produce a different valid construction. The FastAPI service keys verdicts by candidate slot, sample and comparison policy. Its `load()` replaces the candidate pool without clearing the cache, so an in-process reload can reuse a stale verdict if a slot gets a different source. A process restart clears that cache. Both limitations were documented without a cache-policy change.

### 2026-11-24 Verifier audit and execution-performance reference

**Scope.** New executions use only the 5,000 training queries and snippets from the local CoIR APPS snapshot at `f22508f96b7a36c2415181ed8bb76f76e04ae2d5`. The original production verifier is the version at `0586e97`. The local lightweight tag `verifier-ref` points to `9a926ec680bda776f5a2cfd0a16a0620d411bcdf` and resolves to that commit. The staged deletions of `Problem Statement.pdf` and `src/rerank.py` and the unstaged `.gitignore` and `README.md` edits were already present.

**Archived scores and cached mode.** `evaluate.py --mode cached` constructs E5 and loads historical passers from `exec_results.jsonl`, mapping positional records through `doc_ids.npy`. It found 924 queries with nonempty passer lists. The wrapper can rerank semantic retrieval results without candidate execution. MTEB's default `only-missing` policy first checks the aggregate cache for `prism/e5-base-v2-exec-rerank`, revision `1`, task `AppsRetrieval`. On a complete hit it returns the cached result before evaluating the split. A miss could still load the test split and encode uncached text. The normal CLI unconditionally writes `artifacts/appsretrieval_results.json`.

A guarded invocation of the real CLI main and real MTEB cache path replaced E5 construction with a minimal stand-in, trapped evaluation, task data loading, search and encoding, and redirected the output to a disposable path. It returned **19.96 NDCG@10 and 19.05 MRR@10**, with raw cache values 0.19956 and 0.190474. The frozen JSON carries 0.19956 and 0.1904737663104197. This reproduces the reported numbers from the aggregate cache. The archived execution verdicts alone do not specify semantic ranks, and the original notebook's `base_ranking.npz` is absent from the checked-in artifacts. This is not an independent reranking recomputation. The frozen JSON, published result JSON, execution ledger, and document-ID file were preserved.

**Separate web implementation.** The prior FastAPI runner (`server.py`), service module (`src/service.py`), and browser client (`web/app.js`, `web/index.html`, `web/style.css`) were previously untracked. All five are absent from the current worktree and Git index, and their paths have no commits in local refs. No current tracked entry point imports or launches them. Gradio's `demo.py` has its own retrieval and verification implementation. The old service's slot-keyed cache limitation remains historical.

**Rejected experiments and controls.** Metrics are NDCG@10 / MRR@10 on a 0–100 scale. Non-gold and gold counts below use each study's declared population.

| Experiment and population | Verification, before → after | Ranking effect and decision |
|---|---|---|
| Colon-label parser, 5,000 train queries and fixed 25,000 random pairs | Coverage 511 → 845. Gold 391 → 667 on the fixed denominator, wrong 3 → 5. Executed-only gold 391/511 → 667/845 and wrong 3/2,555 → 5/4,225 use different denominators. Identical-example subset 3/2,540 → 3/2,540 wrong. | Ranking unmeasured. Two new clean-exit collisions violate the fixed-panel no-increase criterion. Earlier 50-case narrow colon probe found 38 gold passes and had no wrong or ranking arm. |
| First two or all published examples, 240 stratified train queries, 12,000 candidates | Gold 40/111 → 17/111, non-gold 13/11,889 → 0/11,889, and inconclusive queries 80 → 160 in either arm. Gold census on 845 parser-expanded queries had 662 first passes and 184 all-case passes, with 472 first passes lacking a second pair and six valid alternative outputs failing exact comparison. | 35.60 / 33.70 → 30.34 / 28.25 in both arms. Weighted 49.89 / 46.64 → 47.96 / 44.63. Lost gold promotions and poorer ranking. The 845-row census includes expanded parsing and cannot be compared directly with the 511-row audit. |
| Independently validated multi-example gate, same 240 queries, fixtures for 10 queries and 22 cases | Gold 40/111 → 4/111, non-gold 13/11,889 → 0/11,889, and inconclusive queries 80 → 230. | 35.60 / 33.70 → 27.48 / 25.27, weighted 49.89 / 46.64 → 47.54 / 44.20. Coverage too sparse. The vetted subset is a convenience sample. |
| Unconditional per-line ASCII space/tab trim, 511 gold and 2,555 random wrong pairs, 100 top-50 queries | One paired run recovered 16 gold, 392 → 408, with wrong 2 → 2. Another recovered 16, 390 → 406, with wrong 2 → 2. | First run 28.63 / 28.50 → 29.63 / 29.50, with top-50 gold passes 28 → 29 and non-gold 1 → 1. No separately tabulated unrestricted ranking arm for the second audit count. Synthetic fixed-width rows demonstrated a real false acceptance outside the measured candidates. Scoped trimming is retained. |
| Wrap syntactically invalid candidates in a function, seven diagnosed failures | Exact passes 0/7 → 6/7. The seventh wrapped program produced a valid alternative whose stdout still differed from the sample. | No wrong-pair or ranking arm. The source transformation was diagnostic and would change which program executes. It remains an ingestion question. |
| Retry all initial timeouts, 11 audit candidates and 25 top-50 candidates, independent 1 s and 2 s arms | No extra matches at 1 s. At 2 s audit gold 394/511 → 395/511, wrong 2/2,555 unchanged, and no top-50 promotion changes at either budget. | Both arms 30.00 / 30.00. Added audit/ranking wall time 7.79/22.86 s at 1 s and 11.16/30.70 s at 2 s. Little benefit for the additional cost. A blanket timeout increase was not tested. |
| Early exit on first passing snippet, archived initial study | Wrong candidate passed before gold in 7/15 top-200 cases (47%). | No comparable paired ranking score for the early-exit rule. Historical study, outside the new training audit. |
| Worker oversubscription, archived two-core environment | 32 workers on two cores caused timeouts. The archived summary gives no fixed-population before/after counts or ranking comparison. | Not comparable to the current 12-logical-CPU machine. The 12-candidate, 1/2/4-worker light-load control from the later study was startup-only and cannot establish ranking or timeout effects at K=50. |

Other proposals did not reach a measured promotion experiment. Global tokenization loses layout by construction. Generic numeric tolerance was not tried as a ranking arm. Two of the 50 diagnosed gold errors involved numeric formatting or explicit tolerance and do not justify a blanket rule. Scalar-output rejection and input-consumption checks had case-level counterexamples only. Upstream independently curated fixture coverage was unmeasured, and gold-generated answers would leak relevance labels. No before/after ranking figures exist for these ideas.

**Original versus current verifier.** Reconstructed and matched all candidate IDs to the saved ledger, including the five wrong candidates for each of 511 gold snippets. Confirmed both parsers return identical samples across all 5,000 training queries. Ran each of the 3,066 pairs exactly once with the current runner, two workers, and a one-second timeout under Windows Python 3.13.13 in UTF-8 mode. Both rules evaluated the same captured stdout, exit status, and timeout. The comparison calls the original `verify` from Git with the pre-fix stdout contract and the current `verify` with the accepted exit and padding policy. The temporary-directory lifecycle was common to the observation run. A second execution with the historical directory cleanup was unnecessary for the policy comparison and could change timeout and randomized output observations. Neither arm used a verdict cache. [Candidate-level paired decisions](../artifacts/verification_original_current_train.json)

| Training audit | Original | Current | Difference |
|---|---:|---:|---:|
| Gold | 390/511 (76.3209%) | 393/511 (76.9080%) | +3 (+0.5871 pp) |
| Wrong | 3/2,555 (0.1174%) | 2/2,555 (0.0783%) | −1 (−0.0391 pp) |

Scoped padding recovered `q118/d118`, `q1276/d1276` and `q2176/d2176`. The successful-exit rule rejected the matching stdout of `q2111/d1437`, which exited 1 with `EOFError`. `q2150/d2108` and `q69/d1491` passed under both policies. The fresh round contained two gold timeouts (`q50` and `q2312`), seven wrong timeouts, 22 gold nonzero exits and 955 wrong nonzero exits. These two gold timeouts explain the difference from the saved original run's 392 gold passes. The randomized `q2010` matched the canonical string in this round. Prior measured variation in its output and in cutoff-sensitive runs affects fresh-run counts, and cannot affect the paired comparisons. The same Windows text-decoding warning recurred.

**The existing 40 ranking queries.** Recreated the ordered query digest `4786e3867cede3566ca454e12ae67d6830cc989f2743551b03742ed603c65b2d` and candidate digest `698507b8dc25e36e8198fcfe196f6f8eac1e6bd574aa41890da13f0126721710`. Replayed 2,000 candidates with two workers and a one-second timeout. Gold ranks match the previous 40-query experiment. Cosine alone had three golds at rank 1, one at 2, two at 3, one at 4, one at 10, five at 11–50, and 27 outside 50 (NDCG@10 13.38, MRR@10 11.29). Both exact and scoped execution moved the same candidates and yielded the following counts. [40-query evidence](../artifacts/verification_rank40_train.jsonl)

| Gold rank after execution | Queries |
|---|---:|
| 1 | 12 |
| 2–10 | 0 |
| 11–50 | 1 (`q2392`, rank 18) |
| Outside 50 | 27 |

There is one gold per query. At rank 1, its ideal-normalized DCG and reciprocal rank both equal 1. Each of the 28 queries outside the top 10 contributes zero to both metrics. Thus both means equal exactly `12 / 40 × 100 = 30.00`. The 27 unrerankable golds were `q105`, `q2153`, `q47`, `q83`, `q2138`, `q2383`, `q35`, `q2015`, `q2033`, `q2261`, `q1276`, `q2008`, `q115`, `q2129`, `q2206`, `q13`, `q63`, `q2280`, `q96`, `q2255`, `q100`, `q2256`, `q2391`, `q71`, `q2057`, `q2252`, and `q2161`. Together with `q2392` these are all 28 top-10 misses. The 12 gold passes were 12/13 among retrieved golds. The non-gold match `q2153/d2083` was 1/1,987 and could not change either metric because its gold was rank 664. The replay had 22 timeouts and 1,357 nonzero exits.

**Latency measurement protocol.** Selected 200 of the 511 parseable training queries without replacement using `random.Random(20261129).sample(sorted(parseable_ids), 200)`. This is a stage-2 workload. A representative random sample of all 5,000 would mostly skip subprocess execution. The 40-query ranking study uses its own size-40 draw with the same seed. Python's sampling algorithm can use different strategies at these sizes, so the two panels were planned separately. Use E5-base-v2 at local model snapshot `f52bf8ec8c7124536f0efb74aca902b2995e5bcd` and cached document embeddings for all 5,000 training snippets, CPU query embeddings, stable document-ID tie order, and K=50. Ordered-query SHA-256 for the 200 panel is `bd29b441a1159a43ccdd2c05933017c0c0cf194f19f331b6caa98bc131f7264a`. Ordered query and candidate SHA-256 is `6f1f11b7d5f286074c68c817ed9a06837d0137c159a9b8420937ce11e80026fa`, using `json.dumps([(qid, ordered_candidate_ids), ...])`.

The machine is Windows 11 build 26300 with a 13th Gen Intel Core i5-13420H. Windows reports eight physical cores (four performance and four efficiency), and Python reports 12 system and 12 process-available logical CPUs. The execution run uses Python 3.13.13 with `-X utf8`, the production default of 12 workers, and the one-second per-candidate timeout. The earlier policy and ranking audits use two workers. Model loading, dataset reads, embedding, and cosine search are complete before wall-clock timing. The harness supplies precomputed candidate order to the live wrapper and replaces its similarity values with positive descending scores of equal order. Both modes time the same real checkpoint path and reranker. Cold starts from an empty disposable checkpoint. Warm reopens the resulting checkpoint with an unchanged plan and no subprocesses. The batch wall timer includes checkpoint read/open and request-level setup. Per-query timers cover sample parsing, candidate key lookup, verification, checkpoint append and flush on a miss, and promotion. They exclude one-time checkpoint loading and setup. P95 is the nearest-rank 95th percentile. The production persistent cache and frozen artifacts are not used or rewritten.

**Reproduction.** Run with new output paths. The final performance JSON stores the ordered query and candidate IDs, gold ranks, both cold passer lists and all query timings. Candidates absent from a passer list failed verification. The warm verdicts equal the immediately preceding cold verdicts and are checked against all returned ranking scores.

```powershell
.venv/Scripts/python.exe -B -X utf8 artifacts/verification_performance.py audit NEW_AUDIT_JSON
.venv/Scripts/python.exe -B -X utf8 artifacts/verification_completion.py plan NEW_RANK40_PLAN
.venv/Scripts/python.exe -B -X utf8 artifacts/verification_performance.py rank NEW_RANK40_JSONL --plan NEW_RANK40_PLAN
.venv/Scripts/python.exe -B -X utf8 artifacts/verification_performance.py plan NEW_PERFORMANCE_PLAN
.venv/Scripts/python.exe -B -X utf8 artifacts/verification_performance.py performance NEW_PERFORMANCE_JSON --plan NEW_PERFORMANCE_PLAN --repeats 2
```

Fresh runs can vary with host load and randomized candidates. Warm timing covers a 200-query checkpoint at this size, so it should not be extrapolated to a much larger accumulated checkpoint.

**Measured stage-2 latency.** Per-query values are seconds on cold runs and milliseconds on warm runs. Total wall time is in seconds. All 200 queries were misses on each cold run and hits on each warm run. [Exact workload and verdicts](../artifacts/verification_performance_train.json)

| Run | Median per query | P95 per query | Mean per query | Total wall | Subprocesses | Timeouts |
|---|---:|---:|---:|---:|---:|---:|
| Cold 1 | 1.202 s | 1.740 s | 1.228 s | 245.644 s | 10,000 | 164 |
| Cold 2 | 1.220 s | 1.772 s | 1.239 s | 247.887 s | 10,000 | 166 |
| Warm 1 | 0.296 ms | 0.633 ms | 0.337 ms | 0.0691 s | 0 | 0 |
| Warm 2 | 0.269 ms | 0.573 ms | 0.321 ms | 0.0660 s | 0 | 0 |

Cold wall time varied by 2.243 s (0.91%). The two runs recorded 7,032 and 7,031 nonzero exits and zero caught launch errors. Windows emitted a subprocess reader decoding warning during both cold runs. The same 50 of 59 retrievable golds and nine of 9,941 non-golds passed in both runs, despite the two-count difference in timeouts. Each warm replay returned score dictionaries identical to its cold run. A cached timeout is still a non-pass and can survive a change in host load. No optimization was applied. All 21 regression tests passed after the timed runs. The aggregate frozen scores were read without fresh test-set evaluation, and every newly benchmarked subprocess used training inputs.
