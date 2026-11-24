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

### 2026-11-24 Execution optimization study

All cold measurements in this study use the saved 200 training queries, their original ordered top 50, a fresh disposable live checkpoint, the one-second candidate timeout, and the existing candidate interpreter. The existing stage-1 scores are represented by scores in the same precomputed order, as in the saved latency baseline. No model or data preparation is timed. Compare the complete passer set for every query against both identical saved cold verdict lists. Treat timeouts separately from nonzero exits. The Windows subprocess reader can issue a decoding diagnostic even when the parent receives a nonzero return code. Such diagnostics were present in the reference run.

**Profile before changing production.** Timed the existing runner on 50 no-op candidates and the first 12 fixed queries (600 candidates) at the original 12 workers. Wrapped the standard-library process and temporary-directory calls without replacing their behavior. The worker-level medians below overlap in wall time because workers execute in parallel. The communicate interval includes interpreter startup, child work, pipes, and wait. The no-op control gives a lower bound for the normally provisioned interpreter. The profiler did not instrument individual candidate code or change stdin.

| Phase | No-op candidate median | Training candidate median |
|---|---:|---:|
| Create working directory | 1.04 ms | 0.83 ms |
| Open source file, excluding its write | 0.83 ms | 0.66 ms |
| Construct and start subprocess | 19.81 ms | 16.30 ms |
| Communicate and wait | 81.61 ms | 134.72 ms |
| Directory cleanup | 1.98 ms | 1.73 ms |
| Entire candidate runner | 108.48 ms | 159.38 ms |

The no-op batch took 0.514 s. The 12 real batches took 11.754 s. There were 606 communicate calls for 600 real candidates because six timeout paths called communicate again after termination. Per-query aggregation and checkpoints are bounded by the earlier 0.32–0.34 ms mean warm run, including query parsing and reranking. Filesystem setup is small relative to process launch and wait. Work inside communicate remains a mix of interpreter and candidate work, not isolated candidate CPU time. [Raw phase profile](../artifacts/verification_optimization_profile.json)

**Syntax opportunity.** Compiling the UTF-8 source bytes without executing them found 365 syntax-invalid candidate occurrences (3.65%) across 49 of 1,469 distinct source texts. Compiling all 10,000 occurrences took 5.315 s. Compiling each distinct source once took 0.891 s. This bounds the launch savings and includes syntax errors that an AST-only check could miss. A separate experiment measures actual cold wall time rather than extrapolating from the invalid count. The in-process compiler is a research classifier. It is not an isolation mechanism or a substitute for child compilation. Production adoption would also need to address arbitrary parser resource use, environment-dependent startup behavior, warnings, and source-encoding semantics.

**Cache inventory.** The live checkpoint key hashes source, both sample fields, ordered candidates, policy, verifier source, timeout, worker count, interpreter identity, and selected encoding/hash environment variables. Exact and scoped output verdicts remain separate. The checkpoint already avoids every launch on an unchanged warm batch. Gradio and version history have separate process-local verdict maps. The embedding cache is stage 1 and was excluded. The current live format stores pass lists and does not distinguish a wrong answer from a transient timeout in a reusable negative result. Arbitrary inherited environment settings and mutable external dependencies are outside this identity. Extending cache reuse across candidates or conditions would change semantics, and work on lookup overhead would not achieve a cold speedup.

**Worker control methodology.** Changed only the worker count seen by the existing live wrapper in the research process. Each arm reuses saved candidate IDs, source, sample, and policy, creates an empty checkpoint for its cold run, then repeats from that checkpoint for a warm consistency check. Ran the arms sequentially without model work in parallel. Query latency includes the live verification and reranking path. P95 uses the same nearest-rank statistic as the reference. Shell process time and general host memory are not candidate-process CPU or peak memory. Win32 system counters sampled every three seconds are reported only as coarse host activity.

| Cold arm | Median/query | Mean/query | P95/query | Total wall | Pairs/s | Launches | Timeouts | Nonzero exits | Verdict differences |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Saved 12, run 1 | 1.202 s | 1.228 s | 1.740 s | 245.644 s | 40.71 | 10,000 | 164 | 7,032 | 0 |
| Saved 12, run 2 | 1.220 s | 1.239 s | 1.772 s | 247.887 s | 40.34 | 10,000 | 166 | 7,031 | 0 |
| 6 workers | 1.340 s | 1.385 s | 1.964 s | 276.958 s | 36.11 | 10,000 | 152 | 7,039 | 0 |
| 8 workers | 1.192 s | 1.228 s | 1.806 s | 245.601 s | 40.72 | 10,000 | 156 | 7,036 | 0 |
| 4 workers | 1.529 s | 1.540 s | 2.191 s | 308.011 s | 32.47 | 10,000 | 135 | 7,048 | 0 |
| 2 workers | 2.369 s | 2.443 s | 3.187 s | 488.648 s | 20.46 | 10,000 | 60 | 7,112 | 0 |
| Fresh 12 control | 1.207 s | 1.329 s | 1.736 s | 265.776 s | 37.63 | 10,000 | 169 | 7,031 | 0 |
| 12 workers with compile precheck | 1.195 s | 1.230 s | 1.725 s | 246.003 s | 40.65 | 9,635 | 162 | 6,670 | 0 |

All arms used Windows Python 3.13.13 and a one-second candidate timeout. No launch errors or other caught subprocess exceptions were counted. The recurring Windows reader-thread encoding diagnostic is separate from these counts. Every arm preserved the 50 gold and nine non-gold passes and **25.6267 NDCG@10 / 25.3714 MRR@10**. Each warm replay produced the identical scores, used 200 checkpoint hits and zero subprocesses, and took 52–149 ms in total. Cold has 200 misses in every arm. [Worker sweep, per-query timings and verdicts](../artifacts/verification_optimization_worker_sweep.jsonl)

Reducing concurrency traded fewer timeouts for longer batches. Some former timeouts complete with nonzero exit codes, so a fall in timeouts does not imply an equal reduction in candidate failures. The 8-worker result was within the earlier 0.91% wall-time variation and had worse P95. The fresh 12-worker control includes `q113` at 21.935 s, compared with 1.914 s and 1.773 s in the two saved runs. Its other queries' combined time and its median/P95 were close to the reference. The source of this isolated stall was not observed. The full 265.776 s total is retained. Use a matched repeat to check the small 8-worker difference.

The syntax probe skips 365 processes and preserves all verdicts on this training panel. The total of 246.003 s is within ordinary variation of the two saved controls, so the outlier-contaminated fresh control alone cannot establish a repeatable gain. Skipped syntax failures move out of the subprocess failure count. Under this arm the precheck runs in the research parent, outside the candidate timeout and subprocess boundary. That is unsuitable for arbitrary source in production without independent bounds. No production syntax screening was installed.

Coarse host busy CPU was 73.2% for a partial 6-worker interval, 74.0% at 8, 50.1% at 4, 27.7% at 2, 74.8% at 12, and 77.2% for the syntax arm. Sampled peak physical memory in use across the machine was about 13.9–15.3 GiB. These include the surrounding system and cannot identify candidate peak RSS or attribute background CPU to any one process.

**Interpreter startup controls.** Ran three interleaved K=50 no-op batches per launcher, each with 12 workers. Median batch time was 0.473 s for the stock launcher, 0.409 s for `-S`, and 0.394 s for invoking the base interpreter directly with the Windows virtual-environment launcher variable. The measured gains apply to no-op launches only. `-S` removed the `site` module and the virtual-environment prefix. Direct launch preserved the sampled virtual-environment settings but changed the observable parent process and relied on an internal launcher variable. Neither was eligible for a full training rollout. Reusing a Python interpreter would share import, interpreter, environment, thread and descriptor state. A fresh subprocess is retained for each candidate. [Startup control observations](../artifacts/verification_optimization_startup.json)

The bounded descendant probe launched a child that wrote a marker after the parent timeout. The marker appeared under all three launchers, including stock. The stock subprocess returned a timeout at about 0.419 s under a 0.4 s budget, and a descendant still wrote to the probe directory. This shows a pre-existing process-tree isolation gap on this Windows setup. Separate interpreters and temporary working directories protect ordinary per-candidate Python state. They do not guarantee termination of an entire process tree or containment of arbitrary filesystem effects. No launcher change was retained. A separate correctness control set a synthetic `sitecustomize` hook that printed the expected answer and exited 0 before source compilation. The baseline subprocess passed even though standalone compilation returned `SyntaxError`. This demonstrates a possible policy difference with parent-process syntax gating when interpreter startup hooks are present. The first fixture attempt encountered Windows returning timeout stdout as text, then attempted cleanup while a bounded child was still using its directory. The corrected probe accepts bytes or text and gives the bounded child time to exit.

**Matched repeats and decision.** Repeated the 12- and 8-worker arms with fresh checkpoints in that order, after the startup diagnostic and existing test suite. [Repeat observations and verdicts](../artifacts/verification_optimization_repeats.jsonl)

| Cold arm | Median/query | Mean/query | P95/query | Total wall | Pairs/s | Launches | Timeouts | Nonzero exits | Verdict differences |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 workers, repeat | 1.142 s | 1.194 s | 1.699 s | 238.848 s | 41.87 | 10,000 | 152 | 7,039 | 0 |
| 8 workers, repeat | 1.219 s | 1.302 s | 1.871 s | 260.422 s | 38.40 | 10,000 | 154 | 7,037 | 0 |

The repeat returned `q113` to 1.758 s at 12 workers. The 8-worker P95 was worse in both measured rounds, and its repeat took longer than either saved 12-worker run. The 12-worker references average 246.766 s and 1.234 s/query. The fastest new full run was the unchanged 12-worker control at 238.848 s, about 1.03× relative to that reference average. The span across the two fresh stock controls was 26.928 s. Keep both controls, including the outlier. There is no repeatable speedup attributable to a code or concurrency change, and no tested full arm approaches the 0.617 s/query target. Retain the existing production verifier in full. Warm runs duplicated scores and ranks after all 200 cache hits and zero launches. All 21 existing regression and integration tests passed again after the repeats. The actual-verifier smoke test also showed that sequential candidates do not share builtins mutations, environment edits, or relative files. In that same stock path, a candidate that printed the expected answer and timed out still failed verification, and its bounded descendant left a marker. This is a pre-existing safety gap for separate work. The test split and frozen benchmarks were untouched.

**Reproduction.** The [research driver](../artifacts/verification_optimization.py) uses the saved plan and passers directly. The saved performance reference SHA-256 remained `9fb287b54fdd25bcdb2b475bf10387adb05a32772324ce9e2843eae16cf70b38`. Use new output paths when replaying.

```powershell
.venv/Scripts/python.exe -B -X utf8 artifacts/verification_optimization.py NEW_PROFILE_JSON
.venv/Scripts/python.exe -B -X utf8 artifacts/verification_optimization.py NEW_SWEEP_JSONL 6 8 4 2 12 12:syntax
.venv/Scripts/python.exe -B -X utf8 artifacts/verification_optimization.py NEW_STARTUP_JSON --startup
.venv/Scripts/python.exe -B -X utf8 artifacts/verification_optimization.py NEW_REPEATS_JSONL 12 8
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

### 2026-11-24 Windows process-tree containment and startup feasibility

The preceding study reproduced a Windows failure where an ordinary descendant wrote into the candidate directory after candidate verification returned a timeout. The original launcher remains the performance control. The first isolated change is the Windows job lifecycle. Retrieval, K, example parsing, output comparison, interpreter choice, and the candidate's file, arguments, cwd, stdin, stdout and stderr are otherwise kept.

**Design.** A new per-candidate job starts with kill-on-close and no breakaway permissions. The existing Python executable starts suspended, is enrolled before it executes, and is resumed only after successful assignment. CPython's `Popen` does not expose its primary thread handle. This implementation finds and reopens the suspended primary thread using documented Toolhelp and thread APIs, while leaving setup of file handles and text pipes to `subprocess`. A job setup failure rejects the candidate and terminates any suspended process. Exit, timeout, capture exceptions and cancellation unwind through tree termination, accounting drain and handle closure before the outer temporary-directory cleanup. A watchdog also enforces the deadline when Windows stdio is blocked writing a large input. Cancelling a verifier batch cancels pending futures and waits for running workers to finish their own bounded cleanup. Ordinary candidate/verifier exceptions still fail a single candidate. The checkpoint format is unchanged, and its implementation fingerprint now covers the Windows runner so incompatible old live verdicts are missed. The Linux subprocess path is unchanged.

**Alternatives.** Launching first and assigning afterward leaves a race before job enrollment. Killing only the original PID leaves the already observed gap. The Windows native process-creation job-list attribute also attaches a job at launch, but Python's supported `Popen` startup-info interface exposes a handle list and does not expose a job list. A native launch shim would require maintaining additional inherited-pipe and process-creation code. The supported suspended-process path can be tested without a compiler or additional dependency. Prestarted supervisors still launch fresh Python processes for candidates, so their potential savings are on scheduling and supervisor overhead only. Reusing interpreters that execute candidate code would share Python state and is outside this experiment. The previous no-site and bypass-launcher controls changed environment or observable process semantics and remain rejected.

**Measurement.** Use the saved 200 training queries, ordered top 50, 12 workers, Python 3.13.13 UTF-8 mode and the one-second timeout. Give each cold run a separate fresh checkpoint. Compare full passer sets and rank metrics with both saved identical verdict snapshots. Each run immediately replays the same cold checkpoint for warm validation. Model load, embedding and similarity are outside timing. No new test-split data was accessed. [Containment and supervisor harness](../artifacts/verification_containment.py)

**Teardown investigation.** The first prototype returned too early for some children after a normal exit or an injected capture error. Member handles confirmed that the Python venv launcher, the candidate and its child were all inside the outer job while running. A post-return process handle initially reported an unsignaled child, then signaled after about 1.9 ms. Thus zero active processes in the accounting query was insufficient to confirm physical process exit. The replacement teardown blocks new admissions while members are live, retains synchronization handles for the listed members, stops the job, then waits for those handles and confirms the job drains. The timer and the calling thread each use the same teardown, with one owner. The breakaway test now requests breakaway from the job root. CPython's Windows venv launcher can create an inner job that permits breakaway while the outer CtrlFind job still contains the process. [CPython venv launcher](https://github.com/python/cpython/blob/v3.13.13/PC/venvlauncher.c#L360-L435) · [Microsoft job nesting](https://learn.microsoft.com/en-us/windows/win32/procthread/nested-jobs)

The first prototype's two timings and exact pass sets are preserved, but the failed adversarial tests rule it out. Its 272.065 s and 248.102 s cold runs had 172 and 161 timeouts and zero changed verdicts. The final runner requires a separate validation. [Prototype measurements](../artifacts/verification_containment_job_train.json)

Twelve prestarted Python supervisors passed three interleaved 50-candidate no-op batches, while launching a fresh candidate interpreter for each task. Startup including the readiness tasks took 0.848 s. Median batch time was 0.650 s with the warmed supervisors and 0.673 s with the original thread scheduler in this probe. The small favorable microbenchmark did not remove interpreter startup, and a long-lived supervisor also needs explicit environment and cancellation propagation. It is not a demonstrated cold-workload improvement. The production thread scheduler remains. [Supervisor measurements](../artifacts/verification_containment_supervisors.json)

An initial fresh stock control reported 248.231 s cold with 158 timeouts. The helper source changed during that attempt while we fixed the teardown race. Its warm run correctly missed the changed implementation digest and the trial aborted at the warm-equivalence check before saving a complete artifact. The cold console timing is recorded here as an interrupted observation. The stock control was restarted with source files held fixed through both halves.

The completed fresh stock control took 250.834 s cold, with 1.254 s mean, 1.237 s median, 1.785 s P95, 39.87 pairs/s, 10,000 candidate launches, 159 timeouts and 7,036 nonzero exits. Its warm replay took 0.104 s with zero launches. The complete verdicts and ranking matched the two saved controls. [Fresh stock measurements](../artifacts/verification_containment_stock_train.json)

**Final benchmark.** The plan and both source fingerprints were checked against the final files. Times exclude retrieval and setup, and all timed cold rows had 200 fresh cache misses.

| Runner | Mean/query | Median/query | P95/query | Cold wall | Pairs/s | Launches | Timeouts | Nonzero exits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Saved stock 1 | 1.228 s | 1.202 s | 1.740 s | 245.644 s | 40.71 | 10,000 | 164 | 7,032 |
| Saved stock 2 | 1.239 s | 1.220 s | 1.772 s | 247.887 s | 40.34 | 10,000 | 166 | 7,031 |
| Fresh stock | 1.254 s | 1.237 s | 1.785 s | 250.834 s | 39.87 | 10,000 | 159 | 7,036 |
| Final Windows job 1 | 1.207 s | 1.142 s | 1.764 s | 241.340 s | 41.44 | 10,000 | 164 | 7,032 |
| Final Windows job 2 | 1.298 s | 1.251 s | 1.857 s | 259.603 s | 38.52 | 10,000 | 160 | 7,033 |

Both final cold runs preserved all 10,000 verdicts, including 50 gold and nine non-gold passes. Both preserved NDCG@10 at 25.6267 and MRR@10 at 25.3714. Each warm replay had 200 hits, zero launches and identical score dictionaries to its cold run. Total warm timings were 0.085 s and 0.160 s. There were zero synchronous launch or capture errors in the measured runs. The recurring Windows decoding diagnostics and one reader-thread `MemoryError` were emitted inside the standard-library pipe reader. They are outside the synchronous error counter and did not change any verdict. The runner does not impose an output memory cap. [Final raw data](../artifacts/verification_containment_final_train.json)

The average contained run was 1.252 s/query and 250.471 s cold. That is 0.985× the two saved runs' speed and 1.001× the fresh stock control's speed. The spread between contained runs was 18.263 s. The 0.617 s/query target was missed. Retain the Windows containment correction and the existing fresh-interpreter, thread-based execution architecture. Repeating the supervisor microbenchmark with the final teardown gave median 0.701 s for warmed supervisors and 0.808 s for threads, with 0.860 s supervisor readiness cost. This remained an already-warmed synthetic control with 50 fresh candidate launches. No input-dependent supervisor reuse or alternate interpreter environment was deployed. [Final supervisor repeat](../artifacts/verification_containment_supervisors_final.json)

**Validation and limits.** All 30 regressions passed. The Windows tests confirmed child and grandchild handles were signaled before exit from the runner, including on a matched-output timeout, capture failure, cancellation and both normal and nonzero exits. They cover blocked stdin, failed job assignment, breakaway requests, stdout and stderr, inherited environment and cwd, relative-file cleanup, and independently launched candidate state. MTEB and Gradio cache tests and the history integration also passed. Windows file and process isolation remains limited to the job boundary. Jobs are not a filesystem, network, privilege, or brokered-process sandbox. No performance variant was adopted and no frozen artifacts or unrelated files were edited.

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
.venv/Scripts/python.exe -B -X utf8 artifacts/verification_containment.py stock NEW_STOCK_JSON
.venv/Scripts/python.exe -B -X utf8 artifacts/verification_containment.py job NEW_JOB_JSON --repeats 2
.venv/Scripts/python.exe -B -X utf8 artifacts/verification_containment.py supervisors NEW_SUPERVISOR_JSON
```

### 2026-11-25 Final Windows containment review

**Fault found and corrected.** With stock Windows pipe handling, the caller can block in the synchronous stdin write before `communicate` reaches its wait deadline. Fault injection made `TerminateJobObject` return failure. A 0.2-second deadline then returned only after 1.590 s, when the synthetic candidate finished its 1.5-second sleep. The watchdog used to record the error and leave its only job handle open until the caller's cleanup, which was waiting on the blocked write. The watchdog now closes the kill-on-close job immediately on explicit termination failure. It holds already acquired synchronization handles long enough to wait for those members before surfacing the error. The same fault probe returned an `OSError` in 0.246 s after the correction. Other stop failures also trigger early job closure. The cleanup owner lock prevents a watchdog and an interrupted caller from using a job handle concurrently with closing it. Termination or final job-handle close failures invalidate even a matching stdout and zero exit. No path retries launch without containment. [Microsoft Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects)

**Scope and regression checks.** Job creation and limit-setting failure prevent launch. Assignment failure kills the process before resuming it. Tests exercise explicit termination failure during a blocked stdin write with two descendant generations, failure after matching output, ordinary timeout, capture exception, cancellation, and successful and nonzero exit. Pending-future cancellation is Windows-only. A simulated POSIX import confirmed that Windows bindings are skipped, and dispatch and cancellation tests checked the normal subprocess arguments and executor behavior on that branch. This Windows host could not run a native POSIX test. WSL access was denied and the Docker Linux engine was unavailable.

**Checkpoint compatibility.** The live key includes the main verifier source fingerprint and the Windows helper fingerprint. A synthetic previous helper version wrote the same JSONL record shape with an incompatible key. The current helper missed it and executed once, then reused the new record on a warm hit. The reader accepts the old record shape. The schema and archived score cache files were preserved. As before, live cached negative verdicts do not distinguish an incorrect candidate from a transient timeout or infrastructure failure.

**Timeout counts.** The saved stock runs timed out 164 and 166 times, the fresh stock control 159 times, and the preceding job runs 164 and 160 times. All cold rows independently attempted 10,000 pairs and had identical pass sets and ranking scores. Their warm replays launched zero processes. In the controlled worker sweep, the 2-worker arm timed out 60 times and the 12-worker control 169 times, with identical pass sets. These observations support wall-clock scheduling and interpreter/pipe completion variability around the deadline. They rule out warm-checkpoint hits as an explanation for the cold differences. The artifacts contain aggregate process-state counts and per-query wall times, so they cannot attribute each particular timeout to scheduler delay, startup, candidate runtime, or a pipe wait. A timeout, nonzero exit, or mismatched output all remain a non-pass.

**Security boundary.** Windows Job Objects contain the process tree assigned to each job. Filesystem, network, privilege, and launches brokered by external services require separate controls. Temporary-directory removal is best effort. Output capture is still unbounded. Under the injected single termination API failure, kill-on-close handled the tree and rejected the candidate. If Windows fails both job termination and the last-handle close, the runner cannot guarantee a blocked pipe will unblock or prove that all descendants have exited. Unsupported kernel/host configurations and hostile same-privilege code are outside the tested boundary. No performance optimization or architecture change was made in this review. The **2× cold target was not achieved**.

**Final tests and replay.** All 35 tests passed. One fresh checkpoint replay on the unchanged 200-query training plan completed in 246.191 s cold (1.231 s mean, 1.163 s median, 1.770 s P95, 40.62 pairs/s), with 10,000 attempts, 149 timeouts, 7,040 nonzero exits and no counted launch errors. All 10,000 verdicts matched the saved baseline. NDCG@10 and MRR@10 remained 25.6267 and 25.3714. The warm replay took 0.136 s, performed zero launches and returned identical score dictionaries. The prior Windows reader-thread decoding diagnostic recurred. [Final review replay](../artifacts/verification_containment_review_train.json)

**Per-candidate timeout follow-up.** Selected the eight saved query rows containing sources previously identified as timing-sensitive. Replayed their exact top 50 without checkpoints in a 2, 12, 12, 2 worker sequence. The four runs recorded 3, 10, 11 and 1 timeouts. Ten candidate/query pairs switched between timeout and a completed outcome, and every switched verdict remained a non-pass. For example, `q2045/d2312` exited nonzero under two workers in both repeats and timed out under twelve in both repeats. `q1673/d2346` exited zero under two workers with mismatching output and timed out under twelve. All four passer sets matched the saved reference on these eight queries. This targeted diagnostic explains how contention changes timeout tallies without altering verdicts. It is not a representative timeout-rate estimate for the full panel. [Diagnostic script](../artifacts/verification_containment_timeout_review.py) · [Per-candidate observations](../artifacts/verification_containment_timeout_review_train.json)

### 2026-11-25 E5 retrieval and live verification funnel

Sampled 500 of 5,000 training query IDs from `CoIR-Retrieval/apps` at `f22508f96b7a36c2415181ed8bb76f76e04ae2d5` using `random.Random(20261125).sample(sorted(train_ids), 500)`. Gold IDs come from the train qrels. Search used all 8,765 corpus documents in snapshot order, matching MTEB's candidate pool. The sampled IDs, gold IDs, top-50 lists, ranks, and passing candidate IDs are in the measurement JSON.

The run used CtrlFind `d5274bdbc42fafea0ccf6e8f683c7bc9e0a41427`, `intfloat/e5-base-v2` at `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`, E5's `query: ` and `passage: ` prefixes, 512 tokens, normalized embeddings, and MTEB 2.21.0 cosine search at depth 1,000. The CPU batch size was 16 with four Torch threads and one interop thread. All 8,765 corpus embeddings were cache hits and the 500 sampled query embeddings were computed in memory. The live verifier used the original top 50, the production parser and reranker, a one-second candidate timeout, 12 workers, and a fresh scratch checkpoint. Runtime and dependency versions, source fingerprints, and the embedding-cache hash are recorded with the measurements. [Training measurements and query IDs](../artifacts/retrieval_diagnostic_train.json)

**Stage 1.** Recall@10 was **259/500 (51.8%)**, Recall@50 **311/500 (62.2%)**, Recall@100 **335/500 (67.0%)**, and Recall@200 **364/500 (72.8%)**. The verification cutoff excluded 189/500 golds. Another 53/500 were ranked 51–200 and 136/500 were beyond 200.

**Token lengths.** Raw counts use the dataset text with no prefix or special tokens. Formatted counts use the MTEB encoder text, E5 prefix and tokenizer special tokens before truncation. Effective counts use the production attention mask after truncation, excluding padding.

| Population | Raw >512 | Formatted >512 | Effective >512 |
|---|---:|---:|---:|
| Sampled train queries | 133/500 (26.60%) | 135/500 (27.00%) | 0/500 (0%) |
| All train queries | 1,218/5,000 (24.36%) | 1,242/5,000 (24.84%) | 0/5,000 (0%) |
| Indexed corpus | 581/8,765 (6.63%) | 587/8,765 (6.70%) | 0/8,765 (0%) |
| Train corpus subset | 247/5,000 (4.94%) | 249/5,000 (4.98%) | 0/5,000 (0%) |

**Live verification funnel.** The parser consumed the full statement. Only parsed queries launched candidates.

| Outcome | Conditional rate | Share of sample |
|---|---:|---:|
| Gold in stage-1 top 50 | 311/500 (62.20%) | 311/500 (62.20%) |
| Gold retrieved and example parsed | 8/311 (2.57%) | 8/500 (1.60%) |
| Gold retrieved and example skipped | 303/311 (97.43%) | 303/500 (60.60%) |
| Gold passed | 7/8 parsed (87.50%) and 7/311 retrieved (2.25%) | 7/500 (1.40%) |
| Gold failed after parsing | 1/8 parsed (12.50%) | 1/500 (0.20%) |
| At least one candidate passed | 7/8 parsed (87.50%) and 7/311 retrieved (2.25%) | 7/500 (1.40%) |
| Zero passers, including parser skips | 304/311 (97.75%) | 304/500 (60.80%) in the retrieved cohort |

Eight of 400 attempted pairs passed (2.00%), including one of 392 non-gold pairs (0.26%). Mean passers were **8/311 = 0.0257** per retrieved query including zeros, or **8/8 = 1.00** per parsed query including its one zero-pass case. Among the seven passing golds, six ranked first (85.71%) and one ranked second (14.29%), both overall and within the passing block. On `q2131`, `d5733` passed ahead of gold `d2131`, which moved from rank 41 to 2. The failed gold stayed at rank 2. Across all 311 retrieved golds, 5 improved (1.61%), 306 stayed in place (98.39%), and none dropped. The final median rank was 1, the mean was 5.61, and the top-10 count increased from 259/311 (83.28%) to 264/311 (84.89%).

**Bottlenecks.** The top-50 cutoff excluded 189 golds and the parser prevented 303 further gold checks. Post-run source grouping by dataset URL host shows that 262/311 retrieved gold queries came from Codewars or LeetCode and zero of those parsed. All eight parsed queries in this cohort came from Codeforces. Only 8/303 skipped statements matched the existing colon-label probe. Prioritize a source-stratified audit of example formats and availability. Stage-1 Recall@50 was 56/135 (41.48%) for truncated queries and 255/365 (69.86%) for the rest. Compare within source and use a paired diagnostic before attributing that gap to truncation. Use the 53 golds at ranks 51–200 to estimate how often extra retrieval depth would actually produce a verifiable example.

**Frozen baseline.** `python evaluate.py --mode cached` printed NDCG@10 **19.96** and MRR@10 **19.05** in a scratch copy with MTEB set to `only-cache`. The replay JSON differs in metadata and precision from the frozen artifact, and both headline scores match at the reported precision. The scratch run could only load the cached test score. [Cached output](../artifacts/retrieval_diagnostic_cached.json)

Run from the repository root with the pinned local Hugging Face snapshots, `artifacts/emb_cache.npz`, and the local MTEB result `results/prism__e5-base-v2-exec-rerank/1/AppsRetrieval.json`. Both measurement files use fresh destinations.

```powershell
.venv/Scripts/python.exe -B -X utf8 artifacts/retrieval_diagnostic.py artifacts/retrieval_diagnostic_train_repro.json
.venv/Scripts/python.exe -B -X utf8 artifacts/retrieval_diagnostic_baseline.py artifacts/retrieval_diagnostic_cached_repro.json
```

### 2026-11-25 Semantic retrieval ablation for example removal

**Protocol and baseline.** Used only the 5,000 queries, snippets and relevance labels in CoIR APPS training revision `f22508f96b7a36c2415181ed8bb76f76e04ae2d5`. The production evaluation entry points were not called because they load test data. Drew 500 from lexically sorted training query IDs with `random.Random(20261125).sample`, and kept the separately defined 200-query parseable verification workload in its saved order. The union has 685 distinct queries. The two panels overlap by 15 and are reported separately. The 500-query panel represents the training mix, while the 200-query panel deliberately selects queries for which the current verifier can extract a first example. Their ordered query ID SHA-256 values are `d463ba25ae4db0782402be587a550026ae9eef7c7922c64cdd1cb02c55a5a8c3` and `bd29b441a1159a43ccdd2c05933017c0c0cf194f19f331b6caa98bc131f7264a`, respectively.

Used the production E5-base-v2 encoder with local model revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`, the existing query and document prefixes, 512-token limit and L2 normalization. Ran on Windows Python 3.13.13 with CPU device, four PyTorch intraop threads, one interop thread, and batches of 16. Every one of the 5,000 training document vectors was a cache hit. The run selected only training document vectors from the existing cache and never wrote to that cache. The ordered training corpus ID SHA-256 is `a8a7748b4669d0b92d2fa28ac00b5e227e480ac33889b4bf793bc60aacebd7b1`. Ranked by cosine with document-ID tie order, held K at 50, and used the single training relevance label per query for scoring only. The unmodified cold baseline ran before the experiment parser or tests were added. Its scoped stage-1 wall time, excluding downloads, model load and document setup, was 921.962 s across 685 queries. [Baseline ranks, candidate lists and timing](../artifacts/query_example_baseline_train.json)

| Unmodified panel | Queries | Recall@10 | Recall@50 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Random training panel | 500 | 56.60 | 69.00 | 47.0568 | 44.0040 |
| Saved verification panel | 200 | 14.00 | 29.50 | 8.7183 | 7.1442 |

**Intervention.** The research parser recognizes dashed Example or Examples sections with complete alternating standalone Input and Output labels, or adjacent matching dashed Example Input/Output or Sample Input/Output sections. Sample data must be numbers, with YES and NO additionally allowed as output tokens. It leaves a section with prose, inline labels, empty payloads, or missing/disordered fields untouched. Other sections, including the problem's Input, Output, Constraints, Notes and Explanation, are preserved verbatim. Material on the original statement side remains available to the unchanged first-example parser and output comparator. The experiment feeds the modified text only to the E5 query embedder. Nothing changes in the execution and Windows containment path. [Research-only implementation](../artifacts/query_example_preprocessing.py)

The paired run encodes the 253 changed queries in both representations with separate cold in-memory caches and identical query order and batch shape. Unchanged query rows reuse their original ranks because the encoder would receive the same text. Original ranks are also compared with the fresh paired originals to detect numerical or retrieval drift. Paired bootstrap intervals for Recall@50 use 5,000 iid query-level resamples within each population, seed 20261125. Separately time 24 fixed edited queries, one at a time, in alternating arm order with an empty query cache and resident model and corpus. The preprocessed timed arm includes the stripping call. An actual-model check puts both representations in the same cache and checks two independent first misses and two later hits in reverse order. Before seeing the ablation outputs, the deployment criterion was a gain of at least 1.0 percentage point in Recall@50 on the random panel with a paired interval above zero, compatible secondary metrics and CPU cost, and downstream verification evidence on the fixed workload.

**Results.** All metrics are on a 0–100 scale, with the same 500 and 200 denominators. Gains and losses count the single gold document's full-corpus rank, so movement beyond K=50 is included in those counts.

| Panel | Recall@10 | Recall@50 | NDCG@10 | MRR@10 | Better / worse / unchanged gold ranks |
|---|---:|---:|---:|---:|---:|
| Random 500 | 56.60 → 56.80 | 69.00 → 68.40 | 47.0568 → 47.1155 | 44.0040 → 44.0226 | 30 / 42 / 428 |
| Saved verification 200 | 14.00 → 12.50 | 29.50 → 29.00 | 8.7183 → 8.3520 | 7.1442 → 7.1083 | 37 / 50 / 113 |
| Edited union, 253 distinct rows | 11.46 → 10.67 | 26.09 → 24.51 | 7.4041 → 7.2306 | 6.1746 → 6.1829 | 67 / 87 / 99 |

Recall@50 lost 3/500 hits in the random sample and 1/200 on the verification panel. No query gained a top-50 gold in either panel. The paired bootstrap intervals for the change were [−1.40, 0.00] and [−1.50, 0.00] percentage points, respectively. Recall@10 gained one hit in the random panel. In the verification panel, one entered the top ten and four left. The parser changed 110/500 and 154/200 statements, or 1,108/5,000 in the grammar census. Of the 253 edited benchmark queries, 83 retained identical effective token sequences at the 512-token limit and identical gold ranks. The freshly encoded original arm reproduced every gold rank and every ordered top 50 from the unmodified baseline. [Paired metrics, candidate lists, timings and token details](../artifacts/query_example_experiment_train.json)

| Representative training case | Gold rank | Removed example |
|---|---:|---|
| `q632`, waiter arithmetic | 27 → 8 | The input `5858 1234` and answer `1624`. |
| `q2081`, shopping discounts | 12 → 10 | Three labelled input/output pairs with item prices. |
| `q1509`, numeric pattern generation | 44 → 68 | The example directly showed the shape of the required rows, including `01`, `10` and `012`. |
| `q15`, rectangles cut from a square | 21 → 85 | Three dimension pairs with answers `Yes`, `Yes` and `No`. |

The original `q15` ordered top 50 matches the saved verification replay exactly, and `d15` passed in both saved cold runs. After stripping, rank 85 makes that gold unreachable for the existing verifier at K=50. No new full stage-2 run was triggered because Recall@50 declined on both preregistered panels. The source, first-example parser, output policy, checkpoint handling, and Windows containment were unchanged.

**Cost and cache checks.** Across the same 253 edited queries, cold batched stage-1 wall time was 171.377 s for originals and 176.497 s after removal. All 253 texts were cold misses in each arm. On the 24 alternating single-query trials per arm, original versus stripped mean was 404.8 → 391.7 ms, median 431.2 → 430.1 ms, and P95 463.1 → 474.0 ms. These mixed timings do not establish a consistent CPU benefit. The grammar census for 5,000 raw statements took about 69 ms. Two actual-model representations were distinct first misses and returned their own bit-identical vectors on reverse-order cache hits. A fixture also verified that the same text under the document prefix has its own cache entry. All 39 regression tests passed.

**Decision: REVERT.** The preprocessor stays in research only. There is no production diff to revert. The loss of top-50 gold hits fails the primary criterion, and the small random-panel top-10 gain did not reproduce in the verification panel. The original cache, frozen results, and test split were preserved. The measured result applies to this grammar, corpus and model setup.

```powershell
.venv/Scripts/python.exe -B -X utf8 artifacts/query_example_ablation.py baseline NEW_BASELINE_JSON
.venv/Scripts/python.exe -B -X utf8 artifacts/query_example_ablation.py experiment NEW_EXPERIMENT_JSON --baseline NEW_BASELINE_JSON
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

### 2026-11-25 Example parser coverage and Recall@50 audit

**Protocol.** Ran the retained `src.execute.parse_example` on raw `CoIR-Retrieval/apps` statements from local revision `f22508f96b7a36c2415181ed8bb76f76e04ae2d5`. The training panel is `random.Random(20261125).sample(sorted(train_ids), 500)` from 5,000 training IDs. All 500 IDs and parser flags matched the earlier funnel ledger, and all queries counted regardless of retrieval. The ordered IDs are in the audit JSON with SHA-256 `d463ba25ae4db0782402be587a550026ae9eef7c7922c64cdd1cb02c55a5a8c3`. The test reader projected only `text` under the `partition=test` filter and emitted aggregate counts. The parser audit's sole dataset input is the query parquet. [Measurements and IDs](../artifacts/example_parser_audit.json) · [Audit script](../artifacts/example_parser_audit.py)

| Population | Parsed | Unparsed |
|---|---:|---:|
| Fixed training sample | 43/500 (8.60%) | 457/500 (91.40%) |
| Full test-query text | 2,936/3,765 (77.98%) | 829/3,765 (22.02%) |

The grouped counts below use parsed statements divided by all statements bearing the marker. Flags can overlap across a query, including between its specifications and examples. The JSON includes percentages, other flags, and exclusive format buckets with their precedence specified in the script.

| Marker | Train parsed / with marker | Test parsed / with marker |
|---|---:|---:|
| `Input:` | 1/147 | 4/85 |
| `Output:` | 0/147 | 2/80 |
| Both colon markers anywhere | 0/144 | 0/77 |
| `Example` or `Examples` word | 43/390 | 2,936/3,133 |
| Dashed `Example` heading | 43/75 | 2,936/2,958 |
| Numbered or plain example/sample heading | 0/182 | 4/43 |
| Dashed example with standalone colon-labelled I/O | 0/27 | 0/12 |
| Dashed example with an inline colon-labelled payload | 0/3 | 0/1 |
| Separate dashed sample/example I/O headings | 0/80 | 0/697 |
| Line-start `Input:` followed by line-start `Output:` | 0/99 | 0/52 |
| Narrative cue such as `for example` | 14/131 | 778/904 |
| Arrow or function-result cue | 0/132 | 29/45 |
| No recognized marker | 0/54 | 0/29 |

**Manual review.** Reviewed 20 training misses selected to span distinct formats. Fifteen contained a complete example, including five explicit stdin/stdout pairs and ten function, conceptual or worked examples. One contained an output-only illustration and four supplied no concrete example in the available text. The retained parser expects a dashed `Example` or `Examples` heading, bare standalone `Input` and `Output` labels and nonempty extracted payloads. For instance, `q754` has inline `Input:1` and `Output:7`, `q775` places separate sample headings inside its dashed Example section, and `q3923` puts the arguments and expected result in a sentence. `q1695` supplies only specifications and refers to the easy version for its input. The review includes verbatim excerpts and reasons for each case. These cases illustrate failure modes. Marker counts and the curated review cannot estimate the prevalence of genuinely missing examples, and a textual extraction alone gives no evidence about execution compatibility. [20 reviewed cases](../artifacts/example_parser_audit_cases.json)

**Recall@50 provenance.** The README's roughly 31% traces to the historical 31.29% from the stage-1 notebook. Its data-loading and recall cells use every test query and the test-only corpus, historically 3,765 of each. A dictionary selects the last qrel row per query, ignores `score`, and ranks the selected doc as `1 + count(similarity > gold_similarity)` across the whole corpus. An equal score shares the better rank at the cutoff. The unweighted mean of `rank <= 50` uses all test query rows. The historical log documents one binary relevant doc per query. This audit inspected the notebook code, with the historical test qrels and ranks left outside the audit. The notebook did not pin its dataset or model revision. [README limitation](../README.md#limitations-and-future-work) · [Historical notebook source](../notebook/baseline_recall_and_execution_validation.ipynb)

The funnel reports 311/500 (62.20%) from its pinned uniform training sample against the full 8,765-snippet corpus. `load_train` keeps positive scores and asserts 5,000 qrel rows with unique query IDs and unique gold docs. The funnel uses an unweighted mean of whether that single gold ID is in each materialized stage-1 top 50. MTEB cosine search produces the hits and the script sorts by score, with ties following the returned hit order. The historical notebook used a dot product of normalized vectors, raw passage text and a shared rank for tied scores. Both use binary per-query hits under the documented one-gold dataset. Multiple golds would be overwritten in the notebook dictionary and would fail the funnel's training-data assertions. A quality comparison would require the same query population, corpus and tie rule. [Funnel retrieval and aggregation](../artifacts/retrieval_diagnostic.py) · [Training qrel checks](../artifacts/verification_research.py)

**Query representation.** The funnel sends full `queries[qid]` strings into the MTEB search dataset and again into the batch passed to `LiveExecReranker.search`. The stage-1 encoder creates its own `query: ` strings and the Sentence Transformer applies the 512-token cap while encoding. The reranker independently builds `qtext` from the incoming `queries["text"]` and calls `parse_example(qtext[qid])`. The ledger also calls `execute.parse_example(queries[qid])` directly. The demo follows the same separation. Example visibility to the parser is therefore governed by the original statement and its format. [Live reranker](../src/mteb_wrappers.py#L100) · [E5 encoder](../src/encoder.py#L24) · [Funnel script](../artifacts/retrieval_diagnostic.py#L198)

Reproduce from the repository root with the pinned local query snapshot. The script checks the 20 review IDs, excerpts and parser outcomes. All 39 regression tests passed.

```powershell
.venv/Scripts/python.exe -B -X utf8 artifacts/example_parser_audit.py artifacts/example_parser_audit_repro.json
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

### 2026-11-25 Source-specific example availability and verification audit

Used the same ordered 500 training IDs as the saved retrieval diagnostic and query ablation. The script reproduces them with `random.Random(20261125).sample(sorted(training_query_ids), 500)` and checks both saved lists and the gold IDs. The SHA-256 of the newline-joined ordered IDs is `d463ba25ae4db0782402be587a550026ae9eef7c7922c64cdd1cb02c55a5a8c3`. CoIR APPS snapshot `f22508f96b7a36c2415181ed8bb76f76e04ae2d5` provides a statement, URL and optional starter code for each training query and one positive qrel per query. It supplies no separate expected-output fixture or judge. Source is the normalized URL host. New reads of statements, metadata and candidate source are filtered to `train`, and new executions use only training candidate source. Saved top-50 ranks come from the 8,765-document diagnostic corpus. The same training IDs have 345 hits against the 5,000-document training-only corpus, compared with 311 saved diagnostic hits. The opportunity and near-match analysis below uses the saved 311. No retrieval or test evaluation was rerun. [Saved retrieval diagnostic](../artifacts/retrieval_diagnostic_train.json) · [Training-corpus comparison](../artifacts/query_example_baseline_train.json) · [Audit runner](../artifacts/source_example_audit.py)

| Source | Queries | Saved top 50 | Train-corpus top 50 | Potential stdin/stdout total / saved top 50 | Current raw parses total / saved top 50 | Potential stdin/stdout misses total / saved top 50 |
|---|---:|---:|---:|---:|---:|---:|
| Codewars | 263 | 213 | 226 | 0 / 0 | 0 / 0 | 0 / 0 |
| LeetCode | 68 | 49 | 55 | 0 / 0 | 0 / 0 | 0 / 0 |
| CodeChef | 114 | 31 | 40 | 114 / 31 | 2 / 0 | 112 / 31 |
| Codeforces | 42 | 9 | 12 | 40 / 7 | 41 / 8 | 0 / 0 |
| HackerRank | 9 | 6 | 8 | 2 / 1 | 0 / 0 | 2 / 1 |
| AtCoder | 4 | 3 | 4 | 4 / 3 | 0 / 0 | 4 / 3 |
| **Total** | **500** | **311** | **345** | **160 / 42** | **43 / 8** | **118 / 35** |

Potential stdin/stdout means a structurally extracted or manually reconstructed first pair compatible with the documented stream interface. It is an availability estimate before establishing judge validity, execution success or discriminating power. The parser recognized 42 such cases, seven with gold in the saved top 50. It also recognized `q1694`, whose published input is truncated. The CodeChef census includes six manually recovered pairs that the shadow grammar leaves out, plus two pairs tested successfully only after manual stdin cleanup. The callable sources have a separate pool. All 331 Codewars and LeetCode statements supply starter code, and 262 have gold in the saved top 50. A narrow AST literal check keyed to the starter function found 29 Codewars call-and-return examples, 28 retrieved. LeetCode has 62 colon-labelled statements, 43 retrieved. These counts describe possible material for a typed runner. They give no estimate of callable verification pass rates.

| Source | Explicit pairs | Colon-labelled | Narrative | Constraints/spec alone | Ambiguous | No usable marker |
|---|---:|---:|---:|---:|---:|---:|
| Codewars | 0 | 5 | 208 | 1 | 0 | 49 |
| LeetCode | 0 | 62 | 5 | 0 | 0 | 1 |
| CodeChef | 77 | 31 | 1 | 1 | 4 | 0 |
| Codeforces | 39 | 0 | 0 | 1 | 2 | 0 |
| HackerRank | 0 | 0 | 9 | 0 | 0 | 0 |
| AtCoder | 4 | 0 | 0 | 0 | 0 | 0 |
| **Total** | **120** | **98** | **223** | **3** | **6** | **50** |

The exclusive format census labels the first recognizable syntax under the audit's precedence. Separate named input/output headings count as explicit pairs. Narrative cues can come from tutorials, worked reasoning or illustrations. Absence of a recognizable marker is a heuristic. Reviewed exceptions include the actual width-three grid row `...` in `q79`, which the prototype conservatively abstains on, and `q1695`, which references another version and gives an answer tolerance without a local runnable pair. The current parser requires a dashed Example or Examples heading, bare standalone Input and Output labels and nonempty payloads. `q303` has labelled point assignments for `Solution.validSquare`, `q2549` has a single number and a Boolean method return, and `q3406` gives a callable result list. Sending those representations to stdin assumes a contract absent from the statements and candidate code. HackerRank `q2621` shows deque tutorial calls instead of the task's command batch. [Per-query formats and payloads](../artifacts/source_example_audit_train.json) · [Hand-labelled examples](../artifacts/source_example_labels_train.json)

**Experimental extraction.** A research-only variant admits anchored combined blocks with standalone or inline colons and separately headed Sample or Example Input/Output with matching numbering. It stops at clear commentary boundaries, abstains on known placeholders, gates by stream-oriented source and has an AtCoder-specific first-paragraph check for numeric or yes/no output. It adds 110 pairs (76 separate headings, 30 combined blocks and four AtCoder paragraph cases), or 153/500 raw recognitions including the current parser. On the saved retrieved set it adds 33, from 8/311 raw recognitions to 41/311. The historical narrow colon probe proposes 27 new pairs, of which eight are retrieved. An ungated line-start colon scan flags 99 overall and 58 retrieved, including 67 Codewars or LeetCode statements and 48 of their retrieved queries. The unscreened figures include unsupported payloads.

| Parser probe | Precisely reproduced labelled pairs (13) | Precisely reproduced contract-dependent pairs (11) | Unsafe controls with extraction (11) |
|---|---:|---:|---:|
| Current | 1 | 1 | 1 |
| Earlier narrow colon matcher | 2 | 0 | 0 |
| Ungated colon labels, first output line | 0 | 0 | 3 |
| Source-gated shadow prototype | 13 | 0 | 0 |

The 36 purposefully selected review cases include one visual case outside those three denominators. The labelled controls are a diagnostic challenge set. They cannot estimate population precision. The gated prototype extracted two manual cases inaccurately. `q944` retained an internal display blank and `q658` retained per-row Markdown spaces. The gold snippets threw `ValueError` on those inputs and passed after manual formatting consistent with each input contract. A hand-isolated sample recovered `q1211` (raw heading includes editorial guidance, gold at rank 7), and hand-reconstructed `q2631` passed at rank 1. The explicit empty-input contract `q2650` also passed in a manual run. This evidence does not justify global removal of input blanks or spaces. `q1077` also permits any satisfying node, so a literal extraction can still reject valid outputs. [Label comparisons and manual replays](../artifacts/source_example_summary_train.json)

**Live shadow execution.** Ran the 110 proposed pairs with the current one-second timeout, two workers, Windows containment and production comparison policy selected from the original full statement. The diagnostic reused captured stdout for its exact-comparison control and wrote no verdict checkpoints. Gold passed 88/110 (80.00%) and 30/33 (90.91%) among new pairs with saved top-50 gold. The latter includes 27/30 CodeChef and 3/3 AtCoder. One gold, `q1363`, passed only with existing scoped output padding. Gold exited cleanly 98 times, exited nonzero 11 times and timed out once. `q827` timed out on three further standalone runs. Its stored code fills a large combination table before reading input. This audit measures the effect of enabling an example with existing execution settings. It did not implement input normalization or modify the verifier.

| Control on proposed pairs | Passing pairs / attempts |
|---|---:|
| Five seeded random training snippets per query | 0 / 550 |
| Available distinct training-source non-gold hits from the saved top 50 | 3 / 897 |
| A known wrong training snippet, separately selected | 1 / 1 |

Random controls use `random.Random(f"{SEED}:{qid}")`, sorted training doc IDs and exclude the intended gold and exact source duplicates of it. Near controls use saved top-50 order, exclude the same gold/duplicate cases and skip IDs outside the training corpus. After one cross-cohort duplicate is executed once, the raw run contains 1,557 candidate executions. The control denominators describe different populations. The known collision is not a rate estimate. Three observed near matches and the known control reproduced cleanly on the published fixture. Independently derived inputs established the wrong matches below. [Raw outcomes](../artifacts/source_example_audit_train.json) · [Witness replays with inputs and expected results](../artifacts/source_example_summary_train.json)

| Published match | Reason | Independent witness |
|---|---|---|
| `q1465 / d2133` | A singleton tree computation prints the same `1` as a pill schedule beginning March 31. | March 29 allows two correct doses before April 1. Only the gold matched `2`. |
| `q692 / d556` | A Fibonacci-like streak has length `3`, coinciding with the task's maximum divisibility count. | For `[2,4,6]` the divisibility count is `0`. Only gold matched. |
| `q541 / d1030` | An unfinished-job program prints job `3` while the MEX task's optimal subset size is also `3`. | For `[1,2,3,5]` and MEX `3`, the subset `[1,2,5]` gives size `3`. Only gold matched. |
| `q949 / d814` | An unrelated program's tie branch prints `1`, the published number of square-valued pairs. | `1 2` allows values 2 and 3, so the count is `0`. Only gold matched. |

The same MEX task exposed a gold defect. For `[1,2,3]` and target MEX `3`, the optimal subset `[1,2]` gives size `2`. Gold executed cleanly and printed `-1` because its search for an absent element ends at `N`. Published-example agreement and gold labels therefore leave independent uncertainty about correctness. The targeted witnesses are diagnostic fixtures with derivations, separate from the published-example runs.

**Decision. CHANGE VERIFICATION STRATEGY.** Of the 35 retrieved-gold queries with potentially recoverable stream examples, 33 reached the shadow run, 30 golds matched and 23 of those could move up from below rank 1. Twenty of those 23 had no observed rival match in the sampled controls. This screens roughly 6.4% of the 311 retrieved-gold queries for potential upside. The 20 is neither an observed ranking gain nor a safety guarantee, since some saved neighbors were unavailable as training controls and a single fixture misses errors. No end-to-end ranking metric was measured. Prioritize explicit invocation and return contracts and independently checked fixtures for callable sources, with abstention where a harness or judge semantics cannot be established. Retain grammar changes as a shadow experiment pending candidate-discrimination evidence. No production change was made.

The nonexecution inventory replay reproduced all 500 rows exactly, and all 39 repository regression tests passed, including Windows containment. The commands below create new files and leave the saved evidence intact.

```powershell
.venv/Scripts/python.exe -B -X utf8 artifacts/source_example_audit.py artifacts/source_example_audit_repro.json --execute
.venv/Scripts/python.exe -B -X utf8 artifacts/source_example_audit.py artifacts/source_example_summary_repro.json --summarize artifacts/source_example_audit_repro.json --witnesses
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

### 2026-11-25 Typed callable verification feasibility on training queries

**Population and prior counts.** Reused the 500 query IDs from seed `20261125` on APPS revision `f22508f96b7a36c2415181ed8bb76f76e04ae2d5`. The full-corpus diagnostic had 311 top-50 golds at 8,765 documents and the training-corpus ablation had 345 at 5,000 documents. All 311 remain hits in the train-corpus list. Each of the 34 additional hits had at least one non-training document occupying a position in its full-corpus list. Filtering the saved full-corpus top 50 to training IDs gives the same candidate set as the corresponding train-list prefix for all 500 queries, and identical order for 496. The archived retrieval runs used the same E5 revision and prefixes, with MTEB cosine in the diagnostic and a stable normalized dot product in the ablation. The corpus accounts for the hit-set difference in these lists. The saved evidence does not establish the cause of the four ordering differences. For example, `q272` moved from saved rank 81 with 36 non-training top-50 slots to rank 24 on the training corpus. Separately, 41 of 42 Codeforces statements had a raw production parse, and 40 contained potentially suitable pairs. The extra parse `q1694` contains two literal “input is truncated” lines. [Source audit and raw examples](../artifacts/source_example_audit_train.json) · [Prefix audit and gained IDs](../artifacts/callable_feasibility_summary_train.json)

The evaluation population is all 331 Codewars (263) and LeetCode (68) statements in the panel, with 281 retrieved golds in the training corpus and 262 in the older diagnostic. New query, metadata and candidate code reads were partition-filtered to training. The primary arm uses the saved all-training lists. A secondary replay preserves previously saved full-corpus ranks and leaves candidates outside the training partition ineligible. It never loads or executes their source. No model inference, new corpus search, test evaluation or production writes were performed. [Training-corpus orders](../artifacts/query_example_baseline_train.json) · [Saved diagnostic orders](../artifacts/retrieval_diagnostic_train.json)

**Fixture selection.** The source audit supplied 29 Codewars statements with literal Python calls and 62 LeetCode statements with colon labels. In original panel order, draw 12 from each of these pools with `random.Random(f"{SEED}:{source}")`. Draw six more per source from the remaining queries with a gold rank of 2–50 in either saved list using `random.Random(f"{SEED}:rank:{source}")`. The latter deliberately enriches for measurable rank changes. A targeted review admitted 13/18 Codewars and 14/18 LeetCode queries, for 27/36 reviewed and 27/331 operational coverage in this replay. Unreviewed statements abstained. These counts do not estimate the prevalence of trustworthy fixtures in the remaining workload.

Codewars examples such as `q3930` use a starter's standalone name and Python literal call/return pairs. LeetCode starters establish the exact `Solution` method, parameter order and sometimes types. Examples such as `q1956` spell out assignments. A bare list in `q268` maps to its single declared argument. The review checks each of 87 fixture arguments and expected returns against the published statement and an independent small-input reference. References use direct search, exhaustive enumeration or exact arithmetic where possible, including reachable string permutations for `q1956`, all triple choices for `q193`, a bit-state search for `q138` and rational arithmetic for `q452`. No candidate output supplied the expected value. The nine abstentions cover alternate or ambiguous outputs (four), tree-node object contracts (two), unsupported numerical representations (two) and a missing recurrence (one). Examples include the multiple peaks allowed by `q383`, the tree serialization in `q1841` and the missing recurrence in `q4076`. [Reviewed IDs, starters and typed fixtures](../artifacts/callable_verified_fixtures_train.json) · [Independent references and abstention reasons](../artifacts/callable_fixture_reference.py)

**Candidate interfaces and execution.** Static AST inspection of the 16,550 training top-50 slots found 6,166 standalone-function slots, 2,311 direct `Solution` method slots, 173 other-class-method slots, 146 with standalone functions and other class methods, 7,569 without a direct callable and 185 invalid Python. These are candidate slots across queries. The prototype admits only the supplied callable name, owner and compatible positional arity. It abstains on unsupported constructors, variadic or decorated entry points and class inheritance. It does not rename functions, synthesize object adapters, supply third-party modules or repair submissions. `q4170`'s gold uses a variadic entry and also imports the removed `fractions.gcd`. The compatible gold for `q279` exits nonzero on all four examples because it assumes a `Counter` import.

Each candidate–fixture case runs in a new contained Windows Python process with a private temporary working directory, `-I -S -B`, empty stdin and the existing Job Object launcher and one-second deadline. The trusted harness clears inherited environment variables before loading candidate code, retaining Windows system paths and setting its temp variables to the private directory. The only judge prelude is `typing.List` for `Solution` references. Explicit typed values support integers, Booleans, strings, lists, tuples and `None`. The parent supplies arguments and keeps expected values out of the invocation. It requires exit code zero and an exact typed return record, with stdout/stderr captured separately. Research controls confirmed current-directory cleanup, fresh state per fixture, no inherited sentinel variable, rejection of stdout-only and nonzero results, distinct Boolean and integer types, distinct list and tuple types, and termination of a launched descendant after timeout. Job Objects constrain process lifetimes. The study did not enforce filesystem, network or privilege isolation or attempt to authenticate return records against a hostile interpreter. [Runner](../artifacts/callable_candidate_runner.py) · [Control outcomes](../artifacts/callable_feasibility_summary_train.json)

| Source | Reviewed | Fixture queries | Compatible gold / gold slots | Gold passing all / gold slots | Compatible non-gold / non-gold slots | Non-gold passing first / all cases |
|---|---:|---:|---:|---:|---:|---:|
| Codewars | 18 | 13 | 12 / 13 | 12 / 13 | 2 / 637 | 0 / 2 and 0 / 2 |
| LeetCode | 18 | 14 | 14 / 14 | 13 / 14 | 1 / 686 | 1 / 1 and 0 / 1 |
| **Total** | **36** | **27** | **26 / 27** | **25 / 27 (92.59%)** | **3 / 1,323** | **1 / 3 and 0 / 3** |

Twenty-nine of 1,350 primary shortlist slots were callable-compatible. The remaining statuses were 703 with no direct callable, 456 missing or ambiguous names, 154 missing or ambiguous owners, three unsupported signatures, two unsupported owners and three invalid Python. The all-case gold pass rate among runnable golds was 25/26 (96.15%). Reporting non-gold compatibility separately matters. All cases rejected the three compatible non-golds, and the other 1,320 non-gold slots were screened out before execution. The secondary lists had 24/26 gold passes among retrieved golds and 0/3 compatible non-gold passes. Another 828 slots on those fixture queries were outside the training partition and skipped. The only six extra training candidates with a compatible interface outside both lists matched none of their `q3930` fixtures.

**Incorrect passes and gold defects.** The retrieved `q373/d158` wildcard matcher accepts the same method signature as the regular-expression task and passes three of its five examples, including the first. It fails `("ab", ".*")` because it treats the dot literally and fails `("aab", "c*a*b")` because its star semantics differ. All-case verification rejects it. The real lists placed this candidate below gold, so already rejecting it had no ranking effect. A synthetic `q452` method that always returns true passes all three published rational-equality cases. The independently derived inequality `"0.(3)" != "0.3"` rejects it. This control illustrates a coverage gap and stays outside the observed non-gold rate. The CodeChef `q541` contradiction was independently reproduced with exhaustive subsets on `[1,2,3]` and target MEX 3. The maximum size is 2 using `[1,2]`. The designated gold prints `-1` with exit zero because its scan never finds an absent value within `1..N`. That fixture stays outside the callable and ranking denominators. [Per-case returns and subprocess outcomes](../artifacts/callable_feasibility_train.json) · [Positive and negative controls](../artifacts/callable_feasibility_summary_train.json)

**Ranking.** The production parser abstains on all 331 target statements, so its baseline makes no stage-2 promotions. Saved final ranks confirm this on the secondary lists. For the experimental replay, stably group passing IDs at the top, matching the reranker's promotion order while leaving the existing lists and order within each group intact. Gold missing from the top 50 contributes zero. With one gold per query, NDCG@10 is mean discounted gain and MRR@10 is mean reciprocal rank, both reported on a 0–100 scale. The signature-only arm promotes the same candidates that would be eligible to run before considering returns.

| Saved candidate list / policy | Queries | NDCG@10 | MRR@10 | Gold improved / regressed | Entered top 10 |
|---|---:|---:|---:|---:|---:|
| Training corpus, production | 331 | 63.66 | 60.20 | 0 / 0 | 0 |
| Training corpus, signature only | 331 | 65.55 | 62.43 | 12 / 0 | 3 |
| Training corpus, all fixtures | 331 | 65.70 | 62.63 | 12 / 0 | 3 |
| Training corpus, minimum two fixtures | 331 | 65.48 | 62.33 | 10 / 0 | 3 |
| Full-corpus diagnostic, production | 331 | 59.13 | 55.85 | 0 / 0 | 0 |
| Full-corpus diagnostic, signature only | 331 | 61.37 | 58.33 | 11 / 0 | 5 |
| Full-corpus diagnostic, all fixtures | 331 | 61.52 | 58.54 | 11 / 0 | 5 |
| Full-corpus diagnostic, minimum two fixtures | 331 | 61.11 | 58.08 | 9 / 0 | 4 |

The first-fixture policy has the same gold ranking metrics as all fixtures while accepting the known wrong wildcard candidate. The training-corpus all-fixture gain over production is 2.04 NDCG and 2.43 MRR points. Over the signature-only arm it is 0.15 and 0.20 points, respectively. The single additional move over the signature control is `q3930`: both unrelated `reverse` functions were compatible and ahead of gold, and execution filtered them. Gold moved from baseline rank 15 to rank 3 with signature matching and rank 1 with execution. In the training-only list, `q3246` and `q272` also entered the top 10. On the 36 reviewed queries, training-corpus scores changed from 67.28/61.90 to 86.08/84.26 NDCG/MRR, with the same 12 improved and zero regressed. Selection enriched this subset for rank opportunity. [All rank orders, policies and populations](../artifacts/callable_feasibility_train.json) · [Signature-only control](../artifacts/callable_feasibility_summary_train.json)

**Cold overhead and reproducibility.** Two sequential runs on Windows with 12 logical processors, Python 3.13.13, two workers and no verdict cache each launched 93 candidate subprocesses. Both had 83 matching case verdicts, six mismatches, four nonzero exits and zero timeouts in the main study. All 93 case verdicts and all rankings reproduced. Total experiment wall time was 18.20/18.88 seconds, excluding dataset load and independent controls. Mean, median and P95 per fixture-eligible query were 0.637/0.621/1.254 seconds on the first run and 0.657/0.619/1.227 seconds on the repeat. The production baseline spent under 2 ms parsing all 331 statements and launched no processes. Main study time includes scanning training source and subprocess setup, capture and cleanup. The reported timeout in the descendant control is separate from the main study. [Cold run](../artifacts/callable_feasibility_train.json) · [Cold repeat](../artifacts/callable_feasibility_repeat_train.json)

**Decision. REJECT production integration on this evidence.** Most of the measured gain comes from the callable interface. Execution adds one rank improvement beyond that control, materially increases latency where enabled and has only three compatible retrieved non-golds for estimating false promotions. Published examples alone admit a known wrong callable even when every example is checked. The reviewed sample and partial mixed-corpus candidate coverage limit extrapolation. All 39 repository regression tests passed, including the Windows containment suite. Reproduce with new output filenames so that saved evidence stays intact.

```powershell
.venv/Scripts/python.exe -B -X utf8 artifacts/callable_feasibility.py fixtures artifacts/callable_fixtures_repro.json
.venv/Scripts/python.exe -B -X utf8 artifacts/callable_feasibility.py run artifacts/callable_run_repro.json
.venv/Scripts/python.exe -B -X utf8 artifacts/callable_feasibility.py run artifacts/callable_repeat_repro.json
.venv/Scripts/python.exe -B -X utf8 artifacts/callable_feasibility_analysis.py artifacts/callable_run_repro.json artifacts/callable_summary_repro.json --repeat artifacts/callable_repeat_repro.json
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

### 2026-11-25 Training source distribution and retrieval consistency

Read the 5,000 training query and corpus rows and `train` qrels from `CoIR-Retrieval/apps` revision `f22508f96b7a36c2415181ed8bb76f76e04ae2d5`. Source is the normalized URL hostname. Ran the retained parser on full statements and counted characters and untruncated E5 input tokens using `query: ` and the model's special tokens at revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`. Inputs longer than 512 tokens are classified as truncated. The audit reproduced the prior parser flags and token lengths for every fixed sample ID. The 500 IDs use `random.Random(20261125).sample(sorted(train_ids), 500)` and match both saved ranking runs. Their ordered SHA-256 is `d463ba25ae4db0782402be587a550026ae9eef7c7922c64cdd1cb02c55a5a8c3`. [Audit script](../artifacts/training_distribution_audit.py) · [Summary and sample IDs](../artifacts/training_distribution_audit_summary.json) · [5,000 per-query rows](../artifacts/training_distribution_audit_rows.jsonl)

**Source breakdown.** The prior funnel column reuses the saved scalar ranks of these training queries from the 8,765-document run. The training-only column uses the saved 5,000-document baseline on the same IDs. The audit rerun checked that all parquet reads selected only training data. [Prior funnel](../artifacts/retrieval_diagnostic_train.json) · [Training-only ranks](../artifacts/query_example_baseline_train.json)

| Source | Sample share | Census share | Parsed sample / census | Truncated sample / census | Prior Recall@50 | Training-only Recall@50 |
|---|---:|---:|---:|---:|---:|---:|
| AtCoder | 4/500 (0.80%) | 61/5,000 (1.22%) | 0/4 · 0/61 | 0/4 · 19/61 | 3/4 (75.00%) | 4/4 (100.00%) |
| CodeChef | 114/500 (22.80%) | 1,112/5,000 (22.24%) | 2/114 · 37/1,112 | 53/114 · 464/1,112 | 31/114 (27.19%) | 40/114 (35.09%) |
| Codeforces | 42/500 (8.40%) | 477/5,000 (9.54%) | 41/42 · 474/477 | 30/42 · 314/477 | 9/42 (21.43%) | 12/42 (28.57%) |
| Codewars | 263/500 (52.60%) | 2,515/5,000 (50.30%) | 0/263 · 0/2,515 | 44/263 · 369/2,515 | 213/263 (80.99%) | 226/263 (85.93%) |
| HackerRank | 9/500 (1.80%) | 96/5,000 (1.92%) | 0/9 · 0/96 | 3/9 · 21/96 | 6/9 (66.67%) | 8/9 (88.89%) |
| LeetCode | 68/500 (13.60%) | 739/5,000 (14.78%) | 0/68 · 0/739 | 5/68 · 55/739 | 49/68 (72.06%) | 55/68 (80.88%) |
| All | 500 | 5,000 | 43/500 · 511/5,000 | 135/500 · 1,242/5,000 | 311/500 (62.20%) | 345/500 (69.00%) |

**Representativeness.** Parser coverage was 8.60% in the sample and 10.22% in the census. Truncation was 27.00% and 24.84%, respectively. Median character lengths were 1,027 and 1,051.5. Median formatted lengths were 311 and 318.5 tokens. The largest source-share shift was Codewars at +2.30 percentage points. Across 20,000 random draws of 500 without replacement from the fixed census (seed `20261126`), the source-mix Pearson statistic gave p=0.732, parser count p=0.208, truncation count p=0.254, character-length KS p=0.832, and formatted-token KS p=0.559. The KS statistic compares each draw against its 4,500-query complement. All five Holm-adjusted p-values were 1.00. These checks found no evidence of a material sample imbalance on the measured margins. Small sources still have wide sampling uncertainty. For example, prior Recall@50 Wilson intervals are 30.1–95.4% for 3/4 AtCoder queries and 35.4–87.9% for 6/9 HackerRank queries. The summary includes intervals and joint parser × truncation × retrieval counts by source.

**Parser and retrieval.** The prior funnel's parseable and unparseable Recall@50 values were 8/43 (18.60%, Wilson 95% 9.74–32.62%) and 303/457 (66.30%, 61.85–70.48%). Forty-one of the 43 parseable queries were Codeforces. The only within-source comparisons were Codeforces (8/41 parseable and 1/1 unparseable) and CodeChef (0/2 and 31/112). The 1/1 and 0/2 cells have Wilson intervals of about 21–100% and 0–66%. Under exchangeability of parser flags within each source, the pooled source recall rates predict 9.33 parsed hits and a −44.31-point gap of the observed −47.70 points. The residual is −3.38 points with exact conditional p=0.170. This is a descriptive decomposition that holds source margins fixed and convolves the corresponding hypergeometric hit counts. The same approach on the 5,000-document baseline gave 11/43 versus 334/457 and predicted −43.90 of the −47.50 points (p=0.206).

Parseable queries had median 579 versus 292 formatted tokens, and 30/43 versus 105/457 exceeded 512. Truncation status alone predicts −13.28 of the prior −47.70-point parser gap. Source plus length bands (≤256, 257–512, >512) predicts −45.81 points, with both parser outcomes represented in only one of 16 populated strata. All 43 parsed queries lacked starter code, whereas the 340 queries with starter code had 268/340 prior hits (78.82%). Starter-code presence closely tracks source and was used only for this analysis. The parser runs downstream of retrieval and cannot affect these ranks. The observed overlap is too sparse to isolate the effects of statement format or length.

**Miss concentration.** CodeChef and Codeforces made up 156/500 queries (31.20%) and accounted for 116/189 prior misses (61.38%) and 104/155 training-only misses (67.10%). In the prior run, 35/189 misses were parseable (18.52%), 79/189 were truncated (41.80%), and 24 belonged to both groups. The corresponding training-only counts were 32/155 (20.65%), 72/155 (46.45%), and 22 in both. Truncated recall trailed shorter-query recall by 28.38 points in the prior run and 30.59 in the training-only run. Source composition predicts 22.21 and 21.78 points of those gaps. The remaining within-source association is −6.17 points (exact p=0.113) and −8.82 (exploratory p=0.022), respectively. On the training-only CodeChef panel, truncated queries reached the top 50 at 12/53 (22.64%) and shorter ones at 28/61 (45.90%). Source stratification addresses source composition. Statement content and difficulty may still confound the length comparison.

**Recommended training-only protocol.** Use the existing `query_example_ablation.py baseline` runner with the pinned revision and fixed uniform 500 query IDs. Report the separately saved parseable verification panel as a separate workload. Use all 5,000 training snippets in lexicographic document-ID order, keep positive train qrels, and assert exactly one relevant document per query and complete ID coverage. Use raw statement and snippet text through the original E5 `query: ` and `passage: ` prefixes, a 512-token limit and L2 normalization. Fix the model revision, CPU, batch size 16, four Torch threads and one interop thread. The runner selects the training document vectors from the read-only cache, holds query embeddings in memory, and saves ranks to a fresh artifact. Rank all candidates by descending normalized dot product using stable sorting and lexicographic document ID for exact ties. Compute query-macro Recall@K as `mean(rank <= K)` over all 500 queries, NDCG@10 as the mean of `1/log2(rank+1)` at ranks 1–10, and MRR@10 as mean `1/rank` at ranks 1–10. Both latter metrics assign zero outside the cutoff. Preserve per-query ranks, sample order, corpus digest, and model configuration. [Training-only runner](../artifacts/query_example_ablation.py)

The historical README figure used all 3,765 test queries and 3,765 test candidates, an unpinned dataset/model revision, a dictionary that kept the last qrel per query and ignored its relevance score, and `1 + count(scores > gold_score)` as rank. That rank lets exact ties share a cutoff position. The historical notebook also averaged reciprocal rank without the @10 cutoff. The prior diagnostic on the same 500 training queries used an 8,765-document MTEB search, MTEB corpus-text formatting and MTEB's returned tie order. The training-only ledger has 34 additional top-50 hits and zero losses against the prior ledger. The corpus, formatting and ranking implementation differ between these saved runs, so this is a protocol difference and cannot isolate a corpus-size effect. The historical comparison uses the recorded definition. [Historical evaluation trace](#2026-11-25-example-parser-coverage-and-recall50-audit)

**Next experiment.** Preregister a new source-stratified training panel with enough CodeChef and Codeforces queries above and below 512 tokens. Evaluate the same queries in paired arms using the existing head truncation and a fixed head-plus-tail token allocation at the same 512-token budget. Hold corpus, model, prefixes and tie rules constant. Report paired Recall@50 differences and intervals within each source. Short inputs can serve as an identical-input control. This tests whether information outside the current window is recoverable while controlling for the large source-composition difference.

Reproduce the audit from the saved ledgers and pinned local snapshots. The first command emits new audit files. The second is the training-only baseline command for generating a new rank ledger. All 39 repository regression tests passed.

```powershell
.venv/Scripts/python.exe -B -X utf8 artifacts/training_distribution_audit.py artifacts/training_distribution_summary_repro.json artifacts/training_distribution_rows_repro.jsonl
.venv/Scripts/python.exe -B -X utf8 artifacts/query_example_ablation.py baseline artifacts/training_rank_baseline_repro.json
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

### 2026-11-25 Exact rank attribution and dev-stdin

**311 versus 345.** The archived full-corpus replay reproduced all 500 ordered top-50 lists and all recorded gold ranks, including the markers for ranks beyond depth 1,000. The replay returns 311/500 (62.20%). Holding its cosine scores fixed and dropping the 3,765 non-training distractors gives 345/500 (69.00%), with the exact same 34 entrants as the training-only archive and zero losses. The per-query count of removed documents ahead of gold exactly accounts for all 288 changed gold ranks. For the 34 entrants, the removed counts range from 2 to 173. The attribution script verified the saved full-corpus witnesses and rescored against the training corpus exclusively. The full-corpus scoring evidence is retained as an archived training-query replay. [Full-corpus witness](../artifacts/retrieval_consistency_run/historical_check.json) · [Per-query attribution and score controls](../artifacts/dev_stdin_attribution.json)

**Inputs and numeric controls.** Both runs use the same seed `20261125` IDs, verbatim statements, `query: ` prefix, 512-token cap, pinned `intfloat/e5-base-v2` revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`, and exact same positive training qrel. The historical full-corpus run used MTEB-formatted passage text in parquet order and MTEB cosine/top-k/heap ordering. The training-only ablation used raw passages in lexicographic ID order and NumPy dot product over normalized vectors. The historical production cache had zero sampled query hits. Both sets of 5,000 document inputs resolve from the pinned content-addressed cache, with 1,710 raw/formatted text differences and 524 distinct vector pairs. Their maximum component difference was `2.66e-7`. The archived 16-document fresh encoding probe had minimum cosine `0.99999994` versus the cached vectors. Rescoring the training pool with formatted cosine or raw cosine preserved all 500 original training gold ranks. Reversing document input order with fixed lexical tie resolution preserved every tested order, and the historical MTEB tie behavior caused zero Recall@50 changes.

**Two numerical exceptions in the shared-vector replay.** The historical ablation calls E5 on individual blocks of 16, while the shared query-vector cache lets E5 length-sort all 500 inputs before making batches. Recreating the original blocks recovered both exceptions and all 500 archived gold ranks and ordered top-50 lists. For `q4336`, float32 rounding switches `d249` and `d3696` at ranks 42/43. Gold stays first and the same 50 candidates remain. For `q4630`, cosine over the shared vectors puts `d252` above gold `d4630`. Raw dot product over the same vectors reverses them and gives gold rank 139. With the exact original batch and dot product, both score `0.803173303604126` and lexical ID puts `d252` first, restoring archived gold rank 140. The full-corpus gold rank is 358, with 218 excluded distractors ahead of gold. Neither exception changes any gold top-50 hit.

**Training-only protocol.** Pin the dataset revision `f22508f96b7a36c2415181ed8bb76f76e04ae2d5`. Select every training query from AtCoder, CodeChef, Codeforces and HackerRank by normalized URL hostname, leaving Codewars and LeetCode outside the query panel. Codewars and LeetCode training snippets remain eligible distractors in the fixed 5,000-document training corpus. Validate that each query has exactly one positive training qrel. Preserve raw query text and MTEB passage formatting, the production E5 prefixes, batch size 16 and normalized float32 embeddings. Rank all 5,000 candidates with MTEB cosine, descending score and ascending lexical document ID on exact ties. Recall@K uses one gold-membership indicator per query and averages over the complete query group. The parser consumes full statements. Truncation counts the inputs whose E5 prefix and tokenizer special tokens place them above 512 tokens before truncation. The source rule includes all eligible queries independently of parser output or individual interface compatibility. [Training-only audit runner](../artifacts/dev_stdin_audit.py)

| Source | Queries / 1,746 | Recall@10 | Recall@50 | Recall@100 | Recall@200 | Parsed | Over 512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AtCoder | 61 (3.49%) | 15/61 (24.59%) | 22/61 (36.07%) | 31/61 (50.82%) | 41/61 (67.21%) | 0/61 (0%) | 19/61 (31.15%) |
| CodeChef | 1,112 (63.69%) | 233/1,112 (20.95%) | 370/1,112 (33.27%) | 477/1,112 (42.90%) | 594/1,112 (53.42%) | 37/1,112 (3.33%) | 464/1,112 (41.73%) |
| Codeforces | 477 (27.32%) | 58/477 (12.16%) | 140/477 (29.35%) | 192/477 (40.25%) | 245/477 (51.36%) | 474/477 (99.37%) | 314/477 (65.83%) |
| HackerRank | 96 (5.50%) | 68/96 (70.83%) | 79/96 (82.29%) | 84/96 (87.50%) | 88/96 (91.67%) | 0/96 (0%) | 21/96 (21.88%) |
| **All** | **1,746 (100%)** | **374/1,746 (21.42%)** | **611/1,746 (34.99%)** | **784/1,746 (44.90%)** | **968/1,746 (55.44%)** | **511/1,746 (29.27%)** | **818/1,746 (46.85%)** |

**Deterministic replay.** The separate cached scoring runs produced byte-identical ledgers for all 1,746 gold ranks and 349,200 ordered top-200 positions. Ledger SHA-256 was `e9a5f695da8a82106e62761024ac19ce05163263998da78d0b99a741f84e9b68`. Reversing document input order preserved the ranks. The ordered query ID SHA-256 was `c7453c957102b794c7ddf7c7fe56bcd709712650f7e019b4ee971dda7e932aff`. The independent replay guarded parquet reads to the training partition and confirmed that source, parser and length measurements match the prior training census on every selected ID. The query-vector manifest validates the model files, package versions, prefixed inputs, vector bytes, shape, finiteness and normalization. Both rank runs used the same validated vectors. The two historical witness files and the independent attribution repeat also matched byte for byte. All 39 regression tests passed. [Per-query measurements and configuration](../artifacts/dev_stdin_train.json) · [Replay fingerprints](../artifacts/dev_stdin_replay.json)

**INVALID PROXY for the historical full-test result.** Recall@50 is 34.99% versus the README's 31.29%, while parser coverage is 29.27% versus the historical 2,936/3,765 (77.98%), a 48.71-point gap. CodeChef and Codeforces make up 91.01% of `dev-stdin` and 511/511 of its parser hits. The historical test run used the full split and the saved evaluation description gives no source-level counts to establish a comparable mix. `dev-stdin` uses 5,000 training distractors. The notebook used 3,765 test distractors, selected one qrel by dictionary assignment without checking its relevance score, and gave tied scores the same rank via `1 + count(score > gold_score)`. The historical research log describes one binary relevant document per query. `dev-stdin` verifies that property using positive train qrels and assigns unique positions by lexical ID on ties. Median and p95 statement lengths are 1,644 and 3,110.75 characters, compared with the early research summary's 1,316 and 2,903. That summary lacks pinned split and tokenizer provenance, so the length comparison is contextual. The current E5 median is 495.5 formatted tokens. The verified panel is suitable for repeatable, source-wise stdin development measurements and has its own baseline. [Historical trace](#2026-11-25-example-parser-coverage-and-recall50-audit)

Reproduce with the pinned local snapshots and saved sample/dev query vectors. `attribute` uses the archived historical full-corpus evidence and opens only training parquet rows. The two `dev` commands run fresh scoring passes with the same validated inputs. The optional final command regenerates query vectors in a fresh directory using training queries only.

```powershell
.venv/Scripts/python.exe -B -X utf8 artifacts/dev_stdin_audit.py attribute artifacts/dev_stdin_attribution_repro.json
.venv/Scripts/python.exe -B -X utf8 artifacts/dev_stdin_audit.py dev artifacts/dev_stdin_train_repro.json --reference artifacts/dev_stdin_train.json
.venv/Scripts/python.exe -B -X utf8 artifacts/dev_stdin_audit.py dev artifacts/dev_stdin_train_repeat.json --reference artifacts/dev_stdin_train_repro.json
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
.venv/Scripts/python.exe -B -X utf8 artifacts/retrieval_consistency.py encode artifacts/dev_stdin_cold_vectors_repro
```

### 2026-11-25 Retrieval consistency replay and dev-stdin proxy

**Root cause.** Both archival runs used the same ordered 500 training queries and one positive training qrel per query. The 311 run searched all 8,765 candidates. The 345 run searched the 5,000 training candidates. A paired control froze the full-corpus cosine scores and the document-ID tie rule, then removed the 3,765 non-training candidates. Exactly the same 34 queries gained a top-50 hit, and all 311 original hits stayed in the top 50. For example, `q632` moved from rank 90 to 27 after 63 distractors ahead of its gold were removed. Every query and qrel read in the replay was from training. [Full-corpus ledger](../artifacts/retrieval_diagnostic_train.json) · [Training-only ablation ledger](../artifacts/query_example_baseline_train.json)

| Fixed 500 queries | Recall@10 | Recall@50 | Recall@100 | Recall@200 |
|---|---:|---:|---:|---:|
| 8,765 full-corpus candidates | 259/500 (51.8%) | 311/500 (62.2%) | 335/500 (67.0%) | 364/500 (72.8%) |
| 5,000 training candidates | 283/500 (56.6%) | 345/500 (69.0%) | 368/500 (73.6%) | 395/500 (79.0%) |

**Other protocol differences.** The ordered sample ID SHA-256 was `d463ba25ae4db0782402be587a550026ae9eef7c7922c64cdd1cb02c55a5a8c3`. Both query paths use verbatim statements, the same pinned E5-base-v2 checkpoint, `query: ` and `passage: ` prefixes, 512 tokens, and normalized embeddings. The full run used parquet corpus order, MTEB passage formatting and cosine, and stable descending score with MTEB's returned order for ties. The ablation used lexical corpus order, raw code, NumPy normalized dot product, and stable lexical document-ID ties. Passage formatting changed 1,710 of the 5,000 training cache keys' input strings. Of those, 524 vectors differed, with a maximum component difference of `2.66e-7`. Rescoring with either passage representation and either similarity path preserved the same top-50 gold IDs. Reversing candidate order preserved the deterministic rankings. Exact score ties at the 50th position appeared for one query in each controlled arm and did not change which gold IDs were retrieved.

**Archive fidelity and cache checks.** The unmodified MTEB search reproduced all 500 full-corpus top-50 lists and archived gold ranks at depth 1,000. Reusing the full-run query vectors for the ablation swapped two non-gold documents at ranks 42 and 43 for `q4336`, and placed the gold for `q4630` at 139 instead of 140. Re-encoding those queries in their original ablation batches reproduced both archived orders and ranks. Both batch effects were outside the gold top-50 boundary. The document cache matched the archived SHA-256, every required passage key was present, and none of the 500 prefixed queries was a hit in that cache. Re-encoding 16 fixed training passages with the pinned model gave a maximum absolute component difference of `2.64e-7` against cached vectors. Query inputs, model files, vector digests, dependencies, corpus IDs and the frozen input hashes are recorded with the replay. [Runner](../artifacts/retrieval_consistency.py) · [Reproduction commands](../artifacts/retrieval_consistency_run/README.md)

**dev-stdin census.** Selected every training query from four stdin/stdout sources by normalized hostname. Excluded 2,515 Codewars and 739 LeetCode queries. All 5,000 training documents remained candidates, including documents associated with the two excluded query sources. The dataset is `CoIR-Retrieval/apps@f22508f96b7a36c2415181ed8bb76f76e04ae2d5`. The model is `intfloat/e5-base-v2@f52bf8ec8c7124536f0efb74aca902b2995e5bcd`, CPU float32, batches of 16, four Torch intraop threads and one interop thread. The diagnostic uses production stage-1 cosine and MTEB passage formatting, then ranks score ties by lexical document ID. Recall is the share of all queries with their single positive training qrel in the first K positions. The existing parser sees the verbatim statement. Truncation means the prefixed input including special tokens exceeds 512 tokens before the model cap. [Raw rows, fixed IDs and exact configuration](../artifacts/retrieval_consistency_run/measurements.json)

| Source | Queries | Recall@50 | Parser coverage | Over 512 formatted tokens |
|---|---:|---:|---:|---:|
| AtCoder | 61 (3.49%) | 22/61 (36.07%) | 0/61 (0.00%) | 19/61 (31.15%) |
| CodeChef | 1,112 (63.69%) | 370/1,112 (33.27%) | 37/1,112 (3.33%) | 464/1,112 (41.73%) |
| Codeforces | 477 (27.32%) | 140/477 (29.35%) | 474/477 (99.37%) | 314/477 (65.83%) |
| HackerRank | 96 (5.50%) | 79/96 (82.29%) | 0/96 (0.00%) | 21/96 (21.88%) |
| **Total** | **1,746** | **611/1,746 (34.99%)** | **511/1,746 (29.27%)** | **818/1,746 (46.85%)** |

| Cutoff | dev-stdin | Documented historical test |
|---|---:|---:|
| Recall@10 | 374/1,746 (21.42%) | 19.04% |
| Recall@50 | 611/1,746 (34.99%) | 31.29% |
| Recall@100 | 784/1,746 (44.90%) | 38.57% |
| Recall@200 | 968/1,746 (55.44%) | 48.07% |

**Deterministic replay.** Two independent processes used the validated full query-vector archives and document cache. Both reproduced every ordered top-200 list and all metrics, and independently re-encoded the two historical query batches and the fixed training-passage probe. The complete measurement files were byte-identical with SHA-256 `8981a5054d2c9176352846a512da75e56ab357a59710b252bb22ce2fcdcc6c8d`. The `dev-stdin` ordered query ID SHA-256 is `c7453c957102b794c7ddf7c7fe56bcd709712650f7e019b4ee971dda7e932aff` and its ordered top-200 SHA-256 is `f9a2c8da7822f0bbe5a50d23ec47ecaade1e90a783d4a2644417a0b90cce9c75`. Production MTEB search agreed on every top-200 candidate set and every gold-hit flag at K=10, 50, 100 and 200. It ordered exact-score ties differently for 308 queries. All 39 regression tests passed. Measurements and reproduction commands use read-only production and frozen benchmark inputs. [Reproduction commands](../artifacts/retrieval_consistency_run/README.md)

**INVALID PROXY for the historical test evaluation.** The historical 31.29% came from the notebook's 3,765 test queries against 3,765 test candidates. The separate full MTEB benchmark used 8,765 candidates. Both historical and current protocols are effectively single-gold under the documented binary labels. The notebook kept one qrel by dictionary assignment, scored normalized vectors by dot product, and gave exact ties the same better rank with `1 + count(scores > gold_score)`. It used raw passages, batch size 64 on CUDA and unpinned revisions. The pinned development run filters for positive training qrels and assigns one position per candidate. These are different workloads even when an aggregate recall is close. [Historical stage-1 results](#results) · [Notebook methodology and parser audit](#2026-11-25-example-parser-coverage-and-recall50-audit)

The documented test parser coverage was 2,936/3,765 (77.98%), a 48.71-point difference from `dev-stdin`. The notebook measured regex matches, and the earlier documented audit found the same count with the retained parser. CodeChef and Codeforces make up 91.01% of `dev-stdin` and account for all 511 parser hits. The historical records give no per-source test counts for a matched comparison. The development median statement length was 1,644 characters and the p95 was 3,110.75, versus the early summary's 1,316 and 2,903. Formatted development input had a median of 495.5 tokens and p95 of 959. The development truncation rate was 46.85%, versus the historical estimate of about 27%, whose tokenizer and formatting details are unavailable. Within development, Recall@50 was 31.42% on the longer inputs and 38.15% on the rest. This is a descriptive difference across queries. The frozen panel supports paired, source-level development measurements on its own population. The test comparison uses existing documented measurements and methodology. [Historical data summary](#measurements) · [Historical parser audit](#2026-11-25-example-parser-coverage-and-recall50-audit)

#### Independent training-only completion checks

Re-encoded the ordered 500-query sample from the pinned CPU model. The new query-vector file was byte-identical to the archive. Verified that the two archived 8,765-document replays are byte-identical and regenerated the attribution with read-only training text, labels and candidate keys. It reproduced 311/500 in the archived full-candidate evidence and 345/500 with 5,000 training candidates, the same 34 crossings and zero losses. The new attribution file was byte-identical to its reference. The original two ablation batches recovered the archived order and gold-rank exceptions for `q4336` and `q4630`. No new scoring against the non-training corpus was needed. [Reproduced attribution](../artifacts/dev_stdin_validation/attribution_reproduced.json) · [Fresh sample vectors](../artifacts/dev_stdin_validation/sample_vectors.npz)

Two more independent train-only scoring passes reused the validated `dev-stdin` vectors and reproduced all 349,200 ordered top-200 positions, all metrics and parser and token counts. Both result files match the earlier standalone census byte for byte at SHA-256 `e9a5f695da8a82106e62761024ac19ce05163263998da78d0b99a741f84e9b68`. There were no `dev-stdin` query keys in the production document cache. The production MTEB stage-1 search returned the same top-200 candidate sets and gold-hit flags at all four cutoffs. For all 308 queries whose order differed, every changed position had an exactly equal cosine score. Exact ties crossed the four cutoffs on 2, 10, 8 and 3 queries, respectively. Assigning tied documents the same best rank under the notebook rule added no gold hits on this panel. The notebook's raw regex and the current nonempty-pair parser both found exactly 511 examples on `dev-stdin`. These controls leave the **INVALID PROXY** decision unchanged. All 39 regression tests passed. Frozen input hashes and production files were unchanged. [Two scoring passes](../artifacts/dev_stdin_validation/measurements.json) · [Repeat](../artifacts/dev_stdin_validation/measurements_repeat.json) · [Tie and cache checks](../artifacts/dev_stdin_validation/checks.json)

Reproduce from the repository root with new output filenames.

```powershell
.venv/Scripts/python.exe -B -X utf8 artifacts/dev_stdin_audit.py attribute artifacts/dev_stdin_validation/attribution_repro.json
.venv/Scripts/python.exe -B -X utf8 artifacts/dev_stdin_audit.py dev artifacts/dev_stdin_validation/measurements_repro.json --reference artifacts/dev_stdin_validation/measurements.json
.venv/Scripts/python.exe -B -X utf8 artifacts/dev_stdin_audit.py dev artifacts/dev_stdin_validation/measurements_repro_repeat.json --reference artifacts/dev_stdin_validation/measurements_repro.json
.venv/Scripts/python.exe -B -X utf8 artifacts/dev_stdin_validation_checks.py artifacts/dev_stdin_validation/measurements_repro.json artifacts/dev_stdin_validation/measurements_repro_repeat.json artifacts/dev_stdin_validation/checks_repro.json --vectors artifacts/retrieval_consistency_run --cold-sample artifacts/dev_stdin_validation/sample_vectors.npz
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

### 2026-11-25 Full-corpus dev-stdin baseline and preregistration check

**Historical corpus.** The 31.29% Recall@50 is from `notebook/baseline_recall_and_execution_validation.ipynb`. Code cell indexes are zero-based. Cell 5 filters the corpus with `corpus_t = corpus[corpus.partition == "test"]` and prints 3,765 candidates and 3,765 queries. Cell 9 passes `corpus_t.text` with the `passage: ` prefix into the encoder, and its saved output shows both matrices at `(3765, 768)`. Cell 11 computes `sim = q_emb @ c_emb.T`, ranks each gold as `1 + count(similarity > gold_score)`, and its saved output reports 31.29%. The notebook matches its last tracked bytes at `9a9d20e^`. The separate `evaluate.py` path calls MTEB's AppsRetrieval task and its documented benchmark uses 8,765 candidates. This check read notebook source and archived aggregate evidence. It did not execute the notebook or open test queries or test relevance labels. [Notebook filtering](../notebook/baseline_recall_and_execution_validation.ipynb#L476) · [Notebook metric output](../notebook/baseline_recall_and_execution_validation.ipynb#L987) · [MTEB entry point](../evaluate.py#L67)

**Split saved before comparisons.** Used the fixed 1,746-query training census on APPS revision `f22508f96b7a36c2415181ed8bb76f76e04ae2d5`. Within each hostname, sort IDs and shuffle with `random.Random(f"{20261125}:{source}")`. Assign each source's first half to A, alternating the extra member of odd strata to A then B in hostname order. Membership was determined from query ID and source alone. The split is disjoint, exhaustive and exactly balanced. No per-half retrieval result was used for selection. [Split IDs, source counts and fingerprints](../artifacts/dev_stdin_ab_split.json)

| Source | A | B |
|---|---:|---:|
| AtCoder | 31 | 30 |
| CodeChef | 556 | 556 |
| Codeforces | 238 | 239 |
| HackerRank | 48 | 48 |
| **Total** | **873** | **873** |

**Full-corpus stage-1 baseline.** Reused validated original query embeddings from pinned E5-base-v2 revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`. The embeddings use the verbatim statement, `query: `, the production tokenizer's 512-token cap, batch size 16, CPU float32 and L2 normalization. All 8,765 MTEB-formatted `passage: ` inputs resolved from the same read-only content-addressed document cache. Ranked all candidates by MTEB cosine with lexical document ID on exact score ties. Recall uses a single positive training gold per query. NDCG@10 is the mean of `1/log2(rank+1)` through rank 10 and zero thereafter, on the 0–100 scale. Both independent runs and a reversed-corpus-order control reproduced the deterministic rankings. The two saved outputs have the same SHA-256 `0fae2558c285aa96c6e154ebf2f5b5fdc862b87cfb11f52f6dba3d5e4f7feb07`. Unmodified MTEB search agreed on all gold-hit flags and aggregate NDCG@10, and returned the same top-200 membership for all 1,746 queries. Its order matched on 1,565 queries. The fixed lexical tie rule defines this baseline. [Full raw ranks and configuration](../artifacts/dev_stdin_full_baseline.json) · [Independent replay](../artifacts/dev_stdin_full_baseline_repeat.json)

| Metric | Result over 1,746 training queries |
|---|---:|
| Recall@10 | 293/1,746 (16.78%) |
| Recall@50 | 461/1,746 (26.40%) |
| Recall@100 | 578/1,746 (33.10%) |
| Recall@200 | 722/1,746 (41.35%) |
| NDCG@10 | 12.30 |

The earlier 5,000-candidate panel uses the exact same query-vector manifest and gives 611 top-50 golds. All 461 full-pool hits are among those, with 150 additional hits on the smaller pool. This is a comparison of candidate pools under the unchanged query representation. [Prior training-corpus ranks](../artifacts/dev_stdin_train.json)

**Preregistration gap and decision.** The available repository instructions, research artifacts and tracked history supplied a next-experiment note for current head truncation versus an unspecified fixed head-plus-tail allocation at the same 512-token budget. That note gives neither an exact allocation nor a third arm or decision thresholds. The separate example-removal ablation recorded a gain of at least 1.0 Recall@50 point and a paired interval above zero on its original random panel, with CPU and downstream criteria tied to that study. It did not register criteria for this split. The exact three input recipes and their parameters, the paired inference and acceptance rules, and how to select and confirm a winner could not be established. No representation comparison, source effect estimate, CPU overhead estimate, or winner selection was produced on A. B was left for confirmation after the required unchanged full-panel baseline. **REJECT adoption on the available evidence.** No representation has been tested against the requested rules. All 39 regression tests passed. Production code, caches and frozen benchmark files were unchanged.

Reproduce with the existing pinned vectors and a new output filename for each step.

```powershell
.venv/Scripts/python.exe -B -X utf8 artifacts/dev_stdin_full_baseline.py split artifacts/dev_stdin_ab_split_repro.json --reference artifacts/dev_stdin_ab_split.json
.venv/Scripts/python.exe -B -X utf8 artifacts/dev_stdin_full_baseline.py baseline artifacts/dev_stdin_full_baseline_repro.json --split artifacts/dev_stdin_ab_split_repro.json --reference artifacts/dev_stdin_full_baseline.json
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

### Query representation preregistration

Registered at **2026-11-25 02:22:06 +05:30** before any scoring for this experiment. Research only. No production changes or commit.

- **Data.** Use all 1,746 `dev-stdin` training queries, the full 8,765-candidate pool and the saved source-stratified A/B split with seed `20261125`. Keep the document embeddings, positive training qrels, MTEB float32 cosine and lexical document-ID tie rule fixed. The primary analysis is the subset exceeding 512 formatted tokens, including the E5 prefix and special tokens. Choose on A only. Use B once for the locked winner only.
- **C0 current.** First 512 formatted tokens with the unchanged E5 truncation.
- **V1 head+tail.** First 320 text tokens plus the final `budget - 320` text tokens, maintaining their order. Reserve the original `query: ` prefix and tokenizer special tokens within the same total budget of 512. Use the original token IDs without a decode/re-encode round trip.
- **V2 windows.** Use a maximum of 512 formatted tokens per window with the same `query: ` prefix and special tokens. Start at text offsets 0, 256, 512 and so on, ending when the last window covers the final text token. The final window may be shorter. Compute each document score as maximum cosine across the windows. The first window has the same IDs as C0 and can reuse its validated vector for ranking.
- Queries at or below the cap reuse their unchanged C0 rankings in both variants. Full-half metrics include them. The variant-only scores and top-200 orders are saved separately from C0.

Keep a variant only if all gates hold:

1. Truncated A Recall@50 increases by at least **3.0 percentage points** over C0. Its two-sided 95% paired percentile bootstrap interval excludes zero, with **10,000** query-level resamples and seed `20261125`.
2. The locked variant has the same positive sign on truncated B and improves Recall@50 by at least **1.5 percentage points**.
3. Full-half NDCG@10 does not drop on either evaluated half.
4. No source with at least **100 truncated queries** in the evaluated half loses more than **2.0 Recall@50 percentage points**. Report full-half source metrics as well.
5. Query-side latency increases by no more than **0.1 seconds/query** on this CPU in each evaluated half. Measure the mean over a fixed selection of 32 truncated queries from that half, sampled from sorted IDs with `random.Random(f"20261125:latency:{half}")`. Use the pinned resident CPU model, four Torch threads, one interop thread, batch size 16 and no query-vector cache in the timed arms. Include query formatting, encoding, document cosine and window aggregation. Exclude corpus/cache loading and final ranking. Time two matched batches with the arm order reversed for the second batch. V2 recomputes every window in the timed arm.

Apply the available A gates before opening B. If both variants pass on A, lock the higher A Recall@50. If both pass and their A Recall@50 is exactly equal, the written rule gives no winner and B stays closed pending a tie rule. Report Recall@10/50/100/200, per-source results, plus per-query gold-rank and cutoff transitions. Record **REJECTED** if neither variant qualifies or if the locked variant fails confirmation. No fallback variant is tested on B.

### Query representation experiment results

**REJECTED on A.** V1 failed the preregistered Recall@50 gate. V2 passed all three retrieval and ranking gates and failed the 0.100-second/query latency limit. No winner was locked and no representation was evaluated on B. [Decision and raw timing batches](../artifacts/dev_stdin_query_repr_run/A_results.json)

The A half has 873 queries, 412 over the formatted token limit. The other 461 use their archived C0 results in all arms. The 512-token budget reserves two prefix and two BERT special tokens, leaving 508 for text. V1 used 320 head and 188 tail tokens. V2 used starts spaced 256 text tokens apart and combined the 412 original vectors with 583 newly encoded windows. A compact C0 cosine calculation initially reordered three top-200 lists with no changes to gold ranks. Before evaluating variants, scoring was aligned with the archived matrix shape, which recovered all 412 C0 lists exactly. Shape padding used copies of an A vector. The token-ID encoder matched production C0 IDs and produced byte-identical embeddings on four calibration queries. [Runner](../artifacts/dev_stdin_query_repr.py) · [Recorded calibration and input digests](../artifacts/dev_stdin_query_repr_run/A_results.json)

| Truncated A, n=412 | Recall@10 | Recall@50 | Recall@100 | Recall@200 | Recall@50 delta and 95% paired CI |
|---|---:|---:|---:|---:|---:|
| C0 | 63 (15.29%) | 96 (23.30%) | 120 (29.13%) | 158 (38.35%) | — |
| V1 head+tail | 60 (14.56%) | 97 (23.54%) | 118 (28.64%) | 154 (37.38%) | +0.24 pp [−1.70, +2.18] |
| V2 windows | 71 (17.23%) | 117 (28.40%) | 139 (33.74%) | 170 (41.26%) | +5.10 pp [+2.43, +8.01] |

| Full A, n=873 | Recall@10 hits | Recall@50 hits | Recall@100 hits | Recall@200 hits | NDCG@10 (0–100) |
|---|---:|---:|---:|---:|---:|
| C0 | 151 | 229 | 278 | 351 | 12.487 |
| V1 | 148 | 230 | 276 | 347 | 12.507 |
| V2 | 159 | 250 | 297 | 363 | 13.254 |

| Source, truncated A | Queries | C0 top-50 hits | V1 hits (delta pp) | V2 hits (delta pp) |
|---|---:|---:|---:|---:|
| AtCoder | 6 | 1 | 1 (0.00) | 1 (0.00) |
| CodeChef | 239 | 61 | 64 (+1.26) | 72 (+4.60) |
| Codeforces | 155 | 27 | 25 (−1.29) | 36 (+5.81) |
| HackerRank | 12 | 7 | 7 (0.00) | 8 (+8.33) |

V1 improved 177 gold ranks, worsened 196 and left 39 unchanged on truncated A. Eight queries entered the top 50 and seven left. V2 improved 198 gold ranks, worsened 175 and left 39 unchanged. Twenty-eight entered the top 50 and seven left. The recorded metrics include both source scopes and query IDs for every transition at 10, 50, 100 and 200. [V1 per-query ranks](../artifacts/dev_stdin_query_repr_run/A_V1_rankings.json) · [V2 per-query ranks](../artifacts/dev_stdin_query_repr_run/A_V2_rankings.json)

The mean query-side wall times over the registered 32-query CPU control on the Intel Core i5-13420H were **1.296 s/query for C0**, **1.251 for V1** (−0.046 s) and **2.812 for V2** (+1.516 s). The two batches processed 32 total windows each for C0 and V1 and 79 for V2. Ranking reused V2's validated first-window vector, while the timing control recomputed all windows to measure a cold query. V2's favorable A interval is exploratory evidence. There is no B confirmation. [Per-batch times and sampled IDs](../artifacts/dev_stdin_query_repr_run/A_results.json)

An independent score replay from the saved embeddings matched all 412 gold ranks and ordered top-200 lists for each arm. All 39 repository regression tests passed. The experiment used training queries and relevance labels, and its changes are confined to research artifacts and this log. [Score replay](../artifacts/dev_stdin_query_repr_run/A_replay.json)

Reproduce treatment scoring with a fresh output directory. Only the A command applies to this decision. The saved split, C0 vectors and document cache retain their pinned identities.

```powershell
.venv/Scripts/python.exe -B -X utf8 artifacts/dev_stdin_query_repr.py A artifacts/dev_stdin_query_repr_repro
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

### Query representation latency amendment

Registered at **2026-11-25 03:01:52 +05:30**, before any experimental work on B for this study.

> AMENDMENT (latency rule only): rule 5 becomes: mean added query-side latency of V2 over ALL dev-stdin queries <= 0.3 s/query vs C0, measured as below. Rules 1-4 unchanged. B unseen at time of amendment.

