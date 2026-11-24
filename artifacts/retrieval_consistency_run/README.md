# CtrlFind retrieval consistency replay

The reference measurements contain the full `dev-stdin` census, both historical 500-query arms and the corpus-only control. Each query has its gold ID and rank, ordered top 200, cutoff scores and exact cutoff-tie flags. The census also records each source, parser outcome, character and token counts, and original and formatted query fingerprints. The configuration records exact corpus IDs, model and cache fingerprints, and the input versions.

## Reproduce from the repository root

Use the existing `.venv`, `artifacts/emb_cache.npz`, and the local Hugging Face snapshots for `CoIR-Retrieval/apps@f22508f96b7a36c2415181ed8bb76f76e04ae2d5` and `intfloat/e5-base-v2@f52bf8ec8c7124536f0efb74aca902b2995e5bcd`. The two historical training ledgers remain inputs. Use a fresh output filename for each replay.

```powershell
.venv/Scripts/python.exe -B -X utf8 artifacts/retrieval_consistency.py replay artifacts/retrieval_consistency_repro.json --vectors artifacts/retrieval_consistency_run --reference artifacts/retrieval_consistency_run/measurements.json
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

To regenerate the query vectors from the pinned CPU model, use a fresh directory. Full encoding takes substantially longer than replay. The encode step first processes the ordered 500-query sample as one E5 call and then processes the source-filtered census, reusing exact content hits from the sample. Each call uses the existing E5 encoder and batch size 16.

```powershell
.venv/Scripts/python.exe -B -X utf8 artifacts/retrieval_consistency.py encode artifacts/retrieval_consistency_cold_repro
.venv/Scripts/python.exe -B -X utf8 artifacts/retrieval_consistency.py replay artifacts/retrieval_consistency_cold_repro/measurements.json --vectors artifacts/retrieval_consistency_cold_repro --reference artifacts/retrieval_consistency_run/measurements.json
```

`--historical-only` produces the discrepancy evidence with just `sample_vectors.npz`. Complete replay checks the ordered query IDs, query text and prefix digest, model files, package versions, shape, dtype, norms and vector fingerprint. It validates the pinned historical document cache and re-encodes 16 fixed training documents as a numerical check. It re-encodes the original ablation batches for the two archival rank differences. `--reference` compares the complete output byte for byte, including both historical control rankings and the census top 200. Production code and cache files are read-only inputs. The full-corpus arm adds 3,765 non-training document vectors as distractors for the 500 training queries. All query and qrel reads are from training.

## Ranking and coverage

The historical full-corpus replay uses the original MTEB search with the archived corpus order. The ablation replay uses raw code and NumPy normalized dot product with lexicographic document-ID tie order. The corpus-only control freezes the full-corpus cosine scores, changes corpus membership, and applies the same tie rule to both arms.

The census includes every training query from AtCoder, CodeChef, Codeforces and HackerRank, selected by URL hostname. It uses all 5,000 training documents as candidates and one positive training qrel per query. It embeds verbatim statements with `query: ` and passages with `passage: `, MTEB passage formatting, 512 tokens and L2 normalization. It scores with the production MTEB float32 cosine function and ranks exact ties by ascending lexical document ID. The same validated inputs are separately checked with unmodified MTEB search at depth 1,000. Recall is the fraction of queries with the single labelled document in the first K positions. Parser coverage is the fraction for which `src.execute.parse_example` returns a pair from the full verbatim statement. The truncation count uses the formatted E5 input, including the prefix and special tokens, before the 512-token cap.
