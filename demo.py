"""Interactive demo: paste a problem statement, get ranked Python solutions.

  python demo.py        then open http://127.0.0.1:7860

If a version history exists (python versions.py build), the page can search
one version or all of them at once.
"""
from __future__ import annotations

import html
import json
import os
import random
import time

import numpy as np

from src.config import DEFAULT_K, VERSIONS_DIR
from src.data import load_apps
from src.encoder import E5
from src.evolution import History, rank_verified
from src.execute import parse_example, verify
from src.versions import VersionStore

SHOW = 10          # results rendered on the page
CODE_LINES = 40    # lines of code shown per result
ALL = "All versions"

print("loading dataset and model...")
APPS = load_apps()
E = E5()
QID_BY_TEXT = {t.strip(): q for q, t in zip(APPS.query_ids, APPS.query_texts)}

# The searchable pool: every version's entries, or just the original corpus.
STORE = VersionStore(VERSIONS_DIR)
if STORE.exists():
    HIST = History(STORE, E)
    POOL_IDS = [e.doc_id for e in HIST.entries]
    POOL_TEXT = [e.text for e in HIST.entries]
    POOL_VERS = [e.versions for e in HIST.entries]
    POOL_EMB = HIST.emb
    SCOPES = {v: HIST.rows[v] for v in reversed(HIST.names)}
    SCOPES[ALL] = np.arange(len(POOL_IDS))
    BUGS = json.loads((VERSIONS_DIR / "bugs.json").read_text())
    print(f"versions: {', '.join(HIST.names)}, {len(POOL_IDS)} distinct entries, "
          f"{HIST.encoded} encoded now, {HIST.build_seconds:.0f}s")
else:
    HIST, BUGS = None, None
    POOL_IDS, POOL_TEXT = APPS.doc_ids, APPS.doc_texts
    POOL_VERS = [["original"]] * len(POOL_IDS)
    POOL_EMB, n_new = E.encode_docs(APPS.doc_texts)
    SCOPES = {"Original": np.arange(len(POOL_IDS))}
    if n_new:
        E.save()
    print(f"index: {len(POOL_IDS)} snippets, no version history")
print(f"device {E.device}")

_VERDICTS: dict[tuple[int, str], bool] = {}   # (entry, example) -> passed

VERDICT_TEXT = {
    "passed": "Produced the expected output",
    "failed": "Wrong output or crashed",
    "skipped": "Not run",
}


def _is_buggy(i: int) -> bool:
    return bool(BUGS) and BUGS[POOL_VERS[i][0]].get(POOL_IDS[i], False)


def _card(rank, i, score, verdict, was, gold_id) -> str:
    lines = POOL_TEXT[i].splitlines()
    code = "\n".join(lines[:CODE_LINES])
    if len(lines) > CODE_LINES:
        code += f"\n# ... {len(lines) - CODE_LINES} more lines"
    tags = []
    if was and was != rank:
        tags.append(f'<span class="moved">was {was}</span>')
    if POOL_IDS[i] == gold_id:
        tags.append('<span class="gold">Known answer</span>')
    if _is_buggy(i):
        tags.append('<span class="bug">Injected bug</span>')
    vers = ", ".join(POOL_VERS[i]) if HIST else ""
    where = f"{html.escape(POOL_IDS[i])}" + (f" in {vers}" if vers else "")
    return (
        f'<article class="hit {verdict}">'
        f'<header><span class="rank">{rank}</span>'
        f'<span class="verdict">{VERDICT_TEXT[verdict]}</span>{"".join(tags)}'
        f'<span class="meta">{where}, similarity {score:.3f}</span></header>'
        f'<pre><code>{html.escape(code)}</code></pre></article>'
    )


def search(query: str, scope: str, k: int, do_verify: bool):
    query = (query or "").strip()
    if not query:
        return "Paste a problem statement, or load one from the test set.", ""
    k = int(k)
    rows = SCOPES[scope]

    t = time.perf_counter()
    q_emb, _ = E.encode_queries([query])
    t_encode = time.perf_counter() - t

    t = time.perf_counter()
    sims = POOL_EMB[rows] @ q_emb[0]
    order = [int(rows[j]) for j in np.argsort(-sims)]
    score = {int(rows[j]): float(sims[j]) for j in range(len(rows))}
    t_search = time.perf_counter() - t

    cand = order[:k]
    before = {i: r for r, i in enumerate(cand, 1)}
    verdicts, t_verify, note, reused = {}, 0.0, "", 0
    example = parse_example(query) if do_verify else None
    if do_verify and example is None:
        note = "No example input and output found in the query, so results keep the similarity order."
    if example:
        ex_key = example[0] + "\x00" + example[1]
        todo = [i for i in cand if (i, ex_key) not in _VERDICTS]
        reused = len(cand) - len(todo)
        t = time.perf_counter()
        for i, ok in zip(todo, verify([POOL_TEXT[i] for i in todo], example)):
            _VERDICTS[(i, ex_key)] = ok
        t_verify = time.perf_counter() - t
        passed = {i: _VERDICTS[(i, ex_key)] for i in cand}
        verdicts = {i: ("passed" if ok else "failed") for i, ok in passed.items()}
        ranked = rank_verified(cand, score, passed)
    else:
        ranked = cand

    gold_id, gold_line = None, ""
    qid = QID_BY_TEXT.get(query)
    if qid and qid in APPS.gold:
        gold_id = APPS.gold[qid]
        fam_before = [r for r, i in enumerate(order, 1) if POOL_IDS[i] == gold_id]
        fam_after = [r for r, i in enumerate(ranked, 1) if POOL_IDS[i] == gold_id]
        if fam_before:
            after = fam_after[0] if fam_after else fam_before[0]
            gold_line = (f"| Known answer | rank {fam_before[0]} by similarity, "
                         f"rank {after} after running |\n")

    n_pass = sum(v == "passed" for v in verdicts.values())
    stats = (
        "| | |\n|---|---|\n"
        f"| Encode query | {t_encode*1000:.0f} ms |\n"
        f"| Search {len(rows):,} snippets | {t_search*1000:.1f} ms |\n"
        + (f"| Run top {k} on the example | {t_verify:.2f} s, {n_pass} produced the expected output"
           + (f", {reused} reused from earlier runs" if reused else "") + " |\n"
           if example else "")
        + gold_line
    )
    if note:
        stats += f"\n{note}"

    cards = "".join(
        _card(r, i, score[i], verdicts.get(i, "skipped"), before.get(i), gold_id)
        for r, i in enumerate(ranked[:SHOW], 1)
    )
    return stats, cards


def load_example(idx):
    i = int(idx or 0) % len(APPS.query_texts)
    return APPS.query_texts[i]


def random_example():
    i = random.randrange(len(APPS.query_texts))
    return i, APPS.query_texts[i]


CSS = """
.hit { border-left: 3px solid #94A3B8; padding: 0.6rem 0.9rem; margin: 0 0 1rem;
       background: #FFFFFF; }
.hit.passed { border-left-color: #0F766E; }
.hit.failed { border-left-color: #CBD5E1; }
.hit header { display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: baseline;
              margin-bottom: 0.4rem; font-size: 0.92rem; }
.hit .rank { font-weight: 600; font-size: 1.15rem; color: #1E2A32; min-width: 1.6rem; }
.hit.passed .verdict { color: #0F766E; font-weight: 600; }
.hit.failed .verdict, .hit.skipped .verdict { color: #64748B; }
.hit .moved { color: #64748B; }
.hit .gold { background: #FEF3C7; color: #78350F; padding: 0 0.4rem; border-radius: 3px; }
.hit .bug { background: #FEE2E2; color: #7F1D1D; padding: 0 0.4rem; border-radius: 3px; }
.hit .meta { color: #94A3B8; margin-left: auto; }
.hit pre { margin: 0; max-height: 22rem; overflow: auto; font-size: 0.82rem;
           background: #F8FAFC; padding: 0.6rem; }
"""


def build_ui():
    import gradio as gr

    theme = gr.themes.Base(
        primary_hue="teal", neutral_hue="slate",
        font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    )
    scopes = list(SCOPES)
    default_scope = scopes[-2] if HIST else scopes[0]   # the original version
    with gr.Blocks(title="Find the code that solves it") as app:
        gr.Markdown(
            "# Find the code that solves it\n"
            "Paste a programming problem. Python solutions are ranked by meaning first, "
            "then the top candidates are run on the problem's own example, and any that "
            "print the expected output move to the top."
        )
        with gr.Row():
            with gr.Column(scale=5):
                query = gr.Textbox(label="Problem statement", lines=16,
                                   placeholder="Paste a problem, including its example input and output")
                with gr.Row():
                    idx = gr.Number(label="Test set problem", value=0, precision=0, minimum=0,
                                    maximum=len(APPS.query_texts) - 1)
                    load_btn = gr.Button("Load problem")
                    rand_btn = gr.Button("Load a random problem")
                scope = gr.Radio(scopes, value=default_scope, label="Search in",
                                 visible=bool(HIST))
                k = gr.Slider(10, 100, value=DEFAULT_K, step=10, label="Candidates to run")
                do_verify = gr.Checkbox(value=True, label="Run candidates on the example")
                run_btn = gr.Button("Find solutions", variant="primary")
                stats = gr.Markdown()
            with gr.Column(scale=7):
                results = gr.HTML()

        load_btn.click(load_example, [idx], [query])
        rand_btn.click(random_example, None, [idx, query])
        run_btn.click(search, [query, scope, k, do_verify], [stats, results])
    return app, theme


if __name__ == "__main__":
    app, theme = build_ui()
    app.launch(server_name=os.environ.get("HOST", "127.0.0.1"), server_port=7860,
               theme=theme, css=CSS)
