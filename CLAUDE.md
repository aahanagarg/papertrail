# PaperTrail

RAG question-answering over research papers (QASPER), with a reliability
benchmark: retrieval recall, answer-F1, hallucination rate on unanswerable
questions, and retrieval-vs-generation failure decomposition. See PLAN.md
for the full project plan.

## Stack

- Python 3.11
- Dependency management: `uv` (never `pip install` directly — edit
  `pyproject.toml` and run `uv sync`, or `uv add <package>`)
- Tests: `pytest`, run with `uv run pytest`

## Layout

- `src/` — library code (data loading, retrieval, generation, evaluation)
- `scripts/` — one-off experiments and analysis scripts, run to produce
  results
- `results/` — CSVs and other output artifacts produced by scripts
- `tests/` — pytest tests

## Data decisions (QASPER)

- **Unanswerable definition:** `unanswerable` is true only when *all*
  annotators agree the question is unanswerable (the strict/unanimous
  definition). A separate field, `unanswerable_any`, records the looser
  definition (at least one annotator said unanswerable) so sensitivity to
  this choice can be reported later. On the validation split: 60 unanimous
  vs. 135 any-annotator (out of 1005 questions).
- **Retrieval scoring exclusion:** a small number of answerable questions
  end up with an empty `evidence_idx` after matching (evidence text that
  doesn't correspond to any paragraph — e.g. only `FLOAT SELECTED` spans, or
  text that doesn't match under exact/whitespace-normalized comparison).
  These are excluded from retrieval recall scoring. Each record carries a
  `scorable_retrieval` boolean (`True` iff not unanswerable and
  `evidence_idx` is non-empty); recall@k must only be computed over records
  where `scorable_retrieval` is `True`.

## Working agreement

- After any experiment, always print the headline numbers AND 5 concrete
  example cases including failures, with raw text, so results can be
  sanity-checked by eye.
- Never describe a metric as "high", "good", or "well above threshold" —
  always print the number.
- Before starting a step, state what range of result is expected and why.
  Flag it if the actual result is outside that range instead of accepting
  it silently.
- Never write a number into README.md by hand; read it from `results/`.
- Never add Co-Authored-By trailers, "Generated with Claude Code" lines, or
  any attribution footer to commit messages. Commit messages should contain
  only the conventional commit message itself.

## Hard rule: no fabricated numbers

**Never fabricate, hardcode, or estimate a metric.** Every number that
appears in `results/` must be the output of a script that ran against real
data (the actual QASPER dataset, actual model outputs, etc.). If a number
isn't available yet, say so explicitly — do not fill in a plausible-looking
placeholder. This applies to answer-F1, recall@k, hallucination rate,
latency, cost, and every other metric in this project.
