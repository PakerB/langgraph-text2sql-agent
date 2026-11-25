# langgraph-text2sql-agent

A text-to-SQL agent built with [LangGraph](https://github.com/langchain-ai/langgraph) and evaluated on **BIRD Mini-Dev (2026 update, SQLite)**: 500 questions over 11 real-world databases, scored by execution accuracy (EX).

The agent starts from BIRD's official single-call baseline (same prompt template, same scoring) and adds the pieces that the baseline lacks: an enriched schema, database exploration through a tool, and an execution check with revision. Comparing the two on the same model isolates what the agent adds.

## Why an agent

The 2026 update of BIRD Mini-Dev removed evidence that gave away column names, query plans or answers. As a result, a model has to explore the database to learn how values are actually stored (for example `'+'` vs `'carcinogenic'`, numbers stored as text, lists stored in a single cell). A single model call cannot look at the data, but an agent can.

## Design

```
START -> prepare_prompt -> agent <-> tools -> check_final_sql -> submit -> END
                             ^                      |
                             +----- revise ---------+
```

| Step | Type | What it does |
|---|---|---|
| `prepare_prompt` | workflow | Official BIRD prompt with an **enriched schema** (types, keys, column descriptions, stored values, value meanings), the question and its evidence, plus three extra instructions |
| `agent` | LLM | Reasons step by step and decides whether to call `run_sql` (at most 6 calls per question) |
| `tools` | tool | `run_sql`: read-only SQLite, 20 rows, 100 chars per cell, 10 s timeout |
| `check_final_sql` | workflow | Runs the final SQL (30 s timeout). On an error, an empty result or an all-NULL row, it asks for a revision (at most 3 rounds). Resubmitting the same SQL is accepted, because an empty result can be correct |
| `submit` | workflow | Returns the final SQL, which is what EX scores |

Design choices, mapped to known failure modes of text-to-SQL:

| Failure mode | Choice |
|---|---|
| Schema linking (BIRD's largest error class) | Full schema in the prompt. The databases have only 3 to 13 tables, so no schema pruning is needed |
| Cryptic column names | Full names and descriptions from BIRD's `database_description` files, inlined next to each column |
| Unknown stored values | All values for columns with at most 10 distinct values, otherwise the 3 most frequent. The agent can query anything else with `run_sql` |
| Evidence taken literally (e.g. `DIVIDE(...)`) | Instruction that evidence describes meaning, not SQL syntax |
| Broken or empty SQL | Execution check with revision |
| Output contract (exact columns, rounding, ties) | BIRD's official instructions, kept unchanged |
| Common SQL mistakes (join fan-out, NULL ordering, "latest non-NULL value", keeping rows with no matches) | Optional generic checklist (`--pitfalls`). It contains no table, column or question specific content |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env              # then add your GEMINI_API_KEY
python -m text2sql_agent.download_data   # BIRD Mini-Dev SQLite (~800 MB download)
python -m text2sql_agent.build_schema    # enriched schemas -> schemas/
```

The default model is `gemini-3.5-flash-lite` through `langchain-google-genai`. The native Gemini client is used rather than the OpenAI-compatible endpoint, because tool calling needs Gemini's thought signatures to be passed back on the next turn.

## Usage

```bash
# Official baseline (1 call per question) and the agent on the same 100 questions,
# disjoint from the 80-question development sample used for error analysis
python -m text2sql_agent.run --mode baseline --n 100 --exclude-n 80
python -m text2sql_agent.run --mode agent    --n 100 --exclude-n 80
python -m text2sql_agent.run --mode agent    --n 100 --exclude-n 80 --pitfalls

# Score with the official EX rule
python -m text2sql_agent.evaluate runs/agent__gemini-3.5-flash-lite --n 100 --exclude-n 80
```

Runs are resumable: questions already present in `predictions.jsonl` are skipped, so re-running a command after a rate-limit stop continues where it left off.

## Repository layout

```
text2sql_agent/
  common.py         paths, limits, safe read-only SQL execution with timeout, SQL extraction, model setup
  build_schema.py   enriched schema builder
  baseline.py       official BIRD single-call baseline
  agent.py          LangGraph agent
  run.py            batch runner (sampling, resumable output, token accounting)
  evaluate.py       EX scoring (set equality of result rows, as in BIRD's evaluation_ex.py)
  download_data.py  BIRD Mini-Dev download
```

## Results

Coming soon: baseline vs. agent EX on a held-out sample, with an ablation of the pitfalls checklist, cost per question, and an error analysis.

## References and data

- BIRD: Li et al., *Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs*, NeurIPS 2023. Mini-Dev data: [birdsql/bird_mini_dev](https://huggingface.co/datasets/birdsql/bird_mini_dev), CC BY-SA 4.0. The data is not included in this repository.
- CHESS: Talaei et al., *CHESS: Contextual Harnessing for Efficient SQL Synthesis*, 2024. This work informed the choice of database exploration and execution-based revision.

## License

MIT
