# langgraph-text2sql-agent

A text-to-SQL agent built with [LangGraph](https://github.com/langchain-ai/langgraph) and evaluated on **BIRD Mini-Dev (2026 update, SQLite)**: 500 questions over 11 real-world databases, scored by execution accuracy (EX).

The agent starts from BIRD's official single-call baseline (same prompt template, same scoring) and adds the pieces that the baseline lacks: an enriched schema, database exploration through a tool, and an execution check with revision. Comparing the two on the same model isolates what the agent adds.

- **v1**: one agent run per question.
- **v2**: three agent runs with different reasoning strategies (`direct`, `grain`, `decompose`), then one answer is picked with CHESS-style unit tests. See [v2](#v2-multi-strategy-generation--unit-test-selection).

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

## v2: multi-strategy generation + unit-test selection

v1 runs the agent once per question. v2 runs it three times with **different reasoning strategies**, then picks one answer with **unit tests**. The strategies are meant to fail on different questions, so that at least one candidate is often right; the selector then has to find it.

| Strategy | What the model does before writing SQL | List-column annotation |
|---|---|---|
| `direct` | Nothing extra: explores the database with `run_sql` as needed, then answers | **no** (see note) |
| `grain` | Writes short notes on the counting unit: what is counted, which columns hold it, whether one stored row or cell is exactly one such unit | yes |
| `decompose` | Splits the question into sub-questions, drafts and checks SQL for each, then combines them (after the divide-and-conquer generator of CHASE-SQL) | yes |

All three use the pitfalls checklist. The **list-column annotation** marks columns whose cells hold several items (e.g. `'Pizza, Water, Soda'`) and tells the model to split them before counting or matching. Candidate columns are found by code (share of values containing `,` `;` `|`) and confirmed by one LLM call per database (`list_columns.py`), because code alone cannot tell a list from a single value such as `'Corvallis, OR'`.

> **Note.** `direct` runs **without** the list-column annotation. It is kept exactly as it was measured: `direct` is the v1 agent with the pitfalls checklist, and the gain from selection was measured with this pool.

**Selection** (`select_unit_tests.py`, after the Unit Tester of CHESS):
1. Execute every candidate and group candidates with the same result set. If there is only one group, take it (no model call; about 75% of questions).
2. Otherwise one model call writes 5 natural-language unit tests for the question, given the schema and the candidates with their results.
3. One more call checks every candidate against every test. The candidate passing the most tests wins (ties: more votes, then strategy order).

The selector uses DeepSeek in thinking mode (no tools are called there). Differences from the CHESS paper: 5 tests instead of 20; tests may check output shape and values, not only query logic; one evaluation call for all candidates instead of one call per test; candidates with identical results are merged first.

### Development results

All design choices were made on a **development set of 180 questions** (`--n 80` plus `--n 100 --exclude-n 80`). The other 320 questions are held out for one final run, reported under [Results](#results). Model: DeepSeek-V4.1-Flash, agent in non-thinking mode.

| Configuration | Candidates | EX on the 180 dev questions |
|---|---|---|
| Single agent run (best single prompt) | 1 | 80.0% |
| Majority vote over `direct`, `grain`, `decompose` | 3 | 79.4% |
| **v2: unit-test selection over `direct`, `grain`, `decompose`** | 3 | **86.1%** |
| Upper bound: any candidate correct | 3 | 88.9% |

Rerunning the same configuration at temperature 0 changed 20 of 180 answers (2.2 points EX), so differences of a few questions are within noise.

Tried on the same set and dropped:
- The same prompt sampled 5 times at temperature 0.5: 83.9% with unit-test selection. Different prompts gave more useful diversity than random sampling.
- Three CHASE-SQL style generators (divide-and-conquer, query plan, synthetic examples): 81.1%. Only divide-and-conquer was kept, as `decompose`; the other two were weaker on their own (72.2%) and failed on the same questions.
- Four or five candidates (adding a rerun and other note variants): no gain over three.
- A self-check of the answer shape (the model declares the expected columns and rows, code compares): the declared shape was wrong in the same way as the SQL.

Cost: three agent runs plus the selector (two thinking-mode calls on the ~25% of questions where candidates disagree), about 3.3 times v1.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env              # then add GEMINI_API_KEY and/or DEEPSEEK_API_KEY
python -m text2sql_agent.download_data   # BIRD Mini-Dev SQLite (~800 MB download)
python -m text2sql_agent.build_schema    # enriched schemas -> schemas/
python -m text2sql_agent.list_columns    # list columns -> schemas/list_columns.json (v2 only, one LLM call per database)
```

Models:
- `gemini-3.5-flash-lite` (default) through `langchain-google-genai`. The native Gemini client is used rather than the OpenAI-compatible endpoint, because tool calling needs Gemini's thought signatures to be passed back on the next turn.
- `deepseek-flash` through `langchain-openai`. The agent runs in non-thinking mode: with tool calls, thinking mode requires the previous turns' reasoning to be sent back, which the client does not do. Thinking mode is available for the baseline (`--thinking`) and is used by the v2 selector.

## Usage

```bash
# Development set: 180 questions = these two samples
python -m text2sql_agent.run --mode baseline --n 80  --model deepseek-flash --workers 8
python -m text2sql_agent.run --mode baseline --n 100 --exclude-n 80 --model deepseek-flash --workers 8

# v1 agent (with the pitfalls checklist)
python -m text2sql_agent.run --mode agent --pitfalls --n 80 --model deepseek-flash --workers 8

# v2: three strategies + unit-test selection
python -m text2sql_agent.run_v2 --n 80 --model deepseek-flash --workers 8

# Score with the official EX rule
python -m text2sql_agent.evaluate runs/v2__deepseek-flash
```

Runs are resumable: questions already present in `predictions.jsonl` are skipped, so re-running a command after a rate-limit stop continues where it left off. Use `--workers 1` on the Gemini free tier (15 requests per minute).

## Repository layout

```
text2sql_agent/
  common.py             paths, limits, safe read-only SQL execution with timeout, SQL extraction, model setup
  build_schema.py       enriched schema builder
  list_columns.py       finds columns that store a list in one cell (code finds candidates, an LLM confirms)
  baseline.py           official BIRD single-call baseline
  agent.py              LangGraph agent and the v2 reasoning strategies
  run.py                batch runner (sampling, parallel workers, resumable output, token accounting)
  select_unit_tests.py  v2 selection among several runs with unit tests
  run_v2.py             runs the three strategies, then the selection
  evaluate.py           EX scoring (set equality of result rows, as in BIRD's evaluation_ex.py)
  download_data.py      BIRD Mini-Dev download
```

## Results

Coming soon: baseline vs. v1 vs. v2 EX on the 320 held-out questions, with cost per question.

## References and data

- BIRD: Li et al., *Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs*, NeurIPS 2023. Mini-Dev data: [birdsql/bird_mini_dev](https://huggingface.co/datasets/birdsql/bird_mini_dev), CC BY-SA 4.0. The data is not included in this repository.
- CHESS: Talaei et al., *CHESS: Contextual Harnessing for Efficient SQL Synthesis*, 2024. Informed database exploration, execution-based revision and the unit-test selector.
- CHASE-SQL: Pourreza et al., *CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection in Text-to-SQL*, 2024. Informed the multi-strategy candidate generation (`decompose`).

## License

MIT
