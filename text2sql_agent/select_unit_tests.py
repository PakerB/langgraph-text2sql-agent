"""Pick one SQL per question from several runs, with unit tests in the spirit of CHESS's Unit Tester.

For each question:
  1. Execute every candidate and group candidates with the same result set (same rule as EX).
  2. One group: take it, no model call.
  3. Several groups: two model calls.
     a) Write K unit tests: concrete checks that a correct answer must pass, preferring checks
        that tell the groups apart.
     b) Judge every group against every test (pass / fail).
     The group passing the most tests wins; ties go to more votes, then to the earlier run.
The model never sees the gold SQL. No tools are called here, so DeepSeek's thinking mode is used.

Differences from the CHESS paper: K=5 tests instead of 20; tests may check output shape and values,
not only query logic; one evaluation call for all candidates instead of one per test; candidates with
identical results are merged first.

Usage:
    python -m text2sql_agent.select_unit_tests RUN1 RUN2 RUN3 --out runs/v2__deepseek-flash [--ids 1,2] [--k 5]
RUN names are directories in runs/; their order is the final tie-break order.
"""
import argparse
import collections
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_core.messages import HumanMessage

from .common import (FINAL_TIMEOUT_S, RICH_SCHEMA_DIR, RUNS_DIR, execute, invoke_with_retry, load_questions,
                     make_llm, message_text, official_question_block)

SELECTOR_MODEL = "deepseek-flash"

GEN_PROMPT = """You are checking candidate answers to a text-to-SQL question on a SQLite database.

Database schema:
{schema}

Question (the line before it, if any, is evidence explaining meanings):
{question}

The candidate SQL queries below return different results:
{candidates}

Write {k} unit tests: short, concrete checks that a correct answer to the question must pass. Cover what the question asks for: which values are returned (one column per requested value, no extra columns), whether the answer is one row or many, what is counted or averaged and at which grain (which entity is counted once), which filters, ties and orderings the question requires. Prefer checks that tell the candidates apart. Do not mention candidate numbers in the tests.

Answer with JSON only: {{"tests": ["test 1", "test 2", ...]}}"""

EVAL_PROMPT = """You are checking candidate answers to a text-to-SQL question on a SQLite database.

Database schema:
{schema}

Question (the line before it, if any, is evidence explaining meanings):
{question}

Candidates (SQL and the result it returns):
{candidates}

Unit tests:
{tests}

For each candidate and each unit test, decide whether the candidate passes the test. Judge from the SQL logic and the result shown.

Answer with JSON only: {{"results": {{"1": [true or false for each test, in order], "2": [...], ...}}}}"""


def show_result(cols, rows, n_rows, max_rows=5):
    lines = [f"columns: {cols}", f"rows: {n_rows}"]
    lines += [str(r)[:200] for r in rows[:max_rows]]
    if n_rows > max_rows:
        lines.append("...")
    return "\n".join(lines)


TOKENS = {"input": 0, "output": 0}   # selector token totals across threads
_tok_lock = threading.Lock()


def ask_json(llm, prompt, key, tries=3):
    for _ in range(tries):
        reply = invoke_with_retry(llm, [HumanMessage(prompt)])
        u = reply.usage_metadata or {}
        with _tok_lock:
            TOKENS["input"] += u.get("input_tokens", 0)
            TOKENS["output"] += u.get("output_tokens", 0)
        m = re.search(r"\{.*\}", message_text(reply.content), re.S)
        try:
            return json.loads(m.group(0))[key]
        except (AttributeError, json.JSONDecodeError, KeyError):
            continue
    return None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories in runs/; order = final tie-break order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids")
    ap.add_argument("--k", type=int, default=5, help="unit tests per question")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    Q = {q["question_id"]: q for q in load_questions()}
    preds = {r: {json.loads(l)["question_id"]: json.loads(l)["sql"]
                 for l in (RUNS_DIR / r / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
             for r in a.runs}
    ids = sorted(set.intersection(*(set(p) for p in preds.values())))
    if a.ids:
        ids = [i for i in ids if i in {int(x) for x in a.ids.split(",")}]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = make_llm(SELECTOR_MODEL, thinking=True)

    def solve(i):
        q = Q[i]
        groups = []          # [{key, sql, votes, cols, rows}]
        for r in a.runs:
            sql = preds[r][i] or ""
            cols, rows, _, err = execute(q["db_id"], sql, FINAL_TIMEOUT_S, max_rows=None)
            if err:
                continue
            key = frozenset(rows)
            g = next((g for g in groups if g["key"] == key), None)
            if g:
                g["votes"] += 1
            else:
                groups.append({"key": key, "sql": sql, "votes": 1, "cols": cols, "rows": rows})
        rec = {"question_id": i, "db_id": q["db_id"], "difficulty": q["difficulty"], "n_groups": len(groups),
               "votes": [g["votes"] for g in groups], "model_calls": 0}
        if not groups:
            return {**rec, "sql": preds[a.runs[0]][i], "how": "no_valid_candidate"}
        vote_pick = max(range(len(groups)), key=lambda j: groups[j]["votes"])   # tie: earlier group
        rec["vote_sql"] = groups[vote_pick]["sql"]
        if len(groups) == 1:
            return {**rec, "sql": groups[0]["sql"], "how": "single_group"}

        schema = (RICH_SCHEMA_DIR / f"{q['db_id']}.txt").read_text(encoding="utf-8")
        cand = "\n\n".join(f"Candidate {j + 1}:\nSQL: {g['sql']}\nResult:\n{show_result(g['cols'], list(g['rows']), len(g['rows']))}"
                           for j, g in enumerate(groups))
        question = official_question_block(q)
        tests = ask_json(llm, GEN_PROMPT.format(schema=schema, question=question, candidates=cand, k=a.k), "tests")
        rec["model_calls"] = 1
        if not tests:
            return {**rec, "sql": rec["vote_sql"], "how": "vote_fallback_no_tests"}
        tests_txt = "\n".join(f"{n + 1}. {t}" for n, t in enumerate(tests))
        res = ask_json(llm, EVAL_PROMPT.format(schema=schema, question=question, candidates=cand, tests=tests_txt), "results")
        rec["model_calls"] = 2
        if not isinstance(res, dict):
            return {**rec, "sql": rec["vote_sql"], "how": "vote_fallback_no_eval", "tests": tests}
        scores = [sum(bool(x) for x in (res.get(str(j + 1)) or [])) for j in range(len(groups))]
        best = max(range(len(groups)), key=lambda j: (scores[j], groups[j]["votes"], -j))
        return {**rec, "sql": groups[best]["sql"], "how": "unit_tests", "tests": tests, "scores": scores}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        recs = list(pool.map(solve, ids))
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hows = collections.Counter(r["how"] for r in recs)
    print(f"{len(recs)} questions, {time.time() - t0:.0f} s, {sum(r['model_calls'] for r in recs)} model calls, {dict(hows)}")
    print(f"selector tokens: {TOKENS['input']:,} input, {TOKENS['output']:,} output (including thinking)")
    (out_dir / "tokens.json").write_text(json.dumps(TOKENS), encoding="utf-8")


if __name__ == "__main__":
    main()
