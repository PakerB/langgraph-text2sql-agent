"""Execution accuracy (EX) with the official BIRD Mini-Dev rule:
a prediction is correct iff set(rows of predicted SQL) == set(rows of gold SQL), 30 s timeout.

Usage:
    python -m text2sql_agent.evaluate runs/agent__gemini-3.5-flash-lite --n 100 --exclude-n 80
Writes eval.jsonl (per-question result) into the run directory.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from .common import FINAL_TIMEOUT_S, execute, load_questions


def judge(db_id: str, pred_sql: str | None, gold_sql: str) -> tuple[int, str]:
    if not pred_sql:
        return 0, "no_sql"
    _, pred, _, err = execute(db_id, pred_sql, FINAL_TIMEOUT_S, max_rows=None)
    if err:
        return 0, "timeout" if "timed out" in err else "error"
    _, gold, _, gerr = execute(db_id, gold_sql, FINAL_TIMEOUT_S, max_rows=None)
    if gerr:
        return 0, "gold_error"
    return (1, "correct") if set(pred) == set(gold) else (0, "wrong_result")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=int, help="score only the run.py sample --n/--seed/--exclude-n")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-n", type=int)
    ap.add_argument("--exclude-seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    gold = {q["question_id"]: q for q in load_questions()}
    preds = [json.loads(l) for l in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.n:
        from .run import pick_questions
        sample = {q["question_id"] for q in pick_questions(args.n, args.seed, args.exclude_n, args.exclude_seed)}
        preds = [p for p in preds if p["question_id"] in sample]
        if len(preds) < len(sample):
            print(f"(warning: {len(sample) - len(preds)} questions of the sample have not been run)")

    rows = []
    for p in preds:
        q = gold[p["question_id"]]
        ok, why = judge(q["db_id"], p["sql"], q["SQL"])
        rows.append({**p, "correct": ok, "why": why})
    with open(run_dir / "eval.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    acc = lambda rs: 100 * sum(r["correct"] for r in rs) / len(rs)   # noqa: E731
    print(f"== {run_dir.name}: {n} questions ==")
    print(f"EX: {acc(rows):.1f}%  ({sum(r['correct'] for r in rows)}/{n})")
    for key in ("difficulty", "db_id"):
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        print(f"\nBy {key}:")
        for k in sorted(groups):
            print(f"  {k:28s} {acc(groups[k]):5.1f}%  ({len(groups[k])})")
    why = defaultdict(int)
    for r in rows:
        why[r["why"]] += 1
    print("\nOutcomes:", dict(why))
    avg = lambda k: sum(r.get(k, 0) for r in rows) / n   # noqa: E731
    print(f"\nPer question: {avg('model_calls'):.1f} model calls, "
          f"{avg('input_tokens'):,.0f} input tokens, {avg('output_tokens'):,.0f} output tokens")
    print(f"Questions with tool calls: {sum(r.get('tool_calls_used', 0) > 0 for r in rows)}/{n}; "
          f"with revisions: {sum(r.get('revisions', 0) > 0 for r in rows)}/{n}")
    print("\nWrong:", [r["question_id"] for r in rows if not r["correct"]])


if __name__ == "__main__":
    main()
