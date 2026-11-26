"""Run the baseline or the agent on a set of questions, with resumable output.

Examples:
    python -m text2sql_agent.run --mode baseline --n 100 --exclude-n 80
    python -m text2sql_agent.run --mode agent --pitfalls --n 100 --exclude-n 80          # v1
    python -m text2sql_agent.run --mode agent --strategy grain --model deepseek-flash --workers 8 --all
    (the whole v2 system: python -m text2sql_agent.run_v2 ...)

Output: runs/<variant>__<model>/predictions.jsonl (one line per question) and traces.jsonl
(full message history). Questions already in predictions.jsonl are skipped, so re-running
the same command after a quota stop continues where it left off.
Questions are processed in db_id order so that consecutive calls share the cached schema prefix.
"""
import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import AIMessage

from .common import DEFAULT_MODEL, RUNS_DIR, load_questions, make_llm


def pick_questions(n: int | None, seed: int, exclude_n: int | None = None, exclude_seed: int = 0) -> list[dict]:
    """n random questions (fixed seed). exclude_n removes the sample (exclude_n, exclude_seed) first,
    which keeps a development sample and an evaluation sample disjoint."""
    qs = load_questions()
    if exclude_n:
        dev = {q["question_id"] for q in random.Random(exclude_seed).sample(qs, exclude_n)}
        qs = [q for q in qs if q["question_id"] not in dev]
    if n is not None:
        qs = random.Random(seed).sample(qs, n)
    return sorted(qs, key=lambda q: (q["db_id"], q["question_id"]))


def usage_of(messages) -> dict:
    calls = [m for m in messages if isinstance(m, AIMessage)]
    inp = sum((m.usage_metadata or {}).get("input_tokens", 0) for m in calls)
    out = sum((m.usage_metadata or {}).get("output_tokens", 0) for m in calls)
    return {"model_calls": len(calls), "input_tokens": inp, "output_tokens": out}


def dump_messages(messages) -> list[dict]:
    out = []
    for m in messages:
        d = {"role": m.type, "content": m.content}
        if getattr(m, "tool_calls", None):
            d["tool_calls"] = [{"name": t["name"], "args": t["args"]} for t in m.tool_calls]
        out.append(d)
    return out


def error_kind(e: Exception) -> str:
    """'429' (rate limit / quota), 'busy' (transient), or 'other'."""
    code = None
    for attr in ("code", "status_code"):
        v = getattr(e, attr, None)
        if isinstance(v, int):
            code = v
    text = f"{type(e).__name__} {e}"
    if code == 429 or "429" in text or "RESOURCE_EXHAUSTED" in text:
        return "429"
    if (code and 500 <= code < 600) or any(s in text for s in ("503", "500", "UNAVAILABLE", "Timeout", "timed out", "Connection")):
        return "busy"
    return "other"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "agent"], required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--n", type=int, help="number of randomly sampled questions")
    g.add_argument("--all", action="store_true", help="all 500 questions")
    g.add_argument("--ids", help="only these question_ids, comma-separated (quick checks)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-n", type=int, help="exclude the development sample of this size")
    ap.add_argument("--exclude-seed", type=int, default=0)
    ap.add_argument("--pitfalls", action="store_true", help="agent only (v1): add the SQL pitfalls checklist to the prompt")
    ap.add_argument("--strategy", choices=["direct", "grain", "decompose"],
                    help="agent only (v2): reasoning strategy; sets the prompt options itself")
    ap.add_argument("--thinking", action="store_true", help="baseline only, DeepSeek: thinking mode")
    ap.add_argument("--workers", type=int, default=1, help="questions run in parallel (keep 1 on the Gemini free tier)")
    ap.add_argument("--tag", default="", help="suffix for the output directory name")
    args = ap.parse_args()

    if args.mode == "agent" and args.thinking:
        raise SystemExit("--thinking is for the baseline only (tool calls in thinking mode are not supported).")
    if args.strategy and (args.mode != "agent" or args.pitfalls):
        raise SystemExit("--strategy is for the agent and sets the prompt options itself; do not add --pitfalls.")
    if args.strategy:
        name = f"agent_{args.strategy}"
    else:
        name = args.mode + ("_pitfalls" if args.pitfalls and args.mode == "agent" else "") + ("_thinking" if args.thinking else "")
    out_dir = RUNS_DIR / f"{name}{args.tag}__{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file, trace_file = out_dir / "predictions.jsonl", out_dir / "traces.jsonl"
    done = set()
    if pred_file.exists():
        done = {json.loads(l)["question_id"] for l in pred_file.read_text(encoding="utf-8").splitlines() if l.strip()}

    if args.ids:
        want = {int(x) for x in args.ids.split(",")}
        sample = sorted((q for q in load_questions() if q["question_id"] in want), key=lambda q: (q["db_id"], q["question_id"]))
    else:
        sample = pick_questions(None if args.all else args.n, args.seed, args.exclude_n, args.exclude_seed)
    questions = [q for q in sample if q["question_id"] not in done]
    print(f"{out_dir.name}: sample of {len(sample)}, {len(questions)} left to run.", flush=True)

    llm = make_llm(args.model, thinking=args.thinking)
    if args.mode == "agent":
        from . import agent
        graph = agent.make_graph(llm, pitfalls=args.pitfalls, strategy=args.strategy)
        solve = lambda q: agent.solve(graph, q)          # noqa: E731
    else:
        from . import baseline
        solve = lambda q: baseline.solve(llm, q)          # noqa: E731

    stop = threading.Event()      # set on quota exhaustion or persistent server errors: skip questions not started
    write_lock = threading.Lock()
    done_count = [0]

    def solve_with_retry(q):
        if stop.is_set():
            return None
        t0 = time.time()
        n429 = nbusy = 0
        while not stop.is_set():
            try:
                result = solve(q)
                break
            except Exception as e:  # noqa: BLE001
                kind = error_kind(e)
                if kind == "other":
                    raise
                if kind == "429":
                    # Repeated 429s usually mean the daily quota is used up.
                    n429 += 1
                    if n429 > 3:
                        stop.set()
                        return None
                    print(f"  [{q['question_id']}] rate limited (429), waiting 65 s ({n429}/3)", flush=True)
                    time.sleep(65)
                    continue
                nbusy += 1
                if nbusy > 5:
                    stop.set()
                    return None
                print(f"  [{q['question_id']}] transient error ({type(e).__name__}), waiting 30 s ({nbusy}/5)", flush=True)
                time.sleep(30)
        else:
            return None

        rec = {
            "question_id": q["question_id"], "db_id": q["db_id"], "difficulty": q["difficulty"],
            "sql": result["sql"], "seconds": round(time.time() - t0, 1),
            **usage_of(result["messages"]),
            "tool_calls_used": result.get("tool_calls_used", 0), "revisions": result.get("revisions", 0),
        }
        with write_lock:   # one complete line per question even with several threads
            with open(pred_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            with open(trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"question_id": q["question_id"], "messages": dump_messages(result["messages"])},
                                   ensure_ascii=False, default=str) + "\n")
            done_count[0] += 1
            print(f"[{done_count[0]}/{len(questions)}] q{q['question_id']} {q['db_id']}: calls={rec['model_calls']} "
                  f"tools={rec['tool_calls_used']} revisions={rec['revisions']} "
                  f"tokens={rec['input_tokens']}+{rec['output_tokens']} "
                  f"{'OK' if rec['sql'] else 'NO SQL'} ({rec['seconds']}s)", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(solve_with_retry, questions))
    if stop.is_set():
        print("Still failing after retries (daily quota exhausted or server overloaded). "
              "Stopping; re-run the same command later to continue.", flush=True)


if __name__ == "__main__":
    main()
