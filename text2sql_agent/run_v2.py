"""Run the whole v2 system: three reasoning strategies, then unit-test selection.

    python -m text2sql_agent.run_v2 --n 100 --exclude-n 80 --model deepseek-flash --workers 8
    python -m text2sql_agent.evaluate runs/v2__deepseek-flash

Sampling options (--n / --all / --ids / --seed / --exclude-n ...) are passed to run.py unchanged.
Each strategy writes runs/agent_<strategy>[tag]__<model>/; the selection goes to
runs/v2[tag]__<model>/predictions.jsonl, whose "sql" field is the final answer.
Re-running the same command resumes (questions already answered are skipped).
"""
import argparse
import subprocess
import sys

from .common import DEFAULT_MODEL, RUNS_DIR

STRATEGIES = ["direct", "grain", "decompose"]   # also the final tie-break order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--k", type=int, default=5, help="unit tests per question")
    args, sample_args = ap.parse_known_args()

    for s in STRATEGIES:
        cmd = [sys.executable, "-u", "-m", "text2sql_agent.run", "--mode", "agent", "--strategy", s,
               "--model", args.model, "--workers", str(args.workers), "--tag", args.tag, *sample_args]
        print(">>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    runs = [f"agent_{s}{args.tag}__{args.model}" for s in STRATEGIES]
    cmd = [sys.executable, "-u", "-m", "text2sql_agent.select_unit_tests", *runs,
           "--out", str(RUNS_DIR / f"v2{args.tag}__{args.model}"), "--k", str(args.k), "--workers", str(args.workers)]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
