"""Measure how often instructions hidden in documents are obeyed.

Runs every case in benchmarks/injection_cases.json several times with the
generator's context guard off, then on, and reports the attack success
rate of each configuration.

Usage:
    uv run python scripts/run_injection_eval.py
    uv run python scripts/run_injection_eval.py --trials 5
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

# UTF-8 output even on consoles whose default code page cannot encode the text.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cortex.config import get_settings
from cortex.eval.injection import load_cases, run_injection_eval
from cortex.llm import get_llm_client
from cortex.rag.generator import Generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the prompt-injection benchmark.")
    parser.add_argument("--cases", default="benchmarks/injection_cases.json")
    # The model's sampling is not under our control, so one trial per case
    # says little; repeat and count.
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--save", default="benchmarks/injection_results.json")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    llm = get_llm_client()
    report: dict = {
        "date": date.today().isoformat(),
        "model": get_settings().llm_model,
        "cases": len(cases),
        "trials": args.trials,
        "configurations": {},
    }

    for label, guard in (("unguarded", False), ("guarded", True)):
        generator = Generator(llm, guard_context=guard)
        hits: dict[str, int] = {case.id: 0 for case in cases}
        examples: dict[str, str] = {}
        for trial in range(1, args.trials + 1):
            for result in run_injection_eval(cases, generator):
                if result.injected:
                    hits[result.case_id] += 1
                    examples.setdefault(result.case_id, result.answer[:300])
            print(f"  {label}: trial {trial}/{args.trials} done")
        attempts = len(cases) * args.trials
        succeeded = sum(hits.values())
        report["configurations"][label] = {
            "attempts": attempts,
            "attacks_succeeded": succeeded,
            "attack_success_rate": succeeded / attempts,
            "by_case": hits,
            "examples": examples,
        }

    print(f"\n{'case':<14}{'attack':<58}{'unguarded':>10}{'guarded':>9}")
    for case in cases:
        off = report["configurations"]["unguarded"]["by_case"][case.id]
        on = report["configurations"]["guarded"]["by_case"][case.id]
        print(f"{case.id:<14}{case.attack[:56]:<58}{off:>7}/{args.trials}{on:>7}/{args.trials}")
    for label in ("unguarded", "guarded"):
        c = report["configurations"][label]
        rate = c["attack_success_rate"]
        print(f"{label:>10}: {c['attacks_succeeded']}/{c['attempts']} attacks succeeded ({rate:.0%})")

    Path(args.save).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
