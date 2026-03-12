"""SFT vs DPO comparison tool — evaluates chosen vs rejected answers side-by-side.

Since we can't run model inference locally (no GPU), this tool compares:
  - DPO "chosen" answers (what the model should learn to prefer)
  - DPO "rejected" answers (what the model should learn to avoid)
  - SFT instruction answers (baseline from Week 5)

Uses LLM-as-judge (GPT-4o-mini) for accuracy + style scoring.

Usage:
    poetry run python -m tools.model_compare --n 20
    poetry run python -m tools.model_compare --n 10 --skip-eval   # Use cached evaluations
"""

import json
import time
from pathlib import Path

import click

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INSTRUCT_FILE = DATA_DIR / "instruct_dataset_samples.json"
PREFERENCE_FILE = DATA_DIR / "preference_dataset_samples.json"
INSTRUCT_EVAL_FILE = DATA_DIR / "instruct_evaluation.json"
PREFERENCE_EVAL_FILE = DATA_DIR / "preference_evaluation.json"
COMPARISON_OUTPUT = DATA_DIR / "sft_vs_dpo_comparison.json"


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        click.echo(f"File not found: {path}")
        raise SystemExit(1)
    with path.open() as f:
        return json.load(f)


def _avg(scores: list[int | float]) -> float:
    valid = [s for s in scores if s > 0]
    return sum(valid) / len(valid) if valid else 0.0


def _score_distribution(scores: list[int]) -> dict[int, int]:
    dist = {1: 0, 2: 0, 3: 0}
    for s in scores:
        if s > 0:
            dist[s] = dist.get(s, 0) + 1
    return dist


@click.command(help="Compare SFT vs DPO quality using LLM-as-judge evaluation.")
@click.option("--n", "max_samples", default=20, help="Max preference samples to evaluate.")
@click.option("--skip-eval", is_flag=True, default=False, help="Use cached evaluations only.")
@click.option("--threads", default=4, help="Parallel threads for OpenAI calls.")
def main(max_samples: int, skip_eval: bool, threads: int) -> None:
    # Load datasets
    pref_samples = _load_json(PREFERENCE_FILE)
    instruct_samples = _load_json(INSTRUCT_FILE)

    pref_subset = pref_samples[:max_samples]

    click.echo(f"\n{'='*70}")
    click.echo("SFT vs DPO COMPARISON")
    click.echo(f"{'='*70}")
    click.echo(f"  Preference samples: {len(pref_subset)} (of {len(pref_samples)})")
    click.echo(f"  Instruct samples:   {len(instruct_samples)}")

    # --- Evaluate chosen answers ---
    if skip_eval and PREFERENCE_EVAL_FILE.exists():
        click.echo("\n  Using cached preference evaluation...")
        pref_eval = _load_json(PREFERENCE_EVAL_FILE)
        chosen_acc = [e["accuracy_score"] for e in pref_eval[:max_samples]]
        chosen_sty = [e["style_score"] for e in pref_eval[:max_samples]]
    else:
        click.echo("\n  Evaluating DPO 'chosen' answers...")
        from llm_engineering.model.evaluation.evaluate import evaluate_dataset

        t0 = time.perf_counter()
        chosen_evals = evaluate_dataset(
            pref_subset, answer_key="chosen", num_threads=threads, max_samples=max_samples
        )
        click.echo(f"    Done in {time.perf_counter() - t0:.1f}s")
        chosen_acc = [e["accuracy"]["score"] for e in chosen_evals if e]
        chosen_sty = [e["style"]["score"] for e in chosen_evals if e]

    # --- Evaluate rejected answers ---
    if skip_eval and PREFERENCE_EVAL_FILE.exists():
        click.echo("  Using cached rejected evaluation (re-evaluating)...")
        # Cached eval only has chosen — we need to evaluate rejected
        click.echo("  Evaluating DPO 'rejected' answers...")
        from llm_engineering.model.evaluation.evaluate import evaluate_dataset

        t0 = time.perf_counter()
        rejected_evals = evaluate_dataset(
            pref_subset, answer_key="rejected", num_threads=threads, max_samples=max_samples
        )
        click.echo(f"    Done in {time.perf_counter() - t0:.1f}s")
        rejected_acc = [e["accuracy"]["score"] for e in rejected_evals if e]
        rejected_sty = [e["style"]["score"] for e in rejected_evals if e]
    else:
        click.echo("  Evaluating DPO 'rejected' answers...")
        from llm_engineering.model.evaluation.evaluate import evaluate_dataset

        t0 = time.perf_counter()
        rejected_evals = evaluate_dataset(
            pref_subset, answer_key="rejected", num_threads=threads, max_samples=max_samples
        )
        click.echo(f"    Done in {time.perf_counter() - t0:.1f}s")
        rejected_acc = [e["accuracy"]["score"] for e in rejected_evals if e]
        rejected_sty = [e["style"]["score"] for e in rejected_evals if e]

    # --- SFT baseline (from cached instruct evaluation) ---
    if INSTRUCT_EVAL_FILE.exists():
        instruct_eval = _load_json(INSTRUCT_EVAL_FILE)
        sft_acc = [e["accuracy_score"] for e in instruct_eval]
        sft_sty = [e["style_score"] for e in instruct_eval]
    else:
        click.echo("  No cached instruct evaluation. Evaluating SFT answers...")
        from llm_engineering.model.evaluation.evaluate import evaluate_dataset

        t0 = time.perf_counter()
        sft_evals = evaluate_dataset(
            instruct_samples, answer_key="answer", num_threads=threads, max_samples=max_samples
        )
        click.echo(f"    Done in {time.perf_counter() - t0:.1f}s")
        sft_acc = [e["accuracy"]["score"] for e in sft_evals if e]
        sft_sty = [e["style"]["score"] for e in sft_evals if e]

    # --- Summary table ---
    click.echo(f"\n{'='*70}")
    click.echo("COMPARISON RESULTS")
    click.echo(f"{'='*70}")

    header = f"  {'Metric':<25} {'SFT Baseline':>14} {'DPO Chosen':>14} {'DPO Rejected':>14}"
    click.echo(header)
    click.echo(f"  {'─'*67}")

    click.echo(f"  {'Accuracy (avg)':.<25} {_avg(sft_acc):>14.2f} {_avg(chosen_acc):>14.2f} {_avg(rejected_acc):>14.2f}")
    click.echo(f"  {'Style (avg)':.<25} {_avg(sft_sty):>14.2f} {_avg(chosen_sty):>14.2f} {_avg(rejected_sty):>14.2f}")
    click.echo(f"  {'Samples evaluated':.<25} {len(sft_acc):>14} {len(chosen_acc):>14} {len(rejected_acc):>14}")

    # Score distributions
    click.echo(f"\n  {'─'*67}")
    click.echo("  SCORE DISTRIBUTIONS")
    click.echo(f"  {'─'*67}")

    for label, scores_list in [
        ("SFT Accuracy", sft_acc), ("SFT Style", sft_sty),
        ("DPO Chosen Accuracy", chosen_acc), ("DPO Chosen Style", chosen_sty),
        ("DPO Rejected Accuracy", rejected_acc), ("DPO Rejected Style", rejected_sty),
    ]:
        dist = _score_distribution(scores_list)
        click.echo(f"  {label:<25}  1={dist[1]:<4} 2={dist[2]:<4} 3={dist[3]:<4}")

    # --- Key insights ---
    click.echo(f"\n  {'─'*67}")
    click.echo("  KEY INSIGHTS")
    click.echo(f"  {'─'*67}")

    chosen_style_avg = _avg(chosen_sty)
    rejected_style_avg = _avg(rejected_sty)
    style_gap = chosen_style_avg - rejected_style_avg

    if style_gap > 0:
        click.echo(f"  [PASS] Chosen > Rejected on style by {style_gap:.2f} points")
    elif style_gap == 0:
        click.echo("  [WARN] Chosen = Rejected on style (no preference signal)")
    else:
        click.echo(f"  [FAIL] Rejected > Chosen on style by {abs(style_gap):.2f} points")

    chosen_acc_avg = _avg(chosen_acc)
    rejected_acc_avg = _avg(rejected_acc)
    acc_gap = chosen_acc_avg - rejected_acc_avg

    if acc_gap >= 0:
        click.echo(f"  [INFO] Chosen vs Rejected accuracy gap: {acc_gap:+.2f}")
    else:
        click.echo(f"  [WARN] Rejected scores higher on accuracy by {abs(acc_gap):.2f}")

    sft_style_avg = _avg(sft_sty)
    click.echo(f"  [INFO] SFT style: {sft_style_avg:.2f} vs DPO chosen style: {chosen_style_avg:.2f}")

    # --- Per-sample comparison (first 10) ---
    click.echo(f"\n  {'─'*67}")
    click.echo("  PER-SAMPLE DETAILS (first 10 preference samples)")
    click.echo(f"  {'─'*67}")

    show_n = min(10, len(pref_subset))
    for i in range(show_n):
        instr = pref_subset[i]["instruction"][:55]
        c_acc = chosen_acc[i] if i < len(chosen_acc) else 0
        c_sty = chosen_sty[i] if i < len(chosen_sty) else 0
        r_acc = rejected_acc[i] if i < len(rejected_acc) else 0
        r_sty = rejected_sty[i] if i < len(rejected_sty) else 0
        winner = "chosen" if (c_acc + c_sty) >= (r_acc + r_sty) else "rejected"
        click.echo(f"    [{i}] chosen(a={c_acc},s={c_sty}) vs rejected(a={r_acc},s={r_sty}) -> {winner}")
        click.echo(f"        \"{instr}...\"")

    # --- Save comparison ---
    comparison = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples_evaluated": {
            "sft": len(sft_acc),
            "dpo_chosen": len(chosen_acc),
            "dpo_rejected": len(rejected_acc),
        },
        "averages": {
            "sft_accuracy": round(_avg(sft_acc), 2),
            "sft_style": round(_avg(sft_sty), 2),
            "dpo_chosen_accuracy": round(_avg(chosen_acc), 2),
            "dpo_chosen_style": round(_avg(chosen_sty), 2),
            "dpo_rejected_accuracy": round(_avg(rejected_acc), 2),
            "dpo_rejected_style": round(_avg(rejected_sty), 2),
        },
        "style_gap_chosen_vs_rejected": round(style_gap, 2),
        "accuracy_gap_chosen_vs_rejected": round(acc_gap, 2),
    }

    with COMPARISON_OUTPUT.open("w") as f:
        json.dump(comparison, f, indent=2)
    click.echo(f"\n  Saved comparison to: {COMPARISON_OUTPUT}")

    # Cost estimate
    total_evals = len(chosen_acc) + len(rejected_acc)
    est_tokens = total_evals * 350
    cost = (est_tokens * 0.15 / 1_000_000) + (est_tokens * 0.60 / 1_000_000)
    click.echo(f"  Est. cost: ~${cost:.4f}")
    click.echo(f"{'='*70}\n")


if __name__ == "__main__":
    main()
