"""Dataset inspection CLI — quality metrics, statistics, evaluation, and sample browsing.

Usage:
    poetry run python -m tools.dataset_inspect stats
    poetry run python -m tools.dataset_inspect samples --n 5
    poetry run python -m tools.dataset_inspect quality
    poetry run python -m tools.dataset_inspect quality --deep       # + near-dedup, decontamination, diversity
    poetry run python -m tools.dataset_inspect evaluate --n 10      # LLM-as-judge on N samples
    poetry run python -m tools.dataset_inspect generate --mock      # Generate + inspect in one step
    poetry run python -m tools.dataset_inspect generate             # Real API generation
"""

import json
import time
from pathlib import Path

import click

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INSTRUCT_FILE = DATA_DIR / "instruct_dataset_samples.json"
PREFERENCE_FILE = DATA_DIR / "preference_dataset_samples.json"


def _load_samples(path: Path) -> list[dict]:
    if not path.exists():
        click.echo(f"No dataset file found at {path}")
        click.echo("Run: poetry run python -m tools.dataset_inspect generate")
        raise SystemExit(1)
    with path.open() as f:
        return json.load(f)


@click.group(help="Inspect and generate instruction/preference datasets.")
def cli():
    pass


@cli.command(help="Show dataset statistics.")
@click.option("--type", "dataset_type", default="instruct", type=click.Choice(["instruct", "preference"]))
def stats(dataset_type: str) -> None:
    path = INSTRUCT_FILE if dataset_type == "instruct" else PREFERENCE_FILE
    samples = _load_samples(path)

    train_samples = [s for s in samples if s["split"] == "train"]
    test_samples = [s for s in samples if s["split"] == "test"]

    click.echo(f"\n{'='*70}")
    click.echo(f"DATASET STATISTICS [{dataset_type.upper()}]")
    click.echo(f"{'='*70}")
    click.echo(f"  Total samples:  {len(samples)}")
    click.echo(f"  Train samples:  {len(train_samples)}")
    click.echo(f"  Test samples:   {len(test_samples)}")
    click.echo(f"  Test ratio:     {len(test_samples)/len(samples):.1%}")

    # Instruction length stats
    instr_lens = [len(s["instruction"]) for s in samples]
    click.echo("\n  Instruction length (chars):")
    click.echo(f"    min={min(instr_lens)}  max={max(instr_lens)}  avg={sum(instr_lens)//len(instr_lens)}")

    # Answer length stats
    answer_key = "answer" if dataset_type == "instruct" else "chosen"
    ans_lens = [len(s[answer_key]) for s in samples]
    click.echo("  Answer length (chars):")
    click.echo(f"    min={min(ans_lens)}  max={max(ans_lens)}  avg={sum(ans_lens)//len(ans_lens)}")

    # Word count stats
    instr_words = [len(s["instruction"].split()) for s in samples]
    ans_words = [len(s[answer_key].split()) for s in samples]
    click.echo(
        f"  Instruction words: min={min(instr_words)}  max={max(instr_words)}  avg={sum(instr_words)//len(instr_words)}"
    )
    click.echo(f"  Answer words:      min={min(ans_words)}  max={max(ans_words)}  avg={sum(ans_words)//len(ans_words)}")
    click.echo(f"{'='*70}\n")


@cli.command(help="Browse dataset samples.")
@click.option("--type", "dataset_type", default="instruct", type=click.Choice(["instruct", "preference"]))
@click.option("--n", default=5, help="Number of samples to show.")
@click.option("--split", default="all", type=click.Choice(["all", "train", "test"]))
def samples(dataset_type: str, n: int, split: str) -> None:
    path = INSTRUCT_FILE if dataset_type == "instruct" else PREFERENCE_FILE
    all_samples = _load_samples(path)

    if split != "all":
        all_samples = [s for s in all_samples if s["split"] == split]

    answer_key = "answer" if dataset_type == "instruct" else "chosen"

    click.echo(f"\n{'='*70}")
    click.echo(
        f"DATASET SAMPLES [{dataset_type.upper()}, split={split}, showing {min(n, len(all_samples))}/{len(all_samples)}]"
    )
    click.echo(f"{'='*70}")

    for i, s in enumerate(all_samples[:n]):
        click.echo(f"\n--- [{i+1}] {s['split'].upper()} ---")
        click.echo(f"Q: {s['instruction']}")
        answer = s[answer_key]
        if len(answer) > 300:
            answer = answer[:300] + "..."
        click.echo(f"A: {answer}")

    click.echo(f"\n{'='*70}\n")


@cli.command(help="Run quality checks on the dataset.")
@click.option("--type", "dataset_type", default="instruct", type=click.Choice(["instruct", "preference"]))
@click.option("--deep", is_flag=True, default=False, help="Run deep checks: near-dedup, decontamination, diversity.")
def quality(dataset_type: str, deep: bool) -> None:
    path = INSTRUCT_FILE if dataset_type == "instruct" else PREFERENCE_FILE
    all_samples = _load_samples(path)
    answer_key = "answer" if dataset_type == "instruct" else "chosen"

    click.echo(f"\n{'='*70}")
    click.echo(f"QUALITY CHECKS [{dataset_type.upper()}]{' (DEEP)' if deep else ''}")
    click.echo(f"{'='*70}")

    issues = []

    # Check 1: Empty instructions or answers
    empty_instr = [i for i, s in enumerate(all_samples) if not s["instruction"].strip()]
    empty_ans = [i for i, s in enumerate(all_samples) if not s[answer_key].strip()]
    click.echo("\n  [1] Empty fields:")
    click.echo(f"      Empty instructions: {len(empty_instr)}")
    click.echo(f"      Empty answers:      {len(empty_ans)}")
    if empty_instr or empty_ans:
        issues.append("empty fields")

    # Check 2: Very short answers (<50 chars)
    short_ans = [i for i, s in enumerate(all_samples) if len(s[answer_key]) < 50]
    click.echo(f"  [2] Short answers (<50 chars): {len(short_ans)}")
    if short_ans:
        issues.append(f"{len(short_ans)} short answers")
        for idx in short_ans[:3]:
            click.echo(f'      Sample {idx}: "{all_samples[idx][answer_key][:80]}"')

    # Check 3: Duplicate instructions
    instructions = [s["instruction"].strip().lower() for s in all_samples]
    unique = len(set(instructions))
    dupes = len(instructions) - unique
    click.echo(f"  [3] Duplicate instructions: {dupes} (unique: {unique}/{len(instructions)})")
    if dupes:
        issues.append(f"{dupes} duplicate instructions")

    # Check 4: Instructions mentioning "context", "extract", "course"
    bad_keywords = ["context", "extract", "the course", "the system"]
    keyword_violations = []
    for i, s in enumerate(all_samples):
        instr_lower = s["instruction"].lower()
        for kw in bad_keywords:
            if kw in instr_lower:
                keyword_violations.append((i, kw))
                break
    click.echo(f"  [4] Instructions mentioning context/extract/course: {len(keyword_violations)}")
    if keyword_violations:
        issues.append(f"{len(keyword_violations)} keyword violations")
        for idx, kw in keyword_violations[:3]:
            click.echo(f"      Sample {idx}: contains \"{kw}\" — \"{all_samples[idx]['instruction'][:80]}\"")

    # Check 5: Answer format (starts with uppercase, ends with punctuation)
    format_issues = []
    for i, s in enumerate(all_samples):
        ans = s[answer_key].strip()
        if ans and (not ans[0].isupper() or ans[-1] not in ".!?"):
            format_issues.append(i)
    click.echo(f"  [5] Format issues (no uppercase start or punctuation end): {len(format_issues)}")
    if format_issues:
        issues.append(f"{len(format_issues)} format issues")

    # Deep checks
    if deep:
        from llm_engineering.application.dataset.utils import (
            check_train_test_contamination,
            compute_diversity_stats,
            find_near_duplicates,
        )

        click.echo(f"\n  {'─'*60}")
        click.echo("  DEEP CHECKS")
        click.echo(f"  {'─'*60}")

        # Check 6: Near-duplicate instructions (n-gram Jaccard > 0.7)
        near_dupes = find_near_duplicates(all_samples, key="instruction", threshold=0.7)
        click.echo(f"  [6] Near-duplicate instructions (Jaccard > 0.7): {len(near_dupes)}")
        if near_dupes:
            issues.append(f"{len(near_dupes)} near-duplicate pairs")
            for a, b, sim in near_dupes[:5]:
                click.echo(f"      ({a},{b}) sim={sim:.2f}")
                click.echo(f"        A: \"{all_samples[a]['instruction'][:70]}\"")
                click.echo(f"        B: \"{all_samples[b]['instruction'][:70]}\"")

        # Check 7: Train/test contamination
        contaminated = check_train_test_contamination(all_samples, key="instruction", threshold=0.8)
        click.echo(f"  [7] Train/test contamination (Jaccard > 0.8): {len(contaminated)}")
        if contaminated:
            issues.append(f"{len(contaminated)} contaminated pairs")
            for tr, te, sim in contaminated[:3]:
                click.echo(f"      train[{tr}] ~ test[{te}] sim={sim:.2f}")
                click.echo(f"        Train: \"{all_samples[tr]['instruction'][:60]}\"")
                click.echo(f"        Test:  \"{all_samples[te]['instruction'][:60]}\"")

        # Check 8: Diversity stats
        div_instr = compute_diversity_stats(all_samples, key="instruction")
        div_ans = compute_diversity_stats(all_samples, key=answer_key)
        click.echo("  [8] Instruction diversity:")
        click.echo(f"      Vocab size: {div_instr['vocab_size']}  TTR: {div_instr['type_token_ratio']}")
        click.echo(f"      Unique bigrams: {div_instr['unique_bigrams']}")
        click.echo(f"      Top words: {', '.join(w for w, _ in div_instr['top_10_words'][:5])}")
        click.echo("  [9] Answer diversity:")
        click.echo(f"      Vocab size: {div_ans['vocab_size']}  TTR: {div_ans['type_token_ratio']}")
        click.echo(f"      Unique bigrams: {div_ans['unique_bigrams']}")
        click.echo(f"      Top words: {', '.join(w for w, _ in div_ans['top_10_words'][:5])}")

    # Summary
    click.echo(f"\n  {'='*60}")
    if issues:
        click.echo(f"  ISSUES FOUND: {', '.join(issues)}")
    else:
        click.echo("  ALL QUALITY CHECKS PASSED")
    click.echo(f"  {'='*60}\n")


@cli.command(help="Run LLM-as-judge evaluation (accuracy + style scoring).")
@click.option("--type", "dataset_type", default="instruct", type=click.Choice(["instruct", "preference"]))
@click.option("--n", "max_samples", default=10, help="Max samples to evaluate (cost control).")
@click.option("--threads", default=4, help="Parallel threads for OpenAI calls.")
def evaluate(dataset_type: str, max_samples: int, threads: int) -> None:
    from llm_engineering.model.evaluation.evaluate import evaluate_dataset

    path = INSTRUCT_FILE if dataset_type == "instruct" else PREFERENCE_FILE
    all_samples = _load_samples(path)
    answer_key = "answer" if dataset_type == "instruct" else "chosen"

    click.echo(f"\n{'='*70}")
    click.echo(f"LLM-AS-JUDGE EVALUATION [{dataset_type.upper()}, n={min(max_samples, len(all_samples))}]")
    click.echo(f"{'='*70}")

    t0 = time.perf_counter()
    evaluations = evaluate_dataset(all_samples, answer_key=answer_key, num_threads=threads, max_samples=max_samples)
    elapsed = time.perf_counter() - t0

    # Compute aggregate scores
    accuracy_scores = [e["accuracy"]["score"] for e in evaluations if e and e["accuracy"]["score"] > 0]
    style_scores = [e["style"]["score"] for e in evaluations if e and e["style"]["score"] > 0]

    click.echo(f"\n  Evaluated {len(evaluations)} samples in {elapsed:.1f}s")
    click.echo(f"  {'─'*50}")

    if accuracy_scores:
        avg_acc = sum(accuracy_scores) / len(accuracy_scores)
        avg_sty = sum(style_scores) / len(style_scores)
        click.echo(f"  Accuracy: avg={avg_acc:.2f}  (1=poor, 2=good, 3=excellent)")
        click.echo(f"  Style:    avg={avg_sty:.2f}  (1=formal, 2=good, 3=excellent)")
        click.echo(f"  {'─'*50}")

        # Score distribution
        for label, scores in [("Accuracy", accuracy_scores), ("Style", style_scores)]:
            dist = {1: 0, 2: 0, 3: 0}
            for s in scores:
                dist[s] = dist.get(s, 0) + 1
            click.echo(f"  {label} distribution: 1={dist[1]}  2={dist[2]}  3={dist[3]}")

    # Show per-sample details
    click.echo(f"\n  {'─'*50}")
    click.echo("  Per-sample scores:")
    for i, ev in enumerate(evaluations):
        if ev and ev["accuracy"]["score"] > 0:
            instr = all_samples[i]["instruction"][:50]
            click.echo(f"    [{i}] acc={ev['accuracy']['score']} sty={ev['style']['score']}  \"{instr}...\"")

    # Save evaluation results
    eval_path = path.parent / f"{dataset_type}_evaluation.json"
    eval_data = []
    for i, ev in enumerate(evaluations):
        eval_data.append(
            {
                "index": i,
                "instruction": all_samples[i]["instruction"],
                "accuracy_score": ev["accuracy"]["score"] if ev else 0,
                "accuracy_analysis": ev["accuracy"]["analysis"] if ev else "",
                "style_score": ev["style"]["score"] if ev else 0,
                "style_analysis": ev["style"]["analysis"] if ev else "",
            }
        )
    with eval_path.open("w") as f:
        json.dump(eval_data, f, indent=2)
    click.echo(f"\n  Saved evaluations to: {eval_path}")

    # Cost estimate
    est_tokens = max_samples * 350
    cost = (est_tokens * 0.15 / 1_000_000) + (est_tokens * 0.60 / 1_000_000)
    click.echo(f"  Est. cost: ~${cost:.4f}")
    click.echo(f"{'='*70}\n")


@cli.command(help="Generate instruction dataset and save to JSON.")
@click.option("--mock", is_flag=True, default=False, help="Use mock LLM responses.")
@click.option("--type", "dataset_type", default="instruct", type=click.Choice(["instruct", "preference"]))
@click.option("--test-size", default=0.1, help="Test split ratio.")
def generate(mock: bool, dataset_type: str, test_size: float) -> None:
    from llm_engineering.application.dataset.generation import (
        InstructionDatasetGenerator,
        PreferenceDatasetGenerator,
    )
    from steps.generate_datasets.query_feature_store import fetch_all_data

    mode = "MOCK" if mock else "REAL"
    click.echo(f"\n{'='*70}")
    click.echo(f"DATASET GENERATION [{mode} mode, type={dataset_type}]")
    click.echo(f"{'='*70}")

    # 1. Fetch documents
    results = fetch_all_data()
    docs = [doc for r in results.values() for doc in r]
    click.echo(f"  Fetched {len(docs)} cleaned documents from feature store")

    # 2. Select generator
    if dataset_type == "instruct":
        generator = InstructionDatasetGenerator
        answer_key = "answer"
        output_path = INSTRUCT_FILE
    else:
        generator = PreferenceDatasetGenerator
        answer_key = "chosen"
        output_path = PREFERENCE_FILE

    # 3. Create prompts
    grouped_prompts = generator.get_prompts(docs)
    total_prompts = sum(len(p) for p in grouped_prompts.values())
    input_tokens = sum(p.num_tokens for prompts in grouped_prompts.values() for p in prompts)
    click.echo(f"  Created {total_prompts} prompts ({input_tokens} input tokens)")

    # 4. Generate
    t0 = time.perf_counter()
    result = generator.generate(grouped_prompts, test_size=test_size, mock=mock)
    elapsed = time.perf_counter() - t0

    # 5. Collect samples
    all_samples = []
    for cat in result.train:
        for s in result.train[cat].samples:
            sample = {"split": "train", "instruction": s.instruction}
            sample[answer_key] = getattr(s, answer_key)
            if dataset_type == "preference":
                sample["rejected"] = s.rejected
            all_samples.append(sample)
        for s in result.test[cat].samples:
            sample = {"split": "test", "instruction": s.instruction}
            sample[answer_key] = getattr(s, answer_key)
            if dataset_type == "preference":
                sample["rejected"] = s.rejected
            all_samples.append(sample)

    train_n = sum(result.train[c].num_samples for c in result.train)
    test_n = sum(result.test[c].num_samples for c in result.test)

    # 6. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(all_samples, f, indent=2)

    click.echo("\n  Results:")
    click.echo(f"    Train: {train_n}  Test: {test_n}  Total: {train_n + test_n}")
    click.echo(f"    Time:  {elapsed:.1f}s ({elapsed/max(total_prompts,1):.1f}s per prompt)")
    click.echo(f"    Saved: {output_path}")

    # Cost estimate (gpt-4o-mini)
    if not mock:
        output_tokens_est = 300 * total_prompts
        cost = (input_tokens * 0.15 / 1_000_000) + (output_tokens_est * 0.60 / 1_000_000)
        click.echo(f"    Cost:  ~${cost:.4f} (input={input_tokens}, output~{output_tokens_est} tokens)")

    click.echo(f"{'='*70}\n")


if __name__ == "__main__":
    cli()
