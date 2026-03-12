"""LLM-as-judge evaluation — scores dataset samples on accuracy (1-3) and style (1-3).

Uses GPT-4o-mini to evaluate instruction-answer pairs:
  - Accuracy: factual correctness (1=poor, 2=good, 3=excellent)
  - Style: blog/social-media tone (1=too formal, 2=good, 3=excellent)

Based on reference repo's evaluation approach (Ch 7).
"""

import concurrent.futures
import json

from loguru import logger
from openai import OpenAI

from llm_engineering.settings import settings

EVAL_SYSTEM_PROMPT = (
    "You are a helpful assistant who evaluates answers based on accuracy and style. "
    "Provide your response in JSON format with a short analysis and score for each criterion."
)

EVAL_USER_PROMPT = """You are an expert judge. Please evaluate the quality of a given answer to an instruction based on two criteria:
1. Accuracy: How factually correct is the information presented in the answer? You are a technical expert in this topic.
2. Style: Is the tone and writing style appropriate for a blog post or social media content? It should use simple but technical words and avoid formal or academic language.

Accuracy scale:
1 (Poor): Contains factual errors or misleading information
2 (Good): Mostly accurate with minor errors or omissions
3 (Excellent): Highly accurate and comprehensive

Style scale:
1 (Poor): Too formal, uses some overly complex words
2 (Good): Good balance of technical content and accessibility, but still uses formal words and expressions
3 (Excellent): Perfectly accessible language for blog/social media, uses simple but precise technical terms when necessary

Instruction: {instruction}

Answer: {answer}

Provide your evaluation in JSON format with the following structure:
{{
    "accuracy": {{
        "analysis": "...",
        "score": 0
    }},
    "style": {{
        "analysis": "...",
        "score": 0
    }}
}}"""


def evaluate_answer(instruction: str, answer: str, client: OpenAI | None = None) -> dict:
    """Evaluate a single instruction-answer pair using GPT-4o-mini."""

    if client is None:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

    prompt = EVAL_USER_PROMPT.format(instruction=instruction, answer=answer)

    completion = client.chat.completions.create(
        model=settings.OPENAI_MODEL_ID,
        messages=[
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=500,
        temperature=0.9,
    )

    return json.loads(completion.choices[0].message.content)


def _evaluate_batch(batch: list[tuple[str, str]], start_index: int, client: OpenAI) -> list[tuple[int, dict]]:
    """Evaluate a batch of (instruction, answer) pairs."""
    results = []
    for i, (instruction, answer) in enumerate(batch, start=start_index):
        try:
            result = evaluate_answer(instruction, answer, client)
            results.append((i, result))
        except Exception as e:
            logger.warning(f"Failed to evaluate sample {i}: {e}")
            results.append(
                (i, {"accuracy": {"analysis": "error", "score": 0}, "style": {"analysis": "error", "score": 0}})
            )
    return results


def evaluate_dataset(
    samples: list[dict],
    answer_key: str = "answer",
    num_threads: int = 4,
    batch_size: int = 5,
    max_samples: int | None = None,
) -> list[dict]:
    """Evaluate a dataset of instruction-answer pairs using LLM-as-judge.

    Args:
        samples: List of dicts with 'instruction' and answer_key fields.
        answer_key: Key for the answer field ('answer' for instruct, 'chosen' for preference).
        num_threads: Number of parallel threads for OpenAI calls.
        batch_size: Samples per batch per thread.
        max_samples: If set, only evaluate this many samples (for cost control).

    Returns:
        List of evaluation dicts with accuracy/style scores and analysis.
    """

    eval_samples = samples[:max_samples] if max_samples else samples
    pairs = [(s["instruction"], s[answer_key]) for s in eval_samples]

    batches = []
    for i in range(0, len(pairs), batch_size):
        batches.append((i, pairs[i : i + batch_size]))

    evaluations: list[dict | None] = [None] * len(eval_samples)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for start_index, batch in batches:
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            futures.append(executor.submit(_evaluate_batch, batch, start_index, client))

        for future in concurrent.futures.as_completed(futures):
            for index, evaluation in future.result():
                evaluations[index] = evaluation

    return evaluations
