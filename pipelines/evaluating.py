"""Evaluation pipeline — generates answers via vLLM and scores with LLM-as-judge."""

from zenml import pipeline

from steps import evaluating as evaluating_steps


@pipeline
def evaluating(
    is_dummy: bool = False,
) -> None:
    evaluating_steps.evaluate(
        is_dummy=is_dummy,
    )
