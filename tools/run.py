from datetime import datetime as dt

import click
from loguru import logger


@click.command(
    help="""
MoodSwarm LLM Engineering CLI v0.1.0.

Main entry point for pipeline execution.
"""
)
@click.option("--no-cache", is_flag=True, default=False, help="Disable caching for the pipeline run.")
@click.option("--run-smoke-test", is_flag=True, default=False, help="Run the smoke test pipeline.")
def main(
    no_cache: bool = False,
    run_smoke_test: bool = False,
) -> None:
    assert run_smoke_test, "Please specify an action to run. Available: --run-smoke-test"

    pipeline_args = {
        "enable_cache": not no_cache,
    }

    if run_smoke_test:
        from pipelines.smoke_test import smoke_test_pipeline

        pipeline_args["run_name"] = f"smoke_test_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        logger.info("Running smoke test pipeline...")
        smoke_test_pipeline.with_options(**pipeline_args)()


if __name__ == "__main__":
    main()
