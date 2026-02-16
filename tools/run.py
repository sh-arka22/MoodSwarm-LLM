from datetime import datetime as dt
from pathlib import Path

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
@click.option("--run-etl", is_flag=True, default=False, help="Run the ETL pipeline.")
@click.option(
    "--etl-config-filename",
    default="digital_data_etl.yaml",
    help="Filename of the ETL config file.",
)
def main(
    no_cache: bool = False,
    run_smoke_test: bool = False,
    run_etl: bool = False,
    etl_config_filename: str = "digital_data_etl.yaml",
) -> None:
    assert run_smoke_test or run_etl, "Please specify an action to run. Available: --run-smoke-test, --run-etl"

    pipeline_args = {
        "enable_cache": not no_cache,
    }
    root_dir = Path(__file__).resolve().parent.parent

    if run_smoke_test:
        from pipelines.smoke_test import smoke_test_pipeline

        pipeline_args["run_name"] = f"smoke_test_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        logger.info("Running smoke test pipeline...")
        smoke_test_pipeline.with_options(**pipeline_args)()

    if run_etl:
        from pipelines.digital_data_etl import digital_data_etl

        pipeline_args["config_path"] = root_dir / "configs" / etl_config_filename
        assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = f"digital_data_etl_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        logger.info(f"Running ETL pipeline with config: {etl_config_filename}")
        digital_data_etl.with_options(**pipeline_args)()


if __name__ == "__main__":
    main()
