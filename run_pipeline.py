# .ndmh/run_pipeline.py
import argparse
import sys
from marketml.utils import logger_setup # For initial setup if needed
from marketml.configs import configs # To access any global flags

# Optionally, set up a basic logger for this orchestrator script itself
# This is separate from the per-pipeline loggers
orchestrator_logger = logger_setup.setup_basic_logging(log_file_name="orchestrator.log")

def run_stage(stage_name):
    orchestrator_logger.info(f"Attempting to run stage: {stage_name}")
    if stage_name == "build_features":
        from marketml.pipelines import builder_features_01  # Use updated name
        builder_features_01.main()
    elif stage_name == "train_forecasting":
        from marketml.pipelines import train_forecasting_models_02 # Use updated name
        train_forecasting_models_02.main() # Assuming the main function is called 'main'
    elif stage_name == "generate_forecasts":
        from marketml.pipelines import generate_forecasts_03
        generate_forecasts_03.main()
    elif stage_name == "generate_signals":
        from marketml.pipelines import generate_signals_04
        generate_signals_04.main()
    elif stage_name == "run_backtests":
        from marketml.pipelines import run_portfolio_backtests_05
        run_portfolio_backtests_05.main()
    elif stage_name == "analyze_performance":
        from marketml.analysis import analyze_model_performance
        analyze_model_performance.main()
    # Add more stages as needed
    else:
        orchestrator_logger.error(f"Unknown stage: {stage_name}")
        sys.exit(1)
    orchestrator_logger.info(f"Successfully completed stage: {stage_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MarketML Pipeline Orchestrator")
    parser.add_argument(
        "stages",
        nargs="+",
        help="Name of the pipeline stage(s) to run (e.g., build_features, train_forecasting, all)",
        choices=["build_features", "train_forecasting", "generate_forecasts", "generate_signals", "run_backtests", "analyze_performance", "all"]
    )
    # Example: python run_pipeline.py build_features train_forecasting

    args = parser.parse_args()

    all_stages_ordered = ["build_features", "train_forecasting", "generate_forecasts", "generate_signals", "run_backtests", "analyze_performance"]

    stages_to_run = []
    if "all" in args.stages:
        stages_to_run = all_stages_ordered
    else:
        stages_to_run = [s for s in all_stages_ordered if s in args.stages] # Keep order

    orchestrator_logger.info(f"Pipeline starting with stages: {', '.join(stages_to_run)}")
    for stage in stages_to_run:
        try:
            run_stage(stage)
        except Exception as e:
            orchestrator_logger.error(f"Error running stage {stage}: {e}", exc_info=True)
            # Decide if you want to stop or continue with other stages
            # sys.exit(1) # Uncomment to stop on first error
    orchestrator_logger.info("Pipeline finished.")