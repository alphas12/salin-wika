# AI Amended: Connect the project through a minimal two-command CLI driven only by config.yaml.
import argparse

from pathlib import Path

import yaml

from utils.inference import translate_from_results
from utils.pipelines import TrainingPipeline


CONFIG_PATH = Path("config.yaml")
RESULTS_DIR = Path("results")


def load_config(config_path=CONFIG_PATH):
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"'{config_path}' must contain a YAML mapping.")

    return config


def run_training(config):
    RESULTS_DIR.mkdir(exist_ok=True)
    TrainingPipeline(config=config).train(results_dir=RESULTS_DIR)


def run_translation(config):
    translation = translate_from_results(config=config, results_dir=RESULTS_DIR)
    print(translation)


def create_parser():
    parser = argparse.ArgumentParser(description="Train or use the Seq2Seq translator.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("training", help="Train and evaluate the configured model.")
    subparsers.add_parser("translation", help="Translate the text in config.yaml.")
    return parser


def main():
    args = create_parser().parse_args()
    config = load_config()

    if args.command == "training":
        run_training(config)
    else:
        run_translation(config)


if __name__ == "__main__":
    main()
