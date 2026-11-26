"""Command-line interface for training, conversion, prediction, and demo runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from nids.config import NIDSConfig
from nids.data import convert_csv_to_parquet, generate_synthetic_traffic, load_dataset
from nids.predict import load_model, predict_traffic
from nids.train import train_and_evaluate


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="nids",
        description="Network Intrusion Detection System (UNSW-NB15) CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # demo
    demo_cmd = subparsers.add_parser(
        "demo",
        help="Run an end-to-end demo using synthetic network traffic.",
    )
    demo_cmd.add_argument("--samples", type=int, default=600, help="Number of synthetic rows.")

    # train
    train_cmd = subparsers.add_parser("train", help="Train and benchmark NIDS models.")
    train_cmd.add_argument("--train-path", type=Path, default=Path("dataset/train_full.parquet"))
    train_cmd.add_argument("--test-path", type=Path, default=Path("dataset/test_full.parquet"))

    # predict
    predict_cmd = subparsers.add_parser("predict", help="Score traffic records with a saved model.")
    predict_cmd.add_argument("--input", type=Path, required=True, help="Input .parquet or .csv file.")
    predict_cmd.add_argument("--model", type=Path, default=Path("artifacts/best_model.joblib"))
    predict_cmd.add_argument("--output", type=Path, default=Path("artifacts/predictions.csv"))

    # convert
    convert_cmd = subparsers.add_parser("convert", help="Convert CSV dataset to Parquet.")
    convert_cmd.add_argument("--csv", type=Path, required=True, help="Input CSV path.")
    convert_cmd.add_argument("--parquet", type=Path, required=True, help="Output Parquet path.")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = NIDSConfig()

    if args.command == "demo":
        df = generate_synthetic_traffic(n_samples=args.samples, random_state=config.random_state)
        train_df, test_df = train_test_split(
            df,
            test_size=0.25,
            random_state=config.random_state,
            stratify=df[config.target_column],
        )
        summary = train_and_evaluate(train_df, test_df, config=config, save_artifacts=True)
        print(f"Best model: {summary['best_model_name']}")
        print(summary["leaderboard"].to_string(index=False))

        scored = predict_traffic(test_df.head(5), summary["best_model"], config=config)
        print("\nSample predictions:")
        print(scored.to_string(index=False))
        return 0

    if args.command == "train":
        train_df = load_dataset(args.train_path)
        test_df = load_dataset(args.test_path)
        summary = train_and_evaluate(train_df, test_df, config=config, save_artifacts=True)
        print(f"Best model: {summary['best_model_name']}")
        print(summary["leaderboard"].to_string(index=False))
        return 0

    if args.command == "predict":
        df = load_dataset(args.input)
        model = load_model(args.model)
        scored = predict_traffic(df, model, config=config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(args.output, index=False)
        print(f"Saved {len(scored)} predictions to {args.output}")
        return 0

    if args.command == "convert":
        out_path = convert_csv_to_parquet(args.csv, args.parquet)
        print(f"Converted {args.csv} -> {out_path}")
        return 0

    parser.print_help()
    return 1
