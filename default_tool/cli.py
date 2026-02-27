from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_DATA_ROOT, DEFAULT_MODEL_DIR, DYNAMIC_MODEL_ORDER
from .inference import explain_static_features, load_models, predict_path
from .static_features import extract_static_features_from_keras
from .trainer import train_all_models


def _print_prediction_table(results: list[dict[str, Any]]) -> None:
    header = f"{'File':56} {'Fault?':7} Categories"
    print(header)
    print("-" * len(header))
    for row in results:
        categories = ",".join(row["predicted_categories"]) if row["predicted_categories"] else "-"
        file_name = Path(row["file"]).name
        print(f"{file_name[:56]:56} {str(row['detected_fault']):7} {categories}")

    print("\nClassifier probabilities:")
    for row in results:
        print(f"- {Path(row['file']).name}")
        for name in DYNAMIC_MODEL_ORDER:
            clf = row["classifiers"][name]
            print(
                f"  {name:14} p={clf['probability']:.4f} "
                f"thr={clf['threshold']:.2f} "
                f"pred={int(clf['predicted_positive'])}"
            )


def _cmd_train(args: argparse.Namespace) -> int:
    manifest = train_all_models(
        data_root=Path(args.data_root),
        model_dir=Path(args.model_dir),
        random_state=args.random_state,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    models = load_models(Path(args.model_dir))
    results = predict_path(Path(args.input), models)
    if args.json:
        payload = json.dumps(results, indent=2)
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return 0

    _print_prediction_table(results)
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    models = load_models(Path(args.model_dir))
    if args.keras_model:
        features = extract_static_features_from_keras(Path(args.keras_model))
    else:
        features = pd.read_csv(args.static_features_csv).head(1)

    report = explain_static_features(features, models, top_n=args.top_n)
    payload = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="default-tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and persist all models.")
    train_parser.add_argument(
        "--data-root", default=str(DEFAULT_DATA_ROOT), help="Path to d_DEFault data root."
    )
    train_parser.add_argument(
        "--model-dir", default=str(DEFAULT_MODEL_DIR), help="Output directory for model artifacts."
    )
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.set_defaults(func=_cmd_train)

    predict_parser = subparsers.add_parser("predict", help="Run dynamic prediction on CSV file(s).")
    predict_parser.add_argument("--input", required=True, help="CSV file or directory of CSV files.")
    predict_parser.add_argument(
        "--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory containing trained artifacts."
    )
    predict_parser.add_argument("--json", action="store_true", help="Print predictions as JSON.")
    predict_parser.add_argument("--output", help="Optional output path.")
    predict_parser.set_defaults(func=_cmd_predict)

    explain_parser = subparsers.add_parser(
        "explain", help="Run static root-cause explanation from features or a Keras model."
    )
    source_group = explain_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--static-features-csv", help="CSV with static feature columns.")
    source_group.add_argument("--keras-model", help="Path to .h5/.keras model.")
    explain_parser.add_argument(
        "--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory containing trained artifacts."
    )
    explain_parser.add_argument("--top-n", type=int, default=5)
    explain_parser.add_argument("--output", help="Optional JSON output path.")
    explain_parser.set_defaults(func=_cmd_explain)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

