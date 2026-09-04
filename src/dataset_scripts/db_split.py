import argparse
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./data_splits",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--stratify",
        type=str,
        default='MateIn',
        help="If set, preserve the column distribution across splits (default: MateIn).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float):
    total = train_ratio + val_ratio + test_ratio
    if not (0.999 <= total <= 1.001):
        raise ValueError(
            f"Split ratios must sum to 1.0. Current sum: {total:.4f}"
        )
    if any(r <= 0 for r in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("All split ratios must be strictly positive.")


if __name__ == "__main__":
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    if not os.path.exists(args.input_path):
        print(f"Error: Input file {args.input_path} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset from: {args.input_path}")
    df = pd.read_csv(args.input_path)
    total_samples = len(df)
    print(f"Loaded {total_samples:,} records.")

    stratify_target = None
    if args.stratify:
        if args.stratify not in df.columns:
            raise KeyError(
                f"Column '{args.stratify}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )
        # convert null entries to a common class
        # WARNING:
        # Default value -1 is a temporary choice, we'll assess it later
        df[args.stratify] = df[args.stratify].fillna(-1)
        stratify_target = df[args.stratify]

    # split off the training set
    train_size = args.train_ratio
    temp_size = args.val_ratio + args.test_ratio

    df_train, df_temp = train_test_split(
        df,
        train_size=train_size,
        stratify=stratify_target,
        random_state=args.seed,
    )

    # split into validation and test
    temp_stratify = df_temp[args.stratify] if args.stratify else None
    val_proportion_in_temp = args.val_ratio / temp_size

    df_val, df_test = train_test_split(
        df_temp,
        train_size=val_proportion_in_temp,
        stratify=temp_stratify,
        random_state=args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train.csv")
    val_path = os.path.join(args.output_dir, "val.csv")
    test_path = os.path.join(args.output_dir, "test.csv")

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    print("\n--- Split Summary ---")
    print(f"Train set:      {len(df_train):,} rows ({len(df_train)/total_samples:.1%}) -> {train_path}")
    print(f"Validation set: {len(df_val):,} rows ({len(df_val)/total_samples:.1%}) -> {val_path}")
    print(f"Test set:       {len(df_test):,} rows ({len(df_test)/total_samples:.1%}) -> {test_path}")

    if args.stratify:
        print(f"\n--- {args.stratify} Distribution (%) ---")
        dist = pd.DataFrame({
            "Train": df_train[args.stratify].value_counts(normalize=True) * 100,
            "Val": df_val[args.stratify].value_counts(normalize=True) * 100,
            "Test": df_test[args.stratify].value_counts(normalize=True) * 100,
        }).round(2)
        print(dist.to_string())
