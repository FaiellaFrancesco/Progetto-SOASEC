# Run (TRAINING split only):
#   python src/chess_eval/preprocessing/db_balance.py \
#       -i dataset/train.csv -o dataset/train_balanced.csv --cap 60000
#
# Sub-samples the training split so the unrolled examples are not swamped by
# mate-in-1 and one epoch fits a Colab session.
#
# Unrolling makes every puzzle, whatever its depth, contribute exactly one
# n_remaining=1 example, so ~60% of all examples are "mate in one from here":
# a model that only learns the immediate mate scores ~60% and looks good, and
# the full set is 3.24M examples (~32 min/epoch just to read). Capping the rows
# per MateIn flattens the depth distribution and shrinks the set; the rare
# depths (n=4, 5) are kept in full. Never balance val/test -- they must keep the
# real distribution or their accuracy means nothing.

import argparse
from collections import Counter

import pandas as pd


def examples_by_depth(rows_by_mate):
    """Turn {MateIn: n_rows} into {n_remaining: n_examples}.

    A row with MateIn=n yields one example at each depth n, n-1, ..., 1, so the
    examples at depth k come from every row with MateIn >= k.
    """
    out = Counter()
    for n, count in rows_by_mate.items():
        for k in range(1, n + 1):
            out[k] += count
    return out


def show(title, rows_by_mate):
    ex = examples_by_depth(rows_by_mate)
    total_rows = sum(rows_by_mate.values())
    total_ex = sum(ex.values())
    print(f"\n{title}")
    print(f"{'n':>3} {'righe':>12} {'esempi a n_remaining=n':>24} {'%':>8}")
    for n in sorted(set(rows_by_mate) | set(ex)):
        print(f"{n:>3} {rows_by_mate.get(n, 0):>12,} {ex.get(n, 0):>24,} "
              f"{ex.get(n, 0) / max(total_ex, 1):>7.1%}")
    print(f"{'tot':>3} {total_rows:>12,} {total_ex:>24,}")
    return total_ex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_csv", required=True,
                        help="the TRAINING split only")
    parser.add_argument("-o", "--output_csv", required=True)
    parser.add_argument("--cap", type=int, default=60000,
                        help="max rows kept per MateIn value")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    before = Counter(df["MateIn"].astype(int))
    total_before = show("PRIMA", before)

    parts = []
    for n, group in df.groupby(df["MateIn"].astype(int)):
        if len(group) > args.cap:
            group = group.sample(args.cap, random_state=args.seed)
        parts.append(group)

    out = pd.concat(parts).sample(frac=1, random_state=args.seed)  # shuffle
    out.to_csv(args.output_csv, index=False)

    after = Counter(out["MateIn"].astype(int))
    total_after = show("DOPO", after)

    print(f"\nscritto {args.output_csv}")
    print(f"esempi: {total_before:,} -> {total_after:,} "
          f"({total_after / max(total_before, 1):.1%})")
    # measured on this machine: the DataLoader serves ~1700 examples/s
    print(f"tempo di lettura stimato per epoca: "
          f"{total_after / 1700 / 60:.1f} min  (era {total_before / 1700 / 60:.1f})")


if __name__ == "__main__":
    main()