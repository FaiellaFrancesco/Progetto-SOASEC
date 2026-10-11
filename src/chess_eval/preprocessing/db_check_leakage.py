# Run (after splitting, before building any graph):
#   python src/chess_eval/preprocessing/db_check_leakage.py \
#       -s dataset/train.csv dataset/val.csv dataset/test.csv
#
# Checks that no PuzzleId appears in more than one split. Unrolled examples from
# one puzzle are near-identical positions, so a puzzle in both train and test
# means the model has already seen the answer and every number in the report is
# worthless -- and leakage makes a model look BETTER, so accuracy can never
# catch it. Takes seconds; exits non-zero on any overlap, so it can go in CI.

import argparse
import sys
from itertools import combinations

import pandas as pd


def load_ids(paths):
    """Read the PuzzleId set of each CSV: returns {path: set of ids}."""
    ids = {}
    for p in paths:
        df = pd.read_csv(p, usecols=["PuzzleId"])
        ids[p] = set(df["PuzzleId"])
        dupes = len(df) - len(ids[p])
        print(f"{p:<40} {len(df):>10,} righe, {len(ids[p]):>10,} id unici"
              f"{f'  ATTENZIONE: {dupes:,} duplicati interni' if dupes else ''}")
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--splits", nargs="+", required=True,
                        help="the split CSVs, e.g. train.csv val.csv test.csv")
    args = parser.parse_args()

    print()
    ids = load_ids(args.splits)
    print()

    ok = True
    for a, b in combinations(args.splits, 2):
        shared = ids[a] & ids[b]
        if shared:
            ok = False
            print(f"[FAIL] {a} e {b} condividono {len(shared):,} PuzzleId")
            print(f"       esempi: {sorted(shared)[:5]}")
        else:
            print(f"[PASS] {a} e {b} sono disgiunti")

    # a puzzle repeated inside one split is not leakage, but it does inflate
    # its weight in training and its count in the test report
    for p, s in ids.items():
        n = len(pd.read_csv(p, usecols=["PuzzleId"]))
        if n != len(s):
            ok = False
            print(f"[FAIL] {p} contiene {n - len(s):,} PuzzleId ripetuti")

    print()
    if ok:
        print("nessun leakage: gli split sono disgiunti")
        return 0
    print("LEAKAGE TROVATO: non addestrare finche' non e' risolto")
    return 1


if __name__ == "__main__":
    sys.exit(main())