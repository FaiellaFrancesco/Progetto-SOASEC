"""
db_check_leakage.py

Checks that no PuzzleId appears in more than one split.

    python db_check_leakage.py -s train.csv val.csv test.csv

The examples unrolled from one puzzle are almost identical positions: the same
board one ply apart. If a puzzle lands in both train and test, the model has
already seen the answer and every number in the report is worthless. This is
the single failure that cannot be detected after the fact by looking at the
accuracy, because a leaking model looks BETTER, not worse.

Run it once after splitting, before building any graph. It takes seconds.
Exits non-zero on any overlap, so it can be wired into CI.
"""

import argparse
import sys
from itertools import combinations

import pandas as pd


"""
INPUT: a list of CSV paths
OUTPUT: a dict {path: set of PuzzleId}
"""


def load_ids(paths):
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
