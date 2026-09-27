"""
db_dropped_stats.py

Counts the rows that row_to_graph throws away, and says WHY.

db_rows_to_graphs.py already prints a drop total, but to get it you have to
build the whole dataset: ~15 minutes and several GB of parquet for a single
number. This script runs the same guards, in the same order, without building
a single graph, so the full CSV takes about a minute.

    python db_dropped_stats.py -i ../../dataset/lichess_db_puzzle_mates_only.csv

The reasons are mutually exclusive and evaluated in the order row_to_graph
uses, so a row is attributed to the FIRST guard that rejects it.

WARNING: this file duplicates the guards of row_to_graph. If a guard changes
there, change it here too, or the numbers quietly stop meaning anything.
"""

import argparse
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import chess
import pandas as pd
from tqdm import tqdm

from fen_to_graph import normalize_row

CHUNK_SIZE = 20000
WORKERS = 8

KEPT = "kept"
REASONS = [
    "no moves / odd move count",
    "MateIn tag disagrees with the move count",
    "invalid FEN or illegal setup move",
    "solver under-promotion",
    "corrupt solution line",
]


"""
INPUT: one row of the CSV, as a dict
OUTPUT: the string KEPT, or the reason the row would be dropped

Mirrors the guards at the top of row_to_graph, in the same order.
"""


def diagnose_row(row):
    row = normalize_row(row)
    moves = row["Moves"]

    # guard 1: there has to be at least one full move pair
    n_total = len(moves) // 2
    if n_total < 1:
        return REASONS[0]

    # guard 2: the MateIn column has to agree with the length of the line
    try:
        tagged = int(row["MateIn"])
    except (KeyError, ValueError, TypeError):
        tagged = None
    if tagged is not None and tagged > 0 and tagged != n_total:
        return REASONS[1]

    # guard 3: the FEN has to parse and the opponent move has to be legal
    try:
        board = chess.Board(row["FEN"])
        board.push_uci(moves[0])
    except ValueError:
        return REASONS[2]
    solution = moves[1:]

    # guard 4: we only keep queen promotions
    if any(len(m) == 5 and m[4] in "rbn" for m in solution[::2]):
        return REASONS[3]

    # guard 5: the whole line has to replay cleanly
    replay = board.copy()
    try:
        for uci in solution:
            replay.push_uci(uci)
    except ValueError:
        return REASONS[4]

    return KEPT


"""
INPUT: a chunk of the CSV as a DataFrame
OUTPUT: (Counter of reasons, Counter of mate depths among the kept rows)

Returns Counters rather than touching a shared total: this runs in a separate
process and cannot see the parent's variables.
"""


def diagnose_chunk(chunk):
    reasons = Counter()
    depths = Counter()
    for record in chunk.to_dict("records"):
        why = diagnose_row(record)
        reasons[why] += 1
        if why == KEPT:
            depths[len(str(record["Moves"]).split()) // 2] += 1
    return reasons, depths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_csv", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--rows", type=int, default=None,
                        help="only look at the first N rows (for a quick try)")
    args = parser.parse_args()

    with open(args.input_csv, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f) - 1
    if args.rows:
        total_lines = min(total_lines, args.rows)
    total_chunks = (total_lines + args.chunk_size - 1) // args.chunk_size

    reader = pd.read_csv(args.input_csv, chunksize=args.chunk_size,
                         nrows=args.rows)

    reasons = Counter()
    depths = Counter()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r, d in tqdm(pool.map(diagnose_chunk, reader),
                         total=total_chunks, desc="Scanning rows"):
            reasons.update(r)
            depths.update(d)

    total = sum(reasons.values())
    kept = reasons[KEPT]
    dropped = total - kept

    print(f"\nfile   : {os.path.basename(args.input_csv)}")
    print(f"rows   : {total:,}")
    print(f"kept   : {kept:,}  ({kept / max(total, 1):.4%})")
    print(f"dropped: {dropped:,}  ({dropped / max(total, 1):.4%})")

    print("\nwhy the dropped rows were dropped")
    print(f"{'reason':<45} {'rows':>10} {'% of file':>11} {'% of drops':>11}")
    for reason in REASONS:
        n = reasons[reason]
        print(f"{reason:<45} {n:>10,} {n / max(total, 1):>10.4%} "
              f"{n / max(dropped, 1):>10.2%}")

    print("\nmate depth of the rows that survive")
    print(f"{'MateIn':<10} {'rows':>12} {'%':>9} {'examples':>14}")
    examples = 0
    for n in sorted(depths):
        rows_n = depths[n]
        examples += n * rows_n          # unrolling gives n examples per row
        print(f"{n:<10} {rows_n:>12,} {rows_n / max(kept, 1):>8.2%} "
              f"{n * rows_n:>14,}")
    print(f"{'total':<10} {kept:>12,} {'':>9} {examples:>14,}")
    if kept:
        print(f"\nexamples per kept row: {examples / kept:.3f}")
        print("this is the number db_rows_to_graphs.py must print when it "
              "builds the unrolled dataset: if it does not match, the driver "
              "is losing examples somewhere.")


if __name__ == "__main__":
    main()
