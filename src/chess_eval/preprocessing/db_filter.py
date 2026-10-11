# Run (no arguments; paths are hardcoded, relative to the working directory):
#   python db_filter.py          # run it where ../dataset/ holds the raw dump
#   in:  ../dataset/lichess_db_puzzle.csv
#   out: ../dataset/lichess_db_puzzle_mates_only.csv
#
# Keeps only the mate-in-1..5 puzzles from the raw Lichess dump and adds a
# MateIn column (1..5) pulled from the Themes. First step of the pipeline.

import pandas as pd
import re

if __name__ == '__main__':
    df = pd.read_csv('../dataset/lichess_db_puzzle.csv')

    # keep only mate-in-1..5 puzzles
    pattern = r'\b(mateIn[1-5])\b'
    df = df[df['Themes'].str.contains(
        r'\b(mateIn[1-5])\b', regex=True, na=False)].copy()

    # pull the mate depth into its own column
    df['MateIn'] = df['Themes'].str.extract(
        r'\bmateIn([1-5])\b', expand=False).astype('Int64')

    counts = df['MateIn'].value_counts()

    for n in range(1, 6):
        c = int(counts.get(n, 0))
        pct = (c / len(df)) if len(df) else 0.0
        print(f"MateIn{n}: {c} - {pct:.3f}%")

    df.to_csv("../dataset/lichess_db_puzzle_mates_only.csv", index=False)