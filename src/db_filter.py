import pandas as pd
import re

if __name__ == '__main__':
    df = pd.read_csv('../dataset/lichess_db_puzzle.csv')

    # get mateInN problems
    pattern = r'\b(mateIn[1-5])\b'
    df = df[df['Themes'].str.contains(
        r'\b(mateIn[1-5])\b', regex=True, na=False)].copy()

    # extract mate tag into a new column
    df['MateIn'] = df['Themes'].str.extract(
        r'\bmateIn([1-5])\b', expand=False).astype('Int64')

    counts = df['MateIn'].value_counts()

    for n in range(1, 6):
        c = counts.get(n)
        print(f'MateIn{n}: {c} - {c/len(df) or 0:.3f}%')

    df.to_csv('../dataset/lichess_db_puzzle_mates_only.csv')
