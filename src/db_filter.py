import pandas as pd
import re

themes = [
    'mateIn1',
    'mateIn2',
    'mateIn3',
    'mateIn4',
    'mateIn5'
]


if __name__ == '__main__':
    df = pd.read_csv('../dataset/lichess_db_puzzle.csv')

    df = df[df['Themes'].str.contains(r'mateIn\d', na=False)]

    c = []
    for t in themes:
        c.append(len(df[df['Themes'].str.contains(t)]))

    for i, n in enumerate(c):
        print(f'mateIn{i+1}: {n/len(df)}% - {n}')

    df.to_csv('../dataset/lichess_db_puzzle_mates_only.csv')
