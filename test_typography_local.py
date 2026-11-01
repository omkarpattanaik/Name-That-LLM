import pandas as pd
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/train.csv')
print(f'Train shape: {df.shape}')

marks = {
    'curly_apostrophe': '\u2019',
    'left_single_quote': '\u2018',
    'left_double_quote': '\u201c',
    'right_double_quote': '\u201d',
    'em_dash': '\u2014',
    'en_dash': '\u2013',
    'ellipsis': '\u2026',
    'straight_single': "'",
    'straight_double': '"'
}

for name, mark in marks.items():
    rates = df.groupby('model')['text'].apply(lambda s: (s.str.contains(mark, regex=False)).mean())
    top_models = rates.sort_values(ascending=False).head(3)
    bottom_models = rates.sort_values().head(3)
    print(f'=== {name} ({mark}) ===')
    print('  Highest:', ', '.join([f'{m}: {r:.1%}' for m, r in top_models.items()]))
    print('  Lowest: ', ', '.join([f'{m}: {r:.1%}' for m, r in bottom_models.items()]))

