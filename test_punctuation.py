import pandas as pd
import numpy as np
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

train = pd.read_csv('data/train.csv')
print(f"Loaded {len(train)} rows across {train['model'].nunique()} models.")

# Check apostrophes
models = sorted(train['model'].unique())
print(f"\n{'Model':25s} | {'Straight Apos':13s} | {'Curly Apos':10s} | {'Straight Quot':13s} | {'Curly Quot':10s} | {'Em Dash':7s}")
print("-" * 90)

for m in models:
    df_m = train[train['model'] == m]
    straight_apos = df_m['text'].apply(lambda t: t.count("'")).sum()
    curly_apos = df_m['text'].apply(lambda t: t.count("\u2019") + t.count("\u2018")).sum()
    straight_quot = df_m['text'].apply(lambda t: t.count('"')).sum()
    curly_quot = df_m['text'].apply(lambda t: t.count("\u201c") + t.count("\u201d")).sum()
    em_dash = df_m['text'].apply(lambda t: t.count("\u2014")).sum()
    print(f"{m:25s} | {straight_apos:13d} | {curly_apos:10d} | {straight_quot:13d} | {curly_quot:10d} | {em_dash:7d}")

