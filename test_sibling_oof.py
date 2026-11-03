import re
import numpy as np
import pandas as pd
from functools import lru_cache
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score

df = pd.read_csv('data/train.csv')
texts = df.text.values
labels = sorted(df.model.unique())
y = np.array([labels.index(m) for m in df.model.values])
groups = df.prompt_id.values

# Version 1: Zbigniew baseline (corrupted characters)
MARKERS_ZB = [',','.', ';', ':', '!', '?','-','"',"'",'(',')','[',']','{','}','*','`','/','\\',':','\n','\n\n','\n-','\n1.','\n2.','##','```','**','> ','\t',' = ', ' e.g.', ' i.e.', 'However,','Overall,','In conclusion','Therefore,','Moreover,','First,','Finally,',' - ',': ', ' ; ', '  ']
WORDS = ['the','and','but','that','which','with','into','from','for','you','your','our','we','I','it','its','they','their','there','this','these','those','however','therefore','moreover','although','because','perhaps','might','would','could','should','like','such','just','very','also','not','if','or','as','so','by','of','to','in','on','at','is','are','be','was','were','have','has','had','may','must','all','any','each','every','many','some','more','most','only','even','still','then','when','where','why','how','what','who']
WRE_ZB = re.compile(r"[A-Za-z]+(?:['][A-Za-z]+)?")
SRE = re.compile(r'(?<=[.?!])\s+')

def row_zb(t):
    t = str(t)
    s = WRE_ZB.findall(t); lo=[z.lower() for z in s]; n=max(len(s),1); nt=max(len(t),1); ns=max(len(t.splitlines()),1)
    lens=np.array([len(z) for z in s] or [0]); sent_lens=np.array([len(WRE_ZB.findall(z)) for z in SRE.split(t) if z.strip()] or [0])
    base = [np.log1p(n),np.log1p(nt), nt/n,ns/n,len(set(lo))/n, float(np.mean(lens)), float(np.std(lens)), float(np.mean(sent_lens)), float(np.std(sent_lens)),float(np.max(sent_lens)),float(np.mean(lens>=8)),sum(c.isupper() for c in t)/nt, sum(c.isdigit() for c in t)/nt,sum(ord(c)>127 for c in t)/nt, sum(c.isspace() for c in t)/nt, sum(c=='\n' for c in t)/nt]
    base += [min(t.count(m)/n,3) for m in MARKERS_ZB]
    cnt = {}
    for w in lo: cnt[w]=cnt.get(w,0)+1
    base += [cnt.get(w,0)/n for w in WORDS]
    return np.array(base, dtype=np.float32)

# Version 2: Uncorrupted Unicode + 16 Sibling Cliff Features
MARKERS_V14 = [
    ',', '.', ';', ':', '!', '?',
    '\u2014', '\u2013', '-', '\u2018', '\u2019', '\u201c', '\u201d',
    '"', "'", '(', ')', '[', ']', '{', '}', '*', '`',
    '\u2026', '\u00b7', '/', '\\',
    '\n', '\n\n', '\n-', '\n1.', '\n2.', '##', '```', '**', '> ',
    '\t', ' = ', ' e.g.', ' i.e.', 'However,', 'Overall,', 'In conclusion',
    'Therefore,', 'Moreover,', 'First,', 'Finally,', ' - ', ': ', ' ; ', '  '
]
HEDGES = {'might', 'could', 'perhaps', 'generally', 'typically', 'often', 'usually', 'arguably', 'may'}
CONNECTIVES = {'consequently', 'furthermore', 'moreover', 'specifically', 'therefore', 'thus', 'hence', 'notably', 'crucially', 'ultimately'}
DEMONSTRATIVES = {'this', 'these', 'it', 'they'}
WRE_V14 = re.compile(r"[A-Za-z]+(?:[\u2019'][A-Za-z]+)?")

def row_v14(t):
    t = str(t)
    words = WRE_V14.findall(t)
    words_lower = [w.lower() for w in words]
    nw = max(len(words), 1)
    nt = max(len(t), 1)
    fac = 100.0 / nw
    lens = np.array([len(w) for w in words], dtype=np.float32) if words else np.zeros(1, np.float32)
    sents = [st.strip() for st in SRE.split(t) if st.strip()] or [t]
    slens = np.array([len(st.split()) for st in sents], dtype=np.float32)
    sm = float(slens.mean())
    ss = float(slens.std())
    siqr = float(np.percentile(slens, 75) - np.percentile(slens, 25)) if len(slens) > 1 else 0.0
    first_sent_ratio = slens[0] / max(sm, 1.0)
    last_sent_ratio = slens[-1] / max(sm, 1.0)
    demo_cnt = sum(1 for st in sents if st.split() and st.split()[0].lower() in DEMONSTRATIVES)
    ly_cnt = sum(1 for w in words_lower if w.endswith('ly'))
    ing_cnt = sum(1 for w in words_lower if w.endswith('ing'))
    tion_cnt = sum(1 for w in words_lower if w.endswith(('tion', 'sion')))
    unspaced_emdash = len(re.findall(r'\S\u2014\S', t)) * fac
    spaced_emdash = len(re.findall(r'\s\u2014\s', t)) * fac
    semi_to_comma = t.count(';') / max(t.count(','), 1)
    long_word_rate = sum(1 for w in words if len(w) >= 9) / nw
    hedge_rate = sum(1 for w in words_lower if w in HEDGES) * fac
    conn_rate = sum(1 for w in words_lower if w in CONNECTIVES) * fac
    sent_initial_conn = sum(1 for s in sents if s.split() and s.split()[0].lower().strip(',') in CONNECTIVES) / max(len(sents), 1)
    if len(sents) >= 2:
        first_set = set(re.findall(r'\b\w{4,}\b', sents[0].lower()))
        rest_set = set(re.findall(r'\b\w{4,}\b', ' '.join(sents[1:]).lower()))
        answer_first_overlap = len(first_set & rest_set) / max(len(first_set), 1)
    else:
        answer_first_overlap = 0.0
    dist_to_round_50 = min(abs(nw - 50), abs(nw - 100), abs(nw - 150), abs(nw - 200)) / 50.0
    v = [
        float(len(t)), float(nw), float(nt) / nw, float(lens.mean()), float(lens.std()),
        float(len(sents)), sm, ss, siqr, float(len(set(words_lower)) / nw),
        t.count(chr(39)) * fac, (t.count('\u2019') + t.count('\u2018')) * fac,
        t.count(chr(34)) * fac, (t.count('\u201c') + t.count('\u201d')) * fac,
        t.count('\u2014') * fac, t.count('\u2013') * fac, t.count(';') * fac, t.count(':') * fac,
        (t.count('(') + t.count(')')) * fac, demo_cnt * fac, ly_cnt * fac, ing_cnt * fac, tion_cnt * fac,
        first_sent_ratio, last_sent_ratio, unspaced_emdash, spaced_emdash, semi_to_comma,
        long_word_rate, hedge_rate, conn_rate, sent_initial_conn, answer_first_overlap, dist_to_round_50
    ]
    v.extend(t.count(m) * fac for m in MARKERS_V14)
    c = pd.Series(words_lower).value_counts()
    v.extend(float(c.get(w, 0)) * fac for w in WORDS)
    return np.array(v, dtype=np.float32)

print("Extracting features...")
X_zb = np.stack([row_zb(t) for t in texts])
X_v14 = np.stack([row_v14(t) for t in texts])
print(f"X_zb shape: {X_zb.shape}, X_v14 shape: {X_v14.shape}")

lab = np.asarray(labels)
yy = np.asarray(y)
g = np.asarray(groups)

print("\n--- Evaluating 5-Fold OOF Balanced Accuracy ---")
for family, C in [('claude', 0.01), ('gemini', 0.005), ('gpt', 0.01), ('glm', 0.03)]:
    cols = np.flatnonzero(np.char.startswith(lab.astype(str), family))
    train_mask = np.isin(yy, cols)
    inds = np.flatnonzero(train_mask)
    target = np.searchsorted(cols, yy[inds])
    grp = g[inds]
    chance = 1.0 / len(cols)
    
    # Eval Zbigniew
    oof_zb = np.empty(len(target), dtype=np.int32)
    for a, b in GroupKFold(n_splits=5).split(X_zb[inds], target, groups=grp):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500, class_weight='balanced'))
        m.fit(X_zb[inds][a], target[a])
        oof_zb[b] = m.predict(X_zb[inds][b])
    bacc_zb = balanced_accuracy_score(target, oof_zb)
    
    # Eval V14
    oof_v14 = np.empty(len(target), dtype=np.int32)
    for a, b in GroupKFold(n_splits=5).split(X_v14[inds], target, groups=grp):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500, class_weight='balanced'))
        m.fit(X_v14[inds][a], target[a])
        oof_v14[b] = m.predict(X_v14[inds][b])
    bacc_v14 = balanced_accuracy_score(target, oof_v14)
    
    print(f"Family {family.upper():7s} (n={len(cols)}, chance={chance:.3f}): Zbigniew={bacc_zb:.4f} -> V14={bacc_v14:.4f} (diff = {bacc_v14 - bacc_zb:+.4f})")
