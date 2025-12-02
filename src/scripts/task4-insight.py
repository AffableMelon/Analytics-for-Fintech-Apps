#!/usr/bin/env python3
"""
Improved Task 4 script: theme extraction with n-grams + stopwords, KPI evaluation,
cleaner visuals, per-bank drivers & painpoints with examples, and a rich markdown report.

Requirements:
 - python packages: pandas, sqlalchemy, matplotlib, seaborn, wordcloud,
   vaderSentiment, scikit-learn, numpy
 - optional: pandoc + LaTeX for markdown -> PDF conversion

Usage:
 - Ensure DB env vars are set (PGHOST, PGPORT, PG_DB, PG_USER, PG_PASS)
 - Run: python scripts/task4-insight.py
 - Outputs are written to reports/figures/ and reports/data/ and reports/task4_insights_report.md
"""

import os
import re
import textwrap
import subprocess
from datetime import timedelta
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sqlalchemy import create_engine
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

# -------------------------
# Configuration (tunable)
# -------------------------
PG_HOST = os.environ.get("PGHOST", "localhost")
PG_PORT = os.environ.get("PGPORT", "15432")
PG_DB = os.environ.get("PG_DB", "bank_reviews")
PG_USER = os.environ.get("PG_USER", "postgres")
PG_PASS = os.environ.get("PG_PASS", "pass")

# N-gram / tokenization settings
NGRAM_RANGE = (1, 2)  # unigrams + bigrams
TOP_NGRAMS = 60
MIN_DF = 2  # min documents that must contain the token
ADDITIONAL_STOPWORDS = {
    "the", "and", "this", "have", "has", "you", "not", "but", "very", "a", "an", "is", "it",
    "for", "to", "of", "in", "on", "that", "with", "by", "as", "at", "from", "be", "are", "were",
    "was", "i", "we", "our", "your", "they", "their", "app"  # app can be useful but often noisy; adjust as needed
}

# KPI & thresholds
RECENT_DAYS = 30
BASELINE_DAYS = 90
MIN_MENTIONS = 10
POS_SENT_THRESH = 0.20
NEG_SENT_THRESH = -0.20
REQUIRED_IMPROVEMENT_PCT = 0.20
REQUIRED_SENTIMENT_DELTA = 0.05

# Output dirs
FIG_DIR = "reports/figures"
DATA_DIR = "reports/data"
REPORT_MD = "reports/task4_insights_report.md"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def db_engine():
    return create_engine(f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}")

def safe_name(s):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', (s or "").strip())

def contains_phrase_mask(series, phrase):
    # match whole word or phrase boundaries; for bigrams this still works
    pat = fr'\b{re.escape(phrase)}\b'
    return series.str.contains(pat, case=False, na=False)

def try_pandoc(md_path, out_pdf_path):
    try:
        subprocess.run(["pandoc", md_path, "-o", out_pdf_path, "--pdf-engine=xelatex"], check=True)
        return True
    except Exception:
        return False

# -------------------------
# Load data + sentiment
# -------------------------
engine = db_engine()
df = pd.read_sql("SELECT review_id, bank_id, rating, review_text, review_date FROM reviews", engine, parse_dates=['review_date'])
bnks = pd.read_sql("SELECT bank_id, bank_name FROM banks;", engine)
df = df.merge(bnks[['bank_id', 'bank_name']], on='bank_id', how='left')

analyzer = SentimentIntensityAnalyzer()
if 'sentiment' not in df.columns:
    df['review_text'] = df['review_text'].fillna('').astype(str)
    df['sentiment'] = df['review_text'].apply(lambda t: analyzer.polarity_scores(t)['compound'])

# sanity
if df['review_date'].isna().all():
    raise RuntimeError("review_date missing: KPI windows require review_date values.")

# -------------------------
# Create main visualizations
# -------------------------
df['month'] = df['review_date'].dt.to_period('M').dt.to_timestamp()
sent_trend = df.groupby(['bank_name', 'month'])['sentiment'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=sent_trend, x='month', y='sentiment', hue='bank_name', marker='o')
plt.title('Average Sentiment Over Time by Bank')
plt.ylabel('Avg Sentiment (compound)')
plt.xlabel('Month')
plt.legend(title='Bank', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/sentiment_trend.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='bank_name', y='rating')
plt.title('Rating Distribution by Bank')
plt.ylabel('Rating')
plt.xlabel('Bank')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/rating_distribution.png", dpi=150)
plt.close()

summary = df.groupby('bank_name').agg(avg_sentiment=('sentiment', 'mean'),
                                      avg_rating=('rating', 'mean'),
                                      review_count=('review_id', 'count')).reset_index()
plt.figure(figsize=(10, 6))
ax = plt.gca()
width = 0.35
x = np.arange(len(summary))
ax.bar(x - width/2, summary['avg_rating'], width=width, label='Avg Rating', alpha=0.85)
ax.bar(x + width/2, summary['avg_sentiment'], width=width, label='Avg Sentiment (compound)', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(summary['bank_name'], rotation=45, ha='right')
ax.set_title('Avg Rating and Avg Sentiment by Bank')
ax.set_ylabel('Value')
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/avg_rating_sentiment_by_bank.png", dpi=150)
plt.close()

# -------------------------
# Tokenization & per-bank theme extraction
# -------------------------
stop_words = sorted(list(ADDITIONAL_STOPWORDS))
vectorizer = CountVectorizer(ngram_range=NGRAM_RANGE, stop_words=stop_words, token_pattern=r'(?u)\b[A-Za-z][A-Za-z]+\b', min_df=MIN_DF)

banks = df['bank_name'].dropna().unique().tolist()
bank_insights = {}

for bank in banks:
    sub = df.loc[df['bank_name'] == bank].copy()
    if sub.empty:
        continue
    texts = sub['review_text'].fillna('').astype(str).values
    # Fit vectorizer on texts for this bank
    try:
        X = vectorizer.fit_transform(texts)
        terms = np.array(vectorizer.get_feature_names_out())
        counts = np.array(X.sum(axis=0)).flatten()
        idx = counts.argsort()[::-1][:TOP_NGRAMS]
        top_terms = [(terms[i], int(counts[i])) for i in idx if counts[i] > 0]
    except Exception:
        # fallback tokenization (simple regex)
        c = Counter()
        for t in texts:
            toks = re.sub(r'[^a-zA-Z0-9\s]', ' ', t.lower()).split()
            toks = [w for w in toks if len(w) > 2 and w not in ADDITIONAL_STOPWORDS]
            c.update(toks)
        top_terms = c.most_common(TOP_NGRAMS)

    # Build dataframe of term stats (count + avg_sentiment)
    rows = []
    for term, count in top_terms:
        mask = contains_phrase_mask(sub['review_text'], term)
        if mask.sum() == 0:
            continue
        avg_sent = float(sub.loc[mask, 'sentiment'].mean())
        rows.append({'term': term, 'count': int(count), 'avg_sentiment': float(avg_sent)})
    words_df = pd.DataFrame(rows).sort_values('count', ascending=False)

    # Save word stats CSV
    safe = safe_name(bank)
    words_df.to_csv(f"{DATA_DIR}/word_stats_{safe}.csv", index=False)

    # Visualization: top keywords barh
    top15 = words_df.head(15)
    if not top15.empty:
        plt.figure(figsize=(8, 6))
        plt.barh(top15['term'][::-1], top15['count'][::-1], color=plt.cm.viridis(np.linspace(0.2, 0.8, len(top15))))
        plt.title(f'Top Keywords (unigrams & bigrams) for {bank}')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/top_keywords_{safe}.png", dpi=150)
        plt.close()

    # Word cloud
    big_text = " ".join(texts)
    if big_text.strip():
        wc = WordCloud(width=800, height=400, background_color='white').generate(big_text)
        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {bank}')
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/wordcloud_{safe}.png", dpi=150)
        plt.close()

    # Drivers/painpoints selection
    drivers_df = words_df[(words_df['avg_sentiment'] >= POS_SENT_THRESH) & (words_df['count'] >= MIN_MENTIONS)].sort_values(['avg_sentiment', 'count'], ascending=[False, False]).head(10)
    pain_df = words_df[(words_df['avg_sentiment'] <= NEG_SENT_THRESH) & (words_df['count'] >= MIN_MENTIONS)].sort_values(['avg_sentiment', 'count'], ascending=[True, False]).head(10)

    # fallback: ensure at least 1 driver & 1 painpoint
    if drivers_df.empty:
        drivers_df = words_df[words_df['count'] >= 2].nlargest(2, 'avg_sentiment').head(2)
    if pain_df.empty:
        pain_df = words_df[words_df['count'] >= 2].nsmallest(2, 'avg_sentiment').head(2)

    # For each candidate provide up to 3 example reviews (review_id + text)
    def examples_for_term(term, limit=3):
        mask = contains_phrase_mask(sub['review_text'], term)
        ex = sub.loc[mask, ['review_id', 'review_text']].dropna().head(limit)
        return [ {'review_id': int(r['review_id']), 'text': str(r['review_text']).replace('\n',' ') } for _, r in ex.iterrows() ]

    drivers = []
    for _, r in drivers_df.iterrows():
        term = r['term']
        drivers.append({'term': term, 'count': int(r['count']), 'avg_sentiment': float(r['avg_sentiment']), 'examples': examples_for_term(term)})

    pains = []
    for _, r in pain_df.iterrows():
        term = r['term']
        pains.append({'term': term, 'count': int(r['count']), 'avg_sentiment': float(r['avg_sentiment']), 'examples': examples_for_term(term)})

    bank_insights[bank] = {
        'total_reviews': int(sub.shape[0]),
        'drivers': drivers,
        'painpoints': pains,
        'words_df': words_df,
        'df': sub
    }

# -------------------------
# KPI evaluation (per painpoint)
# -------------------------
kpi_rows = []
now = df['review_date'].max()
recent_end = now
recent_start = recent_end - timedelta(days=RECENT_DAYS)
baseline_end = recent_start
baseline_start = baseline_end - timedelta(days=BASELINE_DAYS)

for bank, insight in bank_insights.items():
    sub = insight['df']
    baseline_reviews = sub[(sub['review_date'] >= baseline_start) & (sub['review_date'] < baseline_end)]
    recent_reviews = sub[(sub['review_date'] >= recent_start) & (sub['review_date'] <= recent_end)]

    for p in insight['painpoints']:
        term = p['term']
        mask_base_term = (baseline_reviews['review_text'].notna()) & contains_phrase_mask(baseline_reviews['review_text'], term)
        mask_recent_term = (recent_reviews['review_text'].notna()) & contains_phrase_mask(recent_reviews['review_text'], term)

        base_count = int(baseline_reviews.loc[mask_base_term].shape[0])
        recent_count = int(recent_reviews.loc[mask_recent_term].shape[0])
        base_total = int(baseline_reviews.shape[0]) if not baseline_reviews.empty else 0
        recent_total = int(recent_reviews.shape[0]) if not recent_reviews.empty else 0

        if base_total == 0 or recent_total == 0:
            base_rate = None
            recent_rate = None
            pct_change = None
            insufficient = True
        else:
            base_rate = base_count / base_total
            recent_rate = recent_count / recent_total
            pct_change = (base_rate - recent_rate) / base_rate if base_rate > 0 else None
            insufficient = base_count < MIN_MENTIONS

        base_sent = float(baseline_reviews.loc[mask_base_term, 'sentiment'].mean()) if base_count > 0 else None
        recent_sent = float(recent_reviews.loc[mask_recent_term, 'sentiment'].mean()) if recent_count > 0 else None
        sent_delta = (recent_sent - base_sent) if (base_sent is not None and recent_sent is not None) else None

        kpi_pass = False
        notes = []
        if insufficient:
            notes.append(f"insufficient baseline mentions (base_count={base_count} < {MIN_MENTIONS})")
        else:
            if pct_change is not None and pct_change >= REQUIRED_IMPROVEMENT_PCT:
                kpi_pass = True
                notes.append(f"mention rate reduced by {pct_change:.2%}")
            else:
                notes.append(f"mention rate change {pct_change:.2%}" if pct_change is not None else "no rate info")
            if sent_delta is not None:
                notes.append(f"sentiment delta {sent_delta:+.3f}")

        # Save monthly trend chart for this painpoint
        mon = sub.copy()
        mon['month'] = mon['review_date'].dt.to_period('M').dt.to_timestamp()
        mon_mask = contains_phrase_mask(mon['review_text'], term)
        monthly_counts = mon.loc[mon_mask].groupby('month')['review_id'].count().reset_index(name='count')
        if not monthly_counts.empty:
            plt.figure(figsize=(10, 4))
            sns.lineplot(data=monthly_counts, x='month', y='count', marker='o')
            plt.title(f"Monthly mentions of '{term}' — {bank}")
            plt.ylabel('Mentions')
            plt.xlabel('Month')
            plt.tight_layout()
            plt.savefig(f"{FIG_DIR}/{safe_name(bank)}_kpi_trend_{safe_name(term)}.png", dpi=150)
            plt.close()

        kpi_rows.append({
            'bank': bank,
            'term': term,
            'base_count': base_count,
            'recent_count': recent_count,
            'base_total_reviews': base_total,
            'recent_total_reviews': recent_total,
            'base_rate': base_rate,
            'recent_rate': recent_rate,
            'pct_change': pct_change,
            'base_sent': base_sent,
            'recent_sent': recent_sent,
            'sent_delta': sent_delta,
            'insufficient_data': insufficient,
            'kpi_pass': kpi_pass,
            'kpi_notes': " ; ".join(notes)
        })

kpi_df = pd.DataFrame(kpi_rows)
kpi_df.to_csv(f"{DATA_DIR}/kpis_by_bank.csv", index=False)

# -------------------------
# Write per-bank CSVs (drivers & painpoints)
# -------------------------
for bank, insight in bank_insights.items():
    safe = safe_name(bank)
    drv = []
    for d in insight['drivers']:
        drv.append({'term': d['term'], 'count': d['count'], 'avg_sentiment': d['avg_sentiment'], 'examples': str(d['examples'])})
    pain = []
    for p in insight['painpoints']:
        pain.append({'term': p['term'], 'count': p['count'], 'avg_sentiment': p['avg_sentiment'], 'examples': str(p['examples'])})
    pd.DataFrame(drv).to_csv(f"{DATA_DIR}/{safe}_drivers.csv", index=False)
    pd.DataFrame(pain).to_csv(f"{DATA_DIR}/{safe}_painpoints.csv", index=False)

# -------------------------
# Compose markdown report (detailed)
# -------------------------
md = []
md.append("# Task 4 — Insights and Recommendations\n")
md.append("\nGenerated by scripts/t4.py (improved theme extraction)\n")
md.append("\nFigures: saved to `reports/figures/`.\n")
md.append("\nData files: saved to `reports/data/`.\n")

md.append("\n## Executive summary\n")
md.append("This report summarizes derived drivers and pain points from user reviews across banks, provides prioritized recommendations and evaluates KPIs (mention rates and sentiment deltas) for top pain points.\n")

md.append("\n## KPI configuration\n")
md.append(f"- recent window: last {RECENT_DAYS} days (ending {now.date()})\n")
md.append(f"- baseline window: preceding {BASELINE_DAYS} days (from {baseline_start.date()} to {baseline_end.date()})\n")
md.append(f"- min mentions threshold: {MIN_MENTIONS}\n")
md.append(f"- positive sentiment threshold: {POS_SENT_THRESH}\n")
md.append(f"- negative sentiment threshold: {NEG_SENT_THRESH}\n")

md.append("\n## Bank-level findings\n")
for bank, insight in bank_insights.items():
    safe = safe_name(bank)
    md.append(f"\n### {bank}\n")
    md.append(f"- Total reviews analyzed: {insight['total_reviews']}\n")
    md.append(f"- Figures: `reports/figures/top_keywords_{safe}.png`, `reports/figures/wordcloud_{safe}.png`\n")

    # Drivers
    md.append("\n#### Drivers (positive themes)\n")
    if insight['drivers']:
        for d in insight['drivers'][:5]:
            md.append(f"- **{d['term']}** (count={d['count']}, avg_sent={d['avg_sentiment']:.3f})\n")
            for ex in d['examples']:
                txt = ex['text'][:300]
                md.append(f"  - example (id={ex['review_id']}): {txt}\n")
    else:
        md.append("- No drivers detected with current thresholds.\n")

    # Painpoints
    md.append("\n#### Pain points (negative themes)\n")
    if insight['painpoints']:
        for p in insight['painpoints'][:5]:
            md.append(f"- **{p['term']}** (count={p['count']}, avg_sent={p['avg_sentiment']:.3f})\n")
            for ex in p['examples']:
                txt = ex['text'][:300]
                md.append(f"  - example (id={ex['review_id']}): {txt}\n")
    else:
        md.append("- No pain points detected with current thresholds.\n")

    # Recommendations (heuristic)
    md.append("\n#### Recommendations\n")
    # simple keyword mapping for actionable recs
    recs = set()
    for term in [x['term'] for x in insight['painpoints']] + [x['term'] for x in insight['drivers']]:
        t = term.lower()
        if any(k in t for k in ("crash", "bug", "fix", "crashes")):
            recs.add("Prioritize crash stability and increase crash reporting/monitoring; add graceful retry UX.")
        if any(k in t for k in ("slow", "time", "lag", "delay")):
            recs.add("Investigate performance hotspots and add visible progress/timeout handling.")
        if any(k in t for k in ("login", "auth", "password")):
            recs.add("Harden and simplify login flow; improve error messages and session handling.")
        if "branch" in t:
            recs.add("Coordinate with branch ops to address staffing/hours and add appointment scheduling / in-app branch feedback.")
        if "fee" in t:
            recs.add("Clarify fees in-app and show fees before confirming transactions.")
    if not recs:
        recs.add("Collect targeted telemetry/support-ticket correlation for top pain points, and run in-app micro-surveys to gather structured feedback.")
    for r in recs:
        md.append(f"- {r}\n")

# KPI summary notes
md.append("\n## KPI summary (per painpoint)\n")
md.append(f"See `reports/data/kpis_by_bank.csv` for per-term KPI results (base_count, recent_count, base_rate, recent_rate, pct_change, sent deltas, flags).\n")

# Ethics / Bias
md.append("\n## Ethics and bias considerations\n")
md.append(textwrap.dedent("""
- Reviewers are a self-selecting group and may skew negative or positive.
- Small sample sizes for some terms reduce confidence (these are flagged as 'insufficient_data' in the KPI CSV).
- Linguistic noise (short reviews, non-native phrasing) can cause spurious tokens; we mitigated by using n-grams and stopwords but recommend manual review of top findings.
- Recommendations should be validated against telemetry (crash logs, performance traces) and support tickets before major product decisions.
"""))

# Append Appendix with listings and instructions to produce PDF
md.append("\n## Appendix — Files generated\n")
md.append("- Figures: `reports/figures/`\n")
md.append("- Per-bank word stats: `reports/data/word_stats_<bank>.csv`\n")
md.append("- Per-bank drivers/painpoints: `reports/data/<bank>_drivers.csv` and `<bank>_painpoints.csv`\n")
md.append("- KPI table: `reports/data/kpis_by_bank.csv`\n")

# Write markdown out
with open(REPORT_MD, "w", encoding="utf-8") as fh:
    fh.write("\n".join(md))

# Try to create PDF (optional)
pdf_path = REPORT_MD.replace(".md", ".pdf")
if try_pandoc(REPORT_MD, pdf_path):
    print("PDF created at", pdf_path)
else:
    print("Pandoc PDF conversion not available or failed. Markdown saved to", REPORT_MD)

print("Done. Figures ->", FIG_DIR, "| Data ->", DATA_DIR, "| Report ->", REPORT_MD)