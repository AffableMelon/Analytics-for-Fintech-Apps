#!/usr/bin/env python3
"""
scripts/t4.py

Fixed: add_examples now accepts either a DataFrame or a list of dicts.
Replaced seaborn barplot call that caused FutureWarning with matplotlib barh.
"""

import os
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sqlalchemy import create_engine
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import re
from datetime import timedelta

# -------------------------
# Configuration (tunable)
# -------------------------
PG_HOST = os.environ.get("PGHOST", "localhost")
PG_PORT = os.environ.get("PGPORT", "15432")
PG_DB = os.environ.get("PG_DB", "bank_reviews")
PG_USER = os.environ.get("PG_USER", "postgres")
PG_PASS = os.environ.get("PG_PASS", "pass")
engine = create_engine(f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}")

fig_dir = "reports/figures"
data_dir = "reports/data"
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# KPI windows and thresholds (kept from prior script)
RECENT_DAYS = 30
BASELINE_DAYS = 90
MIN_MENTIONS = 10
REQUIRED_IMPROVEMENT_PCT = 0.20
REQUIRED_SENTIMENT_DELTA = 0.05

# -------------------------
# Data load & sentiment
# -------------------------
df = pd.read_sql(
    "SELECT review_id, bank_id, rating, review_text, review_date FROM reviews",
    engine,
    parse_dates=["review_date"],
)
bnks = pd.read_sql("SELECT bank_id, bank_name FROM banks;", engine)
df = df.merge(bnks[["bank_id", "bank_name"]], on="bank_id", how="left")

analyzer = SentimentIntensityAnalyzer()
if "sentiment" not in df.columns:
    df["review_text"] = df["review_text"].fillna("").astype(str)
    df["sentiment"] = df["review_text"].apply(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )

if "review_date" not in df.columns or df["review_date"].isna().all():
    raise ValueError(
        "review_date missing or empty in reviews table. KPI time windows need review_date."
    )

# -------------------------
# Helper functions
# -------------------------


def safe_name(s):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s or "")


def contains_word_mask(series, word):
    return series.str.contains(rf"\b{re.escape(word)}\b", case=False, na=False)


# -------------------------
# Visualizations (existing)
# -------------------------
df["month"] = df["review_date"].dt.to_period("M").dt.to_timestamp()
sent_trend = df.groupby(["bank_name", "month"])["sentiment"].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=sent_trend, x="month", y="sentiment", hue="bank_name", marker="o")
plt.title("Average Sentiment Over Time by Bank")
plt.ylabel("Avg Sentiment (compound)")
plt.xlabel("Month")
plt.legend(title="Bank", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{fig_dir}/sentiment_trend.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="bank_name", y="rating")
plt.title("Rating Distribution by Bank")
plt.ylabel("Rating")
plt.xlabel("Bank")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{fig_dir}/rating_distribution.png", dpi=150)
plt.close()

summary = (
    df.groupby("bank_name")
    .agg(
        avg_sentiment=("sentiment", "mean"),
        avg_rating=("rating", "mean"),
        review_count=("review_id", "count"),
    )
    .reset_index()
)
plt.figure(figsize=(10, 6))
ax = plt.gca()
width = 0.35
x = range(len(summary))
ax.bar(
    [i - width / 2 for i in x],
    summary["avg_rating"],
    width=width,
    label="Avg Rating",
    alpha=0.8,
)
ax.bar(
    [i + width / 2 for i in x],
    summary["avg_sentiment"],
    width=width,
    label="Avg Sentiment (compound)",
    alpha=0.8,
)
ax.set_xticks(x)
ax.set_xticklabels(summary["bank_name"], rotation=45, ha="right")
ax.set_title("Avg Rating and Avg Sentiment by Bank")
ax.set_ylabel("Value")
ax.legend()
plt.tight_layout()
plt.savefig(f"{fig_dir}/avg_rating_sentiment_by_bank.png", dpi=150)
plt.close()

# -------------------------
# Keyword extraction & drivers/painpoints
# -------------------------
banks = df["bank_name"].dropna().unique().tolist()
min_word_freq = 5
bank_insights = {}

for bank in banks:
    sub = df.loc[df["bank_name"] == bank].copy()
    if sub.empty:
        continue

    token_counts = Counter()
    for t in sub["review_text"].dropna().astype(str):
        token_counts.update(
            [w for w in re.sub(r"[^a-zA-Z0-9\s]", " ", t.lower()).split() if len(w) > 2]
        )
    top_tokens = token_counts.most_common(40)
    words_df = pd.DataFrame(top_tokens, columns=["word", "count"])

    # Use matplotlib barh to avoid seaborn palette FutureWarning
    top15 = words_df.head(15)
    if not top15.empty:
        plt.figure(figsize=(8, 6))
        plt.barh(
            top15["word"][::-1],
            top15["count"][::-1],
            color=plt.cm.viridis(range(len(top15))),
        )
        plt.title(f"Top Keywords in Reviews for {bank}")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/top_keywords_{safe_name(bank)}.png", dpi=150)
        plt.close()

    # Word cloud
    text = " ".join(sub["review_text"].dropna().astype(str).values)
    if len(text.strip()) > 0:
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for {bank}")
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/wordcloud_{safe_name(bank)}.png", dpi=150)
        plt.close()

    # per-word sentiment stats
    word_stats = []
    for word, count in top_tokens:
        if count < min_word_freq:
            continue
        mask = contains_word_mask(sub["review_text"], word)
        if mask.sum() == 0:
            continue
        avg_sent = sub.loc[mask, "sentiment"].mean()
        word_stats.append((word, count, float(avg_sent)))
    word_stats_df = pd.DataFrame(
        word_stats, columns=["word", "count", "avg_sentiment"]
    ).sort_values("count", ascending=False)

    if not word_stats_df.empty:
        drivers = (
            word_stats_df[word_stats_df["avg_sentiment"] >= 0.2]
            .sort_values(["avg_sentiment", "count"], ascending=[False, False])
            .head(10)
        )
        painpoints = (
            word_stats_df[word_stats_df["avg_sentiment"] <= -0.2]
            .sort_values(["avg_sentiment", "count"], ascending=[True, False])
            .head(10)
        )
    else:
        drivers = pd.DataFrame(columns=word_stats_df.columns)
        painpoints = pd.DataFrame(columns=word_stats_df.columns)

    # add_examples now accepts DataFrame or list of dicts
    def add_examples(rows):
        # rows can be a DataFrame (from .head() / .loc) OR a list of dicts (from to_dict('records'))
        if isinstance(rows, list):
            rows_df = pd.DataFrame(rows)
        else:
            rows_df = (
                pd.DataFrame(rows)
                if not isinstance(rows, pd.DataFrame)
                else rows.copy()
            )

        examples = []
        for _, r in rows_df.iterrows():
            w = r["word"]
            mask = contains_word_mask(sub["review_text"], w)
            sample = sub.loc[mask, ["review_id", "review_text"]].dropna().head(3)
            sample_records = []
            for _, s in sample.iterrows():
                sample_records.append(
                    {
                        "review_id": str(s["review_id"]),
                        "review_text": str(s["review_text"]),
                    }
                )
            examples.append(
                {
                    "word": w,
                    "count": int(r["count"]),
                    "avg_sentiment": float(r["avg_sentiment"]),
                    "examples": sample_records,
                }
            )
        return examples

    driver_list = add_examples(drivers) if not drivers.empty else []
    pain_list = add_examples(painpoints) if not painpoints.empty else []

    bank_insights[bank] = {
        "total_reviews": int(sub.shape[0]),
        "drivers": driver_list,
        "painpoints": pain_list,
        "word_stats_df": word_stats_df,
        "df": sub,
    }

    word_stats_df.to_csv(f"{data_dir}/word_stats_{safe_name(bank)}.csv", index=False)

# -------------------------
# KPI Evaluation (kept from prior)
# -------------------------
kpi_rows = []
now = df["review_date"].max()
recent_end = now
recent_start = recent_end - timedelta(days=RECENT_DAYS)
baseline_end = recent_start
baseline_start = baseline_end - timedelta(days=BASELINE_DAYS)

for bank, insight in bank_insights.items():
    sub = insight["df"]
    total_reviews = insight["total_reviews"]

    base_mask_period = (sub["review_date"] >= baseline_start) & (
        sub["review_date"] < baseline_end
    )
    recent_mask_period = (sub["review_date"] >= recent_start) & (
        sub["review_date"] <= recent_end
    )

    baseline_reviews = sub.loc[base_mask_period]
    recent_reviews = sub.loc[recent_mask_period]

    baseline_avg_sent = (
        float(baseline_reviews["sentiment"].mean())
        if not baseline_reviews.empty
        else None
    )
    recent_avg_sent = (
        float(recent_reviews["sentiment"].mean()) if not recent_reviews.empty else None
    )

    for p in insight["painpoints"]:
        word = p["word"]
        mask_base_word = base_mask_period & contains_word_mask(sub["review_text"], word)
        mask_recent_word = recent_mask_period & contains_word_mask(
            sub["review_text"], word
        )

        base_count = int(sub.loc[mask_base_word].shape[0])
        recent_count = int(sub.loc[mask_recent_word].shape[0])
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
            pct_change = (
                (base_rate - recent_rate) / base_rate if base_rate > 0 else None
            )
            insufficient = base_count < MIN_MENTIONS

        base_sent = (
            float(sub.loc[mask_base_word, "sentiment"].mean())
            if base_count > 0
            else None
        )
        recent_sent = (
            float(sub.loc[mask_recent_word, "sentiment"].mean())
            if recent_count > 0
            else None
        )
        sent_delta = (
            (recent_sent - base_sent)
            if (base_sent is not None and recent_sent is not None)
            else None
        )

        kpi_pass = False
        kpi_notes = []
        if insufficient:
            kpi_notes.append(
                f"insufficient baseline mentions (base_count={base_count} < {
                    MIN_MENTIONS
                })"
            )
        else:
            if pct_change is not None and pct_change >= REQUIRED_IMPROVEMENT_PCT:
                kpi_pass = True
                kpi_notes.append(f"mention rate reduced by {pct_change:.2%}")
            else:
                kpi_notes.append(
                    f"mention rate change {pct_change:.2%}"
                    if pct_change is not None
                    else "no rate info"
                )

            if sent_delta is not None:
                if sent_delta >= REQUIRED_SENTIMENT_DELTA:
                    kpi_notes.append(f"sentiment improved by {sent_delta:.3f}")
                else:
                    kpi_notes.append(f"sentiment delta {sent_delta:.3f}")

        kpi_rows.append(
            {
                "bank": bank,
                "word": word,
                "base_count": base_count,
                "recent_count": recent_count,
                "base_total_reviews": base_total,
                "recent_total_reviews": recent_total,
                "base_rate": base_rate,
                "recent_rate": recent_rate,
                "pct_change": pct_change,
                "base_sent": base_sent,
                "recent_sent": recent_sent,
                "sent_delta": sent_delta,
                "insufficient_data": insufficient,
                "kpi_pass": kpi_pass,
                "kpi_notes": " ; ".join(kpi_notes),
            }
        )

        mon = sub.copy()
        mon["month"] = mon["review_date"].dt.to_period("M").dt.to_timestamp()
        mon_mask = contains_word_mask(mon["review_text"], word)
        monthly_counts = (
            mon.loc[mon_mask]
            .groupby("month")["review_id"]
            .count()
            .reset_index(name="count")
        )
        if not monthly_counts.empty:
            plt.figure(figsize=(10, 4))
            sns.lineplot(data=monthly_counts, x="month", y="count", marker="o")
            plt.title(f"Monthly mention counts for '{word}' — {bank}")
            plt.ylabel("Mentions")
            plt.xlabel("Month")
            plt.tight_layout()
            plt.savefig(
                f"{fig_dir}/{safe_name(bank)}_kpi_trend_{safe_name(word)}.png", dpi=150
            )
            plt.close()

kpi_df = pd.DataFrame(kpi_rows)
kpi_df.to_csv(f"{data_dir}/kpis_by_bank.csv", index=False)

# -------------------------
# Report generation (simplified)
# -------------------------
report_lines = []
report_lines.append("# Task 4 — Insights and Recommendations\n")
report_lines.append("Generated by scripts/t4.py (fixed add_examples and plotting)\n")
report_lines.append(f"Figures: saved to `{fig_dir}/`.\n")
report_lines.append("Data files: saved to `reports/data/`.\n")

report_path = "reports/task4_insights_report.md"
with open(report_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(report_lines))

print("Done. Figures saved to:", fig_dir)
print("Report saved to:", report_path)
print("Per-bank data and KPI summary saved to:", data_dir)
