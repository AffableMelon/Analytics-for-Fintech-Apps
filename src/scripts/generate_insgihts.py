#!/usr/bin/env python3
"""
scripts/generate_findings.py

Utility to expand the markdown report into a longer, paginated 'final report' (markdown),
including method details, appended appendices (tables + example reviews) to help reach a
multi-page deliverable. It uses the outputs from scripts/t4.py.

Usage:
 - Run after scripts/t4.py (so reports/data/ and reports/figures/ exist)
 - python scripts/generate_findings.py
 - Optionally converts the expanded markdown to PDF (requires pandoc + LaTeX)
"""

import os
import json
import textwrap
import pandas as pd
from datetime import datetime
import subprocess

REPORT_MD = "reports/task4_insights_report.md"
EXPANDED_MD = "reports/task4_insights_report_expanded.md"
DATA_DIR = "reports/data"
FIG_DIR = "reports/figures"

def read_existing_md(path):
    if os.path.exists(path):
        return open(path, "r", encoding="utf-8").read()
    return ""

def make_table_markdown(df, max_rows=20):
    if df.empty:
        return ""
    return df.head(max_rows).to_markdown(index=False)

# Build expanded report
base = read_existing_md(REPORT_MD)
lines = [base, "\n\n---\n\n"]
lines.append("# Expanded Report\n")
lines.append(f"Generated: {datetime.utcnow().isoformat()} UTC\n")
lines.append("\n## Methods\n")
lines.append(textwrap.dedent("""
- Tokenization: CountVectorizer with unigrams + bigrams and a curated stopword list.
- Sentiment: VADER (compound score).
- Themes: top n-grams per bank; average sentiment for reviews containing the n-gram; drivers/painpoints selected by sentiment thresholds and minimum counts.
- KPI evaluation: comparison of a baseline window vs a recent window for mention rates and sentiment deltas; results in reports/data/kpis_by_bank.csv.
"""))

lines.append("\n## Detailed per-bank appendices\n")
word_files = [f for f in os.listdir(DATA_DIR) if f.startswith("word_stats_") and f.endswith(".csv")]
for f in sorted(word_files):
    bank_tag = f[len("word_stats_"):-len(".csv")]
    bank_name = bank_tag.replace("_", " ")
    ws = pd.read_csv(os.path.join(DATA_DIR, f))
    lines.append(f"\n### {bank_name}\n")
    lines.append(f"Top terms (first 30):\n")
    lines.append(make_table_markdown(ws, max_rows=30))
    # drivers/painpoints CSVs
    drv_path = os.path.join(DATA_DIR, f"{bank_tag}_drivers.csv")
    pain_path = os.path.join(DATA_DIR, f"{bank_tag}_painpoints.csv")
    if os.path.exists(drv_path):
        drv = pd.read_csv(drv_path)
        lines.append("\nDrivers (examples):\n")
        lines.append(make_table_markdown(drv, max_rows=10))
    if os.path.exists(pain_path):
        pain = pd.read_csv(pain_path)
        lines.append("\nPain points (examples):\n")
        lines.append(make_table_markdown(pain, max_rows=10))
    # include links to figures
    lines.append(f"\nFigures for this bank: `reports/figures/top_keywords_{bank_tag}.png`, `reports/figures/wordcloud_{bank_tag}.png`\n")

# Add KPI table excerpt
kpi_path = os.path.join(DATA_DIR, "kpis_by_bank.csv")
if os.path.exists(kpi_path):
    kpi = pd.read_csv(kpi_path)
    lines.append("\n## KPI excerpts (first 40 rows)\n")
    lines.append(kpi.head(40).to_markdown(index=False))

# Method limitations + reproducibility
lines.append("\n## Limitations\n")
lines.append(textwrap.dedent("""
- Short reviews and non-standard language cause noisy tokens; n-gram extraction reduces but does not eliminate this.
- VADER performs well on short social-text-style reviews, but domain-specific sentiment nuances may be missed.
- KPI reliability depends on review volume; rows with 'insufficient_data' in kpis_by_bank.csv require corroboration via telemetry.
"""))

# Convert to markdown file
with open(EXPANDED_MD, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))

# Try to produce PDF using pandoc if available
PDF = EXPANDED_MD.replace(".md", ".pdf")
try:
    subprocess.run(["pandoc", EXPANDED_MD, "-o", PDF, "--pdf-engine=xelatex"], check=True)
    print("Expanded PDF created at", PDF)
except Exception:
    print("Pandoc not available or failed; expanded markdown saved to", EXPANDED_MD)

print("Done. Expanded report:", EXPANDED_MD)