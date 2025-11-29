# Analytics-for-Fintech-Apps

A collection of Jupyter notebooks demonstrating analytics and machine learning approaches commonly used in fintech applications. This repository focuses on exploratory analysis, feature engineering, modeling, and visualization for problems like credit scoring, churn prediction, fraud detection, user segmentation, and product analytics.
This repository is notebook-first (most content is in Jupyter Notebooks), supplemented by helper scripts and visualizations.

## Purpose
Analyzing customer satisfaction with mobile banking apps by collecting and processing user reviews from the Google Play Store for three Ethiopian banks:
- Commercial Bank of Ethiopia (CBE)
- Bank of Abyssinia (BOA)
- Dashen Bank

## Notebooks (example list)
- preprocessing_EDA.ipynb
- sentiment_analysis.ipynb

## Setup & installation
Clone and create an environment:
```bash
git clone https://github.com/AffableMelon/Analytics-for-Fintech-Apps.git
cd Analytics-for-Fintech-Apps
python3 -m venv .venv
source .venv/bin/activate.fish
pip install -r requirements.txt
```
## Project structure (suggested)
- src/ 
    - notebooks/
      - preprocessing_EDA.ipynb
      - sentiment_analysis.ipynb
      - ...
    - data/
      - raw/
      - processed/
    - scripts/
      - config.py
      - preprocessing.py
      - ...
- requirements.txt
- .gitignore
- READEM.MD

For collaboration or questions, open an issue or contact the repo owner: AffableMelon.
