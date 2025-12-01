# Analytics-for-Fintech-Apps

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

### Database Information
To get information on how to instance the database, please read the [Database Setup file](./sql/README.md).
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
- sql/
    - README.md -> this is where you will find documentation on database setup
    - docker-compose.yml -> docker file to setup and get database working with postgress
    - init.sql -> database sql file used to create tables.
- requirements.txt
- .gitignore
- READEM.MD


For collaboration or questions, open an issue or contact the repo owner: AffableMelon.
