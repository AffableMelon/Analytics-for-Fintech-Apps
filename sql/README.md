# Database Schema and Setup

## Database Schema Overview

The database contains two main tables: **`banks`** and **`reviews`**, connected
by a one-to-many relationship where one bank can have multiple reviews.

### `banks` Table

| Column Name | Data Type | Constraint/Description |
| :--- | :--- | :--- |
| **`bank_id`** | `SERIAL` | **Primary Key**, Auto-incrementing identifier. |
| **`bank_name`** | `VARCHAR(255)` | **Unique**, **NOT NULL**. Full name of the bank. |
| **`app_name`** | `VARCHAR(255)` | Associated application name (if applicable). |

### `reviews` Table
Stores individual review data, including the text, rating, and sentiment analysis results.

| Column Name | Data Type | Constraint/Description |
| :--- | :--- | :--- |
| **`review_id`** | `SERIAL` | **Primary Key**, Auto-incrementing identifier. |
| **`bank_id`** | `INT` | **Foreign Key** referencing `banks(bank_id)`. `ON DELETE CASCADE`. |
| **`review_text`** | `TEXT` | **NOT NULL**. The full text of the user review. |
| **`rating`** | `INT` | The numerical rating (e.g., 1-5 stars). |
| **`review_date`** | `DATE` | The date the review was published. |
| **`sentiment_label`** | `VARCHAR(50)` | Categorical sentiment result (e.g., 'Positive', 'Negative'). |
| **`sentiment_score`** | `NUMERIC(5,3)` | Numerical score for the sentiment analysis result. |
| **`source`** | `VARCHAR(100)` | The source of the review (e.g., 'Google Play', 'App Store'). |

##  Database Instantiation

The database is set up and managed using **Docker Compose**, as defined in
`sql/docker-compose.yml`.

### Prerequisites

1.  **Docker** and **Docker Compose** installed on your system.
2.  A local file named **`.env`** in the `sql/` directory to store credentials
    (based on your `sql/.env`):

    ```ini POSTGRES_USER=postgres POSTGRES_PASSWORD=pass
    PGADMIN_DEFAULT_EMAIL=user@example.com PGADMIN_DEFAULT_PASSWORD=pass ```

### Running the Database

1.  Navigate to the directory containing `docker-compose.yml`:

    ```bash cd sql ```

2.  Run Docker Compose to build and start the services in detached mode (`-d`):

    ```bash docker compose up -d ```

This command will start two services:

* **`bnk-db`** (PostgreSQL): Accessible on **port `15432`**. The database and
tables are created automatically by the `init.sql` script upon first run.
* **`pgadmin`** (Web interface for PostgreSQL): Accessible at
`http://localhost:15433` using the credentials defined in `.env`.

---

##  Populating the Data

Data is populated using the Python script `src/scripts/insert_to_db.py`, which
reads processed CSV data (e.g., `reviews_with_sentiment_and_theme.csv`) and
bulk-inserts it into the `banks` and `reviews` tables.

### Data Insertion Steps

1.  **Ensure the database containers are running** (`docker compose ps` should
show `Up` status).
2.  **Ensure all required Python dependencies are installed** (based on
`requirements.txt`).
3.  **Run the insertion script** from the project root (the directory
containing `README.md`):

    ```bash python src/scripts/insert_to_db.py ```

The script will handle connecting to the PostgreSQL database on port `15432`,
inserting unique banks, and then efficiently inserting the bulk of the review
data, ensuring all foreign key relationships are correctly established.

