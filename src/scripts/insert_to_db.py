import os
import psycopg2
import pandas as pd
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
from config import DATA_PATHS, APP_IDS, BANK_NAMES


# ------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------
load_dotenv()


# ------------------------------------------------------------
# Database connection helper
# ------------------------------------------------------------
def get_connection():
    """Create and return a PostgreSQL connection."""
    return psycopg2.connect(
        host=os.getenv("HOST", "localhost"),
        port=os.getenv("PORT", 15432),
        dbname=os.getenv("DB_NAME", "bank_reviews"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", "pass"),
    )


# ------------------------------------------------------------
# Insert banks and return { code -> bank_id } mapping
# ------------------------------------------------------------
def insert_banks(cursor):
    """Insert all banks and return a mapping of bank_code → bank_id."""
    bank_insert_query = """
    INSERT INTO banks (bank_name, app_name)
    VALUES (%s, %s)
    ON CONFLICT (bank_name) DO UPDATE SET app_name = EXCLUDED.app_name
    RETURNING bank_id;
    """

    bank_ids = {}

    for code, name in BANK_NAMES.items():
        app_name = APP_IDS[code]
        cursor.execute(bank_insert_query, (name, app_name))
        bank_id = cursor.fetchone()[0]
        bank_ids[code] = bank_id

    print(f"[INFO] Inserted banks: {bank_ids}")
    return bank_ids


# ------------------------------------------------------------
# Insert reviews in bulk
# ------------------------------------------------------------
def insert_reviews(cursor, df, bank_ids):
    """Insert cleaned review rows into the reviews table."""
    insert_query = """
    INSERT INTO reviews (
        bank_id, review_text, rating, review_date,
        sentiment_label, sentiment_score, source
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    rows = []

    for _, row in df.iterrows():
        bank_id = bank_ids.get(row.bank_code)

        rows.append(
            (
                bank_id,
                row.review_text,
                int(row.rating),
                str(row.review_date)[:10],  # yyyy-mm-dd
                row.sentiment_label,
                float(row.sentiment_score),
                row.source,
            )
        )

    execute_batch(cursor, insert_query, rows, page_size=500)

    print(f"[INFO] Inserted {len(rows)} reviews.")


# ------------------------------------------------------------
# Main execution function
# ------------------------------------------------------------
def main():
    # Load CSV
    csv_path = DATA_PATHS.get("sentiment_results")
    if not csv_path:
        raise ValueError("Processed reviews CSV path not found in DATA_PATHS.")

    df = pd.read_csv(f"../{csv_path}")
    print(f"[INFO] Loaded {len(df)} reviews from: {csv_path}")
    print("[DEBUG] Columns:", df.columns.tolist())

    # Connect to database
    conn = get_connection()
    cur = conn.cursor()

    try:
        # Insert banks
        bank_ids = insert_banks(cur)

        # Insert reviews
        insert_reviews(cur, df, bank_ids)

        # Commit all changes
        conn.commit()
        print("[SUCCESS] All data inserted successfully.")

    except Exception as e:
        conn.rollback()
        print("[ERROR] Transaction rolled back due to:", str(e))
        raise

    finally:
        cur.close()
        conn.close()
        print("[INFO] Database connection closed.")


# ------------------------------------------------------------
# Run script
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
