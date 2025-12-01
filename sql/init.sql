-- Create database
CREATE DATABASE bank_reviews;

\connect bank_reviews;

-- Create Banks table
CREATE TABLE IF NOT EXISTS banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(255) UNIQUE NOT NULL,
    app_name VARCHAR(255)
);

-- Create Reviews table
CREATE TABLE IF NOT EXISTS reviews (
    review_id SERIAL PRIMARY KEY,
    bank_id INT REFERENCES banks(bank_id) ON DELETE CASCADE,
    review_text TEXT NOT NULL,
    rating INT,
    review_date DATE,
    sentiment_label VARCHAR(50),
    sentiment_score NUMERIC(5,3),
    source VARCHAR(100)
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_reviews_bank_id ON reviews(bank_id);
CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews(rating);
CREATE INDEX IF NOT EXISTS idx_reviews_date ON reviews(review_date);
