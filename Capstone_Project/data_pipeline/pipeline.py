"""
Zepto Data & AI Platform - Module 1: Data Pipeline
Extracts, cleans, converts, stores, and queries catalog data from books.toscrape.com.
Baseline Currency Conversion: 1 GBP = 105.50 INR (Fixed project constant)
"""

import os
import re
import sqlite3
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ---------------------------------------------------------
BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR_RATE = 105.50
DB_PATH = "zepto_catalog.db"

# 4 distinct categories guaranteed to exceed 60 books (20 items per page)
TARGET_CATEGORIES = {
    "Travel": "catalogue/category/books/travel_2/index.html",
    "Mystery": "catalogue/category/books/mystery_3/index.html",
    "Historical Fiction": "catalogue/category/books/historical-fiction_4/index.html",
    "Sequential Art": "catalogue/category/books/sequential-art_5/index.html"
}

RATING_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5
}

# ---------------------------------------------------------
# 1. SCRAPER
# ---------------------------------------------------------
def scrape_catalog():
    raw_records = []
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    print("[1/5] Scraping catalog data across categories...")
    for category_name, rel_url in TARGET_CATEGORIES.items():
        url = BASE_URL + rel_url
        response = session.get(url, timeout=15)
        if response.status_code != 200:
            print(f"Warning: Failed to fetch {category_name} (Status: {response.status_code})")
            continue

        soup = BeautifulSoup(response.content, "html.parser")
        product_pods = soup.select("article.product_pod")

        for pod in product_pods:
            # Title: Full title is in the 'title' attribute of the <a> tag
            a_tag = pod.select_one("h3 > a")
            title = a_tag.get("title", "").strip() if a_tag else None

            # Price text: e.g. "£51.77" or "Â£51.77"
            price_tag = pod.select_one("p.price_color")
            raw_price = price_tag.get_text(strip=True) if price_tag else None

            # Rating: class list contains 'star-rating' and 'Three', etc.
            rating_tag = pod.select_one("p.star-rating")
            rating_classes = rating_tag.get("class", []) if rating_tag else []
            raw_rating = next((c for c in rating_classes if c.lower() != "star-rating"), None)

            # Availability: e.g. "In stock"
            avail_tag = pod.select_one("p.instock.availability")
            raw_avail = avail_tag.get_text(strip=True) if avail_tag else None

            raw_records.append({
                "title": title,
                "raw_price": raw_price,
                "raw_rating": raw_rating,
                "raw_availability": raw_avail,
                "category": category_name
            })

    print(f"Scraped {len(raw_records)} raw records across {len(TARGET_CATEGORIES)} categories.")
    return raw_records

# ---------------------------------------------------------
# 2. DATA CLEANING & TRANSFORMATION
# ---------------------------------------------------------
def clean_data(raw_records):
    print("[2/5] Cleaning and converting fields...")
    df = pd.DataFrame(raw_records)

    # 1. Price Cleaning: Extract decimal numeric values
    def parse_price(val):
        if not val or pd.isna(val):
            return np.nan
        match = re.search(r"(\d+\.\d+)", str(val))
        return float(match.group(1)) if match else np.nan

    df["price_gbp"] = df["raw_price"].apply(parse_price)

    # Median Imputation for missing/corrupted price_gbp
    if df["price_gbp"].isna().sum() > 0:
        median_price = df["price_gbp"].median()
        print(f"Imputing {df['price_gbp'].isna().sum()} missing prices with median: £{median_price:.2f}")
        df["price_gbp"] = df["price_gbp"].fillna(median_price)

    # 2. Fixed-Rate Currency Conversion
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR_RATE).round(2)

    # 3. Rating Conversion: One..Five -> 1..5
    def parse_rating(val):
        if not val:
            return 3  # Impute median rating if missing
        return RATING_MAP.get(str(val).strip().lower(), 3)

    df["rating"] = df["raw_rating"].apply(parse_rating).astype(int)

    # 4. Availability Parsing: in_stock bool / integer (1/0 for SQLite)
    def parse_availability(val):
        if not val:
            return False
        return "in stock" in str(val).lower()

    df["in_stock"] = df["raw_availability"].apply(parse_availability).astype(int)

    # Drop non-recoverable records missing titles
    df = df.dropna(subset=["title"]).reset_index(drop=True)

    print(f"Data cleaned successfully. Cleaned dataset count: {len(df)} rows.")
    return df

# ---------------------------------------------------------
# 3. DATABASE LOADING (NORMALIZED SCHEMA)
# ---------------------------------------------------------
def setup_and_load_database(df, db_path=DB_PATH):
    print(f"[3/5] Loading normalized tables into SQLite database at '{db_path}'...")
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Enforce foreign key constraints
    cur.execute("PRAGMA foreign_keys = ON;")

    # Create tables
    cur.execute("""
    CREATE TABLE IF NOT EXISTS categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
        in_stock INTEGER NOT NULL CHECK(in_stock IN (0, 1)),
        category_id INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories (category_id)
    );
    """)

    # Populate categories
    unique_categories = sorted(df["category"].unique())
    cur.executemany(
        "INSERT INTO categories (category_name) VALUES (?);",
        [(cat,) for cat in unique_categories]
    )

    # Fetch mapping category_name -> category_id
    cur.execute("SELECT category_name, category_id FROM categories;")
    cat_map = dict(cur.fetchall())
    df["category_id"] = df["category"].map(cat_map)

    # Populate books table
    books_data = df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"]].values.tolist()
    cur.executemany("""
        INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?);
    """, books_data)

    conn.commit()
    conn.close()
    print("Database populated successfully.")

# ---------------------------------------------------------
# 4. SQL QUERIES EXECUTION
# ---------------------------------------------------------
def execute_sql_queries(db_path=DB_PATH):
    print("\n[4/5] Executing 5 benchmark SQL queries covering required clauses...\n")
    conn = sqlite3.connect(db_path)

    queries = {
        "Q1 (SELECT, WHERE, ORDER BY, LIMIT)": """
            SELECT title, price_inr, rating 
            FROM books 
            WHERE in_stock = 1 
            ORDER BY price_inr DESC 
            LIMIT 5;
        """,
        "Q2 (DISTINCT)": """
            SELECT DISTINCT rating 
            FROM books 
            ORDER BY rating ASC;
        """,
        "Q3 (BETWEEN)": """
            SELECT title, price_gbp, price_inr 
            FROM books 
            WHERE price_gbp BETWEEN 20.00 AND 30.00 
            ORDER BY price_gbp ASC 
            LIMIT 5;
        """,
        "Q4 (IN)": """
            SELECT title, rating, category_id 
            FROM books 
            WHERE rating IN (4, 5) 
            ORDER BY rating DESC, title ASC 
            LIMIT 5;
        """,
        "Q5 (JOIN: Inner Join books with categories)": """
            SELECT 
                b.book_id,
                b.title,
                c.category_name,
                b.price_gbp,
                b.price_inr,
                b.rating
            FROM books b
            INNER JOIN categories c ON b.category_id = c.category_id
            WHERE b.rating = 5
            ORDER BY b.price_inr DESC
            LIMIT 10;
        """
    }

    for label, sql in queries.items():
        print("=" * 75)
        print(f"RUNNING: {label}")
        print("SQL Text:")
        print(sql.strip())
        print("-" * 75)
        result_df = pd.read_sql_query(sql, conn)
        print(result_df.to_string(index=False))
        print("=" * 75 + "\n")

    conn.close()
    return queries

# ---------------------------------------------------------
# 5. PANDAS VS SQL EQUIVALENCE CHECK
# ---------------------------------------------------------
def verify_pandas_sql_equivalence(db_path=DB_PATH):
    print("\n[5/5] Verifying equivalence: pd.read_sql vs pd.merge...")
    conn = sqlite3.connect(db_path)

    # 1. SQL Result via pd.read_sql
    join_sql = """
        SELECT 
            b.book_id,
            b.title,
            c.category_name,
            b.price_gbp,
            b.price_inr,
            b.rating
        FROM books b
        INNER JOIN categories c ON b.category_id = c.category_id
        WHERE b.rating = 5
        ORDER BY b.price_inr DESC
        LIMIT 10;
    """
    df_sql = pd.read_sql_query(join_sql, conn)

    # 2. In-memory reproduction via pd.merge
    df_books = pd.read_sql_query("SELECT * FROM books;", conn)
    df_categories = pd.read_sql_query("SELECT * FROM categories;", conn)
    conn.close()

    df_merged = pd.merge(df_books, df_categories, on="category_id", how="inner")
    df_merged = df_merged[df_merged["rating"] == 5]
    df_merged = df_merged.sort_values(by="price_inr", ascending=False).head(10)
    df_merged = df_merged[["book_id", "title", "category_name", "price_gbp", "price_inr", "rating"]].reset_index(drop=True)

    print("\n--- pd.read_sql Result (Top 5 rows) ---")
    print(df_sql.head().to_string(index=False))

    print("\n--- pd.merge Result (Top 5 rows) ---")
    print(df_merged.head().to_string(index=False))

    # Assert structural and numerical equality
    pd.testing.assert_frame_equal(df_sql, df_merged, check_dtype=True)
    print("\nVerification Passed: pd.read_sql and pd.merge produce identical datasets.")

if __name__ == "__main__":
    records = scrape_catalog()
    cleaned_df = clean_data(records)
    setup_and_load_database(cleaned_df)
    execute_sql_queries()
    verify_pandas_sql_equivalence()