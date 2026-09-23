# Data Pipeline Module (`/data_pipeline`)

## 1. Overview & Architecture
This module scrapes catalog and competitive pricing data from `books.toscrape.com`, cleans messy strings, enriches records with fixed-rate currency conversion, and loads them into a normalized SQLite relational database.

### Relational Schema (3NF)
+------------------+             +--------------------+
|    categories    |             |       books        |
+------------------+             +--------------------+
| category_id (PK) | <---+-+---o | book_id (PK)       |
| category_name    |     | |     | title              |
+------------------+     | |     | price_gbp          |
| |     | price_inr          |
| |     | rating             |
| |     | in_stock           |
| +---o | category_id (FK)   |
+-------+--------------------+


## 2. Currency Conversion Baseline
* **Fixed Rate:** `1 GBP = 105.50 INR`
* This rate is a fixed, project-defined baseline constant requiring zero network API calls and no date reference.
* Formula: `price_inr = round(price_gbp * 105.50, 2)`

## 3. Cleaning & Parsing Decisions
* **Price (`price_gbp`):** Cleaned using regex `(\d+\.\d+)` to strip currency marks (`£` and non-breaking space encoding artifacts `Â`). If a row fails to parse, it is imputed with the catalog median price to preserve dataset size without distorting variance.
* **Rating (`rating`):** Extracted from CSS class names (`One` through `Five`) and mapped directly to integers `1` to `5`.
* **Stock Availability (`in_stock`):** Normalized to binary `1` (In stock) or `0` (Out of stock).
* **Missing Title:** Rows missing titles represent critical identifier failures and are dropped rather than imputed.

## 4. Execution Instructions

```bash
# 1. Install dependencies from repo root
pip install -r requirements.txt

# 2. Run the end-to-end pipeline script
python data_pipeline/pipeline.py

# 3. Alternatively, test queries directly via sqlite3 CLI
sqlite3 data_pipeline/zepto_catalog.db < data_pipeline/queries.sql
5. Equivalence Verification
The script extracts the 5-star catalog join via pd.read_sql and independently merges the raw in-memory books and categories tables using pd.merge(..., how="inner"). Both outputs are verified for strict equality using pd.testing.assert_frame_equal.


---

### Execution & Verification Checklist

1. **Volume:** Targeting Travel (11 books), Mystery (32 books), Historical Fiction (26 books), and Sequential Art (20 books) yields **89 books across 4 categories**, cleanly exceeding the $\ge 60$ books / $\ge 3$ categories threshold.
2. **Commit History Check:** Ensure you run the Git branching sequence described in the Capstone Blueprint before submitting so `git log --graph --all` displays the required feature branch and merge commit.