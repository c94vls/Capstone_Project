-- ====================================================================
-- Zepto Data & AI Platform - Data Pipeline Benchmark SQL Queries
-- ====================================================================

-- 1. SELECT, WHERE, ORDER BY, LIMIT
-- High-value in-stock products ordered by INR price
SELECT title, price_inr, rating 
FROM books 
WHERE in_stock = 1 
ORDER BY price_inr DESC 
LIMIT 5;

-- 2. DISTINCT
-- Distinct ratings present in the catalog
SELECT DISTINCT rating 
FROM books 
ORDER BY rating ASC;

-- 3. BETWEEN
-- Mid-tier products priced between 20.00 and 30.00 GBP
SELECT title, price_gbp, price_inr 
FROM books 
WHERE price_gbp BETWEEN 20.00 AND 30.00 
ORDER BY price_gbp ASC 
LIMIT 5;

-- 4. IN
-- Filtering catalog for top-tier review ratings (4 and 5 stars)
SELECT title, rating, category_id 
FROM books 
WHERE rating IN (4, 5) 
ORDER BY rating DESC, title ASC 
LIMIT 5;

-- 5. INNER JOIN
-- Top 10 highest-priced 5-star books joined with category names
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