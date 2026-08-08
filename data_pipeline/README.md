# Zepto Data AI Platform

## Module 1 - Data Pipeline

### Overview

This module implements a complete data pipeline for scraping, cleaning, transforming, storing, and querying book catalog data.

The pipeline follows:

Scraping → Cleaning → Currency Conversion → CSV → SQLite Database → SQL Queries → Pandas Analysis

The data is collected from:

http://books.toscrape.com/

Books to Scrape is a public website created for practicing web scraping.

---

## Objective

The objective of this module is to build a data pipeline that:

- Scrapes book data using Requests and BeautifulSoup
- Cleans and transforms the scraped data
- Converts GBP prices to INR
- Stores the cleaned data in a CSV file
- Loads the data into a normalized SQLite database
- Performs SQL queries
- Reads SQL results using Pandas
- Reproduces a SQL JOIN using Pandas merge

---

## Data Source

The project uses the All Products catalogue from:

http://books.toscrape.com/

The first 5 pages of the catalogue were scraped.

Each page contains 20 books.

Total books collected:

**100 books**

This satisfies the requirement of at least 60 books.

The dataset contains books from multiple categories.

---

## Data Collected

For each book, the following information was collected:

- Title
- Price
- Star Rating
- Availability
- Category

---

## Data Cleaning

### Price

The original price contains the GBP currency symbol.

Example:

£51.77

The currency symbol is removed and the value is converted into a float.

The cleaned column is:

`price_gbp`

Example:

51.77

---

### Star Rating

The website provides ratings as text.

The following conversion is performed:

- One → 1
- Two → 2
- Three → 3
- Four → 4
- Five → 5

The cleaned column is:

`rating`

---

### Availability

The availability text is converted into a boolean-style value.

Example:

In stock → True

The cleaned column is:

`in_stock`

When stored in SQLite, the value is represented as 1 or 0.

---

### Category

The category is extracted from the breadcrumb section of each individual book page.

The category is stored in the:

`Category`

column.

---

## Currency Conversion

The project requires a fixed baseline conversion rate.

The following rate is used:

**1 GBP = 105.50 INR**

This is a project-defined fixed conversion rate and is not a live exchange rate.

The INR price is calculated using:

`price_inr = price_gbp * 105.50`

Example:

`51.77 * 105.50 = 5461.735`

No external currency API is required for this conversion.

---

## CSV Output

After scraping and cleaning, the data is saved as:

`books.csv`

The final dataset contains:

**100 books**

The dataset contains the following important columns:

- Title
- Price
- Rating
- Availability
- Category
- price_gbp
- rating
- in_stock
- price_inr

---

## Database

The cleaned data is stored in a SQLite database:

`books.db`

The database contains two normalized tables:

### Categories Table

`categories`

Columns:

- category_id
- category_name

`category_id` is the primary key.

`category_name` is unique.

---

### Books Table

`books`

Columns:

- book_id
- title
- price_gbp
- price_inr
- rating
- in_stock
- category_id

`book_id` is the primary key.

`category_id` is a foreign key referencing:

`categories(category_id)`

---

## Database Relationship

The relationship between the tables is:

categories

↓

category_id

↓

books.category_id

This provides a normalized relational database structure.

---

## SQL Queries

The database script executes multiple SQL queries.

### Query 1 - SELECT / WHERE

Find books with a rating of 5.

```sql
SELECT title, rating
FROM books
WHERE rating = 5;
Module 1 Data Pipeline completed successfully with 100 scraped books, SQLite normalization, SQL queries, and pandas validation.