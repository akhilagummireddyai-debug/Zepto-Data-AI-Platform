import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

csv_file = BASE_DIR / "books.csv"
db_file = BASE_DIR / "books.db"

conn = sqlite3.connect(db_file)
cursor = conn.cursor()

cursor.execute("PRAGMA foreign_keys = ON")

cursor.execute("DROP TABLE IF EXISTS books")
cursor.execute("DROP TABLE IF EXISTS categories")
conn.commit()

cursor.execute("""
CREATE TABLE categories(
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE
)
""")

cursor.execute("""
CREATE TABLE books(
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY(category_id) REFERENCES categories(category_id)
)
""")

conn.commit()

df = pd.read_csv(csv_file)

print("CSV loaded successfully.")
print(f"Total Books: {len(df)}")

categories = df["Category"].unique()

for category in categories:
    cursor.execute(
        "INSERT OR IGNORE INTO categories(category_name) VALUES(?)",
        (category,)
    )

conn.commit()

print("Categories inserted successfully.")

for index, row in df.iterrows():

    cursor.execute(
        "SELECT category_id FROM categories WHERE category_name=?",
        (row["Category"],)
    )

    category_id = cursor.fetchone()[0]

    cursor.execute("""
    INSERT INTO books
    (title, price_gbp, price_inr, rating, in_stock, category_id)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        row["Title"],
        row["price_gbp"],
        row["price_inr"],
        int(row["rating"]),
        int(row["in_stock"]),
        category_id
    ))

conn.commit()

print("Books inserted successfully.")


print("\nQuery 1 - SELECT / WHERE")

cursor.execute("""
SELECT title, rating
FROM books
WHERE rating = 5
LIMIT 10
""")

print(cursor.fetchall())


print("\nQuery 2 - ORDER BY / LIMIT")

cursor.execute("""
SELECT title, price_inr
FROM books
ORDER BY price_inr DESC
LIMIT 5
""")

print(cursor.fetchall())


print("\nQuery 3 - DISTINCT")

cursor.execute("""
SELECT DISTINCT category_name
FROM categories
""")

print(cursor.fetchall())


print("\nQuery 4 - BETWEEN")

cursor.execute("""
SELECT title, price_inr
FROM books
WHERE price_inr BETWEEN 4000 AND 6000
LIMIT 10
""")

print(cursor.fetchall())


print("\nQuery 5 - IN")

cursor.execute("""
SELECT title, rating
FROM books
WHERE rating IN (4, 5)
LIMIT 10
""")

print(cursor.fetchall())


print("\nQuery 6 - JOIN")

cursor.execute("""
SELECT books.title, categories.category_name
FROM books
JOIN categories
ON books.category_id = categories.category_id
LIMIT 10
""")

print(cursor.fetchall())


books_df = pd.read_sql(
    "SELECT * FROM books",
    conn
)

categories_df = pd.read_sql(
    "SELECT * FROM categories",
    conn
)

print("\nBooks DataFrame")
print(books_df.head())


print("\nCategories DataFrame")
print(categories_df.head())


join_df = pd.read_sql("""
SELECT books.title, categories.category_name
FROM books
JOIN categories
ON books.category_id = categories.category_id
LIMIT 10
""", conn)

print("\nJOIN using pd.read_sql")
print(join_df)


merge_df = pd.merge(
    books_df,
    categories_df,
    on="category_id"
)

merge_df = merge_df[
    ["title", "category_name"]
].head(10)

print("\nJOIN using pd.merge")
print(merge_df)


conn.close()

print("\nDatabase Completed Successfully.")