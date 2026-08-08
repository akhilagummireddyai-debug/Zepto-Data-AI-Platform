import requests
from bs4 import BeautifulSoup
import pandas as pd

book_data = []

for page in range(1, 6):

    url = f"https://books.toscrape.com/catalogue/page-{page}.html"

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    books = soup.find_all("article", class_="product_pod")

    print(f"Page {page} - {len(books)} books")

    for book in books:

        title = book.h3.a["title"]

        price = book.find("p", class_="price_color").text

        rating = book.find("p", class_="star-rating")["class"][1]

        availability = book.find("p", class_="instock availability").text.strip()

        book_link = book.h3.a["href"]
        book_url = "https://books.toscrape.com/catalogue/" + book_link

        book_response = requests.get(book_url)
        book_soup = BeautifulSoup(book_response.text, "html.parser")

        breadcrumb = book_soup.find("ul", class_="breadcrumb")
        items = breadcrumb.find_all("li")
        category = items[2].get_text(strip=True)

        book_data.append([
            title,
            price,
            rating,
            availability,
            category
        ])

df = pd.DataFrame(
    book_data,
    columns=[
        "Title",
        "Price",
        "Rating",
        "Availability",
        "Category"
    ]
)

df["price_gbp"] = df["Price"].str.replace("Â£", "", regex=False).astype(float)

rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["rating"] = df["Rating"].map(rating_map)

df["in_stock"] = df["Availability"].str.contains("In stock")

GBP_TO_INR = 105.50

df["price_inr"] = df["price_gbp"] * GBP_TO_INR

df.to_csv("books.csv", index=False)

print(df.head())

print("\nCSV file saved successfully.")
print("Total Books:", len(df))