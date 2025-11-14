from datetime import datetime, timedelta
import requests
import pandas as pd
from bs4 import BeautifulSoup

headers = {
    "Referer": "https://www.amazon.com/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

base_url = f"https://www.amazon.com/s?k=data+engineering+books"
books = []
page = 1
url = f"{base_url}&page={page}"
response = requests.get(url, headers=headers)

if response.status_code == 200:
    soup = BeautifulSoup(response.content, "html.parser")
    book_containers = soup.find_all('div', {'data-component-type': 's-search-result'})
    for book in book_containers:
        title_tag = book.find('h2')
        title = title_tag.get_text(strip=True) if title_tag else None
        author_tag = book.find('div', class_='a-row a-size-base a-color-secondary')
        author = author_tag.get_text(strip=True).replace('by', '') if author_tag else None
        rating_tag = book.find('span', class_='a-icon-alt')
        rating = rating_tag.get_text(strip=True) if rating_tag else None

        books.append({
            "Title": title,
            "Author": author,
            "Rating": rating,
        })
else:
    raise Exception("Failed to retrieve the page")

books = books[:num_books]
# Remove duplicates
unique_books = {book['Title']: book for book in books}.values()
unique_books_list = list(unique_books)

print(f"Fetched {len(unique_books_list)} unique books")

unique_books_list