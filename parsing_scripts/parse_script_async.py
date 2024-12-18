import csv
import aiohttp
import asyncio

async def fetch_book_data(isbn):
    api_url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
    async with aiohttp.ClientSession() as session:
        async with session.get(api_url) as response:
            if response.status == 200:
                data = await response.json()
                if "items" in data:
                    volume_info = data["items"][0]["volumeInfo"]
                    return {
                        "Title": volume_info.get("title", "N/A"),
                        "Authors": ", ".join(volume_info.get("authors", [])),
                        "Description": volume_info.get("description", "N/A"),
                        "Category": ", ".join(volume_info.get("categories", [])),
                        "Publisher": volume_info.get("publisher", "N/A"),
                        "Publish": volume_info.get("publishedDate", "N/A"),
                    }
    return None

async def parse_first_dataset(input_file, output_file):
    with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ["Title", "Authors", "Description", "Category", "Publisher", "Publish"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        tasks = []
        for row in reader:
            isbn = row.get("ISBN")
            if isbn:
                book_data = await fetch_book_data(isbn)
                if book_data and all(book_data[key] != "N/A" for key in ["Title", "Authors", "Description", "Category", "Publish"]):
                    writer.writerow(book_data)
                else:
                    print(f"Skip {isbn}")
        await asyncio.gather(*tasks)

input_file = "Books.csv"
output_file = "output.csv"
asyncio.run(parse_first_dataset(input_file, output_file))
