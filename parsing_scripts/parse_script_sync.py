import csv
import requests

def fetch_book_data(isbn):
    api_url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
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

def parse_first_dataset(input_file, output_file):
    with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ["Title", "Authors", "Description", "Category", "Publisher", "Publish"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            isbn = row.get("ISBN")
            if isbn:
                book_data = fetch_book_data(isbn)
                if book_data and all(book_data[key] != "N/A" for key in ["Title", "Authors", "Description", "Category", "Publish"]):
                    writer.writerow(book_data)
                else:
                    print(f"Skip {isbn}")
                    
input_file = "Books.csv"
output_file = "output.csv" 
parse_first_dataset(input_file, output_file)
