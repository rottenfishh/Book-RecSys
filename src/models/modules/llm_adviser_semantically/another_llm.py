import requests
import pandas as pd
import json
import math

API_KEY = "" #
MODEL = "" #
CSV_PATH = "output.csv" #pyt koroche
BATCH_SIZE = 4000

def load_book_titles(csv_path: str) -> list:
    df = pd.read_csv(csv_path)
    return df.iloc[:, 0].dropna().tolist()

def query_llm(prompt: str) -> list:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data
    )

    if response.status_code != 200:
        print(f"API error: {response.status_code}")
        return []

    try:
        content = response.json()['choices'][0]['message']['content']
        lines = [line.strip("-•* \n") for line in content.splitlines() if line.strip()]
        return lines
    except Exception as e:
        print("Response parsing error:", e)
        return []

def get_candidates_from_batches(book_title: str, all_titles: list) -> list:
    total = len(all_titles)
    num_batches = math.ceil(total / BATCH_SIZE)
    all_candidates = []

    for i in range(num_batches):
        batch_titles = all_titles[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        prompt = (
            f"I have a book titled: '{book_title}'.\n"
            f"From the following list, return the 10 most semantically similar books:\n\n"
            + "\n".join(batch_titles)
            + "\n\nOnly return the book titles as a list."
        )
        candidates = query_llm(prompt)
        all_candidates.extend(candidates)

    return all_candidates

def get_final_recommendations(book_title: str, candidates: list) -> list:
    prompt = (
        f"I have a book titled: '{book_title}'.\n"
        f"From the following list of candidate books, select the 10 most semantically similar:\n\n"
        + "\n".join(candidates)
        + "\n\nOnly return the titles as a clean list."
    )
    final_recommendations = query_llm(prompt)
    return final_recommendations[:10]

def recommend_books(book_title: str) -> list:
    all_titles = load_book_titles(CSV_PATH)
    candidates = get_candidates_from_batches(book_title, all_titles)
    unique_candidates = list(dict.fromkeys(candidates))  # remove duplicates
    recommendations = get_final_recommendations(book_title, unique_candidates)
    return recommendations
