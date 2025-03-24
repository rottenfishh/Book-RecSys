import pandas as pd
from difflib import get_close_matches

class SearchBooksByTitle:
    def __init__(self, library_path):
        self.dataset_df = pd.read_csv(library_path)  # ← правильный способ загрузки CSV
        self.titles = self.dataset_df["Title"].dropna().to_list()

    def closest_title(self, title, size):
        return get_close_matches(title, self.titles, n=size, cutoff=0.5)