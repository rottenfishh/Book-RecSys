import json
import pandas as pd
class RecallKTitles:
    def __init__(self, k, recs_path, model, dataset_path):
        self.k = k
        self.recs_path = recs_path
        self.model = model
        self.data = pd.read_csv(dataset_path)

    def intersections_with_all_books(self, list2):
        counter = 0
        for book in list2:
            if book in self.data['Title'].str.lower().values:
                counter += 1
        return counter

    def compare_lists(self, list1, list2):
        list1 = list(map(lambda x: str.lower(x[0]), list1))
        list2 = list(map(lambda x: str.lower(x[0]), list2))
        intersection = set(list1) & set(list2)  
        count = len(intersection)
        counter = self.intersections_with_all_books(list1)

        positions = []
        for item in intersection:
            pos1 = [i for i, x in enumerate(list1) if x == item]
            pos2 = [i for i, x in enumerate(list2) if x == item]
            positions.append((item, pos1, pos2))
    
        return {
            "intersect_count": count,
            "intersecting_items": positions,
            "books_in_dataset": counter
        }


    def evaluate(self):
        with open(self.recs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        count = 0
        for entry in data:
            count += 1
            for book_info, recommendations in entry.items():
                try:
                    title, author, genres = eval(book_info)

                    predicted = self.model.recommend_by_title(title, n=self.k)
                    comparison_res = self.compare_lists(recommendations, predicted)
                    print(f"For book {title} found {comparison_res['intersect_count']} intersections with n = {self.k}, dataset have = {comparison_res['books_in_dataset']}")

                    comparison_res = self.compare_lists(recommendations[:10], predicted)
                    print(
                        f"For book {title} found {comparison_res['intersect_count']} intersections with 10 best recommendations and n = {self.k}, dataset have = {comparison_res['books_in_dataset']}")

                    print("\n" + "-" * 50 + "\n")

                except ValueError:
                    print(f"{title} not found")
        print(f"Test was by {count} books")