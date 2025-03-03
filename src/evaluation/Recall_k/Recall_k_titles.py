import json

class RecallKTitles:
    def __init__(self, k, recs_path, model):
        self.k = k
        self.recs_path = recs_path
        self.model = model
    
    def compare_lists(self, list1, list2):
        list1 = list(map(str.lower, list1))
        list2 = list(map(lambda x: str.lower(x[0]), list2))
        intersection = set(list1) & set(list2)  
        count = len(intersection)
        counter = intersections_with_all_books(list2)

        positions = []
        for item in intersection:
            pos1 = [i for i, x in enumerate(list1) if x == item]
            pos2 = [i for i, x in enumerate(list2) if x == item]
            positions.append((item, pos1, pos2))
    
        return {
            "intersect_count": count,
            "intersecting_items": positions,
            "books_in_dataset": all
        }

    def intersections_with_all_books(self, list2):
        counter = 0
        for book in list2:
            if (df['Title'].str.lower() == book):
                counter += 1
        return counter

    def evaluate(self):
        with open("./computed.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            for book_info, recommendations in entry.items():
                title, author, genres = eval(book_info)

                predicted = self.model.recommend_by_title(title, n=self.k)
                comparison_res = self.compare_lists(recommendations, predicted)
                print(f"For book {title} found {comparison_res['intersect_count']} intersections with n = {self.k}, dataset have = {comparison_res['books_in_dataset']}")

                comparison_res = self.compare_lists(recommendations[:10], predicted)
                print(
                    f"For book {title} found {comparison_res['intersect_count']} intersections with 10 best recommendations and n = {self.k}, dataset have = {comparison_res['books_in_dataset']}")

                print("\n" + "-" * 50 + "\n")
