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
    
        positions = []
        for item in intersection:
            pos1 = [i for i, x in enumerate(list1) if x == item]
            pos2 = [i for i, x in enumerate(list2) if x == item]
            positions.append((item, pos1, pos2))
    
        return {
            "intersect_count": count,
            "intersecting_items": positions
        }


    def evaluate(self):
        with open(self.recs_path, "r") as f:
            actual = json.load(f)

        for book in actual.keys():
            try:
                predicted = self.model.recommend_by_title(book, n = self.k)
                comparison_res = self.compare_lists(actual[book], predicted)
                print(f"For book {book} found {comparison_res['intersect_count']} intersections with n = {self.k}")

                comparison_res = self.compare_lists(actual[book][:10], predicted)
                print(f"For book {book} found {comparison_res['intersect_count']} intersections with 10 best recommendations and n = {self.k}")
            except ValueError:
                print(f"{book} not found")
