import json
import seaborn as sns
from copy import deepcopy
from collections import defaultdict

import os

RECS_PATH = os.path.join("..", "..", "data", "test_data", "cleared_gpt.json")
class Metrics():
    def __init__(self, model, books_list, k):
        self.model = model
        self.books_list = books_list
        self.recs_path = RECS_PATH
        self.k = k

        with open(self.recs_path, "r", encoding="utf-8") as f:
            self.recs = json.load(f)


    def __model_results(self, title):
        return self.model.recommend_by_title(title, n=self.k)

    def __hit_ratio(self, preds, recs):
        preds, recs = set(preds), set(recs)
        return any([book in preds for book in recs])

    def __confussion_matrix(self, preds, recs):
        preds, recs = set(preds), set(recs)
        matrix = {}
        matrix["TP"] = len([book for book in preds if book in recs])
        matrix["FP"] = len([book for book in preds if book not in recs])
        matrix["FN"] = len([book for book in recs if book not in preds])
        matrix["TN"] = -1
        return matrix

    def __reciprocal_rank(self, preds, recs):
        indexes = [index for index in range(len(preds)) if preds[index] in recs]
        if len(indexes) == 0:
            return 0
        return 1 / indexes[0]

    def __recall(self, preds, recs):
        preds, recs = set(preds), set(recs)
        return len([book for book in preds if book in recs]) / len(recs)

    def __precision(self, preds, recs):
        preds, recs = set(preds), set(recs)
        return len([book for book in preds if book in recs]) / len(preds)


    def stats(self):
        results=defaultdict(list)

        for book_entry in self.recs:
            title = book_entry["title"]
            preds = [title for title, _ in self.__model_results(title)]
            recs = book_entry["recommendations"]
            hit_ratio = self.__hit_ratio(preds, recs)
            recall = self.__recall(preds, recs)
            precision = self.__precision(preds, recs)
            matrix = self.__confussion_matrix(preds, recs)
            reciprocal_rank = self.__reciprocal_rank(preds, recs)

            results["hit_ratio"].append(hit_ratio)
            results["recall"].append(recall)
            results["precision"].append(precision)
            results["matrix"].append(matrix)
            results["reciprocal_rank"].append(reciprocal_rank)

        return results

    def average_stats(self, results=None):
        if results == None:
            results = self.stats()

        results_average = {}
        results_average["hit_ratio_average"] = sum(results["hit_ratio"]) / len(results["hit_ratio"])
        results_average["recall_average"] = sum(results["recall"]) / len(results["recall"])
        results_average["precision_average"] = sum(results["precision"]) / len(results["precision"])
        results_average["mean_reciprocal_rank"] = sum(results["reciprocal_rank"]) / len(results["reciprocal_rank"])

        return results_average

    def graph_stats(self, results, results_average):


        sns.lineplot(data=results)

        length = len(results["recall"])
        results_average = deepcopy(results_average)
        results_average["hit_ratio_average"] = [results_average["hit_ratio_average"]] * length
        results_average["precision_average"] = [results_average["precision_average"]] * length
        results_average["recall_average"] = [results_average["recall_average"]] * length
        results_average["mean_reciprocal_rank"] = [results_average["mean_reciprocal_rank"]] * length
        sns.lineplot(data=results_average)
