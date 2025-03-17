import json
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from typing import List, Tuple, Dict

RECS_PATH = "../../data/test_data/test_final.json"


def to_lower(obj):
    if isinstance(obj, dict):
        return {to_lower(k): to_lower(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_lower(i) for i in obj]
    elif isinstance(obj, str):
        return obj.lower()
    else:
        return obj


class RecallKTitles:
    def __init__(self, model, books_list):
        self.model = model
        self.books_list = books_list
        self.recs_path = RECS_PATH
        with open(self.recs_path, "r", encoding="utf-8") as f:
            self.recs = json.load(f)

        print(self.recs)
        print(len(self.books_list), self.books_list[0])
        self.recs_distilled = [
            [
                book_entry[0],
                [
                    book_rec
                    for book_rec in book_entry[1]
                    if book_rec[0] in self.books_list
                ],
            ]
            for book_entry in self.recs
            if book_entry[0][0] in self.books_list
        ]

    def stats(self, predicted, gt_recs, min_n=100, step=100, max_n=10000):
        """
        I hate computers.

        Returns:
        List[Tuple[int, Dict[str, int]]]: A list of tuples where the first element is n (number of predictions),
        and the second element is a dictionary containing:
            - "TP" (int): True Positives count.
            - "FN" (int): False Negatives count.
            - "Recall" (float): Recall metric computed as TP / (TP + FN).
        """
        stats_list = []
        for n in range(min_n, max_n, step):
            stats = {}
            print(predicted[0], gt_recs[0])
            stats["TP"] = len(set(predicted[:n]) & set(gt_recs))
            stats["FN"] = len(set(gt_recs) - set(predicted[:n]))
            stats["Recall"] = stats["TP"] / (stats["TP"] + stats["FN"])

            stats_list.append((n, stats))

        return stats_list

    def score_distilled(self, min_n=100, step=100, max_n=10000):
        """
        Computes recommendation statistics for distilled recommendations.

        This function iterates through `self.recs_distilled`, retrieves recommendations
        from the model for each book, and computes statistics using the `stats` method.

        Args:
            min_n (int, optional): Minimum number of recommendations to consider. Defaults to 100.
            step (int, optional): Step size for increasing n. Defaults to 100.
            max_n (int, optional): Maximum number of recommendations to retrieve. Defaults to 10000.

        Returns:
            List[Tuple[Tuple[str, str, str], List[Tuple[int, Dict[str, int]]]]]:
            A list of tuples where:
            - The first element is the key (title, author, genres).
            - The second element is the computed statistics from `stats()`.
        """
        stat_list = []
        for rec in self.recs_distilled:
            title = rec[0][0]
            preds = self.model.recommend_by_title(title, n=max_n)
            stat_list.append((rec[0], self.stats(preds, rec[1], min_n, step, max_n)))

        return stat_list

    def score_all(self, min_n=100, step=100, max_n=10000):
        stat_list = []
        for rec in self.recs_distilled:
            title = rec[0][0]
            try:
                preds = self.model.recommend_by_title(title, n=max_n)
            except:
                continue
            stat_list.append((rec[0], self.stats(preds, rec[1], min_n, step, max_n)))

        return stat_list

    def score(self, min_n=100, step=100, max_n=10000):
        return self.score_agg(min_n, step, max_n, False), self.score_agg(
            min_n, step, max_n, True
        )

    def score_agg(self, min_n=100, step=100, max_n=10000, distilled=False):
        """
        Aggregates the scores from `score_distilled` by computing the average TP, FN, and Recall
        across all books for each N value.

        Returns:
            List[Tuple[int, Dict[str, float]]]: A list of tuples where:
            - The first element is the N value (number of recommendations considered).
            - The second element is a dictionary with the average TP, FN, and Recall.
        """
        if distilled:
            all_scores = self.score_distilled(min_n, step, max_n)
        else:
            all_scores = self.score_all(min_n, step, max_n)

        if not all_scores:
            print(len(self.recs), len(self.recs_distilled))
            return []

        aggregated_scores = defaultdict(
            lambda: {"TP": 0, "FN": 0, "Recall": 0.0, "count": 0}
        )

        for _, book_scores in all_scores:
            for n, metrics in book_scores:
                aggregated_scores[n]["TP"] += metrics["TP"]
                aggregated_scores[n]["FN"] += metrics["FN"]
                aggregated_scores[n]["Recall"] += metrics["Recall"]
                aggregated_scores[n]["count"] += 1

        result = []
        for n in sorted(aggregated_scores.keys()):
            count = aggregated_scores[n]["count"]
            avg_metrics = {
                "TP": aggregated_scores[n]["TP"] / count,
                "FN": aggregated_scores[n]["FN"] / count,
                "Recall": aggregated_scores[n]["Recall"] / count,
            }
            result.append((n, avg_metrics))

        return result

    def plot_score_agg(self, distilled=False):
        """
        Plots the aggregated scores (TP, FN, Recall) as a function of N.

        This function takes the output of `score_agg()` and visualizes
        how TP, FN, and Recall change with increasing N.
        """
        scores = self.score_agg(distilled=distilled)

        if not scores:
            print("No data to plot.")
            return

        n_values = [n for n, _ in scores]
        tp_values = [metrics["TP"] for _, metrics in scores]
        fn_values = [metrics["FN"] for _, metrics in scores]
        recall_values = [metrics["Recall"] for _, metrics in scores]

        plt.figure(figsize=(10, 6))

        plt.plot(n_values, tp_values, label="True Positives (TP)", marker="o")
        plt.plot(n_values, fn_values, label="False Negatives (FN)", marker="s")
        plt.plot(n_values, recall_values, label="Recall", marker="^")

        plt.xlabel("N (Number of Recommendations)")
        plt.ylabel("Score")
        plt.title("Aggregated Recommendation Performance")
        plt.legend()
        plt.grid(True)

        plt.show()
