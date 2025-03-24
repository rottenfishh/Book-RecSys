import os
import sys
import json
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

class RecommendUsingGraph:
    def __init__(self, graph_path, model):
        self.graph = self.load_graph(graph_path)
        self.model = model  

    def load_graph(self, graph_path):
        """Loads a graph from a JSON file and returns a NetworkX graph."""
        with open(graph_path, "r") as f:
            graph_data = json.load(f)

        Graph = json_graph.node_link_graph(graph_data) 
        return Graph

    def find_neighbors_title(self, title):
        """Find books directly connected to the given title in the graph."""
        matching_nodes = [node for node in self.graph.nodes if node.startswith(f"{title} (")]
        all_neighbors = set()  

        for node in matching_nodes:
            neighbors = list(self.graph.neighbors(node))
            all_neighbors.update(neighbors)

        return [neighbor.rsplit(" (", 1)[0] for neighbor in all_neighbors]
    
    def find_closest_books(self, title, n=20):
        """Find the top-N most relevant books based on the model and neighbor graph."""
        predicted = self.model.recommend_by_title(title, n + 1)

        neighbors = self.find_neighbors_title(title)

        embedding = None
        for i in self.model.model:
            if i[0] == title:
                embedding = np.array(i[1:], dtype=np.float32)
                break

        if embedding is None:
            raise ValueError(f"Title '{title}' not found in the model.")

        neighbor_scores = {}
        for neighbor in neighbors:
            if neighbor == title: 
                continue  

            neighbor_embedding = None
            for i in self.model.model:
                if i[0] == neighbor:
                    neighbor_embedding = np.array(i[1:], dtype=np.float32)
                    break

            if neighbor_embedding is None:
                continue

            norm_embedding = np.linalg.norm(embedding)
            norm_neighbor = np.linalg.norm(neighbor_embedding)
            similarity = np.dot(embedding, neighbor_embedding) / (norm_embedding * norm_neighbor) if norm_embedding and norm_neighbor else 0

            neighbor_scores[neighbor] = similarity

        sorted_neighbors = sorted(neighbor_scores.items(), key=lambda x: -x[1])[:10]

        combined = {}

        for book, score in predicted:
            if book != title:  
                combined[book] = score

        for book, similarity in sorted_neighbors:
            if book in combined:
                combined[book] = max(combined[book], similarity) 
            else:
                combined[book] = similarity

        sorted_books = sorted(combined.items(), key=lambda x: -x[1])
        return [(book, score) for book, score in sorted_books[:n]]
