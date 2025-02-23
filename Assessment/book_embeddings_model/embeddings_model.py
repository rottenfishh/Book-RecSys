import numpy as np



class recSysModel:
    def __init__(self, path):
        self.path = path
        self.model = np.load(path, allow_pickle=True)

    import numpy as np

    def check_similarity(self, title1, title2):
        embedding1 = None
        embedding2 = None

        for i in self.model:
            if str.lower(i[0]) == str.lower(title1):
                embedding1 = i[1:]
            if str.lower(i[0]) == str.lower(title2):
                embedding2 = i[1:]
        
        if embedding2 is None or embedding1 is None:
            raise ValueError(f"One or two of the titles {title1} {title2} are incorrect")
        
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        return dot_product / (norm1 * norm2)

    def recommend_by_title(self, title, n=10):
        """Find the top-N most similar books based on cosine similarity."""

        embedding = None
        for i in self.model:
            if str.lower(i[0]) == str.lower(title):
                embedding = np.array(i[1:], dtype=np.float32) 
                break
            
        if embedding is None:
            raise ValueError(f"Title '{title}' not found in the model.")

        titles = [i[0] for i in self.model]
        vectors = np.array([i[1:] for i in self.model], dtype=np.float32)

        dot_products = np.dot(vectors, embedding)
        norms_model = np.linalg.norm(vectors, axis=1)
        norm_embedding = np.linalg.norm(embedding)

        valid_norms = (norms_model > 0) & (norm_embedding > 0)
        similarities = np.zeros_like(dot_products)  
        similarities[valid_norms] = dot_products[valid_norms] / (norms_model[valid_norms] * norm_embedding)

        sorted_indices = np.argsort(similarities)[::-1]
        top_n_indices = [idx for idx in sorted_indices if titles[idx] != title][:n]

        return [(titles[i], similarities[i]) for i in top_n_indices]
