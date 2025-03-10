import sys
import os
from src.evaluation import RecallKTitles
from src.models.modules import BookDescriptionEmbeddingSimilarity

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

    from src.evaluation import RecallKTitles
    from src.models.modules import BookDescriptionEmbeddingSimilarity

k = 1000
recs_path = os.path.join(PROJECT_ROOT, "data/test_data/tests.json")
model_path = os.path.join(PROJECT_ROOT, "data/embeddings/books_embeddings_dataset.npy")
dataset_path = os.path.join(PROJECT_ROOT, "data/raw_data/LEHABOOKS.csv")
model = BookDescriptionEmbeddingSimilarity(model_path)
recs = RecallKTitles(k, recs_path, model, dataset_path)
recs.evaluate()