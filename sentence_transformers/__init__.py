import numpy as np
from types import SimpleNamespace

class SentenceTransformer:
    def __init__(self, model_name: str):
        if "does-not-exist" in model_name:
            raise ValueError(f"Model '{model_name}' not found")
        self.model_card_data = SimpleNamespace(base_model=model_name)

    def encode(self, texts, show_progress_bar=False):
        if not isinstance(texts, list):
            texts = [texts]

        keywords = ["simgrep", "semantic", "information", "apples", "bananas"]
        embeddings = []
        for text in texts:
            text_lower = text.lower()
            vec = [1.0 if kw in text_lower else 0.0 for kw in keywords]
            embeddings.append(vec)

        return (
            np.array(embeddings, dtype=np.float32)
            if embeddings
            else np.empty((0, len(keywords)), dtype=np.float32)
        )
