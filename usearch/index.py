import numpy as np

class Matches:
    def __init__(self, keys: np.ndarray, distances: np.ndarray):
        self.keys = keys
        self.distances = distances

class BatchMatches:
    def __init__(self, keys: np.ndarray, distances: np.ndarray, counts: np.ndarray):
        self.keys = keys
        self.distances = distances
        self.counts = counts

class Index:
    def __init__(self, ndim: int = 0, metric: str = "cos", dtype: str = "f32"):
        self.metric = metric
        self.dtype = dtype
        self.ndim = ndim
        self.keys = []
        self.vectors = np.empty((0, ndim), dtype=np.float32)

    def add(self, keys: np.ndarray, vectors: np.ndarray) -> None:
        if self.ndim == 0:
            self.ndim = vectors.shape[1]
            self.vectors = np.empty((0, self.ndim), dtype=np.float32)
        self.keys.extend(keys.tolist())
        self.vectors = np.vstack([self.vectors, vectors])

    def __len__(self) -> int:
        return len(self.keys)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            np.savez(f, keys=np.array(self.keys), vectors=self.vectors)

    def load(self, path: str) -> None:
        data = np.load(path)
        self.keys = data["keys"].tolist()
        self.vectors = data["vectors"]
        self.ndim = self.vectors.shape[1]

    def search(self, vectors: np.ndarray, count: int = 5):
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if self.metric == "cos":
            norm_db = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8)
            norm_q = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
            sims = norm_db.dot(norm_q.T)[:, 0]
            idxs = np.argsort(-sims)[:count]
            dists = 1.0 - sims[idxs]
        else:
            dists = np.linalg.norm(self.vectors - vectors, axis=1)
            idxs = np.argsort(dists)[:count]
        return Matches(np.array(self.keys)[idxs], dists)
