from scipy.spatial.distance import pdist, squareform
import numpy as np

"""
Estas clases no son utilizadas en el proyecto, dado a que son ineficientes por falta de optimizacion para la data.
Sin embargo, replican el mismo funcionamiento que las implementaciones de scikit-learn.
"""

class KMeansManual:
    def __init__(self, n_clusters=2, max_iter=100, random_state=42, init="kmeans++"):
        self.k = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.init = init

    def _kmeans_plus_plus_init(self, X):
        rng = np.random.default_rng(seed=self.random_state)
        centroids = [X[rng.integers(len(X))]]
        for _ in range(1, self.k):
            dists = np.min([np.linalg.norm(X - c, axis=1)**2 for c in centroids], axis=0)
            probs = dists / dists.sum()
            next_idx = rng.choice(len(X), p=probs)
            centroids.append(X[next_idx])
        return np.array(centroids)

    def _random_init(self, X):
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(X), size=self.k, replace=False)
        return X[indices]

    def fit(self, X):
        if self.init == "kmeans++":
            self.centroids = self._kmeans_plus_plus_init(X)
        else:
            self.centroids = self._random_init(X)

        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels_ = np.argmin(distances, axis=1)

            new_centroids = np.array([
                X[self.labels_ == i].mean(axis=0) if np.any(self.labels_ == i) else self.centroids[i]
                for i in range(self.k)
            ])

            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

class AgglomerativeManual:
    def __init__(self, n_clusters=2, linkage='average'):
        self.k = n_clusters
        self.linkage = linkage

    def _compute_linkage(self, c1, c2, distances):
        dists = [distances[i, j] for i in c1 for j in c2]
        if self.linkage == 'single':
            return min(dists)
        elif self.linkage == 'complete':
            return max(dists)
        elif self.linkage == 'average':
            return sum(dists) / len(dists)
        else:
            raise ValueError(f"Linkage '{self.linkage}' no soportado")

    def fit(self, X):
        n = len(X)
        clusters = [{i} for i in range(n)]
        distances = squareform(pdist(X, metric="euclidean"))
        np.fill_diagonal(distances, np.inf)

        while len(clusters) > self.k:
            min_dist = np.inf
            to_merge = (0, 1)

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    linkage = self._compute_linkage(clusters[i], clusters[j], distances)
                    if linkage < min_dist:
                        min_dist = linkage
                        to_merge = (i, j)

            i, j = to_merge
            clusters[i] = clusters[i].union(clusters[j])
            del clusters[j]

        # Asignar etiquetas
        self.labels_ = np.empty(n, dtype=int)
        for cluster_idx, cluster in enumerate(clusters):
            for i in cluster:
                self.labels_[i] = cluster_idx

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_