from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np
import joblib

class PlotProcessor:
    def __init__(self, reduction_method="pca", cluster_model="kmeans", num_clusters=4, subspace="full", y_train=None, random_state=42):
        if reduction_method not in ["pca", "svd", "lda", "pca+lda"]:
            raise ValueError(f"Metodo de reducción '{reduction_method}' no soportado")
        if cluster_model not in ["kmeans", "agglo"]:
            raise ValueError(f"Modelo de clustering '{cluster_model}' no soportado")
        if num_clusters < 1:
            raise ValueError(f"Numero de clusters '{num_clusters}' invalido")
        if subspace not in ["full", "vt", "kbest"]:
            raise ValueError(f"Metodo de subespacio '{subspace}' no soportado")

        self.scaler = StandardScaler()
        self.reduction_method = reduction_method
        self.cluster_model = cluster_model
        self.num_clusters = num_clusters
        self.subspace = subspace
        self.y_train = y_train
        self.random_state = random_state
        # PCA
        self.pca50 = PCA(n_components=50, random_state=random_state)
        self.pca2 = PCA(n_components=2, random_state=random_state)
        # SVD
        self.svd50 = TruncatedSVD(n_components=50, random_state=random_state)
        self.svd2 = TruncatedSVD(n_components=2, random_state=random_state)
        # LDA
        self.lda_full = LinearDiscriminantAnalysis(n_components=None)
        self.lda2 = LinearDiscriminantAnalysis(n_components=2)
        # Subspace reductors
        self.vt = VarianceThreshold(threshold=1.0)
        self.kbest = SelectKBest(score_func=f_classif, k=500)

    def fit_transform(self, X):
        X_scaled = self.scaler.fit_transform(X)
        X_sub = self.subspace_fit_transform(X_scaled)
        if self.reduction_method == "pca":
            X_2d = self.pca2.fit_transform(X_sub)
        elif self.reduction_method == "svd":
            X_2d = self.svd2.fit_transform(X_sub)
        elif self.reduction_method == "lda":
            X_2d = self.lda2.fit_transform(X_sub, self.y_train)
        elif self.reduction_method == "pca+lda":
            X_2d = self.lda2.fit_transform(X_sub, self.y_train)
        return X_2d
    
    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        X_sub = self.subspace_transform(X_scaled)
        if self.reduction_method == "pca":
            X_2d = self.pca2.transform(X_sub)
        elif self.reduction_method == "svd":
            X_2d = self.svd2.transform(X_sub)
        elif self.reduction_method == "lda":
            X_2d = self.lda2.transform(X_sub)
        elif self.reduction_method == "pca+lda":
            X_2d = self.lda2.transform(X_sub)
        return X_2d
    
    def fit_full_transform(self, X):
        X_scaled = self.scaler.fit_transform(X)
        X_sub = self.subspace_fit_transform(X_scaled)
        if self.reduction_method == "pca":
            X_full = self.pca50.fit_transform(X_sub)
        elif self.reduction_method == "svd":
            X_full = self.svd50.fit_transform(X_sub)
        elif self.reduction_method == "lda":
            X_full = self.lda_full.fit_transform(X_sub, self.y_train)
        elif self.reduction_method == "pca+lda":
            G = self.lda_full.n_components
            X_pca = self.pca50.fit_transform(X_sub)
            X_lda = self.lda_full.fit_transform(X_sub, self.y_train)
            X_full = np.hstack([X_lda, X_pca[:, G:]])
        return X_full

    def full_transform(self, X):
        X_scaled = self.scaler.transform(X)
        X_sub = self.subspace_transform(X_scaled)
        if self.reduction_method == "pca":
            X_full = self.pca50.transform(X_sub)
        elif self.reduction_method == "svd":
            X_full = self.svd50.transform(X_sub)
        elif self.reduction_method == "lda":
            X_full = self.lda_full.transform(X_sub)
        elif self.reduction_method == "pca+lda":
            G = self.lda_full.n_components
            X_pca = self.pca50.transform(X_sub)
            X_lda = self.lda_full.transform(X_sub)
            X_full = np.hstack([X_lda, X_pca[:, G:]])
        return X_full

    def fit_predict(self, X):
        X_full = self.fit_full_transform(X)
        labels = self.get_clusterizer().fit_predict(X_full)
        return X_full, labels
    
    def get_clusterizer(self):
        if self.cluster_model == "kmeans":
            return KMeans(n_clusters=self.num_clusters, random_state=self.random_state)
        if self.cluster_model == "agglo":
            return AgglomerativeClustering(n_clusters=self.num_clusters)

    def subspace_fit_transform(self, X):
        if self.subspace == "full":
            return X
        elif self.subspace == "vt":
            X_sub = self.vt.fit_transform(X)
        elif self.subspace == "kbest":
            X_sub = self.kbest.fit_transform(X, self.y_train)
        return X_sub

    def subspace_transform(self, X):
        if self.subspace == "full":
            return X
        elif self.subspace == "vt":
            X_sub = self.vt.transform(X)
        elif self.subspace == "kbest":
            X_sub = self.kbest.transform(X)
        return X_sub

    def create_title(self):
        if self.reduction_method in ["pca", "svd", "pca+lda"]:
            return f"{self.reduction_method.upper()} 2D + clusterización sobre {self.reduction_method.upper()} 50D"
        else:
            return f"{self.reduction_method.upper()} 2D + clusterización sobre {self.reduction_method.upper()} 18D"