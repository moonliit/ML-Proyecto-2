from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

class PlotProcessor:
    def __init__(self, random_state: int = 42):
        self.scaler = StandardScaler() #joblib.load("dataset/model_scaler_hybrid_562.pkl")
        self.pca2 = PCA(n_components=2, random_state=random_state)
        self.random_state = random_state
    
    def fit_transform(self, X):
        X_scaled = self.scaler.fit_transform(X) # scaler is already trained
        X_2d = self.pca2.fit_transform(X_scaled)
        return X_2d
    
    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        X_2d = self.pca2.transform(X_scaled)
        return X_2d

transformer = PlotProcessor()