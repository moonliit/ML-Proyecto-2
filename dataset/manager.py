from dataclasses import dataclass
from typing import Dict, List
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np

FEATS_FILE = "dataset/files/movie_poster_all_features.pkl"
TRAIN_CSV = "dataset/files/train/movies_train.csv"
TEST_CSV = "dataset/files/test/movies_test.csv"
SAMPLE_CSV = "dataset/files/test/sample_submission.csv"
POSTERS_BASE = "dataset/posters"

@dataclass
class DatasetManager:
    # feats and movie id mapping
    feats_df: pd.DataFrame
    movie_id_to_index: Dict[int, int]

    # train/test initial dfs
    train_initial_df: pd.DataFrame
    test_initial_df: pd.DataFrame

    # train/test dataframes
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    
    # train/test np arrays
    X_train: np.ndarray
    X_test: np.ndarray
    
    # submission df
    submission_df: pd.DataFrame

    def get_all_ids(self) -> List[int]:
        return self.feats_df["movieId"].tolist()

    def get_movie_vector(self, movie_id: int) -> np.ndarray:
        if movie_id not in self.movie_id_to_index:
            print(f"Invalid movie: {movie_id}")
            return None
        index = self.movie_id_to_index[movie_id]
        row = self.feats_df.iloc[index]
        row = row.drop(labels=["movieId"])
        return row.values

    def get_poster_image(self, movie_id: int) -> Image:
        path_str = f"{POSTERS_BASE}/{movie_id}.jpg"
        path = Path(path_str)
        if not path.exists():
            return None
        image = Image.open(path).convert("RGB")
        return image

    @staticmethod
    def load():
        """
        Función que carga los dataframes con movieId y las 512 dimensiones de HSV
        """
        # feats df
        feats_df = pd.read_pickle(FEATS_FILE)

        # movie ids
        movie_ids = set(feats_df["movieId"])
        movie_id_to_index = {mid: idx for idx, mid in enumerate(movie_ids)}

        # train/test dfs
        train_initial_df = pd.read_csv(TRAIN_CSV)
        test_initial_df = pd.read_csv(TEST_CSV)

        train_ids = [train_initial_df.iloc[i]["movieId"] for i in range(len(train_initial_df))]
        test_ids = [test_initial_df.iloc[i]["movieId"] for i in range(len(test_initial_df))]
        
        train_df = feats_df[feats_df["movieId"].isin(train_ids)]
        test_df = feats_df[feats_df["movieId"].isin(test_ids)]

        # submission df
        submission_df = pd.read_csv(SAMPLE_CSV)
        submission_df = submission_df[
            submission_df["query_movie_id"].isin(movie_ids) &
            submission_df["recommended_movie_id"].isin(movie_ids)
        ].reset_index(drop=True)
        
        # train/test np arrays
        X_train = train_df.drop(columns=["movieId"]).values
        X_test = test_df.drop(columns=["movieId"]).values

        return DatasetManager(
            # feats and movie id mapping
            feats_df=feats_df,
            movie_id_to_index=movie_id_to_index,
            # train/test initial dfs
            train_initial_df=train_initial_df,
            test_initial_df=test_initial_df,
            # train/test dataframes
            train_df=train_df,
            test_df=test_df,
            # train/test np arrays
            X_train=X_train,
            X_test=X_test,
            # submission df
            submission_df=submission_df
        )

ds = DatasetManager.load()