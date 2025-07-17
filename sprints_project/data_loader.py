from ucimlrepo import fetch_ucirepo
import pandas as pd
from typing import Tuple

class DataLoader:
    def __init__(self, dataset_id: int = 45):
        self.dataset_id = dataset_id

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        heart_disease = fetch_ucirepo(id=self.dataset_id)
        X = heart_disease.data.features
        y = heart_disease.data.targets
        return heart_disease, X, y
