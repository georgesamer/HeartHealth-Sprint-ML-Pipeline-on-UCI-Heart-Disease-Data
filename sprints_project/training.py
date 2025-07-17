import joblib
from sklearn.linear_model import LogisticRegression

class ModelTrainer:
    def __init__(self, model_path: str = "model.pkl"):
        self.model_path = model_path

    def train_and_save(self, X_train, y_train):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        joblib.dump(model, self.model_path)
        print(f"Model saved to {self.model_path}")
