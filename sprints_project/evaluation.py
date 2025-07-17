import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class ModelEvaluator:
    def __init__(self, model_path: str = "model.pkl"):
        self.model_path = model_path
        self.model = joblib.load(model_path)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"Accuracy: {acc:.2f}")
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", report)
