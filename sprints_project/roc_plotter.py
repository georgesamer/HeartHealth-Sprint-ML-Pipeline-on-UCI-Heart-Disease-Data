import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class ROCPlotter:
    def __init__(self, model_path: str = "model.pkl"):
        self.model = joblib.load(model_path)

    def plot_roc(self, X_test, y_test):
        y_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
