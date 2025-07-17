import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class CorrelationVisualizer:
    @staticmethod
    def heatmap(df: pd.DataFrame):
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        plt.title("Heatmap of Feature Correlations")
        plt.show()

    @staticmethod
    def target_correlation(df: pd.DataFrame):
        correlation = df.corr(numeric_only=True)['target'].drop('target')
        correlation = correlation.reindex(correlation.abs().sort_values(ascending=False).index)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=correlation.values, y=correlation.index, palette='coolwarm')
        plt.title('Feature Correlations with Heart Disease Target')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Feature')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
