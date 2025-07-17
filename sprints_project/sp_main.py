from data_loader import DataLoader
from visualize_correlation import CorrelationVisualizer
from preprocessing import DataPreprocessor
from training import ModelTrainer
from evaluation import ModelEvaluator
from roc_plotter import ROCPlotter

# Load Data
loader = DataLoader()
heart_disease, X, y = loader.load()

# Combine for EDA
df = X.copy()
df['target'] = y

# Visualize Correlations
CorrelationVisualizer.heatmap(df)
CorrelationVisualizer.target_correlation(df)

# Preprocess
preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.split_and_scale(df)

# Train
trainer = ModelTrainer()
trainer.train_and_save(X_train, y_train)

# Evaluate
evaluator = ModelEvaluator()
evaluator.evaluate(X_test, y_test)

# Plot ROC
roc_plotter = ROCPlotter()
roc_plotter.plot_roc(X_test, y_test)
