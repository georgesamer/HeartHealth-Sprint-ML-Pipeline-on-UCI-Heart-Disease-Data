# HeartHealth-Sprint-ML-Pipeline-on-UCI-Heart-Disease-Data
🧠 Built during a sprint by Sprints, this ML project analyzes the UCI Heart Disease dataset using PCA, K-Means, anomaly detection, and Apriori. Modular code, real-world focus, and unsupervised techniques for health insight discovery.
name: "Heart Disease Prediction Project"
description: >
  Machine Learning project for predicting heart disease using the UCI Cleveland dataset.
  Includes full pipeline: preprocessing, EDA, feature selection, model training & evaluation,
  hyperparameter tuning, clustering, saving model, and report generation.
  
features:
  - "📥 Data Preprocessing (missing values, encoding, scaling)"
  - "📊 Exploratory Data Analysis (EDA) with visualizations"
  - "🔍 Feature Selection (Random Forest, RFE, Chi-Square, PCA)"
  - "🤖 Model Training (Logistic Regression, Decision Tree, Random Forest, SVM)"
  - "⚙️ Hyperparameter Tuning with GridSearchCV"
  - "🔗 Clustering Analysis (KMeans + Hierarchical)"
  - "💾 Save model with joblib"
  - "📋 Auto-generated report"

requirements:
  - python>=3.8
  - pandas
  - numpy
  - seaborn
  - matplotlib
  - scikit-learn
  - joblib
  - scipy

usage:
  steps:
    - "Clone repo: git clone https://github.com/your-username/heart-disease-prediction.git"
    - "cd heart-disease-prediction"
    - "Run: python heart_disease_predictor.py"
  outputs:
    - "plots/ : contains generated visualizations"
    - "models/heart_disease_model.pkl : saved trained model"
    - "Console report: summary of dataset, features, and performance"

project_structure:
  - heart_disease_predictor.py: "Main script"
  - plots/: "EDA, feature selection, clustering plots"
  - models/: "Saved ML models"
  - README.yaml: "Project documentation in YAML"

author:
  name: "George Essam"
  note: "Always learning and exploring AI & ML applications in healthcare"
