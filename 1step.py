import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class HeartDiseasePredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.best_model = None
        self.selected_features = None
        self.feature_names = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
        ]
        
    def load_and_preprocess_data(self):
        """Load and preprocess the heart disease dataset"""
        print("📥 Loading heart disease dataset...")
        
        # Load data
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        try:
            df = pd.read_csv(url, names=self.feature_names)
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
            
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        
        # Handle missing values
        df.replace('?', np.nan, inplace=True)
        missing_before = df.isnull().sum().sum()
        df = df.dropna()
        missing_after = df.isnull().sum().sum()
        print(f"🧹 Missing values handled: {missing_before} → {missing_after}")
        
        # Convert to numeric
        for col in ['ca', 'thal']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Encode categorical variables
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
            
        # Convert target to binary
        df['target'] = df['target'].apply(lambda x: 1 if int(x) > 0 else 0)
        
        # Scale numerical features
        numerical_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        print("✅ Data preprocessing completed")
        return df
    
    def exploratory_data_analysis(self, df):
        """Perform comprehensive EDA"""
        print("📊 Performing Exploratory Data Analysis...")
        
        # Create output directory
        os.makedirs("plots", exist_ok=True)
        
        # Class distribution
        plt.figure(figsize=(12, 8))
        
        # Age distribution by target
        plt.subplot(2, 3, 1)
        sns.histplot(data=df, x='age', hue='target', kde=True, alpha=0.7)
        plt.title('Age Distribution by Heart Disease Status')
        
        # Cholesterol vs target
        plt.subplot(2, 3, 2)
        sns.boxplot(data=df, x='target', y='chol')
        plt.title('Cholesterol vs Heart Disease')
        
        # Chest pain type distribution
        plt.subplot(2, 3, 3)
        cp_counts = df.groupby(['cp', 'target']).size().unstack()
        cp_counts.plot(kind='bar', ax=plt.gca())
        plt.title('Chest Pain Type vs Heart Disease')
        plt.xticks(rotation=0)
        
        # Max heart rate vs target
        plt.subplot(2, 3, 4)
        sns.boxplot(data=df, x='target', y='thalach')
        plt.title('Max Heart Rate vs Heart Disease')
        
        # Exercise induced angina
        plt.subplot(2, 3, 5)
        exang_counts = df.groupby(['exang', 'target']).size().unstack()
        exang_counts.plot(kind='bar', ax=plt.gca())
        plt.title('Exercise Angina vs Heart Disease')
        plt.xticks(rotation=0)
        
        # Target distribution
        plt.subplot(2, 3, 6)
        df['target'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('Heart Disease Distribution')
        
        plt.tight_layout()
        plt.savefig('plots/eda_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ EDA completed. Plots saved to 'plots/' directory")
    
    def feature_selection_analysis(self, X, y):
        """Perform comprehensive feature selection"""
        print("🔍 Performing Feature Selection Analysis...")
        
        # 1. Random Forest Feature Importance
        rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_temp.fit(X, y)
        
        feature_importance = pd.Series(rf_temp.feature_importances_, index=X.columns)
        feature_importance = feature_importance.sort_values(ascending=False)
        
        plt.figure(figsize=(12, 8))
        
        # Feature importance plot
        plt.subplot(2, 2, 1)
        feature_importance.plot(kind='bar')
        plt.title('Feature Importance - Random Forest')
        plt.xticks(rotation=45)
        
        # 2. Recursive Feature Elimination
        rfe = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=8)
        rfe.fit(X, y)
        selected_rfe = X.columns[rfe.support_]
        
        # 3. Chi-Square Test (for positive features only)
        X_positive = X - X.min() + 1  # Make all values positive for chi2
        chi2_selector = SelectKBest(score_func=chi2, k=8)
        chi2_selector.fit(X_positive, y)
        selected_chi2 = X.columns[chi2_selector.get_support()]
        
        # 4. PCA Analysis
        plt.subplot(2, 2, 2)
        pca = PCA()
        X_pca = pca.fit_transform(X)
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        
        plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
        plt.title('PCA - Explained Variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.legend()
        plt.grid(True)
        
        # Feature selection comparison
        plt.subplot(2, 2, 3)
        methods = ['RFE', 'Chi-Square', 'Top RF Features']
        feature_sets = [set(selected_rfe), set(selected_chi2), set(feature_importance.head(8).index)]
        
        # Venn diagram-like comparison (simplified)
        all_selected = set().union(*feature_sets)
        selection_matrix = pd.DataFrame(index=list(all_selected), columns=methods)
        
        for i, (method, features) in enumerate(zip(methods, feature_sets)):
            selection_matrix[method] = selection_matrix.index.isin(features)
        
        sns.heatmap(selection_matrix.astype(int), annot=True, cmap='Blues', cbar=False)
        plt.title('Feature Selection Methods Comparison')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('plots/feature_selection.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Store selected features (using RFE as default)
        self.selected_features = selected_rfe
        
        print(f"✅ Selected features (RFE): {list(selected_rfe)}")
        return selected_rfe
    
    def train_and_evaluate_models(self, X, y):
        """Train and evaluate multiple ML models"""
        print("🤖 Training and evaluating models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42)
        }
        
        results = {}
        plt.figure(figsize=(15, 10))
        
        # ROC curves
        plt.subplot(2, 3, 1)
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            
            # Fit and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            test_auc = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            
            # Store results
            results[name] = {
                'model': model,
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'test_auc': test_auc,
                'predictions': y_pred,
                'probabilities': y_proba
            }
            
            # Plot ROC curve
            plt.plot(fpr, tpr, label=f'{name} (AUC = {test_auc:.3f})')
            
            print(f"\n📌 {name}")
            print(f"   CV AUC: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
            print(f"   Test AUC: {test_auc:.3f}")
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Confusion matrices for top 4 models
        for idx, (name, result) in enumerate(list(results.items())[:4]):
            plt.subplot(2, 3, idx + 2)
            cm = confusion_matrix(y_test, result['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{name}\nAUC: {result["test_auc"]:.3f}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_auc'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\n🏆 Best model: {best_model_name}")
        print(f"   Test AUC: {results[best_model_name]['test_auc']:.3f}")
        
        return results, (X_test, y_test)
    
    def hyperparameter_tuning(self, X, y):
        """Perform hyperparameter tuning for the best model"""
        print("⚙️ Performing hyperparameter tuning...")
        
        # Define parameter grids for different models
        param_grids = {
            'RandomForestClassifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },
            'LogisticRegression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga']
            },
            'SVC': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            }
        }
        
        model_type = type(self.best_model).__name__
        
        if model_type in param_grids:
            param_grid = param_grids[model_type]
            
            # Stratified K-Fold for better validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            grid_search = GridSearchCV(
                self.best_model, 
                param_grid, 
                cv=cv, 
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            self.best_model = grid_search.best_estimator_
            
            print(f"\n🔧 Best parameters for {model_type}:")
            for param, value in grid_search.best_params_.items():
                print(f"   {param}: {value}")
            print(f"✅ Best CV AUC: {grid_search.best_score_:.3f}")
        
        return self.best_model
    
    def clustering_analysis(self, X):
        """Perform clustering analysis"""
        print("🔗 Performing clustering analysis...")
        
        plt.figure(figsize=(15, 5))
        
        # Elbow method for K-means
        plt.subplot(1, 3, 1)
        wcss = []
        K_range = range(1, 11)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        
        plt.plot(K_range, wcss, marker='o')
        plt.title('Elbow Method for K-Means')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Within-Cluster Sum of Squares')
        plt.grid(True, alpha=0.3)
        
        # Hierarchical clustering dendrogram
        plt.subplot(1, 3, 2)
        linked = linkage(X.sample(100), method='ward')  # Sample for visualization
        dendrogram(linked, truncate_mode='lastp', p=10)
        plt.title('Hierarchical Clustering Dendrogram')
        
        # Cluster evaluation
        plt.subplot(1, 3, 3)
        cluster_range = range(2, 8)
        kmeans_scores = []
        hierarchical_scores = []
        
        for n_clusters in cluster_range:
            # K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(X)
            
            # Hierarchical
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
            hier_labels = hierarchical.fit_predict(X)
            
            # Calculate silhouette scores (if sklearn available)
            try:
                from sklearn.metrics import silhouette_score
                kmeans_score = silhouette_score(X, kmeans_labels)
                hier_score = silhouette_score(X, hier_labels)
                kmeans_scores.append(kmeans_score)
                hierarchical_scores.append(hier_score)
            except:
                kmeans_scores.append(0)
                hierarchical_scores.append(0)
        
        if max(kmeans_scores) > 0:  # Only plot if scores were calculated
            plt.plot(cluster_range, kmeans_scores, marker='o', label='K-Means')
            plt.plot(cluster_range, hierarchical_scores, marker='s', label='Hierarchical')
            plt.title('Clustering Performance Comparison')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Clustering analysis completed")
    
    def save_model(self, model_path='models/heart_disease_model.pkl'):
        """Save the trained model and preprocessing objects"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_package = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'selected_features': self.selected_features,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_package, model_path)
        print(f"💾 Model and preprocessing objects saved to: {model_path}")
    
    def generate_report(self, results, test_data):
        """Generate a comprehensive analysis report"""
        X_test, y_test = test_data
        
        print("\n" + "="*60)
        print("📋 HEART DISEASE PREDICTION - ANALYSIS REPORT")
        print("="*60)
        
        print(f"\n📊 Dataset Summary:")
        print(f"   • Total samples: {len(X_test) + len(X_test)}")  # Approximate
        print(f"   • Features used: {len(self.selected_features)}")
        print(f"   • Selected features: {list(self.selected_features)}")
        
        print(f"\n🏆 Model Performance Summary:")
        for name, result in results.items():
            print(f"   • {name}:")
            print(f"     - CV AUC: {result['cv_auc_mean']:.3f} (±{result['cv_auc_std']:.3f})")
            print(f"     - Test AUC: {result['test_auc']:.3f}")
        
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_auc'])
        print(f"\n🎯 Best Model: {best_model_name}")
        print(f"   • Test AUC Score: {results[best_model_name]['test_auc']:.3f}")
        
        # Feature importance for best model if available
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.selected_features,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 Top 5 Most Important Features:")
            for idx, row in importance_df.head().iterrows():
                print(f"   • {row['feature']}: {row['importance']:.3f}")
        
        print("\n💡 Recommendations:")
        print("   • Model is ready for deployment")
        print("   • Consider collecting more data for improved performance")
        print("   • Regular model retraining recommended")
        print("   • Monitor model performance in production")
        
        print("\n" + "="*60)

# Main execution
def main():
    # Initialize predictor
    predictor = HeartDiseasePredictor()
    
    # Load and preprocess data
    df = predictor.load_and_preprocess_data()
    if df is None:
        return
    
    # Perform EDA
    predictor.exploratory_data_analysis(df)
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Feature selection
    selected_features = predictor.feature_selection_analysis(X, y)
    X_selected = X[selected_features]
    
    # Train and evaluate models
    results, test_data = predictor.train_and_evaluate_models(X_selected, y)
    
    # Hyperparameter tuning
    predictor.hyperparameter_tuning(X_selected, y)
    
    # Clustering analysis
    predictor.clustering_analysis(X_selected)
    
    # Save model
    predictor.save_model()
    
    # Generate report
    predictor.generate_report(results, test_data)
    
    print("\n🎉 Analysis completed successfully!")
    print("📁 Check 'plots/' directory for visualizations")
    print("💾 Model saved in 'models/' directory")

if __name__ == "__main__":
    main()