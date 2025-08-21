import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(csv_file_path):
    """Load CSV and prepare features and target"""
    df = pd.read_csv(csv_file_path)
    
    # Display basic info about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['target_proxy'].value_counts()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Prepare features (embedding columns) and target
    feature_cols = [col for col in df.columns if col.startswith('embed_')]
    X = df[feature_cols]
    y = df['target_proxy']
    
    print(f"Number of embedding features: {len(feature_cols)}")
    
    return X, y, df

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance"""
    
    # Scale features for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define all models
    models = {
        # Tree-based models (don't need scaling)
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        
        # Linear models (benefit from scaling)
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Ridge Classifier': RidgeClassifier(random_state=42),
        'SGD Classifier': SGDClassifier(random_state=42, max_iter=1000),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
        
        # Distance-based models (need scaling)
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'SVM (RBF)': SVC(random_state=42, probability=True, kernel='rbf'),
        'SVM (Linear)': SVC(random_state=42, probability=True, kernel='linear'),
        'SVM (Poly)': SVC(random_state=42, probability=True, kernel='poly', degree=3),
        
        # Probabilistic models
        'Naive Bayes': GaussianNB(),
        
        # Neural networks (need scaling)
        'Neural Network (Small)': MLPClassifier(hidden_layer_sizes=(50,), random_state=42, max_iter=500),
        'Neural Network (Medium)': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
        'Neural Network (Large)': MLPClassifier(hidden_layer_sizes=(200, 100, 50), random_state=42, max_iter=500),
    }
    
    # Models that don't need scaling
    no_scaling_models = [
        'Random Forest', 'Extra Trees', 'Decision Tree', 
        'Gradient Boosting', 'AdaBoost', 'Naive Bayes'
    ]
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...", end=" ")
        
        try:
            if name in no_scaling_models:
                # Models that don't need scaling
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                # Models that benefit from scaling
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            
            # Calculate cross-validation score for robustness
            if name in no_scaling_models:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            else:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc,
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_proba': y_proba,
                'needs_scaling': name not in no_scaling_models
            }
            
            print(f"✓ Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, CV AUC: {cv_scores.mean():.4f}±{cv_scores.std():.3f}")
            
        except Exception as e:
            print(f"✗ Failed: {str(e)}")
            continue
    
    return results, scaler

def evaluate_best_model(results, y_test):
    """Find and evaluate the best performing model"""
    
    # Find best model by AUC score
    best_model_name = max(results.keys(), key=lambda k: results[k]['auc'])
    best_result = results[best_model_name]
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"{'='*60}")
    print(f"Test AUC: {best_result['auc']:.4f}")
    print(f"Test Accuracy: {best_result['accuracy']:.4f}")
    print(f"CV AUC: {best_result['cv_auc_mean']:.4f} ± {best_result['cv_auc_std']:.3f}")
    
    # Show top 5 models
    print(f"\n{'='*60}")
    print("TOP 5 MODELS BY AUC:")
    print(f"{'='*60}")
    sorted_models = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
    for i, (name, result) in enumerate(sorted_models[:5]):
        print(f"{i+1:2d}. {name:<25} AUC: {result['auc']:.4f} | Acc: {result['accuracy']:.4f} | CV: {result['cv_auc_mean']:.4f}±{result['cv_auc_std']:.3f}")
    
    # Detailed evaluation for best model
    print(f"\n{'='*60}")
    print(f"DETAILED EVALUATION - {best_model_name}")
    print(f"{'='*60}")
    print(classification_report(y_test, best_result['y_pred']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, best_result['y_pred'])
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                No Pest  Pest")
    print(f"Actual No Pest    {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"       Pest       {cm[1,0]:4d}   {cm[1,1]:4d}")
    
    return best_model_name, best_result

def plot_results(results, y_test):
    """Create comprehensive visualizations of model performance"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Model comparison bar plot (sorted by AUC)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
    model_names = [name for name, _ in sorted_results]
    accuracies = [results[name]['accuracy'] for name, _ in sorted_results]
    aucs = [results[name]['auc'] for name, _ in sorted_results]
    cv_aucs = [results[name]['cv_auc_mean'] for name, _ in sorted_results]
    
    x = np.arange(len(model_names))
    width = 0.25
    
    axes[0, 0].bar(x - width, accuracies, width, label='Test Accuracy', alpha=0.8)
    axes[0, 0].bar(x, aucs, width, label='Test AUC', alpha=0.8)
    axes[0, 0].bar(x + width, cv_aucs, width, label='CV AUC', alpha=0.8)
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Model Performance Comparison (Sorted by AUC)')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Best model confusion matrix
    best_model_name = max(results.keys(), key=lambda k: results[k]['auc'])
    cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title(f'Confusion Matrix - {best_model_name}')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # 3. Prediction probability distribution for best model
    y_proba = results[best_model_name]['y_proba']
    axes[0, 2].hist(y_proba[y_test == 0], alpha=0.7, label='No Pest (0)', bins=20, color='blue')
    axes[0, 2].hist(y_proba[y_test == 1], alpha=0.7, label='Pest (1)', bins=20, color='red')
    axes[0, 2].set_xlabel('Predicted Probability')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title(f'Prediction Probability Distribution - {best_model_name}')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Feature importance (for tree-based models)
    tree_based_models = ['Random Forest', 'Extra Trees', 'Decision Tree', 'Gradient Boosting', 'AdaBoost']
    best_tree_model = None
    for model_name in tree_based_models:
        if model_name in results:
            if best_tree_model is None or results[model_name]['auc'] > results[best_tree_model]['auc']:
                best_tree_model = model_name
    
    if best_tree_model:
        model = results[best_tree_model]['model']
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1][:15]  # Top 15 features
        
        axes[1, 0].bar(range(len(indices)), importance[indices])
        axes[1, 0].set_title(f'Top 15 Feature Importances - {best_tree_model}')
        axes[1, 0].set_xlabel('Feature Index')
        axes[1, 0].set_ylabel('Importance')
        axes[1, 0].set_xticks(range(len(indices)))
        axes[1, 0].set_xticklabels([f'embed_{i:02d}' for i in indices], rotation=45)
    else:
        axes[1, 0].text(0.5, 0.5, 'No tree-based model\navailable for\nfeature importance', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Feature Importance')
    
    # 5. Cross-validation scores comparison
    cv_means = [results[name]['cv_auc_mean'] for name, _ in sorted_results]
    cv_stds = [results[name]['cv_auc_std'] for name, _ in sorted_results]
    
    axes[1, 1].errorbar(range(len(model_names)), cv_means, yerr=cv_stds, 
                       fmt='o-', capsize=3, capthick=1)
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('Cross-Validation AUC')
    axes[1, 1].set_title('Cross-Validation Performance with Error Bars')
    axes[1, 1].set_xticks(range(len(model_names)))
    axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Precision-Recall curve for top 3 models
    top_3_models = sorted_results[:3]
    colors = ['blue', 'red', 'green']
    
    for i, (name, result) in enumerate(top_3_models):
        precision, recall, _ = precision_recall_curve(y_test, result['y_proba'])
        axes[1, 2].plot(recall, precision, color=colors[i], 
                       label=f'{name} (AUC: {result["auc"]:.3f})')
    
    axes[1, 2].set_xlabel('Recall')
    axes[1, 2].set_ylabel('Precision')
    axes[1, 2].set_title('Precision-Recall Curve (Top 3 Models)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main(csv_file_path):
    """Main function to run the complete ML pipeline"""
    
    print("=== Pest Prediction ML Pipeline ===\n")
    
    # Load and prepare data
    print("1. Loading and preparing data...")
    X, y, df = load_and_prepare_data(csv_file_path)
    
    # Split data
    print("\n2. Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train models
    print("\n3. Training 18 different models...")
    print("   This may take a few minutes...")
    results, scaler = train_models(X_train, X_test, y_train, y_test)
    
    # Evaluate best model
    print("\n4. Evaluating best model...")
    best_model_name, best_result = evaluate_best_model(results, y_test)
    
    # Plot results
    print("\n5. Creating visualizations...")
    plot_results(results, y_test)
    
    return results, scaler, best_model_name

# Example usage:
if __name__ == "__main__":
    # Replace 'your_file.csv' with your actual CSV file path
    csv_file_path = 'your_file.csv'
    
    try:
        results, scaler, best_model = main(csv_file_path)
        print(f"\nPipeline completed successfully!")
        print(f"Best model: {best_model}")
        
        # You can now use the best model for predictions on new data
        # best_trained_model = results[best_model]['model']
        # predictions = best_trained_model.predict(new_data)
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{csv_file_path}'")
        print("Please make sure the file path is correct.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")