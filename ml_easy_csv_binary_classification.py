import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        if name == 'Random Forest':
            # Random Forest doesn't need scaling
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            # Other models benefit from scaling
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'auc': auc,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    
    return results, scaler

def evaluate_best_model(results, y_test):
    """Find and evaluate the best performing model"""
    
    # Find best model by AUC score
    best_model_name = max(results.keys(), key=lambda k: results[k]['auc'])
    best_result = results[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best AUC: {best_result['auc']:.4f}")
    
    # Detailed evaluation
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, best_result['y_pred']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, best_result['y_pred'])
    print("\nConfusion Matrix:")
    print(cm)
    
    return best_model_name, best_result

def plot_results(results, y_test):
    """Create visualizations of model performance"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Model comparison bar plot
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    aucs = [results[name]['auc'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, accuracies, width, label='Accuracy')
    axes[0, 0].bar(x + width/2, aucs, width, label='AUC')
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Model Performance Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Best model confusion matrix
    best_model_name = max(results.keys(), key=lambda k: results[k]['auc'])
    cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title(f'Confusion Matrix - {best_model_name}')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # 3. Prediction probability distribution
    y_proba = results[best_model_name]['y_proba']
    axes[1, 0].hist(y_proba[y_test == 0], alpha=0.7, label='No Pest (0)', bins=20)
    axes[1, 0].hist(y_proba[y_test == 1], alpha=0.7, label='Pest (1)', bins=20)
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Prediction Probability Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Feature importance (if Random Forest)
    if best_model_name == 'Random Forest':
        model = results[best_model_name]['model']
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1][:10]  # Top 10 features
        
        axes[1, 1].bar(range(len(indices)), importance[indices])
        axes[1, 1].set_title('Top 10 Feature Importances')
        axes[1, 1].set_xlabel('Feature Index')
        axes[1, 1].set_ylabel('Importance')
        axes[1, 1].set_xticks(range(len(indices)))
        axes[1, 1].set_xticklabels([f'embed_{i:02d}' for i in indices], rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, f'Feature importance\nnot available for\n{best_model_name}', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Feature Importance')
    
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
    print("\n3. Training models...")
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