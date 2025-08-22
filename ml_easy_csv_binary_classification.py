import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, ElasticNet, PassiveAggressiveClassifier, Perceptron
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Import additional boosting libraries
import xgboost as xgb
import catboost as cb

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

def create_keras_model(input_dim, model_type='small'):
    """Create different Keras model architectures"""
    
    if model_type == 'small':
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
    elif model_type == 'medium':
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
    elif model_type == 'large':
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

class KerasClassifierWrapper:
    """Wrapper to make Keras models compatible with sklearn interface"""
    
    def __init__(self, model_type='small', epochs=50, batch_size=32, verbose=0):
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.input_dim = None
    
    def fit(self, X, y):
        self.input_dim = X.shape[1]
        self.model = create_keras_model(self.input_dim, self.model_type)
        
        # Add early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=self.verbose
        )
        return self
    
    def predict(self, X):
        predictions = self.model.predict(X, verbose=0)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        predictions = self.model.predict(X, verbose=0).flatten()
        # Return probabilities for both classes
        return np.column_stack([1 - predictions, predictions])
    
    def get_params(self, deep=True):
        """Get parameters for this estimator (required by sklearn)"""
        return {
            'model_type': self.model_type,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'verbose': self.verbose
        }
    
    def set_params(self, **parameters):
        """Set parameters for this estimator (required by sklearn)"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

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
        'Histogram Gradient Boosting': HistGradientBoostingClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        
        # Bagging ensembles
        'Bagging (Decision Tree)': BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=42), 
            n_estimators=50, random_state=42
        ),
        'Bagging (SVM)': BaggingClassifier(
            estimator=SVC(random_state=42, probability=True), 
            n_estimators=10, random_state=42
        ),
        
        # Linear models (benefit from scaling)
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Ridge Classifier': RidgeClassifier(random_state=42),
        'SGD Classifier': SGDClassifier(random_state=42, max_iter=1000, loss='log_loss'),
        'Elastic Net': ElasticNet(random_state=42, max_iter=1000),
        'Passive Aggressive': PassiveAggressiveClassifier(random_state=42, max_iter=1000),
        'Perceptron': Perceptron(random_state=42, max_iter=1000),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
        
        # Distance-based models (need scaling)
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'SVM (RBF)': SVC(random_state=42, probability=True, kernel='rbf'),
        'SVM (Linear)': SVC(random_state=42, probability=True, kernel='linear'),
        'SVM (Poly)': SVC(random_state=42, probability=True, kernel='poly', degree=3),
        'Linear SVC': LinearSVC(random_state=42, max_iter=2000),
        
        # Probabilistic models
        'Naive Bayes (Gaussian)': GaussianNB(),
        'Naive Bayes (Complement)': ComplementNB(),
        'Naive Bayes (Bernoulli)': BernoulliNB(),
        'Gaussian Process': GaussianProcessClassifier(random_state=42),
        
        # Neural networks (need scaling)
        'Neural Network (Small)': MLPClassifier(hidden_layer_sizes=(50,), random_state=42, max_iter=500),
        'Neural Network (Medium)': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
        'Neural Network (Large)': MLPClassifier(hidden_layer_sizes=(200, 100, 50), random_state=42, max_iter=500),
    }
    
    # Add LightGBM models
    models.update({
        'LightGBM': lgb.LGBMClassifier(
            random_state=42,
            n_estimators=100,
            verbosity=-1,
            force_col_wise=True
        ),
        'LightGBM (Tuned)': lgb.LGBMClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbosity=-1,
            force_col_wise=True
        ),
    })
    
    # Add XGBoost models
    models.update({
        'XGBoost': xgb.XGBClassifier(
            random_state=42,
            n_estimators=100,
            eval_metric='logloss',
            verbosity=0
        ),
        'XGBoost (Tuned)': xgb.XGBClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            verbosity=0
        ),
    })
    
    # Add CatBoost models
    models.update({
        'CatBoost': cb.CatBoostClassifier(
            random_state=42,
            iterations=100,
            verbose=False
        ),
        'CatBoost (Tuned)': cb.CatBoostClassifier(
            random_state=42,
            iterations=200,
            depth=6,
            learning_rate=0.1,
            verbose=False
        ),
    })
    
    # Add Keras models
    models.update({
        'Keras Neural Network (Small)': KerasClassifierWrapper(model_type='small', epochs=100, verbose=0),
        'Keras Neural Network (Medium)': KerasClassifierWrapper(model_type='medium', epochs=100, verbose=0),
        'Keras Neural Network (Large)': KerasClassifierWrapper(model_type='large', epochs=100, verbose=0),
    })
    
    # Create ensemble models using some of the best individual models
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42, n_estimators=50)),
        ('lr', LogisticRegression(random_state=42, max_iter=1000))
    ]
    
    models.update({
        'Voting Classifier (Hard)': VotingClassifier(
            estimators=base_models,
            voting='hard'
        ),
        'Voting Classifier (Soft)': VotingClassifier(
            estimators=base_models,
            voting='soft'
        ),
        'Stacking Classifier': StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(random_state=42),
            cv=3
        ),
    })
    
    # Models that don't need scaling
    no_scaling_models = [
        'Random Forest', 'Extra Trees', 'Decision Tree', 
        'Gradient Boosting', 'Histogram Gradient Boosting', 'AdaBoost',
        'Bagging (Decision Tree)', 'Bagging (SVM)',
        'Naive Bayes (Gaussian)', 'Naive Bayes (Complement)', 'Naive Bayes (Bernoulli)',
        'LightGBM', 'LightGBM (Tuned)', 'Gaussian Process',
        'Voting Classifier (Hard)', 'Voting Classifier (Soft)', 'Stacking Classifier'
    ]
    
    # Add XGBoost and CatBoost to no_scaling_models
    no_scaling_models.extend(['XGBoost', 'XGBoost (Tuned)', 'CatBoost', 'CatBoost (Tuned)'])
    
    results = {}
    total_models = len(models)
    
    for i, (name, model) in enumerate(models.items(), 1):
        print(f"Training {name} ({i}/{total_models})...", end=" ")
        
        try:
            if name in no_scaling_models:
                # Models that don't need scaling
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Get probabilities or decision scores
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_proba = model.decision_function(X_test)
                else:
                    y_proba = np.zeros_like(y_pred, dtype=float)
                
                # Cross-validation
                try:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
                except:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
            else:
                # Models that benefit from scaling
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Get probabilities or decision scores
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test_scaled)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_proba = model.decision_function(X_test_scaled)
                else:
                    y_proba = np.zeros_like(y_pred, dtype=float)
                
                # Cross-validation with scaled data
                try:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
                except:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate AUC if we have probability-like scores
            try:
                auc = roc_auc_score(y_test, y_proba)
            except:
                auc = 0.0  # Default if AUC calculation fails
            
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

def evaluate_holdout_dataset(results, scaler, best_model_name, holdout_csv_path, output_csv_path=None):
    """Evaluate the best model on holdout dataset and save predictions"""
    
    print(f"\n{'='*60}")
    print("HOLDOUT DATASET EVALUATION")
    print(f"{'='*60}")
    
    # Load holdout dataset
    print(f"Loading holdout dataset from: {holdout_csv_path}")
    holdout_df = pd.read_csv(holdout_csv_path)
    
    print(f"Holdout dataset shape: {holdout_df.shape}")
    print(f"Holdout target distribution:\n{holdout_df['target_proxy'].value_counts()}")
    
    # Prepare features
    feature_cols = [col for col in holdout_df.columns if col.startswith('embed_')]
    X_holdout = holdout_df[feature_cols]
    y_holdout = holdout_df['target_proxy']
    
    # Get the best model and its configuration
    best_result = results[best_model_name]
    best_model = best_result['model']
    needs_scaling = best_result['needs_scaling']
    
    print(f"\nUsing best model: {best_model_name}")
    print(f"Model needs scaling: {needs_scaling}")
    
    # Make predictions
    print("Making predictions on holdout dataset...")
    
    if needs_scaling:
        X_holdout_processed = scaler.transform(X_holdout)
    else:
        X_holdout_processed = X_holdout
    
    # Get predictions and probabilities
    y_pred_holdout = best_model.predict(X_holdout_processed)
    y_proba_holdout = best_model.predict_proba(X_holdout_processed)[:, 1]
    
    # Calculate holdout performance metrics
    holdout_accuracy = accuracy_score(y_holdout, y_pred_holdout)
    holdout_auc = roc_auc_score(y_holdout, y_proba_holdout)
    
    print(f"\n{'='*60}")
    print("HOLDOUT PERFORMANCE RESULTS")
    print(f"{'='*60}")
    print(f"Holdout Accuracy: {holdout_accuracy:.4f}")
    print(f"Holdout AUC: {holdout_auc:.4f}")
    print(f"Training AUC: {best_result['auc']:.4f}")
    print(f"Performance difference: {best_result['auc'] - holdout_auc:+.4f}")
    
    # Detailed classification report
    print(f"\nDetailed Classification Report (Holdout):")
    print(classification_report(y_holdout, y_pred_holdout))
    
    # Confusion Matrix
    cm_holdout = confusion_matrix(y_holdout, y_pred_holdout)
    print(f"\nConfusion Matrix (Holdout):")
    print(f"                Predicted")
    print(f"                No Pest  Pest")
    print(f"Actual No Pest    {cm_holdout[0,0]:4d}   {cm_holdout[0,1]:4d}")
    print(f"       Pest       {cm_holdout[1,0]:4d}   {cm_holdout[1,1]:4d}")
    
    # Create results dataframe
    results_df = holdout_df.copy()
    results_df['predicted_class'] = y_pred_holdout
    results_df['predicted_probability'] = y_proba_holdout
    
    # Generate output filename if not provided
    if output_csv_path is None:
        output_csv_path = f"holdout_predictions_{best_model_name.lower().replace(' ', '_')}.csv"
    
    # Save to CSV
    print(f"\nSaving results to: {output_csv_path}")
    results_df.to_csv(output_csv_path, index=False)
    
    # Show sample of results
    print(f"\nSample of saved results:")
    sample_cols = ['entity_id', 'target_proxy', 'predicted_class', 'predicted_probability'] + feature_cols[:3]
    sample_df = results_df[sample_cols].head()
    print(sample_df.to_string(index=False, float_format='%.4f'))
    
    # Create holdout visualization
    plot_holdout_results(y_holdout, y_pred_holdout, y_proba_holdout, best_model_name, 
                        best_result['auc'], holdout_auc)
    
    return results_df, holdout_accuracy, holdout_auc

def plot_holdout_results(y_true, y_pred, y_proba, model_name, train_auc, holdout_auc):
    """Create visualizations for holdout dataset results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title(f'Holdout Confusion Matrix - {model_name}')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # 2. Prediction Probability Distribution
    axes[0, 1].hist(y_proba[y_true == 0], alpha=0.7, label='No Pest (0)', bins=20, color='blue')
    axes[0, 1].hist(y_proba[y_true == 1], alpha=0.7, label='Pest (1)', bins=20, color='red')
    axes[0, 1].set_xlabel('Predicted Probability')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Holdout Probability Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. AUC Comparison (Train vs Holdout)
    auc_comparison = ['Training AUC', 'Holdout AUC']
    auc_values = [train_auc, holdout_auc]
    colors = ['lightblue', 'orange']
    
    bars = axes[1, 0].bar(auc_comparison, auc_values, color=colors, alpha=0.7)
    axes[1, 0].set_ylabel('AUC Score')
    axes[1, 0].set_title('Training vs Holdout Performance')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, auc_values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    axes[1, 1].plot(recall, precision, color='blue', linewidth=2, 
                   label=f'Holdout (AUC: {holdout_auc:.3f})')
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].set_title('Holdout Precision-Recall Curve')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

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
    
    # Feature importance (for tree-based models including LightGBM)
    tree_based_models = ['Random Forest', 'Extra Trees', 'Decision Tree', 'Gradient Boosting', 'AdaBoost', 'LightGBM', 'LightGBM (Tuned)']
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

def main(csv_file_path, holdout_csv_path=None, output_csv_path=None):
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
    print(f"\n3. Training multiple models...")
    print("   Keras models may take longer due to neural network training...")
    print("   This may take several minutes...")
    results, scaler = train_models(X_train, X_test, y_train, y_test)
    
    # Evaluate best model
    print("\n4. Evaluating best model...")
    best_model_name, best_result = evaluate_best_model(results, y_test)
    
    # Plot results
    print("\n5. Creating visualizations...")
    plot_results(results, y_test)
    
    # Evaluate on holdout dataset if provided
    holdout_results = None
    if holdout_csv_path:
        print("\n6. Evaluating on holdout dataset...")
        holdout_results = evaluate_holdout_dataset(results, scaler, best_model_name, 
                                                  holdout_csv_path, output_csv_path)
    
    return results, scaler, best_model_name, holdout_results

# Example usage:
if __name__ == "__main__":
    # Install required packages first:
    # pip install pandas numpy scikit-learn matplotlib seaborn lightgbm tensorflow
    
    # Replace with your actual file paths
    csv_file_path = 'your_training_file.csv'
    holdout_csv_path = 'your_holdout_file.csv'  # Optional
    output_csv_path = 'holdout_predictions.csv'  # Optional - will auto-generate if not provided
    
    print("All required libraries are available.")
    print()
    
    try:
        # Run the complete pipeline
        results, scaler, best_model, holdout_results = main(
            csv_file_path, 
            holdout_csv_path=holdout_csv_path,
            output_csv_path=output_csv_path
        )
        
        print(f"\nPipeline completed successfully!")
        print(f"Best model: {best_model}")
        
        if holdout_results:
            holdout_df, holdout_acc, holdout_auc = holdout_results
            print(f"Holdout dataset evaluated and saved!")
            print(f"Holdout accuracy: {holdout_acc:.4f}")
            print(f"Holdout AUC: {holdout_auc:.4f}")
        
        # You can now use the best model for predictions on new data
        # best_trained_model = results[best_model]['model']
        # predictions = best_trained_model.predict(new_data)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find a file - {str(e)}")
        print("Please make sure all file paths are correct.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Alternative: Run just holdout evaluation if you already have trained models
def evaluate_holdout_only(results, scaler, best_model_name, holdout_csv_path, output_csv_path=None):
    """
    Use this function if you've already trained models and just want to evaluate holdout data
    
    Example:
    results, scaler, best_model = main('training_data.csv')  # Train first
    evaluate_holdout_only(results, scaler, best_model, 'holdout_data.csv', 'predictions.csv')
    """
    return evaluate_holdout_dataset(results, scaler, best_model_name, holdout_csv_path, output_csv_path)