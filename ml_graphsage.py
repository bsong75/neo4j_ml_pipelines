import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           f1_score, accuracy_score, precision_score, recall_score)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class PestPredictionMLPipeline:
    def __init__(self, data_file='enhanced_graphsage_entity_features.csv'):
        """Initialize the ML pipeline for pest prediction"""
        self.data_file = data_file
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scalers = {}
        self.models = {}
        self.feature_names = None
        
    def load_and_prepare_data(self):
        """Load the GraphSAGE enhanced features and prepare for ML"""
        print("Loading GraphSAGE enhanced features...")
        
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"Loaded data shape: {self.df.shape}")
            print(f"Columns: {self.df.columns.tolist()}")
            
            # Check for the target variable
            if 'has_pest_ever' not in self.df.columns:
                raise ValueError("Target variable 'has_pest_ever' not found in data")
            
            # Remove rows with missing target values
            initial_rows = len(self.df)
            self.df = self.df.dropna(subset=['has_pest_ever'])
            print(f"Removed {initial_rows - len(self.df)} rows with missing target values")
            
            # Create binary target (ensure it's 0 or 1)
            self.df['pest_binary'] = (self.df['has_pest_ever'] > 0).astype(int)
            
            # Identify feature columns (exclude metadata and target columns)
            exclude_columns = ['nodeId', 'entity_id', 'has_pest_ever', 'pest_rate', 'pest_binary']
            feature_columns = [col for col in self.df.columns if col not in exclude_columns]
            
            # Separate features and target
            self.X = self.df[feature_columns].copy()
            self.y = self.df['pest_binary'].copy()
            self.feature_names = feature_columns
            
            print(f"Feature matrix shape: {self.X.shape}")
            print(f"Target distribution: {self.y.value_counts().to_dict()}")
            print(f"Positive class percentage: {(self.y.sum() / len(self.y) * 100):.2f}%")
            
            # Handle missing values in features
            print("Handling missing values...")
            missing_counts = self.X.isnull().sum()
            if missing_counts.sum() > 0:
                print(f"Features with missing values: {missing_counts[missing_counts > 0].to_dict()}")
                # Fill missing values with median for numerical features
                self.X = self.X.fillna(self.X.median())
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def exploratory_data_analysis(self):
        """Perform basic EDA on the features"""
        print("\n=== Exploratory Data Analysis ===")
        
        # Basic statistics
        print(f"Dataset shape: {self.X.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        
        # Feature types
        graphsage_features = [col for col in self.feature_names if 'graphsage_dim' in col]
        structural_features = [col for col in self.feature_names if col in ['degree', 'entity_degree']]
        other_features = [col for col in self.feature_names if col not in graphsage_features + structural_features]
        
        print(f"GraphSAGE embedding dimensions: {len(graphsage_features)}")
        print(f"Structural features: {len(structural_features)}")
        print(f"Other computed features: {len(other_features)}")
        
        # Class balance
        class_counts = self.y.value_counts()
        print(f"\nClass distribution:")
        print(f"No pest (0): {class_counts[0]} ({class_counts[0]/len(self.y)*100:.1f}%)")
        print(f"Has pest (1): {class_counts[1]} ({class_counts[1]/len(self.y)*100:.1f}%)")
        
        return {
            'graphsage_features': graphsage_features,
            'structural_features': structural_features,
            'other_features': other_features
        }
    
    def prepare_train_test_split(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print(f"\nSplitting data: {100-test_size*100:.0f}% train, {test_size*100:.0f}% test")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, 
            stratify=self.y  # Ensure balanced split
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Training set positive class: {self.y_train.sum()}/{len(self.y_train)} ({self.y_train.mean()*100:.1f}%)")
        print(f"Test set positive class: {self.y_test.sum()}/{len(self.y_test)} ({self.y_test.mean()*100:.1f}%)")
    
    def scale_features(self, scaler_type='standard'):
        """Scale features for algorithms that require it"""
        print(f"\nScaling features using {scaler_type} scaler...")
        
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'robust'")
        
        # Fit on training data and transform both sets
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        self.scalers[scaler_type] = scaler
        
        print("Feature scaling completed")
    
    def train_multiple_models(self):
        """Train multiple ML models for comparison"""
        print("\n=== Training Multiple Models ===")
        
        # Ensure features are scaled
        if not hasattr(self, 'X_train_scaled'):
            self.scale_features()
        
        # Define models to train
        models_config = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM (RBF)': SVC(random_state=42, probability=True)
        }
        
        results = {}
        
        for model_name, model in models_config.items():
            print(f"\nTraining {model_name}...")
            
            try:
                # Use scaled data for SVM and Logistic Regression
                if model_name in ['Logistic Regression', 'SVM (RBF)']:
                    X_train_use = self.X_train_scaled
                    X_test_use = self.X_test_scaled
                else:
                    X_train_use = self.X_train
                    X_test_use = self.X_test
                
                # Train model
                model.fit(X_train_use, self.y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_use)
                y_pred_proba = model.predict_proba(X_test_use)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                auc = roc_auc_score(self.y_test, y_pred_proba)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train_use, self.y_train, 
                                          cv=5, scoring='f1')
                
                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'cv_f1_mean': cv_scores.mean(),
                    'cv_f1_std': cv_scores.std(),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                print(f"  AUC: {auc:.4f}")
                print(f"  CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
                
            except Exception as e:
                print(f"  Error training {model_name}: {e}")
                continue
        
        self.models = results
        return results
    
    def hyperparameter_tuning(self, model_name='Random Forest'):
        """Perform hyperparameter tuning for the best performing model"""
        print(f"\n=== Hyperparameter Tuning for {model_name} ===")
        
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'subsample': [0.8, 0.9, 1.0]
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return None
        
        # Select appropriate model and data
        if model_name == 'Random Forest':
            base_model = RandomForestClassifier(random_state=42)
            X_train_use, X_test_use = self.X_train, self.X_test
        elif model_name == 'Gradient Boosting':
            base_model = GradientBoostingClassifier(random_state=42)
            X_train_use, X_test_use = self.X_train, self.X_test
        elif model_name == 'Logistic Regression':
            base_model = LogisticRegression(random_state=42, max_iter=1000)
            X_train_use, X_test_use = self.X_train_scaled, self.X_test_scaled
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            base_model, 
            param_grids[model_name],
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        print("Running grid search...")
        grid_search.fit(X_train_use, self.y_train)
        
        # Best model results
        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
        
        # Evaluate best model on test set
        y_pred = best_model.predict(X_test_use)
        y_pred_proba = best_model.predict_proba(X_test_use)[:, 1]
        
        test_f1 = f1_score(self.y_test, y_pred)
        test_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"Test set F1 score: {test_f1:.4f}")
        print(f"Test set AUC: {test_auc:.4f}")
        
        # Store the best model
        self.models[f'{model_name} (Tuned)'] = {
            'model': best_model,
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': test_f1,
            'auc': test_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'best_params': grid_search.best_params_
        }
        
        return best_model
    
    def feature_importance_analysis(self, model_name='Random Forest'):
        """Analyze feature importance for tree-based models"""
        print(f"\n=== Feature Importance Analysis for {model_name} ===")
        
        if model_name not in self.models:
            print(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            return None
        
        model = self.models[model_name]['model']
        
        # Get feature importance (works for tree-based models)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create feature importance dataframe
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("Top 20 most important features:")
            print(feature_importance_df.head(20))
            
            # Categorize features
            feature_categories = self.exploratory_data_analysis()
            
            # Calculate importance by category
            category_importance = {}
            for category, features in feature_categories.items():
                cat_importance = feature_importance_df[
                    feature_importance_df['feature'].isin(features)
                ]['importance'].sum()
                category_importance[category] = cat_importance
            
            print(f"\nImportance by feature category:")
            for category, importance in sorted(category_importance.items(), 
                                             key=lambda x: x[1], reverse=True):
                print(f"  {category}: {importance:.4f}")
            
            return feature_importance_df
        
        else:
            print(f"Feature importance not available for {model_name}")
            return None
    
    def generate_model_report(self):
        """Generate a comprehensive model comparison report"""
        print("\n" + "="*60)
        print("           PEST PREDICTION MODEL COMPARISON REPORT")
        print("="*60)
        
        if not self.models:
            print("No models trained yet. Run train_multiple_models() first.")
            return
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, results in self.models.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1'],
                'AUC': results['auc'],
                'CV F1 Mean': results.get('cv_f1_mean', 'N/A')
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.round(4))
        
        # Find best model by F1 score
        best_model_name = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
        print(f"\nBest performing model (by F1-Score): {best_model_name}")
        
        # Detailed report for best model
        best_results = self.models[best_model_name]
        print(f"\nDetailed results for {best_model_name}:")
        print(f"  Accuracy:  {best_results['accuracy']:.4f}")
        print(f"  Precision: {best_results['precision']:.4f}")
        print(f"  Recall:    {best_results['recall']:.4f}")
        print(f"  F1-Score:  {best_results['f1']:.4f}")
        print(f"  AUC:       {best_results['auc']:.4f}")
        
        # Confusion matrix for best model
        print(f"\nConfusion Matrix for {best_model_name}:")
        cm = confusion_matrix(self.y_test, best_results['predictions'])
        print(cm)
        
        # Classification report
        print(f"\nClassification Report for {best_model_name}:")
        print(classification_report(self.y_test, best_results['predictions']))
        
        return comparison_df, best_model_name
    
    def run_full_pipeline(self):
        """Run the complete ML pipeline"""
        print("Starting Pest Prediction ML Pipeline...")
        print("="*50)
        
        # Step 1: Load and prepare data
        if not self.load_and_prepare_data():
            print("Failed to load data. Exiting.")
            return None
        
        # Step 2: EDA
        self.exploratory_data_analysis()
        
        # Step 3: Train-test split
        self.prepare_train_test_split()
        
        # Step 4: Scale features
        self.scale_features()
        
        # Step 5: Train multiple models
        self.train_multiple_models()
        
        # Step 6: Generate report
        comparison_df, best_model_name = self.generate_model_report()
        
        # Step 7: Feature importance analysis
        if 'Random Forest' in self.models:
            self.feature_importance_analysis('Random Forest')
        
        # Step 8: Hyperparameter tuning for best model
        if best_model_name in ['Random Forest', 'Gradient Boosting', 'Logistic Regression']:
            print(f"\nPerforming hyperparameter tuning for {best_model_name}...")
            self.hyperparameter_tuning(best_model_name)
            
            # Update report with tuned model
            print("\nUpdated results after hyperparameter tuning:")
            self.generate_model_report()
        
        print("\n" + "="*50)
        print("Pipeline completed successfully!")
        
        return self
    
    def predict_new_entities(self, new_data_file):
        """Use trained model to predict on new entities"""
        if not self.models:
            print("No trained models available. Train models first.")
            return None
        
        # Get best model
        comparison_data = []
        for model_name, results in self.models.items():
            comparison_data.append({
                'model_name': model_name,
                'f1': results['f1']
            })
        
        best_model_info = max(comparison_data, key=lambda x: x['f1'])
        best_model_name = best_model_info['model_name']
        best_model = self.models[best_model_name]['model']
        
        print(f"Using {best_model_name} for predictions...")
        
        # Load new data
        new_df = pd.read_csv(new_data_file)
        new_X = new_df[self.feature_names]
        
        # Scale if necessary
        if best_model_name in ['Logistic Regression', 'SVM (RBF)']:
            new_X_scaled = self.scalers['standard'].transform(new_X)
            predictions = best_model.predict(new_X_scaled)
            probabilities = best_model.predict_proba(new_X_scaled)[:, 1]
        else:
            predictions = best_model.predict(new_X)
            probabilities = best_model.predict_proba(new_X)[:, 1]
        
        # Create results dataframe
        results_df = new_df[['entity_id']].copy()
        results_df['pest_prediction'] = predictions
        results_df['pest_probability'] = probabilities
        
        return results_df

# Example usage
if __name__ == "__main__":
    # Initialize and run the pipeline
    pipeline = PestPredictionMLPipeline()
    trained_pipeline = pipeline.run_full_pipeline()
    
    # Example of making predictions on new data
    # new_predictions = pipeline.predict_new_entities('new_entities.csv')