import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def get_default_embedding(train_df, embedding_cols, category=None, category_col=None):
    """
    Get default embedding (mean of training embeddings)
    
    Args:
        train_df: Training DataFrame with valid embeddings
        embedding_cols: List of embedding column names
        category: Specific category to get mean for (optional)
        category_col: Column name for categories (optional)
    
    Returns:
        numpy array: Default embedding vector
    """
    # Filter out rows with NaN embeddings
    valid_df = train_df.dropna(subset=embedding_cols)
    
    # Category-specific mean if requested
    if category and category_col and category_col in valid_df.columns:
        category_df = valid_df[valid_df[category_col] == category]
        if len(category_df) > 0:
            return category_df[embedding_cols].mean().values
    
    # Overall mean as fallback
    return valid_df[embedding_cols].mean().values


def train_cold_start_model(train_df, embedding_cols, feature_cols):
    """
    Train a cold start model to predict embeddings from features
    
    Args:
        train_df: Training DataFrame with embeddings and features
        embedding_cols: List of embedding column names
        feature_cols: List of feature column names to use as predictors
    
    Returns:
        tuple: (models_dict, scaler, feature_names)
    """
    # Filter out rows with NaN embeddings
    valid_df = train_df.dropna(subset=embedding_cols).copy()
    
    # Prepare features
    X = valid_df[feature_cols].copy()
    y = valid_df[embedding_cols].values
    
    # Handle categorical variables
    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_names = X_encoded.columns.tolist()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # Train models for each embedding dimension
    models = {}
    for i, emb_col in enumerate(embedding_cols):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y[:, i])
        models[emb_col] = model
    
    return models, scaler, feature_names


def predict_cold_start_embedding(entity_data, models, scaler, feature_names, feature_cols):
    """
    Predict embedding for new entity using cold start model
    
    Args:
        entity_data: Dict with entity features
        models: Dict of trained models (from train_cold_start_model)
        scaler: Fitted StandardScaler
        feature_names: List of feature names from training
        feature_cols: Original feature columns used in training
    
    Returns:
        numpy array: Predicted embedding vector
    """
    # Convert to DataFrame
    entity_df = pd.DataFrame([entity_data])
    
    # Select only the features used in training
    entity_features = entity_df[feature_cols]
    
    # Apply same encoding as training
    entity_encoded = pd.get_dummies(entity_features, drop_first=True)
    
    # Ensure all training features are present
    for col in feature_names:
        if col not in entity_encoded.columns:
            entity_encoded[col] = 0
    
    # Reorder columns to match training
    entity_encoded = entity_encoded[feature_names]
    
    # Scale features
    entity_scaled = scaler.transform(entity_encoded)
    
    # Predict each embedding dimension
    embedding = []
    for model in models.values():
        pred = model.predict(entity_scaled)[0]
        embedding.append(pred)
    
    return np.array(embedding)


# Example usage
def example_usage():
    """Example of how to use the functions"""
    
    # Create sample training data
    np.random.seed(42)
    n_entities = 1000
    
    train_data = {
        'enty_id': [f'entity_{i}' for i in range(n_entities)],
        'category': np.random.choice(['A', 'B', 'C'], n_entities),
        'feature_1': np.random.normal(0, 1, n_entities),
        'feature_2': np.random.uniform(0, 100, n_entities),
        'feature_3': np.random.choice(['X', 'Y', 'Z'], n_entities),
    }
    
    # Add 64 embedding columns (your FastRP embeddings)
    embedding_cols = []
    for i in range(64):
        col_name = f'fastrp_{i}'
        train_data[col_name] = np.random.normal(0, 1, n_entities)
        embedding_cols.append(col_name)
    
    train_df = pd.DataFrame(train_data)
    feature_cols = ['feature_1', 'feature_2', 'feature_3']
    
    print("=== Training Cold Start Model ===")
    # Train cold start model
    models, scaler, feature_names = train_cold_start_model(
        train_df, embedding_cols, feature_cols
    )
    print(f"Trained models for {len(embedding_cols)} embedding dimensions")
    
    # New entity data (from holdout set)
    new_entities = [
        {'category': 'A', 'feature_1': 0.5, 'feature_2': 75, 'feature_3': 'X'},
        {'category': 'B', 'feature_1': -1.2, 'feature_2': 25, 'feature_3': 'Y'},
        {'category': 'D', 'feature_1': 2.0, 'feature_2': 90, 'feature_3': 'Z'},  # New category
    ]
    
    print("\n=== Getting Embeddings for New Entities ===")
    for i, entity in enumerate(new_entities):
        print(f"\nEntity {i+1}: {entity}")
        
        # Method 1: Default embedding
        default_emb = get_default_embedding(
            train_df, embedding_cols, 
            category=entity['category'], 
            category_col='category'
        )
        print(f"Default embedding (first 5): {default_emb[:5]}")
        
        # Method 2: Cold start prediction
        cold_start_emb = predict_cold_start_embedding(
            entity, models, scaler, feature_names, feature_cols
        )
        print(f"Cold start embedding (first 5): {cold_start_emb[:5]}")


if __name__ == "__main__":
    example_usage()