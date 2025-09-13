import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def fill_missing_embeddings(df, embedding_cols, feature_cols):
    """
    Args:
        df: DataFrame with both entities (some with embeddings, some without)
        embedding_cols: List of embedding column names (e.g., ['fastrp_0', 'fastrp_1', ...])
        feature_cols: List of feature column names to use as predictors
    
    Returns:
        DataFrame: Same dataframe with missing embeddings filled
    """
    df_result = df.copy()
    
    # Split data: entities with embeddings vs without
    has_embeddings = ~df[embedding_cols].isnull().any(axis=1)
    train_df = df[has_embeddings].copy()
    predict_df = df[~has_embeddings].copy()
    
    print(f"Entities with embeddings: {len(train_df)}")
    print(f"Entities without embeddings: {len(predict_df)}")
    
    # Prepare training data
    X = train_df[feature_cols].copy()
    y = train_df[embedding_cols].values
    
    # Handle categorical variables
    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_names = X_encoded.columns.tolist()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # Train models for each embedding dimension
    print("Training cold start models...")
    models = {}
    for i, emb_col in enumerate(embedding_cols):
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_scaled, y[:, i])
        models[emb_col] = model
    
    # Prepare prediction data
    X_pred = predict_df[feature_cols].copy()
    X_pred_encoded = pd.get_dummies(X_pred, drop_first=True)
    
    # Ensure all training features are present
    for col in feature_names:
        if col not in X_pred_encoded.columns:
            X_pred_encoded[col] = 0
    
    # Reorder columns to match training
    X_pred_encoded = X_pred_encoded[feature_names]
    X_pred_scaled = scaler.transform(X_pred_encoded)
    
    # Predict embeddings
    print("Predicting embeddings for new entities...")
    predicted_embeddings = np.zeros((len(predict_df), len(embedding_cols)))
    
    for i, emb_col in enumerate(embedding_cols):
        predicted_embeddings[:, i] = models[emb_col].predict(X_pred_scaled)
    
    # Fill in the missing embeddings
    for i, emb_col in enumerate(embedding_cols):
        df_result.loc[~has_embeddings, emb_col] = predicted_embeddings[:, i]
    
    print(f"Successfully filled embeddings for {len(predict_df)} entities")
    return df_result


# Example usage
def example_usage():
    """Example of how to use the function"""
    
    # Create sample data with mixed entities
    np.random.seed(42)
    
    # Entities with embeddings (training data)
    n_with_emb = 800
    with_emb_data = {
        'enty_id': [f'entity_{i}' for i in range(n_with_emb)],
        'feature_1': np.random.normal(0, 1, n_with_emb),
        'feature_2': np.random.uniform(0, 100, n_with_emb),
        'feature_3': np.random.choice(['X', 'Y', 'Z'], n_with_emb),
    }
    
    # Add FastRP embeddings for these entities
    embedding_cols = [f'fastrp_{i}' for i in range(64)]
    for col in embedding_cols:
        with_emb_data[col] = np.random.normal(0, 1, n_with_emb)
    
    # Entities without embeddings (holdout/new entities)
    n_without_emb = 200
    without_emb_data = {
        'enty_id': [f'new_entity_{i}' for i in range(n_without_emb)],
        'feature_1': np.random.normal(0, 1, n_without_emb),
        'feature_2': np.random.uniform(0, 100, n_without_emb),
        'feature_3': np.random.choice(['X', 'Y', 'Z'], n_without_emb),
    }
    
    # Add NaN embeddings for these entities
    for col in embedding_cols:
        without_emb_data[col] = [np.nan] * n_without_emb
    
    # Combine both types of entities
    combined_df = pd.concat([
        pd.DataFrame(with_emb_data),
        pd.DataFrame(without_emb_data)
    ], ignore_index=True)
    
    print("=== Original Data ===")
    print(f"Total entities: {len(combined_df)}")
    print(f"Missing embeddings: {combined_df[embedding_cols].isnull().any(axis=1).sum()}")
    
    # Fill missing embeddings
    feature_cols = ['feature_1', 'feature_2', 'feature_3']
    result_df = fill_missing_embeddings(combined_df, embedding_cols, feature_cols)
    
    print("\n=== After Filling Embeddings ===")
    print(f"Missing embeddings: {result_df[embedding_cols].isnull().any(axis=1).sum()}")
    print(f"Sample predicted embedding: {result_df.loc[800, embedding_cols[:5]].values}")
    
    return result_df


if __name__ == "__main__":
    result = example_usage()