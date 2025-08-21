import json
import pandas as pd
from pprint import pprint

def display_model_results(train_result):
    """Display model training results in a readable format"""
    
    model_info = train_result['modelInfo'].iloc[0]
    model_stats = train_result['modelSelectionStats'].iloc[0]
    
    print("\n" + "="*60)
    print("MODEL TRAINING RESULTS")
    print("="*60)
    
    # Display Model Info
    print("\n📊 MODEL INFORMATION:")
    print("-" * 30)
    
    if isinstance(model_info, dict):
        for key, value in model_info.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
    else:
        print("Raw model info:")
        pprint(model_info)
    
    print("\n🎯 MODEL SELECTION STATISTICS:")
    print("-" * 35)
    
    if isinstance(model_stats, dict):
        # Extract key metrics if available
        if 'bestParameters' in model_stats:
            print("Best Parameters:")
            for param, value in model_stats['bestParameters'].items():
                print(f"  {param}: {value}")
        
        if 'bestTrial' in model_stats:
            print(f"\nBest Trial: {model_stats['bestTrial']}")
            
        if 'trainMetrics' in model_stats:
            print("\nTraining Metrics:")
            for metric, value in model_stats['trainMetrics'].items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        
        if 'validationMetrics' in model_stats:
            print("\nValidation Metrics:")
            for metric, value in model_stats['validationMetrics'].items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        
        if 'testMetrics' in model_stats:
            print("\nTest Metrics:")
            for metric, value in model_stats['testMetrics'].items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        
        # Display any other keys
        displayed_keys = {'bestParameters', 'bestTrial', 'trainMetrics', 'validationMetrics', 'testMetrics'}
        other_keys = set(model_stats.keys()) - displayed_keys
        
        if other_keys:
            print("\nOther Statistics:")
            for key in other_keys:
                value = model_stats[key]
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value}")
                elif isinstance(value, dict) and len(value) < 5:  # Small dicts
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {type(value).__name__} (complex object)")
    
    else:
        print("Raw model selection stats:")
        pprint(model_stats)
    
    print("\n" + "="*60)

# Alternative: JSON formatted display
def display_model_results_json(train_result):
    """Display model results as formatted JSON"""
    
    model_info = train_result['modelInfo'].iloc[0]
    model_stats = train_result['modelSelectionStats'].iloc[0]
    
    print("\n📊 MODEL INFO (JSON):")
    print("-" * 25)
    try:
        print(json.dumps(model_info, indent=2, default=str))
    except:
        pprint(model_info)
    
    print("\n🎯 MODEL SELECTION STATS (JSON):")
    print("-" * 35)
    try:
        print(json.dumps(model_stats, indent=2, default=str))
    except:
        pprint(model_stats)

# Updated train_classification_model method
def train_classification_model(self):
    """Train the node classification model"""
    
    print("Training node classification model...")
    
    # Train the model
    train_result = self.gds.run_cypher(f"""
        CALL gds.nodeClassification.train('{self.enhanced_graph_name}', {{
            pipeline: '{self.pipeline_name}',
            modelName: '{self.model_name}',
            nodeLabels: ['Entity'],
            targetProperty: 'has_pest',
            randomSeed: 42
        }})
        YIELD modelInfo, modelSelectionStats
        RETURN modelInfo, modelSelectionStats
    """)
    
    print("Model training completed!")
    
    # Display results in readable format
    display_model_results(train_result)
    
    # also save to file for later analysis
    try:
        model_info = train_result['modelInfo'].iloc[0]
        model_stats = train_result['modelSelectionStats'].iloc[0]
        
        results_summary = {
            'model_info': model_info,
            'model_selection_stats': model_stats
        }
        
        with open('model_training_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print("\n💾 Detailed results saved to: model_training_results.json")
        
    except Exception as e:
        print(f"Note: Could not save detailed results to file: {e}")
    
    return train_result