import pandas as pd
from typing import Dict, List, Tuple, Optional


class LinkPredictionPipelinePestAnalyzer:
    """
    Advanced link prediction using Neo4j GDS Pipeline for pest analysis.
    Uses proper train/test splits and multiple algorithms in a pipeline.
    """
    
    def __init__(self, gds_instance, graph_name: str):
        """
        Initialize the pipeline analyzer.
        
        Args:
            gds_instance: Neo4j Graph Data Science instance
            graph_name: Name of the graph projection to use
        """
        self.gds = gds_instance
        self.graph_name = graph_name
        self.pipeline_name = "pest_link_prediction_pipeline"
    
    def prepare_pipeline_data(self) -> pd.DataFrame:
        """
        Prepare entity-pest relationships for pipeline training.
        
        Returns:
            DataFrame with training relationships
        """
        # Get all existing entity-pest relationships
        relationships = self.gds.run_cypher("""
            MATCH (e:Entity)-[:HAS_INSPECTION_RESULT]->(t:TargetProxy)
            RETURN id(e) as sourceNodeId, 
                   id(t) as targetNodeId,
                   e.id as entity_id,
                   t.value as pest_value,
                   t.label as pest_label
        """)
        
        print(f"Found {len(relationships)} entity-pest relationships for training")
        return relationships
    
    def create_link_prediction_pipeline(self):
        """Create and configure the link prediction pipeline"""
        
        # Drop existing pipeline if it exists
        try:
            self.gds.run_cypher(f"CALL gds.pipeline.drop('{self.pipeline_name}')")
            print(f"Dropped existing pipeline: {self.pipeline_name}")
        except:
            print(f"No existing pipeline to drop")
        
        # Create new pipeline
        self.gds.run_cypher(f"""
            CALL gds.linkPrediction.create('{self.pipeline_name}')
        """)
        print(f"Created link prediction pipeline: {self.pipeline_name}")
        
        # Add node property steps (optional - adds centrality features)
        self.gds.run_cypher(f"""
            CALL gds.linkPrediction.addNodeProperty('{self.pipeline_name}', 'degree', {{
                mutateProperty: 'degree'
            }})
        """)
        
        # Add link features (the heuristic algorithms)
        feature_configs = [
            {
                'name': 'adamicAdar',
                'config': {}
            },
            {
                'name': 'jaccard', 
                'config': {}
            },
            {
                'name': 'commonNeighbors',
                'config': {}
            },
            {
                'name': 'preferentialAttachment',
                'config': {}
            },
            {
                'name': 'cosine',
                'config': {}
            }
        ]
        
        for feature in feature_configs:
            self.gds.run_cypher(f"""
                CALL gds.linkPrediction.addFeature('{self.pipeline_name}', '{feature['name']}', {{
                    nodeProperties: ['degree']
                }})
            """)
            print(f"Added feature: {feature['name']}")
        
        # Configure the machine learning model
        self.gds.run_cypher(f"""
            CALL gds.linkPrediction.configureSplit('{self.pipeline_name}', {{
                testFraction: 0.2,
                trainFraction: 0.6,
                validationFolds: 3
            }})
        """)
        
        # Add logistic regression model
        self.gds.run_cypher(f"""
            CALL gds.linkPrediction.addLogisticRegression('{self.pipeline_name}', {{
                penalty: 1.0,
                maxIterations: 1000
            }})
        """)
        
        print("Pipeline configuration completed")
    
    def train_pipeline(self, relationship_data: pd.DataFrame):
        """
        Train the link prediction pipeline
        
        Args:
            relationship_data: DataFrame with entity-pest relationships
        """
        print("Training link prediction pipeline...")
        
        # Convert relationship data to format needed for pipeline
        train_relationships = []
        for _, row in relationship_data.iterrows():
            train_relationships.append({
                'sourceNodeId': int(row['sourceNodeId']),
                'targetNodeId': int(row['targetNodeId'])
            })
        
        # Train the pipeline
        train_result = self.gds.run_cypher(f"""
            CALL gds.linkPrediction.train('{self.graph_name}', {{
                pipeline: '{self.pipeline_name}',
                modelName: 'pest_link_prediction_model',
                sourceNodeLabel: 'Entity',
                targetNodeLabel: 'TargetProxy',
                targetRelationshipType: 'HAS_INSPECTION_RESULT',
                randomSeed: 42
            }})
            YIELD modelInfo, modelSelectionStats
            RETURN modelInfo, modelSelectionStats
        """)
        
        print("Pipeline training completed")
        print("Model Info:", train_result['modelInfo'].iloc[0])
        print("Model Selection Stats:", train_result['modelSelectionStats'].iloc[0])
        
        return train_result
    
    def predict_with_pipeline(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Make predictions using the trained pipeline
        
        Args:
            threshold: Probability threshold for positive predictions
            
        Returns:
            DataFrame with predictions
        """
        print("Making predictions with trained pipeline...")
        
        # Stream predictions for all possible entity-pest pairs
        predictions = self.gds.run_cypher(f"""
            CALL gds.linkPrediction.predict.stream('{self.graph_name}', {{
                modelName: 'pest_link_prediction_model',
                sourceNodeLabel: 'Entity',
                targetNodeLabel: 'TargetProxy',
                topN: 10000
            }})
            YIELD node1, node2, probability
            WITH node1, node2, probability
            MATCH (e:Entity) WHERE id(e) = node1
            MATCH (t:TargetProxy) WHERE id(t) = node2
            RETURN e.id as entity_id,
                   id(e) as entity_node_id,
                   t.value as pest_value,
                   t.label as pest_label,
                   probability as link_prediction_probability
            ORDER BY probability DESC
        """)
        
        # Focus on pest predictions (pest_value = 1)
        pest_predictions = predictions[predictions['pest_value'] == 1].copy()
        
        # Create binary predictions based on threshold
        pest_predictions['pipeline_pest_binary'] = (pest_predictions['link_prediction_probability'] >= threshold).astype(int)
        
        # Get actual pest history for comparison
        actual_pest_data = self.gds.run_cypher("""
            MATCH (e:Entity)-[:HAS_INSPECTION_RESULT]->(t:TargetProxy)
            RETURN e.id as entity_id, max(t.value) as has_actual_pest
        """)
        
        # Merge with actual data
        final_predictions = pest_predictions.merge(actual_pest_data, on='entity_id', how='left')
        final_predictions['has_actual_pest'] = final_predictions['has_actual_pest'].fillna(0)
        
        # Rename columns for consistency
        final_predictions = final_predictions.rename(columns={
            'link_prediction_probability': 'pipeline_pest_score'
        })
        
        # Select final columns
        result_columns = [
            'entity_id',
            'pipeline_pest_score',
            'pipeline_pest_binary', 
            'has_actual_pest'
        ]
        
        final_predictions = final_predictions[result_columns].drop_duplicates(subset=['entity_id'])
        
        print(f"Generated predictions for {len(final_predictions)} entities")
        
        # Calculate summary statistics
        total_entities = len(final_predictions)
        predicted_pests = final_predictions['pipeline_pest_binary'].sum()
        actual_pests = final_predictions['has_actual_pest'].sum()
        avg_score = final_predictions['pipeline_pest_score'].mean()
        
        print(f"Results Summary:")
        print(f"  Total entities: {total_entities}")
        print(f"  Predicted pests (threshold={threshold}): {predicted_pests}")
        print(f"  Actual pests in data: {actual_pests}")
        print(f"  Average prediction score: {avg_score:.3f}")
        print(f"  Score range: {final_predictions['pipeline_pest_score'].min():.3f} - {final_predictions['pipeline_pest_score'].max():.3f}")
        
        return final_predictions
    
    def run_full_pipeline_analysis(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Run the complete link prediction pipeline analysis
        
        Args:
            threshold: Probability threshold for predictions
            
        Returns:
            DataFrame with pipeline predictions
        """
        print("Starting Link Prediction Pipeline Analysis...")
        
        try:
            # Prepare data
            relationship_data = self.prepare_pipeline_data()
            
            # Create and configure pipeline
            self.create_link_prediction_pipeline()
            
            # Train pipeline
            train_result = self.train_pipeline(relationship_data)
            
            # Make predictions
            predictions = self.predict_with_pipeline(threshold=threshold)
            
            # Save results
            output_file = 'pipeline_pest_predictions.csv'
            predictions.to_csv(output_file, index=False)
            print(f"Pipeline predictions saved to: {output_file}")
            
            print("Link Prediction Pipeline Analysis completed successfully!")
            return predictions
            
        except Exception as e:
            print(f"Pipeline analysis failed: {str(e)}")
            raise
    
    def cleanup_pipeline(self):
        """Clean up pipeline and model resources"""
        try:
            self.gds.run_cypher(f"CALL gds.model.drop('pest_link_prediction_model')")
            print("Dropped model: pest_link_prediction_model")
        except:
            pass
        
        try:
            self.gds.run_cypher(f"CALL gds.pipeline.drop('{self.pipeline_name}')")
            print(f"Dropped pipeline: {self.pipeline_name}")
        except:
            pass


# Integration function for your main PestDataAnalyzer class
def run_pipeline_pest_prediction(gds_instance, graph_name, threshold=0.5):
    """
    Convenience function to run pipeline analysis
    
    Args:
        gds_instance: Your GDS connection
        graph_name: Your graph projection name
        threshold: Prediction threshold
        
    Returns:
        DataFrame with pipeline predictions
    """
    
    pipeline_analyzer = LinkPredictionPipelinePestAnalyzer(gds_instance, graph_name)
    
    try:
        predictions = pipeline_analyzer.run_full_pipeline_analysis(threshold=threshold)
        return predictions
    finally:
        # Always cleanup
        pipeline_analyzer.cleanup_pipeline()


# Example usage
def example_pipeline_usage():
    """Example of how to use the pipeline analyzer"""
    
    # This would be called from your main PestDataAnalyzer
    # after you've created your graph projection
    
    # Example:
    # analyzer = PestDataAnalyzer(...)
    # analyzer.connect()
    # analyzer.load_data() 
    # analyzer.create_nodes()
    # analyzer.create_inspections_and_relationships()
    # analyzer.create_projection()
    
    # Then run pipeline analysis:
    # predictions = run_pipeline_pest_prediction(
    #     gds_instance=analyzer.gds,
    #     graph_name=analyzer.graph.name(),
    #     threshold=0.3  # Lower threshold if scores are low
    # )
    
    print("Pipeline analysis example - see comments for integration steps")


if __name__ == "__main__":
    example_pipeline_usage()