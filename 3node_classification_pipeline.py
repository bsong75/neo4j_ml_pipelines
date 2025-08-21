import pandas as pd
from typing import Dict, List, Tuple, Optional


class NodeClassificationPestAnalyzer:
    """
    Node Classification pipeline for pest analysis.
    Directly classifies entities as pest/no-pest based on graph features.
    """
    
    def __init__(self, gds_instance, graph_name: str):
        """
        Initialize the node classification analyzer.
        
        Args:
            gds_instance: Neo4j Graph Data Science instance
            graph_name: Name of the graph projection to use
        """
        self.gds = gds_instance
        self.graph_name = graph_name
        self.pipeline_name = "pest_node_classification_pipeline"
        self.model_name = "pest_node_classification_model"
    
    def prepare_node_properties(self):
        """Add pest target property to Entity nodes for training"""
        
        print("Preparing node properties for classification...")
        
        # Add has_pest property to Entity nodes based on their inspection results
        self.gds.run_cypher(f"""
            CALL gds.graph.nodeProperty.write('{self.graph_name}', 'temp_pest_property', 0)
            YIELD writeMillis, graphName, nodeProperty, nodePropertiesWritten
            RETURN writeMillis, graphName, nodeProperty, nodePropertiesWritten
        """)
        
        # Set has_pest property based on inspection results
        result = self.gds.run_cypher("""
            MATCH (e:Entity)-[:HAS_INSPECTION_RESULT]->(t:TargetProxy)
            WITH e, max(t.value) as has_pest_value
            SET e.has_pest = has_pest_value
            RETURN count(e) as entities_updated
        """)
        
        print(f"Updated pest property for {result.iloc[0]['entities_updated']} entities")
        
        # Check the distribution
        pest_distribution = self.gds.run_cypher("""
            MATCH (e:Entity)
            RETURN e.has_pest as pest_value, count(e) as entity_count
            ORDER BY pest_value
        """)
        
        print("Pest distribution:")
        for _, row in pest_distribution.iterrows():
            print(f"  has_pest = {row['pest_value']}: {row['entity_count']} entities")
        
        return pest_distribution
    
    def create_enhanced_graph_projection(self):
        """Create enhanced graph projection with node properties"""
        
        enhanced_graph_name = f"{self.graph_name}_enhanced"
        
        try:
            self.gds.graph.drop(enhanced_graph_name)
            print(f"Dropped existing enhanced graph: {enhanced_graph_name}")
        except:
            print("No existing enhanced graph to drop")
        
        # Create enhanced projection with has_pest property
        enhanced_graph, _ = self.gds.graph.project(
            enhanced_graph_name,
            ["Country", "Month", "Entity", "TargetProxy", "CountryMonth"],
            {
                'SHIPPED_IN': {'orientation': 'UNDIRECTED'},
                'IS_FROM': {'orientation': 'UNDIRECTED'}, 
                'HAS_WEATHER': {'orientation': 'UNDIRECTED'},
                'HAS_INSPECTION_RESULT': {'orientation': 'NATURAL'}
            },
            nodeProperties=['has_pest']
        )
        
        self.enhanced_graph_name = enhanced_graph_name
        print(f"Created enhanced graph projection: {enhanced_graph_name}")
        return enhanced_graph
    
    def add_centrality_features(self):
        """Add centrality measures as node features"""
        
        print("Computing centrality features...")
        
        # List of centrality algorithms to compute
        centrality_algorithms = [
            ('degree', 'gds.degree'),
            ('pageRank', 'gds.pageRank'), 
            ('betweenness', 'gds.betweenness'),
            ('closeness', 'gds.closeness'),
            ('eigenvector', 'gds.eigenvector')
        ]
        
        for feature_name, algorithm in centrality_algorithms:
            try:
                print(f"  Computing {feature_name}...")
                if feature_name == 'degree':
                    self.gds.run_cypher(f"""
                        CALL {algorithm}.mutate('{self.enhanced_graph_name}', {{
                            mutateProperty: '{feature_name}'
                        }})
                        YIELD centralityDistribution
                        RETURN centralityDistribution
                    """)
                else:
                    self.gds.run_cypher(f"""
                        CALL {algorithm}.mutate('{self.enhanced_graph_name}', {{
                            mutateProperty: '{feature_name}'
                        }})
                        YIELD centralityDistribution  
                        RETURN centralityDistribution
                    """)
                print(f"    ✓ {feature_name} computed successfully")
            except Exception as e:
                print(f"    ✗ Failed to compute {feature_name}: {str(e)}")
        
        print("Centrality features computation completed")
    
    def create_node_classification_pipeline(self):
        """Create and configure node classification pipeline"""
        
        # Drop existing pipeline
        try:
            self.gds.run_cypher(f"CALL gds.pipeline.drop('{self.pipeline_name}')")
            print(f"Dropped existing pipeline: {self.pipeline_name}")
        except:
            print("No existing pipeline to drop")
        
        # Create new pipeline
        self.gds.run_cypher(f"""
            CALL gds.nodeClassification.create('{self.pipeline_name}')
            YIELD name, nodePropertySteps, featureSteps, parameterSpace
            RETURN name, nodePropertySteps, featureSteps, parameterSpace
        """)
        
        print(f"Created node classification pipeline: {self.pipeline_name}")
        
        # Configure train/test split
        self.gds.run_cypher(f"""
            CALL gds.nodeClassification.configureSplit('{self.pipeline_name}', {{
                testFraction: 0.2,
                validationFolds: 3
            }})
            YIELD splitConfig
            RETURN splitConfig
        """)
        
        # Add logistic regression model
        self.gds.run_cypher(f"""
            CALL gds.nodeClassification.addLogisticRegression('{self.pipeline_name}', {{
                penalty: 1.0,
                maxIterations: 1000
            }})
            YIELD parameterSpace
            RETURN parameterSpace
        """)
        
        print("Node classification pipeline configured successfully")
    
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
        
        # Print training results
        model_info = train_result['modelInfo'].iloc[0]
        model_stats = train_result['modelSelectionStats'].iloc[0]
        
        print(f"Model Info: {model_info}")
        print(f"Model Selection Stats: {model_stats}")
        
        return train_result
    
    def predict_pest_classifications(self) -> pd.DataFrame:
        """Make pest predictions using the trained model"""
        
        print("Making pest predictions...")
        
        # Stream predictions for all Entity nodes
        predictions = self.gds.run_cypher(f"""
            CALL gds.nodeClassification.predict.stream('{self.enhanced_graph_name}', {{
                modelName: '{self.model_name}',
                includePredictedProbabilities: true
            }})
            YIELD nodeId, predictedClass, predictedProbabilities
            WITH nodeId, predictedClass, predictedProbabilities
            MATCH (e:Entity) WHERE id(e) = nodeId
            RETURN e.id as entity_id,
                   predictedClass as predicted_pest_class,
                   predictedProbabilities[0] as prob_no_pest,
                   predictedProbabilities[1] as prob_pest
            ORDER BY predictedProbabilities[1] DESC
        """)
        
        # Get actual pest values for comparison
        actual_pest_data = self.gds.run_cypher("""
            MATCH (e:Entity)
            RETURN e.id as entity_id, e.has_pest as actual_pest_class
        """)
        
        # Merge predictions with actual data
        final_predictions = predictions.merge(actual_pest_data, on='entity_id', how='left')
        
        # Add risk categories based on pest probability
        final_predictions['pest_risk_category'] = final_predictions['prob_pest'].apply(
            lambda x: 'High Risk' if x >= 0.7 else 
                     'Medium Risk' if x >= 0.4 else 
                     'Low Risk'
        )
        
        # Calculate summary statistics
        total_entities = len(final_predictions)
        predicted_pests = (final_predictions['predicted_pest_class'] == 1).sum()
        actual_pests = (final_predictions['actual_pest_class'] == 1).sum()
        high_risk_entities = len(final_predictions[final_predictions['pest_risk_category'] == 'High Risk'])
        avg_pest_prob = final_predictions['prob_pest'].mean()
        
        print(f"\nPrediction Results Summary:")
        print(f"  Total entities: {total_entities}")
        print(f"  Predicted pest entities: {predicted_pests}")
        print(f"  Actual pest entities: {actual_pests}")
        print(f"  High risk entities (prob ≥ 0.7): {high_risk_entities}")
        print(f"  Average pest probability: {avg_pest_prob:.3f}")
        print(f"  Probability range: {final_predictions['prob_pest'].min():.3f} - {final_predictions['prob_pest'].max():.3f}")
        
        return final_predictions
    
    def run_full_node_classification_analysis(self) -> pd.DataFrame:
        """Run complete node classification analysis"""
        
        print("Starting Node Classification Analysis for Pest Prediction...")
        
        try:
            # Step 1: Prepare node properties
            self.prepare_node_properties()
            
            # Step 2: Create enhanced graph projection
            self.create_enhanced_graph_projection()
            
            # Step 3: Add centrality features
            self.add_centrality_features()
            
            # Step 4: Create classification pipeline
            self.create_node_classification_pipeline()
            
            # Step 5: Train model
            self.train_classification_model()
            
            # Step 6: Make predictions
            predictions = self.predict_pest_classifications()
            
            # Step 7: Save results
            output_file = 'node_classification_pest_predictions.csv'
            predictions.to_csv(output_file, index=False)
            print(f"\nNode classification predictions saved to: {output_file}")
            
            print("Node Classification Analysis completed successfully!")
            return predictions
            
        except Exception as e:
            print(f"Node classification analysis failed: {str(e)}")
            raise
    
    def cleanup_resources(self):
        """Clean up pipeline and model resources"""
        try:
            self.gds.run_cypher(f"CALL gds.model.drop('{self.model_name}')")
            print(f"Dropped model: {self.model_name}")
        except:
            pass
        
        try:
            self.gds.run_cypher(f"CALL gds.pipeline.drop('{self.pipeline_name}')")
            print(f"Dropped pipeline: {self.pipeline_name}")
        except:
            pass
        
        try:
            self.gds.graph.drop(self.enhanced_graph_name)
            print(f"Dropped enhanced graph: {self.enhanced_graph_name}")
        except:
            pass

