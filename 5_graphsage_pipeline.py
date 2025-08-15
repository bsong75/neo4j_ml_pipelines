import pandas as pd
from typing import Dict, List, Tuple, Optional


class GraphSAGEPestAnalyzer:
    """
    GraphSAGE (Graph Neural Network) pipeline for pest analysis.
    Uses deep learning to learn node representations and predict pest likelihood.
    """
    
    def __init__(self, gds_instance, graph_name: str):
        """
        Initialize the GraphSAGE analyzer.
        
        Args:
            gds_instance: Neo4j Graph Data Science instance
            graph_name: Name of the graph projection to use
        """
        self.gds = gds_instance
        self.graph_name = graph_name
        self.pipeline_name = "pest_graphsage_pipeline"
        self.model_name = "pest_graphsage_model"
        self.embedding_model_name = "pest_graphsage_embeddings"
    
    def prepare_graphsage_features(self):
        """Prepare node features for GraphSAGE training"""
        
        print("Preparing GraphSAGE node features...")
        
        # Create target labels for supervised learning
        target_result = self.gds.run_cypher("""
            MATCH (e:Entity)-[:HAS_INSPECTION_RESULT]->(t:TargetProxy)
            WITH e, 
                 max(t.value) as has_pest_binary,
                 avg(toFloat(t.value)) as pest_rate,
                 count(t) as total_inspections
            SET e.has_pest = has_pest_binary,
                e.pest_score = pest_rate,
                e.inspection_count = total_inspections
            RETURN count(e) as entities_updated,
                   sum(has_pest_binary) as pest_entities,
                   avg(pest_rate) as avg_pest_rate
        """)
        
        stats = target_result.iloc[0]
        print(f"Updated features for {stats['entities_updated']} entities")
        print(f"  Entities with pests: {stats['pest_entities']}")
        print(f"  Average pest rate: {stats['avg_pest_rate']:.3f}")
        
        # Add basic node features that GraphSAGE can use
        print("Computing basic node features...")
        
        # Add degree as a simple feature
        self.gds.run_cypher(f"""
            CALL gds.degree.mutate('{self.graph_name}', {{
                mutateProperty: 'degree'
            }})
        """)
        
        # Add PageRank as another feature
        try:
            self.gds.run_cypher(f"""
                CALL gds.pageRank.mutate('{self.graph_name}', {{
                    mutateProperty: 'pageRank',
                    maxIterations: 20
                }})
            """)
        except Exception as e:
            print(f"PageRank computation failed: {str(e)}")
        
        print("Basic features computed")
        return target_result
    
    def create_graphsage_projection(self):
        """Create enhanced graph projection for GraphSAGE"""
        
        enhanced_graph_name = f"{self.graph_name}_graphsage"
        
        try:
            self.gds.graph.drop(enhanced_graph_name)
            print(f"Dropped existing GraphSAGE graph: {enhanced_graph_name}")
        except:
            print("No existing GraphSAGE graph to drop")
        
        # Create projection with node properties for GraphSAGE
        enhanced_graph, _ = self.gds.graph.project(
            enhanced_graph_name,
            ["Country", "Month", "Entity", "TargetProxy", "CountryMonth"],
            {
                'SHIPPED_IN': {'orientation': 'UNDIRECTED'},
                'IS_FROM': {'orientation': 'UNDIRECTED'}, 
                'HAS_WEATHER': {'orientation': 'UNDIRECTED'},
                'HAS_INSPECTION_RESULT': {'orientation': 'NATURAL'}
            },
            nodeProperties=['has_pest', 'pest_score', 'inspection_count', 'degree', 'pageRank']
        )
        
        self.enhanced_graph_name = enhanced_graph_name
        print(f"Created GraphSAGE graph projection: {enhanced_graph_name}")
        print(f"  Nodes: {enhanced_graph.node_count()}")
        print(f"  Relationships: {enhanced_graph.relationship_count()}")
        
        return enhanced_graph
    
    def create_graphsage_pipeline(self):
        """Create GraphSAGE pipeline for node classification"""
        
        # Drop existing pipeline
        try:
            self.gds.run_cypher(f"CALL gds.pipeline.drop('{self.pipeline_name}')")
            print(f"Dropped existing pipeline: {self.pipeline_name}")
        except:
            print("No existing pipeline to drop")
        
        # Create new GraphSAGE pipeline
        self.gds.run_cypher(f"""
            CALL gds.nodeClassification.create('{self.pipeline_name}')
            YIELD name, nodePropertySteps, featureSteps, parameterSpace
            RETURN name
        """)
        
        print(f"Created GraphSAGE pipeline: {self.pipeline_name}")
        
        # Add node features that GraphSAGE will use
        node_features = ['degree', 'pageRank', 'inspection_count']
        
        for feature in node_features:
            try:
                self.gds.run_cypher(f"""
                    CALL gds.nodeClassification.addNodeProperty('{self.pipeline_name}', '{feature}', {{
                        mutateProperty: '{feature}'
                    }})
                """)
                print(f"  Added node feature: {feature}")
            except Exception as e:
                print(f"  Failed to add feature {feature}: {str(e)}")
        
        # Configure train/test split
        self.gds.run_cypher(f"""
            CALL gds.nodeClassification.configureSplit('{self.pipeline_name}', {{
                testFraction: 0.2,
                validationFolds: 3
            }})
        """)
        
        # Add GraphSAGE model with various configurations
        graphsage_configs = [
            {
                'name': 'graphsage_small',
                'aggregator': 'mean',
                'activationFunction': 'relu',
                'sampleSizes': [10, 5],
                'embeddingDimension': 64,
                'epochs': 50
            },
            {
                'name': 'graphsage_medium', 
                'aggregator': 'pool',
                'activationFunction': 'relu',
                'sampleSizes': [25, 10],
                'embeddingDimension': 128,
                'epochs': 100
            }
        ]
        
        for config in graphsage_configs:
            try:
                self.gds.run_cypher(f"""
                    CALL gds.nodeClassification.addGraphSage('{self.pipeline_name}', {{
                        aggregator: '{config['aggregator']}',
                        activationFunction: '{config['activationFunction']}',
                        sampleSizes: {config['sampleSizes']},
                        embeddingDimension: {config['embeddingDimension']},
                        epochs: {config['epochs']},
                        learningRate: 0.01,
                        batchSize: 256
                    }})
                """)
                print(f"  Added GraphSAGE config: {config['name']}")
                break  # Use first successful configuration
            except Exception as e:
                print(f"  Failed to add GraphSAGE config {config['name']}: {str(e)}")
                continue
        
        print("GraphSAGE pipeline configured")
    
    def train_graphsage_model(self):
        """Train the GraphSAGE model"""
        
        print("Training GraphSAGE model...")
        print("  This may take several minutes for graph neural network training...")
        
        try:
            # Train GraphSAGE model
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
            
            print("GraphSAGE model training completed!")
            
            # Print training results
            model_info = train_result['modelInfo'].iloc[0]
            model_stats = train_result['modelSelectionStats'].iloc[0]
            
            print(f"Model Info: {model_info}")
            print(f"Model Selection Stats: {model_stats}")
            
            return train_result
            
        except Exception as e:
            print(f"GraphSAGE training failed: {str(e)}")
            print("Attempting with simpler configuration...")
            
            # Fallback: Try with simpler GraphSAGE configuration
            try:
                # Create new simplified pipeline
                self.gds.run_cypher(f"CALL gds.pipeline.drop('{self.pipeline_name}_simple')")
            except:
                pass
            
            self.gds.run_cypher(f"""
                CALL gds.nodeClassification.create('{self.pipeline_name}_simple')
            """)
            
            # Add only degree feature
            self.gds.run_cypher(f"""
                CALL gds.nodeClassification.addNodeProperty('{self.pipeline_name}_simple', 'degree', {{
                    mutateProperty: 'degree'
                }})
            """)
            
            # Configure split
            self.gds.run_cypher(f"""
                CALL gds.nodeClassification.configureSplit('{self.pipeline_name}_simple', {{
                    testFraction: 0.2,
                    validationFolds: 3
                }})
            """)
            
            # Add simple GraphSAGE
            self.gds.run_cypher(f"""
                CALL gds.nodeClassification.addGraphSage('{self.pipeline_name}_simple', {{
                    aggregator: 'mean',
                    activationFunction: 'relu',
                    sampleSizes: [5, 2],
                    embeddingDimension: 32,
                    epochs: 20,
                    learningRate: 0.01
                }})
            """)
            
            # Train simple model
            train_result = self.gds.run_cypher(f"""
                CALL gds.nodeClassification.train('{self.enhanced_graph_name}', {{
                    pipeline: '{self.pipeline_name}_simple',
                    modelName: '{self.model_name}_simple',
                    nodeLabels: ['Entity'],
                    targetProperty: 'has_pest',
                    randomSeed: 42
                }})
                YIELD modelInfo, modelSelectionStats
                RETURN modelInfo, modelSelectionStats
            """)
            
            # Update model name for predictions
            self.model_name = f"{self.model_name}_simple"
            print("Simple GraphSAGE model trained successfully!")
            return train_result
    
    def generate_graphsage_embeddings(self):
        """Generate node embeddings using trained GraphSAGE model"""
        
        print("Generating GraphSAGE embeddings...")
        
        try:
            # Generate embeddings for all nodes
            embeddings = self.gds.run_cypher(f"""
                CALL gds.nodeClassification.predict.stream('{self.enhanced_graph_name}', {{
                    modelName: '{self.model_name}',
                    includePredictedProbabilities: true
                }})
                YIELD nodeId, predictedClass, predictedProbabilities
                WITH nodeId, predictedClass, predictedProbabilities
                MATCH (e:Entity) WHERE id(e) = nodeId
                RETURN e.id as entity_id,
                       predictedClass as graphsage_predicted_class,
                       predictedProbabilities[0] as graphsage_prob_no_pest,
                       predictedProbabilities[1] as graphsage_prob_pest
                ORDER BY predictedProbabilities[1] DESC
            """)
            
            print(f"Generated embeddings for {len(embeddings)} entities")
            return embeddings
            
        except Exception as e:
            print(f"Embedding generation failed: {str(e)}")
            return pd.DataFrame()
    
    def train_unsupervised_graphsage(self):
        """Train unsupervised GraphSAGE for embeddings"""
        
        print("Training unsupervised GraphSAGE for embeddings...")
        
        try:
            # Train unsupervised GraphSAGE
            embedding_result = self.gds.run_cypher(f"""
                CALL gds.graphSage.train('{self.enhanced_graph_name}', {{
                    modelName: '{self.embedding_model_name}',
                    featureProperties: ['degree', 'pageRank'],
                    embeddingDimension: 64,
                    sampleSizes: [10, 5],
                    epochs: 20,
                    learningRate: 0.01,
                    randomSeed: 42
                }})
                YIELD modelInfo, configuration
                RETURN modelInfo, configuration
            """)
            
            print("Unsupervised GraphSAGE training completed")
            
            # Generate embeddings
            embeddings = self.gds.run_cypher(f"""
                CALL gds.graphSage.stream('{self.enhanced_graph_name}', {{
                    modelName: '{self.embedding_model_name}'
                }})
                YIELD nodeId, embedding
                WITH nodeId, embedding
                MATCH (e:Entity) WHERE id(e) = nodeId
                RETURN e.id as entity_id,
                       nodeId as entity_node_id,
                       embedding as graphsage_embedding
            """)
            
            # Convert embeddings to features
            if not embeddings.empty:
                embedding_df = pd.DataFrame(embeddings['graphsage_embedding'].tolist())
                embedding_df.columns = [f'graphsage_emb_{i}' for i in range(len(embedding_df.columns))]
                
                final_embeddings = pd.concat([
                    embeddings[['entity_id', 'entity_node_id']],
                    embedding_df
                ], axis=1)
                
                print(f"Generated {final_embeddings.shape[1]-2} dimensional embeddings for {len(final_embeddings)} entities")
                return final_embeddings
            
        except Exception as e:
            print(f"Unsupervised GraphSAGE failed: {str(e)}")
            return pd.DataFrame()
    
    def predict_with_graphsage(self) -> pd.DataFrame:
        """Make pest predictions using GraphSAGE"""
        
        print("Making predictions with GraphSAGE...")
        
        # Get supervised predictions
        supervised_predictions = self.generate_graphsage_embeddings()
        
        # Get unsupervised embeddings
        unsupervised_embeddings = self.train_unsupervised_graphsage()
        
        # Get actual values
        actual_values = self.gds.run_cypher("""
            MATCH (e:Entity)
            RETURN e.id as entity_id,
                   e.has_pest as actual_pest_binary,
                   e.pest_score as actual_pest_score,
                   e.inspection_count as inspection_count
        """)
        
        # Merge all data
        if not supervised_predictions.empty:
            final_predictions = supervised_predictions.merge(actual_values, on='entity_id', how='left')
        else:
            final_predictions = actual_values.copy()
            final_predictions['graphsage_predicted_class'] = 0
            final_predictions['graphsage_prob_no_pest'] = 0.5
            final_predictions['graphsage_prob_pest'] = 0.5
        
        # Add unsupervised embeddings if available
        if not unsupervised_embeddings.empty:
            final_predictions = final_predictions.merge(
                unsupervised_embeddings.drop('entity_node_id', axis=1), 
                on='entity_id', 
                how='left'
            )
        
        # Add risk categories
        if 'graphsage_prob_pest' in final_predictions.columns:
            final_predictions['graphsage_risk_category'] = final_predictions['graphsage_prob_pest'].apply(
                lambda x: 'Very High Risk' if x >= 0.8 else
                         'High Risk' if x >= 0.6 else
                         'Medium Risk' if x >= 0.4 else
                         'Low Risk' if x >= 0.2 else
                         'Very Low Risk'
            )
        
        # Calculate summary statistics
        total_entities = len(final_predictions)
        print(f"\nGraphSAGE Results Summary:")
        print(f"  Total entities: {total_entities}")
        
        if 'graphsage_predicted_class' in final_predictions.columns:
            predicted_pests = (final_predictions['graphsage_predicted_class'] == 1).sum()
            print(f"  GraphSAGE predicted pests: {predicted_pests}")
        
        if 'graphsage_prob_pest' in final_predictions.columns:
            avg_pest_prob = final_predictions['graphsage_prob_pest'].mean()
            prob_range_min = final_predictions['graphsage_prob_pest'].min()
            prob_range_max = final_predictions['graphsage_prob_pest'].max()
            print(f"  Average pest probability: {avg_pest_prob:.3f}")
            print(f"  Probability range: {prob_range_min:.3f} - {prob_range_max:.3f}")
        
        actual_pests = (final_predictions['actual_pest_binary'] == 1).sum()
        print(f"  Actual pests: {actual_pests}")
        
        if 'graphsage_risk_category' in final_predictions.columns:
            risk_dist = final_predictions['graphsage_risk_category'].value_counts()
            print(f"\nRisk Distribution:")
            for category, count in risk_dist.items():
                print(f"  {category}: {count} entities")
        
        return final_predictions
    
    def run_full_graphsage_analysis(self) -> pd.DataFrame:
        """Run complete GraphSAGE analysis"""
        
        print("Starting GraphSAGE Analysis for Pest Prediction...")
        print("Note: This is the most advanced approach using Graph Neural Networks")
        
        try:
            # Step 1: Prepare features
            self.prepare_graphsage_features()
            
            # Step 2: Create GraphSAGE projection
            self.create_graphsage_projection()
            
            # Step 3: Create GraphSAGE pipeline
            self.create_graphsage_pipeline()
            
            # Step 4: Train GraphSAGE model
            self.train_graphsage_model()
            
            # Step 5: Make predictions
            predictions = self.predict_with_graphsage()
            
            # Step 6: Save results
            output_file = 'graphsage_pest_predictions.csv'
            predictions.to_csv(output_file, index=False)
            print(f"\nGraphSAGE predictions saved to: {output_file}")
            
            print("GraphSAGE Analysis completed successfully!")
            return predictions
            
        except Exception as e:
            print(f"GraphSAGE analysis failed: {str(e)}")
            raise
    
    def cleanup_resources(self):
        """Clean up GraphSAGE resources"""
        models_to_drop = [self.model_name, f'{self.model_name}_simple', self.embedding_model_name]
        pipelines_to_drop = [self.pipeline_name, f'{self.pipeline_name}_simple']
        
        for model in models_to_drop:
            try:
                self.gds.run_cypher(f"CALL gds.model.drop('{model}')")
                print(f"Dropped model: {model}")
            except:
                pass
        
        for pipeline in pipelines_to_drop:
            try:
                self.gds.run_cypher(f"CALL gds.pipeline.drop('{pipeline}')")
                print(f"Dropped pipeline: {pipeline}")
            except:
                pass
        
        try:
            self.gds.graph.drop(self.enhanced_graph_name)
            print(f"Dropped enhanced graph: {self.enhanced_graph_name}")
        except:
            pass


# Integration function
def run_graphsage_pest_prediction(gds_instance, graph_name):
    """
    Convenience function to run GraphSAGE analysis
    
    Args:
        gds_instance: Your GDS connection
        graph_name: Your graph projection name
        
    Returns:
        DataFrame with GraphSAGE predictions
    """
    
    graphsage_analyzer = GraphSAGEPestAnalyzer(gds_instance, graph_name)
    
    try:
        predictions = graphsage_analyzer.run_full_graphsage_analysis()
        return predictions
    finally:
        # Always cleanup
        graphsage_analyzer.cleanup_resources()


# Example integration
def example_integration():
    """Example of how to integrate with your main PestDataAnalyzer class"""
    
    # Add this method to your PestDataAnalyzer class:
    def run_graphsage_analysis(self):
        """Run GraphSAGE pest prediction"""
        from graphsage_pipeline_pest import run_graphsage_pest_prediction
        
        predictions = run_graphsage_pest_prediction(
            gds_instance=self.gds,
            graph_name=self.graph.name()
        )
        
        return predictions
    
    print("Integration example - see comments for usage")


if __name__ == "__main__":
    example_integration()