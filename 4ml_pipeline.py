import pandas as pd
from typing import Dict, List, Tuple, Optional


class MLPipelinePestAnalyzer:
    """
    Machine Learning Pipeline for pest analysis using traditional ML algorithms.
    Extracts graph features and applies various ML models for pest prediction.
    """
    
    def __init__(self, gds_instance, graph_name: str):
        """
        Initialize the ML pipeline analyzer.
        
        Args:
            gds_instance: Neo4j Graph Data Science instance
            graph_name: Name of the graph projection to use
        """
        self.gds = gds_instance
        self.graph_name = graph_name
        self.pipeline_name = "pest_ml_pipeline"
        self.model_name = "pest_ml_model"
    
    def prepare_ml_features_and_targets(self):
        """Extract comprehensive graph features and prepare targets"""
        
        print("Preparing ML features and targets...")
        
        # Create both binary and continuous targets
        target_result = self.gds.run_cypher("""
            MATCH (e:Entity)-[:HAS_INSPECTION_RESULT]->(t:TargetProxy)
            WITH e, 
                 count(t) as total_inspections,
                 sum(t.value) as pest_inspections,
                 max(t.value) as has_pest_binary,
                 avg(toFloat(t.value)) as pest_rate
            SET e.has_pest = has_pest_binary,
                e.pest_rate = pest_rate,
                e.total_inspections = total_inspections,
                e.pest_inspections = pest_inspections
            RETURN count(e) as entities_updated,
                   avg(pest_rate) as avg_pest_rate,
                   sum(has_pest_binary) as entities_with_pests
        """)
        
        stats = target_result.iloc[0]
        print(f"Updated targets for {stats['entities_updated']} entities")
        print(f"  Entities with pests: {stats['entities_with_pests']}")
        print(f"  Average pest rate: {stats['avg_pest_rate']:.3f}")
        
        return target_result
    
    def create_enhanced_graph_projection(self):
        """Create enhanced graph projection with all properties"""
        
        enhanced_graph_name = f"{self.graph_name}_ml"
        
        try:
            self.gds.graph.drop(enhanced_graph_name)
            print(f"Dropped existing ML graph: {enhanced_graph_name}")
        except:
            print("No existing ML graph to drop")
        
        # Create enhanced projection with all target properties
        enhanced_graph, _ = self.gds.graph.project(
            enhanced_graph_name,
            ["Country", "Month", "Entity", "TargetProxy", "CountryMonth"],
            {
                'SHIPPED_IN': {'orientation': 'UNDIRECTED'},
                'IS_FROM': {'orientation': 'UNDIRECTED'}, 
                'HAS_WEATHER': {'orientation': 'UNDIRECTED'},
                'HAS_INSPECTION_RESULT': {'orientation': 'NATURAL'}
            },
            nodeProperties=['has_pest', 'pest_rate', 'total_inspections', 'pest_inspections']
        )
        
        self.enhanced_graph_name = enhanced_graph_name
        print(f"Created enhanced ML graph projection: {enhanced_graph_name}")
        
        return enhanced_graph
    
    def extract_comprehensive_features(self):
        """Extract comprehensive graph features for ML"""
        
        print("Extracting comprehensive graph features...")
        
        # 1. Basic centrality measures
        centrality_features = [
            ('degree', 'gds.degree'),
            ('pageRank', 'gds.pageRank'),
            ('betweenness', 'gds.betweenness'),
            ('closeness', 'gds.closeness'),
            ('eigenvector', 'gds.eigenvector'),
            ('articleRank', 'gds.articleRank')
        ]
        
        for feature_name, algorithm in centrality_features:
            try:
                print(f"  Computing {feature_name}...")
                if feature_name == 'degree':
                    self.gds.run_cypher(f"""
                        CALL {algorithm}.mutate('{self.enhanced_graph_name}', {{
                            mutateProperty: '{feature_name}'
                        }})
                    """)
                else:
                    self.gds.run_cypher(f"""
                        CALL {algorithm}.mutate('{self.enhanced_graph_name}', {{
                            mutateProperty: '{feature_name}',
                            maxIterations: 20
                        }})
                    """)
                print(f"    ✓ {feature_name}")
            except Exception as e:
                print(f"    ✗ {feature_name}: {str(e)}")
        
        # 2. Community and clustering features
        community_features = [
            ('louvain', 'gds.louvain', 'communityId'),
            ('localClusteringCoefficient', 'gds.localClusteringCoefficient', 'localClusteringCoefficient'),
            ('triangleCount', 'gds.triangleCount', 'triangleCount')
        ]
        
        for feature_name, algorithm, property_name in community_features:
            try:
                print(f"  Computing {feature_name}...")
                self.gds.run_cypher(f"""
                    CALL {algorithm}.mutate('{self.enhanced_graph_name}', {{
                        mutateProperty: '{property_name}'
                    }})
                """)
                print(f"    ✓ {feature_name}")
            except Exception as e:
                print(f"    ✗ {feature_name}: {str(e)}")
        
        # 3. Node embedding features (FastRP)
        try:
            print("  Computing FastRP embeddings...")
            self.gds.run_cypher(f"""
                CALL gds.fastRP.mutate('{self.enhanced_graph_name}', {{
                    mutateProperty: 'fastRP_embedding',
                    embeddingDimension: 32,
                    randomSeed: 42
                }})
            """)
            print("    ✓ FastRP embeddings")
        except Exception as e:
            print(f"    ✗ FastRP embeddings: {str(e)}")
        
        # 4. K-Core decomposition
        try:
            print("  Computing K-Core...")
            self.gds.run_cypher(f"""
                CALL gds.kcore.mutate('{self.enhanced_graph_name}', {{
                    mutateProperty: 'coreValue'
                }})
            """)
            print("    ✓ K-Core")
        except Exception as e:
            print(f"    ✗ K-Core: {str(e)}")
        
        print("Feature extraction completed")
    
    def create_ml_pipeline_classification(self):
        """Create ML pipeline for binary classification"""
        
        # Drop existing pipeline
        try:
            self.gds.run_cypher(f"CALL gds.pipeline.drop('{self.pipeline_name}_classification')")
        except:
            pass
        
        # Create classification pipeline
        self.gds.run_cypher(f"""
            CALL gds.nodeClassification.create('{self.pipeline_name}_classification')
        """)
        
        print(f"Created ML classification pipeline")
        
        # Add all node property features
        node_features = [
            'degree', 'pageRank', 'betweenness', 'closeness', 'eigenvector', 'articleRank',
            'communityId', 'localClusteringCoefficient', 'triangleCount', 'coreValue',
            'total_inspections', 'pest_inspections'
        ]
        
        for feature in node_features:
            try:
                self.gds.run_cypher(f"""
                    CALL gds.nodeClassification.addNodeProperty('{self.pipeline_name}_classification', '{feature}', {{
                        mutateProperty: '{feature}'
                    }})
                """)
                print(f"  Added feature: {feature}")
            except Exception as e:
                print(f"  Failed to add {feature}: {str(e)}")
        
        # Add FastRP embedding features
        try:
            self.gds.run_cypher(f"""
                CALL gds.nodeClassification.addFeature('{self.pipeline_name}_classification', 'fastRP', {{
                    nodeProperties: ['fastRP_embedding']
                }})
            """)
            print("  Added FastRP embedding features")
        except Exception as e:
            print(f"  Failed to add FastRP features: {str(e)}")
        
        # Configure split
        self.gds.run_cypher(f"""
            CALL gds.nodeClassification.configureSplit('{self.pipeline_name}_classification', {{
                testFraction: 0.2,
                validationFolds: 3
            }})
        """)
        
        # Add multiple ML algorithms
        ml_algorithms = [
            ('logisticRegression', 'gds.nodeClassification.addLogisticRegression'),
            ('randomForest', 'gds.nodeClassification.addRandomForest')
        ]
        
        for algo_name, algo_method in ml_algorithms:
            try:
                if algo_name == 'logisticRegression':
                    self.gds.run_cypher(f"""
                        CALL {algo_method}('{self.pipeline_name}_classification', {{
                            penalty: 1.0,
                            maxIterations: 1000
                        }})
                    """)
                elif algo_name == 'randomForest':
                    self.gds.run_cypher(f"""
                        CALL {algo_method}('{self.pipeline_name}_classification', {{
                            numberOfDecisionTrees: 100,
                            maxDepth: 10
                        }})
                    """)
                print(f"  Added ML algorithm: {algo_name}")
            except Exception as e:
                print(f"  Failed to add {algo_name}: {str(e)}")
        
        print("ML classification pipeline configured")
    
    def create_ml_pipeline_regression(self):
        """Create ML pipeline for regression"""
        
        # Drop existing pipeline
        try:
            self.gds.run_cypher(f"CALL gds.pipeline.drop('{self.pipeline_name}_regression')")
        except:
            pass
        
        # Create regression pipeline
        self.gds.run_cypher(f"""
            CALL gds.nodeRegression.create('{self.pipeline_name}_regression')
        """)
        
        print(f"Created ML regression pipeline")
        
        # Add all node property features (same as classification)
        node_features = [
            'degree', 'pageRank', 'betweenness', 'closeness', 'eigenvector', 'articleRank',
            'communityId', 'localClusteringCoefficient', 'triangleCount', 'coreValue',
            'total_inspections', 'pest_inspections'
        ]
        
        for feature in node_features:
            try:
                self.gds.run_cypher(f"""
                    CALL gds.nodeRegression.addNodeProperty('{self.pipeline_name}_regression', '{feature}', {{
                        mutateProperty: '{feature}'
                    }})
                """)
            except Exception as e:
                pass
        
        # Add FastRP embedding features
        try:
            self.gds.run_cypher(f"""
                CALL gds.nodeRegression.addFeature('{self.pipeline_name}_regression', 'fastRP', {{
                    nodeProperties: ['fastRP_embedding']
                }})
            """)
        except Exception as e:
            pass
        
        # Configure split
        self.gds.run_cypher(f"""
            CALL gds.nodeRegression.configureSplit('{self.pipeline_name}_regression', {{
                testFraction: 0.2,
                validationFolds: 3
            }})
        """)
        
        # Add regression algorithms
        try:
            self.gds.run_cypher(f"""
                CALL gds.nodeRegression.addLinearRegression('{self.pipeline_name}_regression', {{
                    penalty: 1.0
                }})
            """)
        except:
            pass
        
        try:
            self.gds.run_cypher(f"""
                CALL gds.nodeRegression.addRandomForest('{self.pipeline_name}_regression', {{
                    numberOfDecisionTrees: 100,
                    maxDepth: 10
                }})
            """)
        except:
            pass
        
        print("ML regression pipeline configured")
    
    def train_ml_models(self):
        """Train both classification and regression models"""
        
        print("Training ML models...")
        
        # Train classification model
        try:
            print("  Training classification model...")
            class_result = self.gds.run_cypher(f"""
                CALL gds.nodeClassification.train('{self.enhanced_graph_name}', {{
                    pipeline: '{self.pipeline_name}_classification',
                    modelName: '{self.model_name}_classification',
                    nodeLabels: ['Entity'],
                    targetProperty: 'has_pest',
                    randomSeed: 42
                }})
                YIELD modelInfo, modelSelectionStats
                RETURN modelInfo, modelSelectionStats
            """)
            print("    ✓ Classification model trained")
        except Exception as e:
            print(f"    ✗ Classification training failed: {str(e)}")
            class_result = None
        
        # Train regression model
        try:
            print("  Training regression model...")
            reg_result = self.gds.run_cypher(f"""
                CALL gds.nodeRegression.train('{self.enhanced_graph_name}', {{
                    pipeline: '{self.pipeline_name}_regression',
                    modelName: '{self.model_name}_regression',
                    nodeLabels: ['Entity'],
                    targetProperty: 'pest_rate',
                    randomSeed: 42
                }})
                YIELD modelInfo, modelSelectionStats
                RETURN modelInfo, modelSelectionStats
            """)
            print("    ✓ Regression model trained")
        except Exception as e:
            print(f"    ✗ Regression training failed: {str(e)}")
            reg_result = None
        
        return class_result, reg_result
    
    def predict_with_ml_models(self) -> pd.DataFrame:
        """Make predictions using both trained models"""
        
        print("Making predictions with ML models...")
        
        # Get classification predictions
        try:
            class_predictions = self.gds.run_cypher(f"""
                CALL gds.nodeClassification.predict.stream('{self.enhanced_graph_name}', {{
                    modelName: '{self.model_name}_classification',
                    includePredictedProbabilities: true
                }})
                YIELD nodeId, predictedClass, predictedProbabilities
                WITH nodeId, predictedClass, predictedProbabilities
                MATCH (e:Entity) WHERE id(e) = nodeId
                RETURN e.id as entity_id,
                       predictedClass as ml_predicted_class,
                       predictedProbabilities[0] as ml_prob_no_pest,
                       predictedProbabilities[1] as ml_prob_pest
            """)
            print("  ✓ Classification predictions completed")
        except Exception as e:
            print(f"  ✗ Classification predictions failed: {str(e)}")
            class_predictions = pd.DataFrame(columns=['entity_id', 'ml_predicted_class', 'ml_prob_no_pest', 'ml_prob_pest'])
        
        # Get regression predictions
        try:
            reg_predictions = self.gds.run_cypher(f"""
                CALL gds.nodeRegression.predict.stream('{self.enhanced_graph_name}', {{
                    modelName: '{self.model_name}_regression'
                }})
                YIELD nodeId, predictedValue
                WITH nodeId, predictedValue
                MATCH (e:Entity) WHERE id(e) = nodeId
                RETURN e.id as entity_id,
                       predictedValue as ml_predicted_score
            """)
            print("  ✓ Regression predictions completed")
        except Exception as e:
            print(f"  ✗ Regression predictions failed: {str(e)}")
            reg_predictions = pd.DataFrame(columns=['entity_id', 'ml_predicted_score'])
        
        # Get actual values
        actual_values = self.gds.run_cypher("""
            MATCH (e:Entity)
            RETURN e.id as entity_id,
                   e.has_pest as actual_pest_binary,
                   e.pest_rate as actual_pest_rate,
                   e.total_inspections as total_inspections
        """)
        
        # Merge all predictions
        if not class_predictions.empty and not reg_predictions.empty:
            final_predictions = class_predictions.merge(reg_predictions, on='entity_id', how='outer')
        elif not class_predictions.empty:
            final_predictions = class_predictions
        elif not reg_predictions.empty:
            final_predictions = reg_predictions
        else:
            final_predictions = pd.DataFrame(columns=['entity_id'])
        
        final_predictions = final_predictions.merge(actual_values, on='entity_id', how='left')
        
        # Add risk categories and multiple thresholds
        if 'ml_prob_pest' in final_predictions.columns:
            final_predictions['ml_pest_risk_category'] = final_predictions['ml_prob_pest'].apply(
                lambda x: 'Very High Risk' if x >= 0.8 else
                         'High Risk' if x >= 0.6 else
                         'Medium Risk' if x >= 0.4 else
                         'Low Risk' if x >= 0.2 else
                         'Very Low Risk'
            )
        
        if 'ml_predicted_score' in final_predictions.columns:
            final_predictions['ml_predicted_score'] = final_predictions['ml_predicted_score'].clip(0, 1)
            final_predictions['ml_score_binary_05'] = (final_predictions['ml_predicted_score'] >= 0.5).astype(int)
            final_predictions['ml_score_binary_03'] = (final_predictions['ml_predicted_score'] >= 0.3).astype(int)
        
        # Calculate summary statistics
        total_entities = len(final_predictions)
        print(f"\nML Pipeline Results Summary:")
        print(f"  Total entities: {total_entities}")
        
        if 'ml_predicted_class' in final_predictions.columns:
            ml_class_pests = (final_predictions['ml_predicted_class'] == 1).sum()
            print(f"  ML Classification predicted pests: {ml_class_pests}")
        
        if 'ml_predicted_score' in final_predictions.columns:
            avg_ml_score = final_predictions['ml_predicted_score'].mean()
            print(f"  ML Regression average score: {avg_ml_score:.3f}")
        
        actual_pests = (final_predictions['actual_pest_binary'] == 1).sum()
        print(f"  Actual pests: {actual_pests}")
        
        return final_predictions
    
    def run_full_ml_pipeline_analysis(self) -> pd.DataFrame:
        """Run complete ML pipeline analysis"""
        
        print("Starting ML Pipeline Analysis for Pest Prediction...")
        
        try:
            # Step 1: Prepare features and targets
            self.prepare_ml_features_and_targets()
            
            # Step 2: Create enhanced graph projection
            self.create_enhanced_graph_projection()
            
            # Step 3: Extract comprehensive features
            self.extract_comprehensive_features()
            
            # Step 4: Create ML pipelines
            self.create_ml_pipeline_classification()
            self.create_ml_pipeline_regression()
            
            # Step 5: Train models
            self.train_ml_models()
            
            # Step 6: Make predictions
            predictions = self.predict_with_ml_models()
            
            # Step 7: Save results
            output_file = 'ml_pipeline_pest_predictions.csv'
            predictions.to_csv(output_file, index=False)
            print(f"\nML pipeline predictions saved to: {output_file}")
            
            print("ML Pipeline Analysis completed successfully!")
            return predictions
            
        except Exception as e:
            print(f"ML pipeline analysis failed: {str(e)}")
            raise
    
    def cleanup_resources(self):
        """Clean up pipeline and model resources"""
        models_to_drop = [f'{self.model_name}_classification', f'{self.model_name}_regression']
        pipelines_to_drop = [f'{self.pipeline_name}_classification', f'{self.pipeline_name}_regression']
        
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
def run_ml_pipeline_pest_prediction(gds_instance, graph_name):
    """
    Convenience function to run ML pipeline analysis
    
    Args:
        gds_instance: Your GDS connection
        graph_name: Your graph projection name
        
    Returns:
        DataFrame with ML pipeline predictions
    """
    
    ml_analyzer = MLPipelinePestAnalyzer(gds_instance, graph_name)
    
    try:
        predictions = ml_analyzer.run_full_ml_pipeline_analysis()
        return predictions
    finally:
        # Always cleanup
        ml_analyzer.cleanup_resources()


# Example integration
def example_integration():
    """Example of how to integrate with your main PestDataAnalyzer class"""
    
    # Add this method to your PestDataAnalyzer class:
    def run_ml_pipeline_analysis(self):
        """Run ML pipeline pest prediction"""
        from ml_pipeline_pest import run_ml_pipeline_pest_prediction
        
        predictions = run_ml_pipeline_pest_prediction(
            gds_instance=self.gds,
            graph_name=self.graph.name()
        )
        
        return predictions
    
    print("Integration example - see comments for usage")


if __name__ == "__main__":
    example_integration()