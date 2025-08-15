import pandas as pd
from typing import Dict, List, Tuple, Optional


class NodeRegressionPestAnalyzer:
    """
    Node Regression pipeline for pest analysis.
    Predicts continuous pest probability scores for entities based on graph features.
    """
    
    def __init__(self, gds_instance, graph_name: str):
        """
        Initialize the node regression analyzer.
        
        Args:
            gds_instance: Neo4j Graph Data Science instance
            graph_name: Name of the graph projection to use
        """
        self.gds = gds_instance
        self.graph_name = graph_name
        self.pipeline_name = "pest_node_regression_pipeline"
        self.model_name = "pest_node_regression_model"
    
    def prepare_regression_target(self):
        """Create continuous pest risk scores as regression targets"""
        
        print("Preparing regression target values...")
        
        # Create pest risk score based on historical pest frequency
        # This gives us a continuous target (0.0 to 1.0) instead of binary (0 or 1)
        result = self.gds.run_cypher("""
            MATCH (e:Entity)-[:HAS_INSPECTION_RESULT]->(t:TargetProxy)
            WITH e, 
                 count(t) as total_inspections,
                 sum(t.value) as pest_inspections,
                 avg(toFloat(t.value)) as pest_rate
            SET e.pest_risk_score = pest_rate
            RETURN count(e) as entities_updated,
                   avg(pest_rate) as avg_pest_rate,
                   min(pest_rate) as min_pest_rate,
                   max(pest_rate) as max_pest_rate
        """)
        
        stats = result.iloc[0]
        print(f"Updated pest risk scores for {stats['entities_updated']} entities")
        print(f"  Average pest rate: {stats['avg_pest_rate']:.3f}")
        print(f"  Range: {stats['min_pest_rate']:.3f} - {stats['max_pest_rate']:.3f}")
        
        # Check distribution of pest risk scores
        distribution = self.gds.run_cypher("""
            MATCH (e:Entity)
            WHERE e.pest_risk_score IS NOT NULL
            WITH e.pest_risk_score as risk_score,
                 CASE 
                     WHEN e.pest_risk_score = 0.0 THEN 'No Risk (0.0)'
                     WHEN e.pest_risk_score > 0.0 AND e.pest_risk_score <= 0.2 THEN 'Low Risk (0.0-0.2)'
                     WHEN e.pest_risk_score > 0.2 AND e.pest_risk_score <= 0.5 THEN 'Medium Risk (0.2-0.5)'
                     WHEN e.pest_risk_score > 0.5 AND e.pest_risk_score <= 0.8 THEN 'High Risk (0.5-0.8)'
                     ELSE 'Very High Risk (0.8-1.0)'
                 END as risk_category
            RETURN risk_category, count(*) as entity_count
            ORDER BY risk_category
        """)
        
        print("Pest risk score distribution:")
        for _, row in distribution.iterrows():
            print(f"  {row['risk_category']}: {row['entity_count']} entities")
        
        return result
    
    def create_enhanced_graph_projection(self):
        """Create enhanced graph projection with regression target"""
        
        enhanced_graph_name = f"{self.graph_name}_regression"
        
        try:
            self.gds.graph.drop(enhanced_graph_name)
            print(f"Dropped existing regression graph: {enhanced_graph_name}")
        except:
            print("No existing regression graph to drop")
        
        # Create enhanced projection with pest_risk_score property
        enhanced_graph, _ = self.gds.graph.project(
            enhanced_graph_name,
            ["Country", "Month", "Entity", "TargetProxy", "CountryMonth"],
            {
                'SHIPPED_IN': {'orientation': 'UNDIRECTED'},
                'IS_FROM': {'orientation': 'UNDIRECTED'}, 
                'HAS_WEATHER': {'orientation': 'UNDIRECTED'},
                'HAS_INSPECTION_RESULT': {'orientation': 'NATURAL'}
            },
            nodeProperties=['pest_risk_score']
        )
        
        self.enhanced_graph_name = enhanced_graph_name
        print(f"Created enhanced graph projection: {enhanced_graph_name}")
        print(f"Graph contains {enhanced_graph.node_count()} nodes and {enhanced_graph.relationship_count()} relationships")
        
        return enhanced_graph
    
    def add_advanced_features(self):
        """Add comprehensive graph features for regression"""
        
        print("Computing advanced graph features...")
        
        # Centrality measures
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
                        YIELD centralityDistribution
                        RETURN centralityDistribution
                    """)
                else:
                    self.gds.run_cypher(f"""
                        CALL {algorithm}.mutate('{self.enhanced_graph_name}', {{
                            mutateProperty: '{feature_name}',
                            maxIterations: 20
                        }})
                        YIELD centralityDistribution  
                        RETURN centralityDistribution
                    """)
                
                print(f"    ✓ {feature_name} computed")
                
            except Exception as e:
                print(f"    ✗ Failed to compute {feature_name}: {str(e)}")
        
        # Community detection features
        try:
            print("  Computing community detection (Louvain)...")
            self.gds.run_cypher(f"""
                CALL gds.louvain.mutate('{self.enhanced_graph_name}', {{
                    mutateProperty: 'communityId'
                }})
                YIELD communityCount, modularity
                RETURN communityCount, modularity
            """)
            print("    ✓ Community detection computed")
        except Exception as e:
            print(f"    ✗ Failed to compute community detection: {str(e)}")
        
        # Local clustering coefficient
        try:
            print("  Computing local clustering coefficient...")
            self.gds.run_cypher(f"""
                CALL gds.localClusteringCoefficient.mutate('{self.enhanced_graph_name}', {{
                    mutateProperty: 'localClusteringCoefficient'
                }})
                YIELD averageClusteringCoefficient
                RETURN averageClusteringCoefficient
            """)
            print("    ✓ Local clustering coefficient computed")
        except Exception as e:
            print(f"    ✗ Failed to compute clustering coefficient: {str(e)}")
        
        # Triangle count
        try:
            print("  Computing triangle count...")
            self.gds.run_cypher(f"""
                CALL gds.triangleCount.mutate('{self.enhanced_graph_name}', {{
                    mutateProperty: 'triangleCount'
                }})
                YIELD globalTriangleCount
                RETURN globalTriangleCount
            """)
            print("    ✓ Triangle count computed")
        except Exception as e:
            print(f"    ✗ Failed to compute triangle count: {str(e)}")
        
        print("Advanced features computation completed")
    
    def create_node_regression_pipeline(self):
        """Create and configure node regression pipeline"""
        
        # Drop existing pipeline
        try:
            self.gds.run_cypher(f"CALL gds.pipeline.drop('{self.pipeline_name}')")
            print(f"Dropped existing pipeline: {self.pipeline_name}")
        except:
            print("No existing pipeline to drop")
        
        # Create new regression pipeline
        self.gds.run_cypher(f"""
            CALL gds.nodeRegression.create('{self.pipeline_name}')
            YIELD name, nodePropertySteps, featureSteps, parameterSpace
            RETURN name, nodePropertySteps, featureSteps, parameterSpace
        """)
        
        print(f"Created node regression pipeline: {self.pipeline_name}")
        
        # Add all computed features
        node_features = [
            'degree', 'pageRank', 'betweenness', 'closeness', 'eigenvector', 'articleRank',
            'communityId', 'localClusteringCoefficient', 'triangleCount'
        ]
        
        for feature in node_features:
            try:
                self.gds.run_cypher(f"""
                    CALL gds.nodeRegression.addNodeProperty('{self.pipeline_name}', '{feature}', {{
                        mutateProperty: '{feature}'
                    }})
                    YIELD name, nodePropertySteps, featureSteps, parameterSpace
                    RETURN name
                """)
                print(f"  Added regression feature: {feature}")
            except Exception as e:
                print(f"  Failed to add feature {feature}: {str(e)}")
        
        # Configure train/test split for regression
        self.gds.run_cypher(f"""
            CALL gds.nodeRegression.configureSplit('{self.pipeline_name}', {{
                testFraction: 0.2,
                validationFolds: 3
            }})
            YIELD splitConfig
            RETURN splitConfig
        """)
        
        # Add linear regression model
        self.gds.run_cypher(f"""
            CALL gds.nodeRegression.addLinearRegression('{self.pipeline_name}', {{
                penalty: 1.0,
                maxIterations: 1000
            }})
            YIELD parameterSpace
            RETURN parameterSpace
        """)
        
        # Add random forest model as alternative
        try:
            self.gds.run_cypher(f"""
                CALL gds.nodeRegression.addRandomForest('{self.pipeline_name}', {{
                    numberOfDecisionTrees: 100,
                    maxDepth: 10
                }})
                YIELD parameterSpace
                RETURN parameterSpace
            """)
            print("Added Random Forest model to pipeline")
        except Exception as e:
            print(f"Could not add Random Forest: {str(e)}")
        
        print("Node regression pipeline configured successfully")
    
    def train_regression_model(self):
        """Train the node regression model"""
        
        print("Training node regression model...")
        
        # Train the model
        train_result = self.gds.run_cypher(f"""
            CALL gds.nodeRegression.train('{self.enhanced_graph_name}', {{
                pipeline: '{self.pipeline_name}',
                modelName: '{self.model_name}',
                nodeLabels: ['Entity'],
                targetProperty: 'pest_risk_score',
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
    
    def predict_pest_scores(self) -> pd.DataFrame:
        """Make pest risk predictions using the trained regression model"""
        
        print("Making pest risk score predictions...")
        
        # Stream regression predictions for all Entity nodes
        predictions = self.gds.run_cypher(f"""
            CALL gds.nodeRegression.predict.stream('{self.enhanced_graph_name}', {{
                modelName: '{self.model_name}'
            }})
            YIELD nodeId, predictedValue
            WITH nodeId, predictedValue
            MATCH (e:Entity) WHERE id(e) = nodeId
            RETURN e.id as entity_id,
                   predictedValue as predicted_pest_score,
                   e.pest_risk_score as actual_pest_score
            ORDER BY predictedValue DESC
        """)
        
        # Ensure predicted scores are within [0, 1] range
        predictions['predicted_pest_score'] = predictions['predicted_pest_score'].clip(0, 1)
        
        # Create binary predictions using threshold
        predictions['predicted_pest_binary_05'] = (predictions['predicted_pest_score'] >= 0.5).astype(int)
        predictions['predicted_pest_binary_03'] = (predictions['predicted_pest_score'] >= 0.3).astype(int)
        predictions['predicted_pest_binary_02'] = (predictions['predicted_pest_score'] >= 0.2).astype(int)
        
        # Add risk categories
        predictions['pest_risk_category'] = predictions['predicted_pest_score'].apply(
            lambda x: 'Very High Risk' if x >= 0.8 else
                     'High Risk' if x >= 0.6 else
                     'Medium Risk' if x >= 0.4 else
                     'Low Risk' if x >= 0.2 else
                     'Very Low Risk'
        )
        
        # Create actual binary for comparison
        predictions['actual_pest_binary'] = (predictions['actual_pest_score'] > 0).astype(int)
        
        # Calculate summary statistics
        total_entities = len(predictions)
        avg_predicted_score = predictions['predicted_pest_score'].mean()
        avg_actual_score = predictions['actual_pest_score'].mean()
        
        # Predictions at different thresholds
        pred_05 = predictions['predicted_pest_binary_05'].sum()
        pred_03 = predictions['predicted_pest_binary_03'].sum()
        pred_02 = predictions['predicted_pest_binary_02'].sum()
        actual_pests = predictions['actual_pest_binary'].sum()
        
        print(f"\nRegression Prediction Results:")
        print(f"  Total entities: {total_entities}")
        print(f"  Average predicted score: {avg_predicted_score:.3f}")
        print(f"  Average actual score: {avg_actual_score:.3f}")
        print(f"  Score range: {predictions['predicted_pest_score'].min():.3f} - {predictions['predicted_pest_score'].max():.3f}")
        print(f"\nBinary Predictions at Different Thresholds:")
        print(f"  Threshold 0.5: {pred_05} entities predicted as pest")
        print(f"  Threshold 0.3: {pred_03} entities predicted as pest")
        print(f"  Threshold 0.2: {pred_02} entities predicted as pest")
        print(f"  Actual pests: {actual_pests} entities")
        
        # Risk category distribution
        risk_dist = predictions['pest_risk_category'].value_counts()
        print(f"\nRisk Category Distribution:")
        for category, count in risk_dist.items():
            print(f"  {category}: {count} entities")
        
        return predictions
    
    def run_full_node_regression_analysis(self) -> pd.DataFrame:
        """Run complete node regression analysis"""
        
        print("Starting Node Regression Analysis for Pest Risk Prediction...")
        
        try:
            # Step 1: Prepare regression targets
            self.prepare_regression_target()
            
            # Step 2: Create enhanced graph projection
            self.create_enhanced_graph_projection()
            
            # Step 3: Add advanced features
            self.add_advanced_features()
            
            # Step 4: Create regression pipeline
            self.create_node_regression_pipeline()
            
            # Step 5: Train model
            self.train_regression_model()
            
            # Step 6: Make predictions
            predictions = self.predict_pest_scores()
            
            # Step 7: Save results
            output_file = 'node_regression_pest_predictions.csv'
            predictions.to_csv(output_file, index=False)
            print(f"\nNode regression predictions saved to: {output_file}")
            
            print("Node Regression Analysis completed successfully!")
            return predictions
            
        except Exception as e:
            print(f"Node regression analysis failed: {str(e)}")
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


# Integration function for your main PestDataAnalyzer class
def run_node_regression_pest_prediction(gds_instance, graph_name):
    """
    Convenience function to run node regression analysis
    
    Args:
        gds_instance: Your GDS connection
        graph_name: Your graph projection name
        
    Returns:
        DataFrame with node regression predictions
    """
    
    regressor = NodeRegressionPestAnalyzer(gds_instance, graph_name)
    
    try:
        predictions = regressor.run_full_node_regression_analysis()
        return predictions
    finally:
        # Always cleanup
        regressor.cleanup_resources()


# Example usage for integration with your PestDataAnalyzer
def example_integration():
    """
    Example of how to integrate with your main PestDataAnalyzer class
    """
    
    # Add this method to your PestDataAnalyzer class:
    def run_node_regression_analysis(self):
        """Run node regression pest prediction"""
        from node_regression_pest import run_node_regression_pest_prediction
        
        predictions = run_node_regression_pest_prediction(
            gds_instance=self.gds,
            graph_name=self.graph.name()
        )
        
        return predictions
    
    # Then use it like:
    # analyzer = PestDataAnalyzer(...)
    # analyzer.connect()
    # analyzer.load_data()
    # analyzer.create_nodes()
    # analyzer.create_inspections_and_relationships()
    # analyzer.create_projection()
    # 
    # # Run node regression
    # predictions = analyzer.run_node_regression_analysis()
    
    print("Integration example - see comments for usage")


if __name__ == "__main__":
    example_integration()