import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional


class LinkPredictionPestAnalyzer:
    """
    Link prediction class for pest analysis using heuristic algorithms.
    Uses Adamic-Adar index and Jaccard coefficient to predict pest likelihood.
    """
    
    def __init__(self, gds_instance, graph_name: str, log_level=logging.INFO):
        """
        Initialize the link prediction analyzer.
        
        Args:
            gds_instance: Neo4j Graph Data Science instance
            graph_name: Name of the graph projection to use
            log_level: Logging level
        """
        self.gds = gds_instance
        self.graph_name = graph_name
        
        # Setup logger
        # self.logger = logging.getLogger(self.__class__.__name__)
        # self.logger.setLevel(log_level)
        
        # # Create console handler if no handlers exist
        # if not self.logger.handlers:
        #     handler = logging.StreamHandler()
        #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #     handler.setFormatter(formatter)
        #     self.logger.addHandler(handler)
    
    def prepare_link_prediction_data(self) -> pd.DataFrame:
        """
        Prepare data for link prediction by identifying entity-pest relationships.
        
        Returns:
            DataFrame with entity pairs and existing pest relationships
        """
        print("Preparing link prediction data...")
        
        # Get all entities and their current pest status
        entity_pest_data = self.gds.run_cypher("""
            MATCH (e:Entity)-[:HAS_INSPECTION_RESULT]->(t:TargetProxy)
            RETURN e.id as entity_id, 
                   id(e) as entity_node_id,
                   t.value as target_value,
                   max(t.value) as has_pest_ever
            """)
        
        # Get all pest proxy nodes, which is just 0 or 1 / pest or no_pest
        pest_nodes = self.gds.run_cypher("""
            MATCH (t:TargetProxy)
            RETURN id(t) as pest_node_id, t.value as pest_value, t.label as pest_label
            """)
        
        print(f"Found {len(entity_pest_data)} entity-pest relationships")
        print(f"Found {len(pest_nodes)} pest proxy nodes")
        
        return entity_pest_data, pest_nodes
    
    def create_link_prediction_pairs(self, entity_pest_data: pd.DataFrame, pest_nodes: pd.DataFrame) -> pd.DataFrame:
        """
        Create entity-pest node pairs for link prediction.
        Args:
            entity_pest_data: DataFrame with entity data
            pest_nodes: DataFrame with pest node data
        Returns:
            DataFrame with entity-pest pairs for prediction
        """
        print("Creating entity-pest pairs for link prediction...")
        
        # Get unique entities
        entities = entity_pest_data[['entity_id', 'entity_node_id']].drop_duplicates()
        
        # Create all possible entity-pest pairs  cartesian product
        prediction_pairs = []
        
        for _, entity in entities.iterrows():
            for _, pest in pest_nodes.iterrows():
                # Does this entity already have a relationship to this pest outcome?
                existing_rel = entity_pest_data[
                    (entity_pest_data['entity_node_id'] == entity['entity_node_id']) & 
                    (entity_pest_data['target_value'] == pest['pest_value'])
                ]
                
                has_existing_relationship = len(existing_rel) > 0
                
                prediction_pairs.append({
                    'entity_id': entity['entity_id'],
                    'entity_node_id': entity['entity_node_id'],
                    'pest_node_id': pest['pest_node_id'],
                    'pest_value': pest['pest_value'],
                    'pest_label': pest['pest_label'],
                    'has_existing_relationship': has_existing_relationship
                })
        
        pairs_df = pd.DataFrame(prediction_pairs)
        # For each entity, 2 rows are created; one for pest and non-pest
        print(f"Created {len(pairs_df)} entity-pest pairs for prediction")
        
        return pairs_df
    
    def compute_adamic_adar_scores(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Adamic-Adar index scores for entity-pest pairs.
        
        Args:
            pairs_df: DataFrame with entity-pest pairs
            
        Returns:
            DataFrame with Adamic-Adar scores added
        """
        print("Computing Adamic-Adar index scores...")
        
        # Prepare node pairs for batch processing
        for _, row in pairs_df.iterrows():
            node_pairs = [
                {'node1': int(row['entity_node_id']), 'node2': int(row['pest_node_id'])} 
                ]
        
        # Compute Adamic-Adar scores using Neo4j GDS
        adamic_adar_results = self.gds.run_cypher("""
            UNWIND $node_pairs as pair
            MATCH (n1) WHERE id(n1) = pair.node1
            MATCH (n2) WHERE id(n2) = pair.node2
            RETURN pair.node1 as node1_id, 
                   pair.node2 as node2_id,
                   gds.linkprediction.adamicAdar(n1, n2) as adamic_adar_score
            """, {'node_pairs': node_pairs})
        
        # Merge scores back to pairs DataFrame
        pairs_df = pairs_df.merge(
            adamic_adar_results.rename(columns={
                'node1_id': 'entity_node_id',
                'node2_id': 'pest_node_id'
            }),
            on=['entity_node_id', 'pest_node_id'],
            how='left'
        )
        
        print("Adamic-Adar scores computed successfully")
        return pairs_df
    
    def compute_jaccard_scores(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Jaccard coefficient scores for entity-pest pairs.
        Args:
            pairs_df: DataFrame with entity-pest pairs
        Returns:
            DataFrame with Jaccard scores added
        """
        print("Computing Jaccard coefficient scores...")
        
        # Prepare node pairs for batch processing
        node_pairs = [
            {'node1': int(row['entity_node_id']), 'node2': int(row['pest_node_id'])}
            for _, row in pairs_df.iterrows()
        ]
        
        # Compute Jaccard scores using Neo4j GDS
        jaccard_results = self.gds.run_cypher("""
            UNWIND $node_pairs as pair
            MATCH (n1) WHERE id(n1) = pair.node1
            MATCH (n2) WHERE id(n2) = pair.node2
            RETURN pair.node1 as node1_id, 
                   pair.node2 as node2_id,
                   gds.linkprediction.jaccard(n1, n2) as jaccard_score
            """, {'node_pairs': node_pairs})
        
        # Merge scores back to pairs DataFrame
        pairs_df = pairs_df.merge(
            jaccard_results.rename(columns={
                'node1_id': 'entity_node_id',
                'node2_id': 'pest_node_id'
            }),
            on=['entity_node_id', 'pest_node_id'],
            how='left'
        )
        
        print("Jaccard coefficient scores computed successfully")
        return pairs_df
    
    def compute_common_neighbors_scores(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute common neighbors count for entity-pest pairs.
        
        Args:
            pairs_df: DataFrame with entity-pest pairs
            
        Returns:
            DataFrame with common neighbors count added
        """
        print("Computing common neighbors scores...")
        
        # Prepare node pairs for batch processing
        node_pairs = [
            {'node1': int(row['entity_node_id']), 'node2': int(row['pest_node_id'])}
            for _, row in pairs_df.iterrows()
        ]
        
        # Compute common neighbors using Neo4j GDS
        common_neighbors_results = self.gds.run_cypher("""
            UNWIND $node_pairs as pair
            MATCH (n1) WHERE id(n1) = pair.node1
            MATCH (n2) WHERE id(n2) = pair.node2
            RETURN pair.node1 as node1_id, 
                   pair.node2 as node2_id,
                   gds.linkprediction.commonNeighbors(n1, n2) as common_neighbors_count
            """, {'node_pairs': node_pairs})
        
        # Merge scores back to pairs DataFrame
        pairs_df = pairs_df.merge(
            common_neighbors_results.rename(columns={
                'node1_id': 'entity_node_id',
                'node2_id': 'pest_node_id'
            }),
            on=['entity_node_id', 'pest_node_id'],
            how='left'
        )
        
        print("Common neighbors scores computed successfully")
        return pairs_df
    
    def compute_preferential_attachment_scores(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute preferential attachment scores for entity-pest pairs.
        
        Args:
            pairs_df: DataFrame with entity-pest pairs
            
        Returns:
            DataFrame with preferential attachment scores added
        """
        print("Computing preferential attachment scores...")
        
        # Prepare node pairs for batch processing
        node_pairs = [
            {'node1': int(row['entity_node_id']), 'node2': int(row['pest_node_id'])}
            for _, row in pairs_df.iterrows()
        ]
        
        # Compute preferential attachment using Neo4j GDS
        pref_attach_results = self.gds.run_cypher("""
            UNWIND $node_pairs as pair
            MATCH (n1) WHERE id(n1) = pair.node1
            MATCH (n2) WHERE id(n2) = pair.node2
            RETURN pair.node1 as node1_id, 
                   pair.node2 as node2_id,
                   gds.linkprediction.preferentialAttachment(n1, n2) as preferential_attachment_score
            """, {'node_pairs': node_pairs})
        
        # Merge scores back to pairs DataFrame
        pairs_df = pairs_df.merge(
            pref_attach_results.rename(columns={
                'node1_id': 'entity_node_id',
                'node2_id': 'pest_node_id'
            }),
            on=['entity_node_id', 'pest_node_id'],
            how='left'
        )
        
        print("Preferential attachment scores computed successfully")
        return pairs_df
    
    def create_composite_pest_prediction_score(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a composite score for pest prediction using multiple link prediction metrics.
        
        Args:
            pairs_df: DataFrame with all link prediction scores
            
        Returns:
            DataFrame with composite prediction scores
        """
        print("Creating composite pest prediction scores...")
        
        # Fill NaN values with 0
        score_columns = ['adamic_adar_score', 'jaccard_score', 'common_neighbors_count', 'preferential_attachment_score']
        for col in score_columns:
            pairs_df[col] = pairs_df[col].fillna(0)
        
        # Normalize scores to 0-1 range
        for col in score_columns:
            if pairs_df[col].max() > 0:
                pairs_df[f'{col}_normalized'] = pairs_df[col] / pairs_df[col].max()
            else:
                pairs_df[f'{col}_normalized'] = 0
        
        # Create composite score (weighted average)
        # Adjust these weights based on domain knowledge
        weights = {
            'adamic_adar_score_normalized': 0.4,
            'jaccard_score_normalized': 0.3,
            'common_neighbors_count_normalized': 0.2,
            'preferential_attachment_score_normalized': 0.1  
        }
        
        pairs_df['composite_pest_prediction_score'] = 0
        for score_col, weight in weights.items():
            pairs_df['composite_pest_prediction_score'] += pairs_df[score_col] * weight
        
        print("Composite pest prediction scores created successfully")
        return pairs_df
    
    def generate_pest_predictions(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Generate pest predictions using link prediction heuristics.
        Args:
            threshold: Threshold for binary prediction (default: 0.5)
        Returns:
            DataFrame with pest predictions for entities
        """
        print("Generating pest predictions using link prediction heuristics...")
        
        # Get base data
        entity_pest_data, pest_nodes = self.prepare_link_prediction_data()
        
        # Create prediction pairs
        pairs_df = self.create_link_prediction_pairs(entity_pest_data, pest_nodes)
        
        # Compute all link prediction scores
        pairs_df = self.compute_adamic_adar_scores(pairs_df)
        pairs_df = self.compute_jaccard_scores(pairs_df)
        pairs_df = self.compute_common_neighbors_scores(pairs_df)
        pairs_df = self.compute_preferential_attachment_scores(pairs_df)
        
        # Create composite score
        pairs_df = self.create_composite_pest_prediction_score(pairs_df)
        
        # Focus on pest predictions (pest_value = 1)
        pest_predictions = pairs_df[pairs_df['pest_value'] == 1].copy()
        
        # Create binary predictions
        pest_predictions['predicted_pest'] = (pest_predictions['composite_pest_prediction_score'] >= threshold).astype(int)
        
        # Aggregate predictions by entity
        entity_predictions = pest_predictions.groupby(['entity_id']).agg({
            'composite_pest_prediction_score': 'max', 

            'predicted_pest': 'max',
            'adamic_adar_score': 'max',
            'jaccard_score': 'max',

            'common_neighbors_count': 'max',
            'preferential_attachment_score': 'max',
            'has_existing_relationship': 'max'
        }).reset_index()
        
        # Rename columns for clarity
        entity_predictions.columns = [
            'entity_id',
            'link_pred_pest_score',

            'link_pred_pest_binary',
            'max_adamic_adar_score',
            'max_jaccard_score',

            'max_common_neighbors',
            'max_preferential_attachment',
            'has_actual_pest'
        ]
        
        # Calculate prediction statistics
        total_entities = len(entity_predictions)
        predicted_pests = entity_predictions['link_pred_pest_binary'].sum()
        actual_pests = entity_predictions['has_actual_pest'].sum()
        
        print(f"Link prediction results:")
        print(f"  Total entities: {total_entities}")
        print(f"  Predicted pests: {predicted_pests}")
        print(f"  Actual pests: {actual_pests}")
        
        if actual_pests > 0:
            # Calculate basic accuracy metrics
            correct_predictions = ((entity_predictions['link_pred_pest_binary'] == 1) & 
                                 (entity_predictions['has_actual_pest'] == 1)).sum()
            precision = correct_predictions / predicted_pests if predicted_pests > 0 else 0
            recall = correct_predictions / actual_pests if actual_pests > 0 else 0
            
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
        
        # Save results
        output_file = 'link_prediction_pest_results.csv'
        entity_predictions.to_csv(output_file, index=False)
        print(f"Link prediction results saved to '{output_file}'")
        
        return entity_predictions
    
    def run_link_prediction_analysis(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Run complete link prediction analysis for pest prediction.
        Args:
            threshold: Threshold for binary prediction
        Returns:
            DataFrame with link prediction features and predictions
        """
        print("Starting complete link prediction analysis...")
        
        try:
            # Generate predictions
            predictions_df = self.generate_pest_predictions(threshold=threshold)
            
            print("Link prediction analysis completed successfully")
            return predictions_df
            
        except Exception as e:
            print(f"Link prediction analysis failed: {str(e)}")
            raise