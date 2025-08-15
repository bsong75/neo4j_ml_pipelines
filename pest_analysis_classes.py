    def extract_node_embeddings(self, method='node2vec'):
        """Extract node embeddings using various methods"""
        self.logger.info(f"Extracting node embeddings using {method}...")
        
        entity_df = self.gds.run_cypher('MATCH (n:Entity) RETURN id(n) as nodeId, n.id as entity_id')
        
        if method == 'node2vec':
            # Node2Vec - better for structural similarity
            result = self.gds.node2vec.stream(
                self.graph,
                embeddingDimension=64,
                walkLength=10,
                walksPerNode=10,
                randomSeed=42
            )
        elif method == 'fastRP':
            # FastRP - faster but less sophisticated
            result = self.gds.fastRP.stream(
                self.graph,
                embeddingDimension=64,
                iterationWeights=[0.8, 1, 1, 1],
                randomSeed=42
            )
        else:
            raise ValueError(f"Unknown embedding method: {method}")
        
        # Process embeddings
        embedding_df = pd.DataFrame(result['embedding'].tolist())
        embedding_df.columns = [f'{method}_dim_{i}' for i in range(len(embedding_df.columns))]
        
        final_df = pd.concat([
            result[['nodeId']],
            embedding_df
        ], axis=1)
        
        # Merge with entity data
        final_df = pd.merge(entity_df, final_df, on="nodeId")
        
        self.logger.info(f"{method} embeddings extracted. Shape: {final_df.shape}")
        return final_dfimport pandas as pd
import logging
from graphdatascience import GraphDataScience


class PestDataAnalyzer:
    def __init__(self, csv_path, neo4j_uri, neo4j_user, neo4j_password, log_level=logging.INFO):
        self.csv_path = csv_path
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.gds = None
        self.df = None
        self.graph = None
        
        # Setup logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        
        # Create console handler if no handlers exist
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def connect(self):
        """Connect to Neo4j database"""
        self.logger.info("Connecting to Neo4j database...")
        self.gds = GraphDataScience(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        self.logger.info("Successfully connected to Neo4j")
    
    def load_data(self):
        """Load CSV data"""
        self.logger.info(f"Loading data from {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        self.logger.info(f"Loaded {len(self.df)} rows")
        unique_count = self.df['ENTY_ID'].nunique()
        self.logger.info(f"Number of distinct ENTY_ID: {unique_count}")
    
    def create_constraints(self):
        """Create database constraints"""
        self.logger.info("Creating database constraints...")
        constraints = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT country_code_unique IF NOT EXISTS FOR (c:Country) REQUIRE c.code IS UNIQUE", 
            "CREATE CONSTRAINT month_name_unique IF NOT EXISTS FOR (m:Month) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT target_value_unique IF NOT EXISTS FOR (t:TargetProxy) REQUIRE t.value IS UNIQUE"
        ]
        
        for constraint in constraints:
            try:
                self.gds.run_cypher(constraint)
                self.logger.debug(f"Created constraint: {constraint[:50]}...")
            except Exception as e:
                self.logger.debug(f"Constraint already exists or failed: {str(e)[:100]}...")
        
        self.logger.info("Database constraints setup complete")
    
    def clear_database(self):
        """Clear existing data"""
        self.logger.warning("Clearing entire database...")
        self.gds.run_cypher("MATCH (n) DETACH DELETE n")
        self.logger.info("Database cleared")
    
    def create_nodes(self):
        """Create all nodes in the graph"""
        self.logger.info("Creating graph nodes...")
        
        # Define node types and their data
        node_configs = [
            {
                'label': 'Country',
                'property': 'code',
                'data': self.df['CTRY_CODE'].unique(),
                'description': 'country'
            },
            {
                'label': 'Month', 
                'property': 'name',
                'data': self.df['MONTH'].unique(),
                'description': 'month'
            },
            {
                'label': 'Entity',
                'property': 'id', 
                'data': self.df['ENTY_ID'].unique(),
                'description': 'entity'
            }
        ]
        
        # Create nodes using loop
        for config in node_configs:
            self.logger.debug(f"Creating {len(config['data'])} {config['description']} nodes")
            self.gds.run_cypher(f"""
            UNWIND $data as item
            MERGE (n:{config['label']} {{{config['property']}: item}})
            """, {'data': config['data'].tolist()})
        
        # TargetProxy nodes (special case with labels)
        targets = self.df['TARGET_PROXY'].unique()
        self.logger.debug(f"Creating {len(targets)} target proxy nodes")
        target_data = [{'value': int(t), 'label': 'no_pest' if t == 0 else 'pest'} for t in targets]
        self.gds.run_cypher("""
        UNWIND $targets as target
        MERGE (t:TargetProxy {value: target.value, label: target.label})
        """, {'targets': target_data})
        
        # CountryMonth composite nodes
        self.logger.debug("Creating country-month composite nodes")
        country_month_data = []
        for _, row in self.df.iterrows():
            ctry_month_id = f"{row['CTRY_CODE']}_{row['MONTH']}"
            country_month_data.append({
                'id': ctry_month_id,
                'country': row['CTRY_CODE'], 
                'month': row['MONTH']
            })
        
        # Remove duplicates
        unique_cm_data = {item['id']: item for item in country_month_data}.values()
        self.gds.run_cypher("""
        UNWIND $data as cm
        MERGE (n:CountryMonth {id: cm.id, country: cm.country, month: cm.month})
        """, {'data': list(unique_cm_data)})
        
        self.logger.info("All nodes created successfully")
    
    def create_inspections_and_relationships(self):
        """Create inspection nodes and all relationships in one operation"""
        self.logger.info(f"Creating {len(self.df)} inspection nodes and relationships...")
        
        # Prepare batch data
        inspection_data = []
        for _, row in self.df.iterrows():
            ctry_month_id = f"{row['CTRY_CODE']}_{row['MONTH']}"
            inspection_data.append({
                'country_code': row['CTRY_CODE'],
                'month': row['MONTH'],
                'target_proxy': int(row['TARGET_PROXY']),
                'entity_id': row['ENTY_ID'],
                'ctry_month_id': ctry_month_id,
                'exams_30d': int(row['ENTY_EXAMS_30D']),
                'pests_30d': int(row['ENTY_PESTS_30D']),
                'exams_90d': int(row['ENTY_EXAMS_90D']),
                'pests_90d': int(row['ENTY_PESTS_90D']),
                'exams_1yr': int(row['ENTY_EXAMS_1YR']),
                'pests_1yr': int(row['ENTY_PESTS_1YR']),
                'has_pest': row['TARGET_PROXY'] == 1
            })
        
        # Create inspections and relationships in one query
        self.gds.run_cypher("""
        UNWIND $inspections as inspection
        CREATE (i:Inspection {
            country_code: inspection.country_code,
            month: inspection.month,
            target_proxy: inspection.target_proxy,
            entity_id: inspection.entity_id,
            exams_30d: inspection.exams_30d,
            pests_30d: inspection.pests_30d,
            exams_90d: inspection.exams_90d,
            pests_90d: inspection.pests_90d,
            exams_1yr: inspection.exams_1yr,
            pests_1yr: inspection.pests_1yr,
            has_pest: inspection.has_pest
        })
        WITH i, inspection
        MATCH (c:Country {code: inspection.country_code}),
              (m:Month {name: inspection.month}),
              (e:Entity {id: inspection.entity_id}),
              (t:TargetProxy {value: inspection.target_proxy}),
              (cm:CountryMonth {id: inspection.ctry_month_id})
        CREATE (i)-[:IN_COUNTRY]->(c),
               (i)-[:IN_MONTH]->(m),
               (i)-[:FOR_ENTITY]->(e),
               (i)-[:HAS_TARGET]->(t)
        MERGE (e)-[:SHIPPED_IN]->(m)
        MERGE (e)-[:IS_FROM]->(c)
        MERGE (e)-[:HAS_WEATHER]->(cm)
        MERGE (e)-[:HAS_INSPECTION_RESULT]->(t)
        """, {'inspections': inspection_data})
        
        self.logger.info("Inspections and relationships created successfully")
    
    def get_stats(self):
        """Get database statistics"""
        self.logger.info("Gathering database statistics...")
        nodes = self.gds.run_cypher("MATCH (n) RETURN count(n) as count").iloc[0]['count']
        rels = self.gds.run_cypher("MATCH ()-[r]->() RETURN count(r) as count").iloc[0]['count']
        self.logger.info(f"Database contains {nodes} nodes and {rels} relationships")
        
        self.logger.info("Relationship type breakdown:")
        rel_stats = self.gds.run_cypher("""
                MATCH ()-[r]->()
                RETURN type(r) as relationship_type, count(r) as count
                ORDER BY count DESC
                """)
        for _, row in rel_stats.iterrows():
            self.logger.info(f"  {row['relationship_type']}: {row['count']}")
    
    def create_projection(self):
        """Create graph projection"""
        self.logger.info("Creating graph projection...")
        try:
            self.gds.graph.drop("pest_graph")
            self.logger.debug("Dropped existing graph")
        except Exception:
            self.logger.debug("No existing graph to drop")

        self.graph, _ = self.gds.graph.project("pest_graph", 
                            ["Country", "Month", "Entity", "TargetProxy", "CountryMonth"],
                            ['SHIPPED_IN', 'IS_FROM', 'HAS_WEATHER', 'HAS_INSPECTION_RESULT'])
        
        self.logger.info("Graph projection created successfully")
        self.logger.info(f"Graph name: {self.graph.name()}")
        self.logger.info(f"Node count: {self.graph.node_count()}")
        self.logger.info(f"Relationship count: {self.graph.relationship_count()}")
        self.logger.info(f"Density: {self.graph.density()}")
        
        wcc_components = self.gds.wcc.stream(self.graph).componentId.nunique()
        self.logger.info(f"WCC Components: {wcc_components}")
        
        self.logger.debug(f"Degree distribution:\n{self.graph.degree_distribution()}")
    
    def extract_structural_features(self):
        """Extract comprehensive structural features from the graph"""
        self.logger.info("Extracting structural features from graph...")
        
        # Start with entity base
        entity_df = self.gds.run_cypher('MATCH (n:Entity) RETURN id(n) as nodeId, n.id as entity_id')
        
        # Centrality measures
        centrality_algorithms = ['pageRank', 'betweenness', 'closeness', 'eigenvector', 'degree']
        self.logger.info(f"Computing centrality measures: {', '.join(centrality_algorithms)}")
        
        for algo in centrality_algorithms:
            self.logger.debug(f"Computing {algo} centrality...")
            method = getattr(self.gds, algo)
            result = method.stream(self.graph).rename(columns={'score': algo})
            entity_df = pd.merge(entity_df, result, on="nodeId")
        
        # Additional structural features
        self.logger.info("Computing additional structural features...")
        
        # Triangle count and clustering coefficient
        triangles = self.gds.triangleCount.stream(self.graph)
        entity_df = pd.merge(entity_df, triangles[['nodeId', 'triangleCount']], on="nodeId")
        
        # Local clustering coefficient
        clustering = self.gds.localClusteringCoefficient.stream(self.graph)
        entity_df = pd.merge(entity_df, clustering[['nodeId', 'localClusteringCoefficient']], on="nodeId")
        
        # Community detection (Louvain)
        communities = self.gds.louvain.stream(self.graph)
        entity_df = pd.merge(entity_df, communities[['nodeId', 'communityId']], on="nodeId")
        
        # Article Rank (variant of PageRank)
        article_rank = self.gds.articleRank.stream(self.graph)
        entity_df = pd.merge(entity_df, article_rank[['nodeId', 'score']].rename(columns={'score': 'articleRank'}), on="nodeId")
        
        # K-Core decomposition
        kcore = self.gds.kcore.stream(self.graph)
        entity_df = pd.merge(entity_df, kcore[['nodeId', 'coreValue']], on="nodeId")
        
        self.logger.info(f"Structural features extracted. Final shape: {entity_df.shape}")
        return entity_df
    
    def run_graphsage(self):
        """Run GraphSAGE for pest prediction"""
        self.logger.info("Starting GraphSAGE pest prediction pipeline")
        
        # Add node properties
        self.logger.info("Adding node properties for GraphSAGE...")
        self.gds.run_cypher(""" MATCH (n) SET n.degree = size([(n)--() | 1]) """)
        self.gds.run_cypher("""  MATCH (e:Entity) SET e.entity_degree = size([(e)--() | 1]) """)
        self.gds.run_cypher("""  MATCH (t:TargetProxy) SET t.pest_value = t.value """)
        self.logger.debug("Node properties added successfully")
        
        # Create prediction projection
        self.logger.info("Creating prediction projection...")
        try:
            self.gds.graph.drop("pest_prediction")
            self.logger.debug("Dropped existing prediction graph")
        except:
            self.logger.debug("No existing prediction graph to drop")
            
        G_pred, _ = self.gds.graph.project(
            "pest_prediction",
            ["Entity", "TargetProxy", "Country", "Month", "CountryMonth"],
            {
                'HAS_INSPECTION_RESULT': {'orientation': 'NATURAL'},
                'SHIPPED_IN': {'orientation': 'NATURAL'},
                'IS_FROM': {'orientation': 'NATURAL'}, 
                'HAS_WEATHER': {'orientation': 'NATURAL'}
            },
            nodeProperties=["degree", "pest_value", "entity_degree"]
        )
        self.logger.info("Prediction projection created successfully")
        
        # Train model
        self.logger.info("Training GraphSAGE model...")
        train_result = self.gds.beta.graphSage.train(
            G_pred,
            modelName="pest_predictor",
            featureProperties=["degree"],
            projectedFeatureDimension=64,
            randomSeed=42,
            epochs=20,
            batchSize=256,
            learningRate=0.01,
            sampleSizes=[25, 10]
        )
        
        self.logger.info("GraphSAGE model training completed")
        self.logger.debug(f"Training metrics: {train_result}")
        
        # Generate embeddings
        self.logger.info("Generating entity embeddings...")
        entity_embeddings = self.gds.run_cypher("""
            CALL gds.beta.graphSage.stream('pest_prediction', 
                                            {modelName: 'pest_predictor' }) 
            YIELD nodeId, embedding
            WITH nodeId, embedding
            MATCH (n) WHERE id(n) = nodeId AND 'Entity' IN labels(n)
            RETURN nodeId, n.id as entity_id, embedding
            """)
        
        # Convert to DataFrame
        self.logger.debug("Converting embeddings to DataFrame format...")
        embedding_df = pd.DataFrame(entity_embeddings['embedding'].tolist())
        embedding_df.columns = [f'graphsage_dim_{i}' for i in range(len(embedding_df.columns))]
        
        final_df = pd.concat([
            entity_embeddings[['nodeId', 'entity_id']],
            embedding_df
        ], axis=1)
        
        # Get pest labels
        self.logger.debug("Retrieving pest labels for entities...")
        entity_labels = self.gds.run_cypher("""
            MATCH (e:Entity)-[:HAS_INSPECTION_RESULT]->(t:TargetProxy)
            RETURN e.id as entity_id, 
                max(t.value) as has_pest_ever,
                avg(t.value) as pest_rate
            """)
        
        final_df = pd.merge(final_df, entity_labels, on='entity_id', how='left')
        
        pest_count = final_df['has_pest_ever'].sum()
        total_entities = len(final_df)
        self.logger.info(f"GraphSAGE analysis complete: {pest_count}/{total_entities} entities have pest history")
        self.logger.info(f"Final DataFrame shape: {final_df.shape}")
        
        output_file = 'graphsage_entity_features.csv'
        final_df.to_csv(output_file, index=False)
        self.logger.info(f"GraphSAGE features saved to '{output_file}'")
        
        return final_dfdegree", "pest_value", "entity_degree"]
        )
        
        # Train model
        print("Step 3: Training GraphSAGE model...")
        train_result = self.gds.beta.graphSage.train(
            G_pred,
            modelName="pest_predictor",
            featureProperties=["degree"],
            projectedFeatureDimension=64,
            randomSeed=42,
            epochs=20,
            batchSize=256,
            learningRate=0.01,
            sampleSizes=[25, 10]
        )
        
        print("Training metrics:")
        print(train_result)
        
        # Generate embeddings
        print("Step 4: Generating node embeddings...")
        entity_embeddings = self.gds.run_cypher("""
            CALL gds.beta.graphSage.stream('pest_prediction', 
                                            {modelName: 'pest_predictor' }) 
            YIELD nodeId, embedding
            WITH nodeId, embedding
            MATCH (n) WHERE id(n) = nodeId AND 'Entity' IN labels(n)
            RETURN nodeId, n.id as entity_id, embedding
            """)
        
        # Convert to DataFrame
        embedding_df = pd.DataFrame(entity_embeddings['embedding'].tolist())
        embedding_df.columns = [f'graphsage_dim_{i}' for i in range(len(embedding_df.columns))]
        
        final_df = pd.concat([
            entity_embeddings[['nodeId', 'entity_id']],
            embedding_df
        ], axis=1)
        
        # Get pest labels
        entity_labels = self.gds.run_cypher("""
            MATCH (e:Entity)-[:HAS_INSPECTION_RESULT]->(t:TargetProxy)
            RETURN e.id as entity_id, 
                max(t.value) as has_pest_ever,
                avg(t.value) as pest_rate
            """)
        
        final_df = pd.merge(final_df, entity_labels, on='entity_id', how='left')
        
        print("Final DataFrame with labels:")
        print(final_df.head())
        print(f"Entities with pest history: {final_df['has_pest_ever'].sum()}")
        print(f"Total entities: {len(final_df)}")
        
        final_df.to_csv('graphsage_entity_features.csv', index=False)
        print("Saved GraphSAGE features to 'graphsage_entity_features.csv'")
        
        return final_df
    
    def cleanup(self):
        """Clean up models and graphs"""
        self.logger.info("Cleaning up resources...")
        try:
            self.gds.run_cypher("CALL gds.model.drop('pest_predictor')")
            self.logger.debug("Model 'pest_predictor' dropped successfully")
        except Exception as e:
            self.logger.debug(f"Model cleanup: {str(e)}")
        
        try:
            self.gds.graph.drop("pest_prediction") 
            self.logger.debug("Graph 'pest_prediction' dropped successfully")
        except Exception as e:
            self.logger.debug(f"Graph cleanup: {str(e)}")
        
        self.logger.info("Resource cleanup completed")
    
    def close(self):
        """Close database connection"""
        if self.gds:
            try:
                # Clean up any remaining graphs
                try:
                    self.gds.graph.drop("pest_graph")
                except:
                    pass
                try:
                    self.gds.graph.drop("pest_prediction")
                except:
                    pass
                
                # Clean up any remaining models
                try:
                    self.gds.run_cypher("CALL gds.model.drop('pest_predictor')")
                except:
                    pass
                
                # Close the connection
                self.gds.close()
                self.logger.info("Database connection closed successfully")
                
                # Force garbage collection
                import gc
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"Error during cleanup: {e}")
            finally:
                self.gds = None
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        self.logger.info("Starting full pest data analysis pipeline")
        
        try:
            # Setup
            self.connect()
            self.load_data()
            self.create_constraints()
            self.clear_database()
            
            # Build graph
            self.create_nodes()
            self.create_inspections()
            self.get_stats()
            
            # Analysis
            self.create_projection()
            df = self.run_fastrp()
            df = self.run_centrality_algorithms(df)
            
            # GraphSAGE prediction
            self.cleanup()
            final_df = self.run_graphsage()
            self.cleanup()
            
            self.logger.info("Full analysis pipeline completed successfully")
            self.logger.info(f"Final result DataFrame shape: {final_df.shape}")
            
            return final_df
            
        except Exception as e:
            self.logger.error(f"Analysis pipeline failed: {str(e)}")
            raise
        finally:
            # Always close connection
            self.close()


def main():
    # Configuration
    CSV_FILE_PATH = "pest_data.csv"
    NEO4J_URI = "bolt://neo4j:7687"  
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pest_analysis.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting pest analysis application")
    
    analyzer = None
    try:
        # Create analyzer and run analysis
        analyzer = PestDataAnalyzer(CSV_FILE_PATH, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        final_df = analyzer.run_full_analysis()
        
        logger.info("Analysis completed successfully")
        return final_df
        
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        raise
    finally:
        # Ensure cleanup always happens
        if analyzer:
            analyzer.close()
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    main()