
###  Neo4j Pipeline 
Node Classification - Most direct for "classify entities as pest/no-pest"
Link Prediction - What you're already trying
Node Regression - If you want continuous probability scores
GraphSAGE - Most advanced, if others don't work well


### Link Prediction explanation
1. Adamic-Adar Index

What it measures: Weighted similarity based on shared neighbors, giving more weight to rare connections
Formula: Σ(1/log(degree(shared_neighbor))) for all shared neighbors
Intuition: "Sharing rare connections is more meaningful than sharing common ones"
Example: Two entities both shipping from a small, uncommon port carry more predictive weight than both shipping from a major hub like Los Angeles

2. Jaccard Coefficient

What it measures: Proportion of shared connections relative to total unique connections
Formula: |intersection(neighbors)| / |union(neighbors)|
Intuition: "How much overlap exists in their connection patterns?"
Example: Entity shipping from Mexico in July with flowers vs pest case from Mexico in July with fruits = high Jaccard score due to shared country and month

3. Common Neighbors

What it measures: Simple count of shared connections between entity and pest cases
Formula: |intersection(neighbors)|
Intuition: "More shared connections = more similar contexts"
Example: Entity and pest case both connected to Mexico, July, and Port_TX = 3 common neighbors

4. Preferential Attachment

What it measures: Likelihood of connection based on node popularity/activity level
Formula: degree(Entity) × degree(PestNode)
Intuition: "Popular entities are more likely to connect to popular outcomes"
Example: High-volume shipping entities might be more likely to encounter pests simply due to volume and frequency of inspections