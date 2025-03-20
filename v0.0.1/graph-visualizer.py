import networkx as nx
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
import random

# Configuration - same as in the main script
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

def visualize_graph():
    """
    Connect to Neo4j and visualize the knowledge graph using NetworkX.
    """
    # Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # Create a NetworkX graph
    G = nx.DiGraph()
    
    try:
        with driver.session() as session:
            # Get all nodes
            result = session.run("""
                MATCH (n)
                RETURN id(n) AS id, labels(n) AS labels, properties(n) AS props
            """)
            
            # Add nodes to the graph
            for record in result:
                node_id = record["id"]
                labels = record["labels"]
                props = record["props"]
                
                # Use the most descriptive property for the node label
                if "name" in props:
                    node_label = props["name"]
                elif "summary" in props:
                    # Truncate long summaries
                    node_label = props["summary"][:30] + "..." if len(props["summary"]) > 30 else props["summary"]
                elif "id" in props:
                    node_label = props["id"]
                else:
                    node_label = f"Node {node_id}"
                
                # Add the node with its attributes
                G.add_node(
                    node_id, 
                    label=node_label, 
                    node_type=labels[0], 
                    properties=props
                )
            
            # Get all relationships
            result = session.run("""
                MATCH (a)-[r]->(b)
                RETURN id(a) AS source, id(b) AS target, type(r) AS rel_type
            """)
            
            # Add edges to the graph
            for record in result:
                source = record["source"]
                target = record["target"]
                rel_type = record["rel_type"]
                
                G.add_edge(source, target, type=rel_type)
                
    finally:
        driver.close()
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    # Define node colors based on type
    color_map = {
        "Activity": "lightblue",
        "Application": "lightgreen",
        "Intention": "salmon",
        "Entity": "yellow"
    }
    
    # Get node positions using a layout algorithm
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    for node_type in color_map:
        nodelist = [n for n, attr in G.nodes(data=True) if attr.get("node_type") == node_type]
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=nodelist,
            node_color=color_map[node_type],
            node_size=500,
            alpha=0.8,
            label=node_type
        )
    
    # Draw edges with different colors based on relationship type
    edge_types = set(nx.get_edge_attributes(G, 'type').values())
    for edge_type in edge_types:
        edgelist = [(u, v) for u, v, attr in G.edges(data=True) if attr.get('type') == edge_type]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edgelist,
            width=1.5,
            alpha=0.7,
            edge_color=[random.random() for _ in range(3)],
            label=edge_type
        )
    
    # Add node labels
    labels = {node: G.nodes[node]["label"] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title("GraphRAG Knowledge Graph Visualization")
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("graphrag_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Graph visualization saved as 'graphrag_visualization.png'")
    print(f"Graph statistics: {len(G.nodes())} nodes, {len(G.edges())} edges")

if __name__ == "__main__":
    visualize_graph()
