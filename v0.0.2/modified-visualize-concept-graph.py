import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from neo4j import GraphDatabase
import os
import sys
from pyvis.network import Network
import json
from datetime import datetime

# Configuration
NEO4J_URI = "bolt://localhost:7687"  # Neo4j connection URI
NEO4J_USER = "neo4j"                 # Neo4j username
NEO4J_PASSWORD = "password"          # Neo4j password

class ConceptGraphVisualizer:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_uri, auth=(neo4j_user, neo4j_password)
        )
    
    def __del__(self):
        if hasattr(self, 'neo4j_driver'):
            self.neo4j_driver.close()
    
    def create_concept_network(self):
        """
        Create a NetworkX graph of concepts and their relationships.
        """
        G = nx.DiGraph()
        
        with self.neo4j_driver.session() as session:
            # Add all concepts as nodes
            result = session.run("""
                MATCH (c:Concept)
                RETURN c.name AS name
            """)
            
            for record in result:
                G.add_node(record['name'], type='Concept')
            
            # Add all topics as nodes
            result = session.run("""
                MATCH (t:Topic)
                RETURN t.name AS name
            """)
            
            for record in result:
                G.add_node(record['name'], type='Topic')
            
            # Add direct relationships between concepts
            result = session.run("""
                MATCH (c1:Concept)-[r]->(c2:Concept)
                RETURN c1.name AS source, c2.name AS target, type(r) AS relation
            """)
            
            for record in result:
                G.add_edge(record['source'], record['target'], 
                           relation=record['relation'], type='direct')
            
            # Add edges between concepts that are both connected to the same Topic
            result = session.run("""
                MATCH (c1:Concept)<-[:INVOLVES_CONCEPT]-(o:Observation)-[:INVOLVES_TOPIC]->(t:Topic)
                MATCH (c2:Concept)<-[:INVOLVES_CONCEPT]-(o)
                WHERE c1 <> c2
                WITH c1, c2, t, count(o) AS weight
                RETURN c1.name AS source, c2.name AS target, t.name AS topic,
                       weight
                ORDER BY weight DESC
            """)
            
            for record in result:
                if not G.has_edge(record['source'], record['target']):
                    G.add_edge(record['source'], record['target'], 
                               relation=f"CO_OCCURS_IN_{record['topic']}", 
                               weight=record['weight'],
                               type='co-occurrence')
        
        return G
    
    def visualize_concepts_matplotlib(self, output_file=None):
        """
        Create a matplotlib visualization of the concept graph.
        """
        G = self.create_concept_network()
        
        # Define node colors by type
        color_map = {'Concept': 'skyblue', 'Topic': 'lightgreen'}
        node_colors = [color_map[G.nodes[n]['type']] for n in G.nodes()]
        
        # Define edge colors by type
        edge_colors = []
        for u, v, data in G.edges(data=True):
            if data.get('type') == 'direct':
                edge_colors.append('red')
            else:
                edge_colors.append('gray')
        
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.5, edge_color=edge_colors, alpha=0.7, 
                              arrowsize=15, connectionstyle='arc3,rad=0.1')
        
        # Draw labels with a slight offset
        label_pos = {k: (v[0], v[1] + 0.04) for k, v in pos.items()}
        nx.draw_networkx_labels(G, label_pos, font_size=10, font_weight='bold')
        
        # Add a legend
        plt.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=15, label='Concept'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15, label='Topic'),
            plt.Line2D([0], [0], color='red', lw=4, label='Direct Relationship'),
            plt.Line2D([0], [0], color='gray', lw=4, label='Co-occurrence')
        ], loc='upper right')
        
        plt.title('Concept Knowledge Graph', fontsize=20)
        plt.axis('off')
        
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            print(f"Saved visualization to {output_file}")
        else:
            plt.show()
    
    def visualize_concepts_interactive(self, output_file=None):
        """
        Create an interactive HTML visualization of the concept graph using pyvis.
        Falls back to networkx if pyvis fails.
        """
        G = self.create_concept_network()
        
        try:
            # Try to use pyvis for interactive visualization
            from pyvis.network import Network
            
            # Create a Network instance
            net = Network(height='800px', width='100%', notebook=False, directed=True)
            net.barnes_hut(gravity=-10000, central_gravity=0.3, spring_length=200)
            
            # Add nodes to the network
            for node, attr in G.nodes(data=True):
                color = 'skyblue' if attr['type'] == 'Concept' else 'lightgreen'
                size = 25
                net.add_node(node, label=node, title=f"{attr['type']}: {node}", 
                             color=color, size=size)
            
            # Add edges to the network
            for u, v, attr in G.edges(data=True):
                color = 'red' if attr.get('type') == 'direct' else '#aaaaaa'
                title = attr.get('relation', 'Related')
                weight = attr.get('weight', 1)
                # Scale width by weight but keep it within reasonable bounds
                width = min(max(weight, 1), 10)
                net.add_edge(u, v, title=title, color=color, width=width)
            
            # Set network options
            net.set_options("""
            {
              "physics": {
                "forceAtlas2Based": {
                  "gravitationalConstant": -50,
                  "centralGravity": 0.01,
                  "springLength": 100,
                  "springConstant": 0.08
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {
                  "enabled": true,
                  "iterations": 1000
                }
              },
              "edges": {
                "smooth": {
                  "type": "continuous",
                  "forceDirection": "none"
                }
              },
              "interaction": {
                "navigationButtons": true,
                "keyboard": true
              }
            }
            """)
            
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"concept_graph_{timestamp}.html"
            
            # Try to render with pyvis, with error handling
            try:
                net.show(output_file)
                print(f"Interactive visualization saved to {output_file}")
                return
            except Exception as e:
                print(f"Pyvis visualization failed: {e}")
                print("Falling back to manual HTML generation...")
                # Fall back to manual HTML generation
        
        except Exception as e:
            print(f"Error using pyvis: {e}")
            print("Falling back to custom HTML visualization...")
        
        # Fallback: Create a simple HTML file with D3.js directly
        self._create_d3_visualization(G, output_file)
    
    def _create_d3_visualization(self, G, output_file=None):
        """
        Fallback method to create a D3.js visualization directly without pyvis.
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"concept_graph_{timestamp}.html"
        
        # Convert networkx graph to D3.js compatible format
        nodes = []
        for node, attr in G.nodes(data=True):
            group = 1 if attr['type'] == 'Concept' else 2
            nodes.append({"id": node, "group": group, "label": node})
        
        links = []
        for u, v, attr in G.edges(data=True):
            # Find node indices
            source_idx = next(i for i, n in enumerate(nodes) if n["id"] == u)
            target_idx = next(i for i, n in enumerate(nodes) if n["id"] == v)
            
            value = attr.get('weight', 1)
            relation = attr.get('relation', 'related')
            links.append({"source": source_idx, "target": target_idx, "value": value, "label": relation})
        
        # Create a simple HTML template with D3.js
        html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Concept Graph Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body { margin: 0; font-family: Arial, sans-serif; }
        .links line { stroke: #999; stroke-opacity: 0.6; }
        .nodes circle { stroke: #fff; stroke-width: 1.5px; }
        .concept { fill: #6baed6; }
        .topic { fill: #74c476; }
        .label { font-size: 10px; pointer-events: none; }
        .tooltip {
            position: absolute;
            text-align: center;
            padding: 8px;
            font: 12px sans-serif;
            background: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 8px;
            pointer-events: none;
            opacity: 0;
        }
    </style>
</head>
<body>
    <h1 style="text-align: center; margin: 20px 0;">Concept Knowledge Graph</h1>
    <div style="text-align: center; margin-bottom: 10px;">
        <span style="display: inline-block; margin-right: 15px;"><span style="display: inline-block; width: 15px; height: 15px; background-color: #6baed6; margin-right: 5px;"></span> Concept</span>
        <span style="display: inline-block;"><span style="display: inline-block; width: 15px; height: 15px; background-color: #74c476; margin-right: 5px;"></span> Topic</span>
    </div>
    <svg width="960" height="700"></svg>
    <script>
    // Graph data
    const graph = {
        "nodes": %s,
        "links": %s
    };

    // Create a force simulation
    const svg = d3.select("svg"),
          width = +svg.attr("width"),
          height = +svg.attr("height");
          
    const tooltip = d3.select("body").append("div")
        .attr("class", "tooltip");
    
    const simulation = d3.forceSimulation()
        .force("link", d3.forceLink().id(d => d.id).distance(150))
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter(width / 2, height / 2));

    // Create links
    const link = svg.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(graph.links)
        .enter().append("line")
        .attr("stroke-width", d => Math.sqrt(d.value));

    // Create nodes
    const node = svg.append("g")
        .attr("class", "nodes")
        .selectAll("circle")
        .data(graph.nodes)
        .enter().append("circle")
        .attr("r", 8)
        .attr("class", d => d.group === 1 ? "concept" : "topic")
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));
    
    // Add labels to nodes
    const text = svg.append("g")
        .attr("class", "labels")
        .selectAll("text")
        .data(graph.nodes)
        .enter().append("text")
        .attr("class", "label")
        .attr("dx", 12)
        .attr("dy", ".35em")
        .text(d => d.label);

    // Add title for tooltips
    node.on("mouseover", function(event, d) {
        tooltip.transition()
            .duration(200)
            .style("opacity", .9);
        tooltip.html(d.label)
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 28) + "px");
    })
    .on("mouseout", function() {
        tooltip.transition()
            .duration(500)
            .style("opacity", 0);
    });

    // Set up simulation
    simulation
        .nodes(graph.nodes)
        .on("tick", ticked);

    simulation.force("link")
        .links(graph.links);

    function ticked() {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("cx", d => d.x = Math.max(10, Math.min(width - 10, d.x)))
            .attr("cy", d => d.y = Math.max(10, Math.min(height - 10, d.y)));
            
        text
            .attr("x", d => d.x)
            .attr("y", d => d.y);
    }

    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
    </script>
</body>
</html>
""" % (json.dumps(nodes), json.dumps(links))
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(html_template)
        
        print(f"Fallback D3.js visualization saved to {output_file}")
        return output_file
    
    def generate_topic_hierarchy(self, output_file=None):
        """
        Generate a JSON representation of the topic hierarchy.
        """
        with self.neo4j_driver.session() as session:
            # Get the top-level work contexts
            result = session.run("""
                MATCH (w:WorkContext)<-[:PART_OF]-(o:Observation)
                WITH w, count(o) AS obs_count
                ORDER BY obs_count DESC
                RETURN w.name AS context, obs_count
            """)
            
            contexts = [record['context'] for record in result]
            
            # Build a hierarchical structure
            hierarchy = {"name": "Knowledge Graph", "children": []}
            
            for context in contexts:
                context_node = {"name": context, "children": []}
                
                # Get topics for this context
                result = session.run("""
                    MATCH (w:WorkContext {name: $context})<-[:PART_OF]-(o:Observation)-[:INVOLVES_TOPIC]->(t:Topic)
                    WITH t, count(o) AS topic_weight
                    RETURN t.name AS topic, topic_weight
                    ORDER BY topic_weight DESC
                """, context=context)
                
                for record in result:
                    topic_node = {"name": record['topic'], "children": []}
                    
                    # Get concepts for this topic
                    result2 = session.run("""
                        MATCH (t:Topic {name: $topic})<-[:INVOLVES_TOPIC]-(o:Observation)-[:INVOLVES_CONCEPT]->(c:Concept)
                        RETURN c.name AS concept, count(o) AS concept_weight
                        ORDER BY concept_weight DESC
                        LIMIT 10
                    """, topic=record['topic'])
                    
                    for record2 in result2:
                        concept_node = {"name": record2['concept'], "value": record2['concept_weight']}
                        topic_node["children"].append(concept_node)
                    
                    if topic_node["children"]:
                        context_node["children"].append(topic_node)
                
                if context_node["children"]:
                    hierarchy["children"].append(context_node)
                    
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(hierarchy, f, indent=2)
                print(f"Topic hierarchy saved to {output_file}")
            
            return hierarchy
            
    def generate_top_concepts_report(self, output_file=None):
        """
        Generate a text report of the most important concepts and their relationships.
        """
        with self.neo4j_driver.session() as session:
            report = []
            report.append("# Concept Graph Analysis Report\n")
            report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Get top work contexts
            report.append("## Main Work Contexts\n")
            result = session.run("""
                MATCH (w:WorkContext)<-[:PART_OF]-(o:Observation)
                WITH w, COUNT(o) AS count
                RETURN w.name AS context, count
                ORDER BY count DESC
                LIMIT 5
            """)
            
            for record in result:
                report.append(f"- {record['context']} ({record['count']} observations)")
            
            # Get top topics
            report.append("\n## Key Topics\n")
            result = session.run("""
                MATCH (t:Topic)<-[:INVOLVES_TOPIC]-(o:Observation)
                WITH t, COUNT(o) AS count
                RETURN t.name AS topic, count
                ORDER BY count DESC
                LIMIT 10
            """)
            
            for record in result:
                report.append(f"- {record['topic']} ({record['count']} occurrences)")
            
            # Get top concepts
            report.append("\n## Central Concepts\n")
            result = session.run("""
                MATCH (c:Concept)
                OPTIONAL MATCH (c)<-[:INVOLVES_CONCEPT]-(o:Observation)
                WITH c, COUNT(o) AS mentions
                RETURN c.name AS concept, mentions
                ORDER BY mentions DESC
                LIMIT 10
            """)
            
            for record in result:
                report.append(f"- {record['concept']} ({record['concept']} mentions)")
            
            # Get concept relationships
            report.append("\n## Key Concept Relationships\n")
            result = session.run("""
                MATCH (c1:Concept)-[r]->(c2:Concept)
                RETURN c1.name AS source, type(r) AS relationship, c2.name AS target
                LIMIT 15
            """)
            
            for record in result:
                rel_type = record['relationship'].replace('_', ' ').lower()
                report.append(f"- {record['source']} {rel_type} {record['target']}")
            
            # Get concept clusters
            report.append("\n## Concept Clusters\n")
            result = session.run("""
                MATCH (t:Topic)<-[:INVOLVES_TOPIC]-(o:Observation)-[:INVOLVES_CONCEPT]->(c:Concept)
                WITH t, collect(DISTINCT c.name) AS concepts
                WHERE size(concepts) > 2
                RETURN t.name AS topic, concepts
                ORDER BY size(concepts) DESC
                LIMIT 5
            """)
            
            for record in result:
                concepts_list = ', '.join(record['concepts'][:5])
                if len(record['concepts']) > 5:
                    concepts_list += f", and {len(record['concepts']) - 5} more"
                report.append(f"- Topic: {record['topic']}")
                report.append(f"  - Concepts: {concepts_list}")
            
            report_text = '\n'.join(report)
            
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(report_text)
                print(f"Report saved to {output_file}")
            
            return report_text

def main():
    """
    Main function to run the concept graph visualization.
    """
    try:
        visualizer = ConceptGraphVisualizer(
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD
        )
        
        # Create output directory if it doesn't exist
        output_dir = "concept_graph_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create static visualization
        try:
            visualizer.visualize_concepts_matplotlib(
                output_file=f"{output_dir}/concept_graph_{timestamp}.png"
            )
        except Exception as e:
            print(f"Error creating static visualization: {e}")
        
        # Create interactive visualization
        try:
            visualizer.visualize_concepts_interactive(
                output_file=f"{output_dir}/concept_graph_interactive_{timestamp}.html"
            )
        except Exception as e:
            print(f"Error creating interactive visualization: {e}")
        
        # Generate topic hierarchy
        try:
            visualizer.generate_topic_hierarchy(
                output_file=f"{output_dir}/topic_hierarchy_{timestamp}.json"
            )
        except Exception as e:
            print(f"Error generating topic hierarchy: {e}")
        
        # Generate report
        try:
            visualizer.generate_top_concepts_report(
                output_file=f"{output_dir}/concept_report_{timestamp}.md"
            )
        except Exception as e:
            print(f"Error generating concept report: {e}")
        
        print(f"All completed visualizations and reports have been saved to the '{output_dir}' directory.")
    
    except Exception as e:
        print(f"Error in visualization process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()