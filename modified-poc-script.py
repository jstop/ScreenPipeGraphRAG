import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple
import requests
from neo4j import GraphDatabase
import langchain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# Configuration
SQLITE_DB_PATH = "/Users/jstein/.screenpipe/db.sqlite"  # Path to your SQLite database
NEO4J_URI = "bolt://localhost:7687"  # Neo4j connection URI
NEO4J_USER = "neo4j"                 # Neo4j username
NEO4J_PASSWORD = "password"          # Neo4j password
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Your OpenAI API key

# Setup the LangChain OpenAI LLM client
def setup_langchain_client():
    """
    Set up and return a LangChain ChatOpenAI client configured for GPT-4o.
    """
    # Set OpenAI API key
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    return llm

# LLM Client setup using LangChain and OpenAI's GPT-4o
def analyze_with_llm(text: str, app_name: str, window_name: str, timestamp: str) -> Dict[str, Any]:
    """
    Use LangChain with GPT-4o to analyze OCR text and extract topics, concepts, and relationships.
    Returns a dictionary with summary, topics, concepts, and relationships.
    """
    # Define output schemas for structured parsing
    summary_schema = ResponseSchema(
        name="summary",
        description="A brief summary of what the user was doing based on the OCR text"
    )

    topics_schema = ResponseSchema(
        name="topics",
        description="List of main topics or subjects being worked on (e.g., 'Knowledge Graphs', 'Data Analysis', 'Machine Learning')"
    )

    concepts_schema = ResponseSchema(
        name="concepts",
        description="List of key concepts mentioned or implicitly present (e.g., 'Graph Database', 'Entity Extraction', 'Semantic Networks')"
    )

    relationships_schema = ResponseSchema(
        name="relationships",
        description="List of relationships between concepts in the format: [concept1, relationship, concept2]"
    )

    work_context_schema = ResponseSchema(
        name="work_context",
        description="The higher-level project or work context (e.g., 'Database Migration', 'AI Research', 'Document Analysis System')"
    )

    # Create the parser with the schemas
    parser = StructuredOutputParser.from_response_schemas([
        summary_schema, topics_schema, concepts_schema, relationships_schema, work_context_schema
    ])

    # Get the format instructions
    format_instructions = parser.get_format_instructions()

    # Create the prompt template
    template = """
    You are an AI specialized in analyzing OCR text data from user screens to understand the conceptual and topical nature of the work being done.

    Analyze the following OCR text captured from a user's screen:

    App: {app_name}
    Window: {window_name}
    Time: {timestamp}

    Text:
    {text}

    Your task is to identify the high-level concepts, topics, and domain knowledge represented in this content. 
    Look beyond what the user is literally doing and identify what field of knowledge or domain they are working in.
    
    For relationships, identify how concepts connect to each other. Each relationship should be a list with exactly 3 elements: 
    [source concept, relationship type, target concept]. For example: ["Knowledge Graph", "REPRESENTS", "Information Structure"].
    
    IMPORTANT: Make sure all lists (topics, concepts, relationships) are properly formatted as valid JSON arrays. 
    For topics and concepts, use simple string arrays. For relationships, use arrays of 3-element arrays.
    
    Example of proper format:
    ```json
    {{
      "summary": "User is analyzing data with Python",
      "topics": ["Data Analysis", "Python Programming"],
      "concepts": ["Pandas", "Data Visualization", "Statistical Analysis"],
      "relationships": [["Python", "USED_FOR", "Data Analysis"], ["Pandas", "PART_OF", "Python Ecosystem"]],
      "work_context": "Data Science Project"
    }}
    ```
    
    {format_instructions}

    Focus on being accurate and specific. If the text is too ambiguous for any category, provide general concepts that might reasonably apply.
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Create the LLM chain
    llm = setup_langchain_client()

    # Format the prompt with the input values
    formatted_prompt = prompt.format(
        app_name=app_name,
        window_name=window_name,
        timestamp=timestamp,
        text=text,
        format_instructions=format_instructions
    )

    # Get the response from the LLM
    response = llm.invoke(formatted_prompt)

    try:
        # Parse the structured output
        parsed_output = parser.parse(response.content)

        # Handle parsing the LLM output more robustly
        # For topics, handle both comma-separated strings and array formats
        if isinstance(parsed_output["topics"], str):
            # First check if it's a JSON array string
            try:
                parsed_output["topics"] = json.loads(parsed_output["topics"])
            except json.JSONDecodeError:
                # If not JSON, treat as comma-separated list
                parsed_output["topics"] = [topic.strip() for topic in parsed_output["topics"].split(",")]
        
        # For concepts, handle both comma-separated strings and array formats
        if isinstance(parsed_output["concepts"], str):
            try:
                parsed_output["concepts"] = json.loads(parsed_output["concepts"])
            except json.JSONDecodeError:
                parsed_output["concepts"] = [concept.strip() for concept in parsed_output["concepts"].split(",")]
        
        # For relationships, handle string formats more robustly
        if isinstance(parsed_output["relationships"], str):
            try:
                parsed_output["relationships"] = json.loads(parsed_output["relationships"])
            except json.JSONDecodeError:
                # If we can't parse it as JSON, use an empty list as fallback
                print(f"Warning: Could not parse relationships from LLM output. Using empty list.")
                parsed_output["relationships"] = []

        return parsed_output

    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Raw response: {response.content}")

        # Try to extract useful information even if the parsing failed
        try:
            # Try to clean up the response content for another parsing attempt
            # Remove markdown code blocks if present
            cleaned_content = response.content
            if "```json" in cleaned_content:
                cleaned_content = cleaned_content.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned_content:
                cleaned_content = cleaned_content.split("```")[1].split("```")[0].strip()
            
            # Try to parse the cleaned content
            import json
            parsed = json.loads(cleaned_content)
            
            # Create a standardized output
            result = {
                "summary": parsed.get("summary", f"User was interacting with {app_name}"),
                "topics": parsed.get("topics", ["computer usage"]),
                "concepts": parsed.get("concepts", ["software application"]),
                "relationships": parsed.get("relationships", []),
                "work_context": parsed.get("work_context", "Digital Workflow")
            }
            
            # Convert any string lists to actual lists
            for key in ["topics", "concepts"]:
                if isinstance(result[key], str):
                    result[key] = [item.strip() for item in result[key].split(",")]
            
            print("Successfully recovered partial data from LLM response")
            return result
            
        except Exception as nested_e:
            print(f"Secondary parsing attempt failed: {nested_e}")
            
            # Ultimate fallback response if all parsing fails
            return {
                "summary": f"User was interacting with {app_name}",
                "topics": ["computer usage"],
                "concepts": ["software application"],
                "relationships": [["user", "INTERACTS_WITH", "application"]],
                "work_context": "Digital Workflow"
            }

class ConceptGraphProcessor:
    def __init__(self, sqlite_path: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.sqlite_path = sqlite_path
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password

        # Connect to Neo4j
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_uri, auth=(neo4j_user, neo4j_password)
        )

    def __del__(self):
        # Close Neo4j connection when object is destroyed
        if hasattr(self, 'neo4j_driver'):
            self.neo4j_driver.close()

    def fetch_ocr_samples(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch sample OCR text chunks from the SQLite database.
        Optimized to get more meaningful and content-rich text samples.
        """
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Query to get OCR text along with related metadata
        # Prioritize longer text entries with meaningful content
        query = """
        SELECT o.frame_id, o.text, o.app_name, o.window_name, f.timestamp
        FROM ocr_text o
        JOIN frames f ON o.frame_id = f.id
        WHERE o.text IS NOT NULL 
          AND o.text != '' 
          AND LENGTH(o.text) > 200
          AND o.text NOT LIKE '%error%404%'
          AND o.text NOT LIKE '%loading%'
          AND (
              o.app_name IN ('Code', 'VSCode', 'Terminal', 'iTerm', 'Chrome', 'Firefox', 'Safari', 'Microsoft Word', 'Google Docs', 'Notion')
              OR o.window_name LIKE '%edit%'
              OR o.window_name LIKE '%document%'
              OR o.window_name LIKE '%project%'
          )
        ORDER BY LENGTH(o.text) DESC, f.timestamp DESC
        LIMIT ?
        """

        cursor.execute(query, (limit,))
        results = [dict(row) for row in cursor.fetchall()]
        
        # If not enough results, fall back to a simpler query
        if len(results) < limit:
            query = """
            SELECT o.frame_id, o.text, o.app_name, o.window_name, f.timestamp
            FROM ocr_text o
            JOIN frames f ON o.frame_id = f.id
            WHERE o.text IS NOT NULL AND o.text != '' AND LENGTH(o.text) > 100
            ORDER BY LENGTH(o.text) DESC, f.timestamp DESC
            LIMIT ?
            """
            cursor.execute(query, (limit - len(results),))
            additional_results = [dict(row) for row in cursor.fetchall()]
            results.extend(additional_results)
        
        conn.close()
        
        print(f"Fetched {len(results)} OCR samples with an average text length of {sum(len(r['text']) for r in results) / len(results) if results else 0:.0f} characters")
        
        return results

    def create_graph_schema(self):
        """
        Set up the Neo4j graph schema with constraints and indexes for concept representation.
        """
        with self.neo4j_driver.session() as session:
            # Create constraints for unique node types
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (o:Observation) REQUIRE o.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (w:WorkContext) REQUIRE w.name IS UNIQUE")
            
            # Create indexes for performance
            session.run("CREATE INDEX IF NOT EXISTS FOR (o:Observation) ON (o.timestamp)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (t:Topic) ON (t.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.name)")

    def process_ocr_to_concept_graph(self, ocr_data: List[Dict[str, Any]]):
        """
        Process OCR chunks and build a conceptual knowledge graph in Neo4j.
        """
        with self.neo4j_driver.session() as session:
            for item in ocr_data:
                # Get LLM analysis with topic and concept extraction
                llm_result = analyze_with_llm(
                    item['text'],
                    item['app_name'] or "Unknown App",
                    item['window_name'] or "Unknown Window",
                    item['timestamp']
                )

                # Create Observation node (renamed from Activity to better reflect the conceptual focus)
                observation_id = f"observation_{item['frame_id']}"
                session.run("""
                    MERGE (o:Observation {id: $id})
                    SET o.summary = $summary,
                        o.timestamp = $timestamp,
                        o.source_text = $text,
                        o.app_name = $app_name,
                        o.window_name = $window_name
                """, {
                    'id': observation_id,
                    'summary': llm_result['summary'],
                    'timestamp': item['timestamp'],
                    'text': item['text'][:200] + ('...' if len(item['text']) > 200 else ''),  # Truncate long texts
                    'app_name': item['app_name'] or "Unknown App",
                    'window_name': item['window_name'] or "Unknown Window"
                })

                # Create WorkContext node and relate to Observation
                session.run("""
                    MERGE (w:WorkContext {name: $name})
                    WITH w
                    MATCH (o:Observation {id: $observation_id})
                    MERGE (o)-[:PART_OF]->(w)
                """, {
                    'name': llm_result.get('work_context', "General Work"),
                    'observation_id': observation_id
                })

                # Create Topic nodes and relate to Observation
                for topic in llm_result['topics']:
                    session.run("""
                        MERGE (t:Topic {name: $name})
                        WITH t
                        MATCH (o:Observation {id: $observation_id})
                        MERGE (o)-[:INVOLVES_TOPIC]->(t)
                    """, {
                        'name': topic,
                        'observation_id': observation_id
                    })

                # Create Concept nodes and relate to Observation
                for concept in llm_result['concepts']:
                    session.run("""
                        MERGE (c:Concept {name: $name})
                        WITH c
                        MATCH (o:Observation {id: $observation_id})
                        MERGE (o)-[:INVOLVES_CONCEPT]->(c)
                    """, {
                        'name': concept,
                        'observation_id': observation_id
                    })

                # Create relationships between concepts
                for rel in llm_result.get('relationships', []):
                    if len(rel) == 3:
                        source_concept, relationship_type, target_concept = rel
                        # Convert relationship type to uppercase and replace spaces with underscores
                        relationship_type = relationship_type.upper().replace(' ', '_')
                        
                        # Create relationship between concepts
                        session.run(f"""
                            MERGE (source:Concept {{name: $source_concept}})
                            MERGE (target:Concept {{name: $target_concept}})
                            MERGE (source)-[:{relationship_type}]->(target)
                        """, {
                            'source_concept': source_concept,
                            'target_concept': target_concept
                        })

                # Connect observations to build a chronological narrative (if not the first one)
                if ocr_data.index(item) > 0:
                    prev_item = ocr_data[ocr_data.index(item) - 1]
                    prev_observation_id = f"observation_{prev_item['frame_id']}"

                    session.run("""
                        MATCH (prev:Observation {id: $prev_id})
                        MATCH (current:Observation {id: $current_id})
                        WHERE prev.timestamp < current.timestamp
                        MERGE (prev)-[:FOLLOWED_BY]->(current)
                    """, {
                        'prev_id': prev_observation_id,
                        'current_id': observation_id
                    })

    def get_concept_graph_summary(self) -> Dict[str, Any]:
        """
        Return a summary of the concept graph.
        """
        with self.neo4j_driver.session() as session:
            # Count nodes by type
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS nodeType, count(*) AS count
                ORDER BY count DESC
            """)
            node_counts = {row["nodeType"]: row["count"] for row in result}

            # Get the most central topics (by degree centrality)
            result = session.run("""
                MATCH (t:Topic)<-[:INVOLVES_TOPIC]-(o:Observation)
                WITH t, count(o) AS degree
                RETURN t.name AS topic, degree
                ORDER BY degree DESC
                LIMIT 5
            """)
            central_topics = {row["topic"]: row["degree"] for row in result}

            # Get the most central concepts
            result = session.run("""
                MATCH (c:Concept)
                WHERE EXISTS((c)<-[:INVOLVES_CONCEPT]-()) OR EXISTS((c)-[]-(:Concept))
                WITH c, 
                     SIZE([(c)<-[r:INVOLVES_CONCEPT]-() | r]) + 
                     SIZE([(c)-[r]-(:Concept) | r]) AS connections
                RETURN c.name AS concept, connections
                ORDER BY connections DESC
                LIMIT 5
            """)
            central_concepts = {row["concept"]: row["connections"] for row in result}

            # Get main work contexts
            result = session.run("""
                MATCH (w:WorkContext)<-[:PART_OF]-(o:Observation)
                WITH w, count(o) AS observations
                RETURN w.name AS context, observations
                ORDER BY observations DESC
            """)
            work_contexts = {row["context"]: row["observations"] for row in result}

            # Get concept relationships
            result = session.run("""
                MATCH (c1:Concept)-[r]->(c2:Concept)
                WHERE c1 <> c2
                RETURN c1.name AS source, type(r) AS relationship, c2.name AS target
                LIMIT 10
            """)
            concept_relationships = [{"source": row["source"], "relationship": row["relationship"], "target": row["target"]} for row in result]

            return {
                "node_counts": node_counts,
                "central_topics": central_topics,
                "central_concepts": central_concepts,
                "work_contexts": work_contexts,
                "concept_relationships": concept_relationships
            }

    def generate_topic_map(self) -> Dict[str, Any]:
        """
        Generate a higher-level topic map from the knowledge graph.
        """
        with self.neo4j_driver.session() as session:
            # Generate a map of topics and their related concepts
            result = session.run("""
                MATCH (t:Topic)<-[:INVOLVES_TOPIC]-(o:Observation)-[:INVOLVES_CONCEPT]->(c:Concept)
                WITH t, collect(DISTINCT c.name) AS concepts
                RETURN t.name AS topic, concepts
                ORDER BY size(concepts) DESC
                LIMIT 5
            """)
            
            topic_concept_map = {row["topic"]: row["concepts"] for row in result}
            
            # Find pairs of topics that share concepts
            result = session.run("""
                MATCH (t1:Topic)<-[:INVOLVES_TOPIC]-(o1:Observation)-[:INVOLVES_CONCEPT]->(c:Concept)<-[:INVOLVES_CONCEPT]-(o2:Observation)-[:INVOLVES_TOPIC]->(t2:Topic)
                WHERE t1 <> t2
                WITH t1, t2, collect(DISTINCT c) AS shared
                WITH t1, t2, size(shared) AS shared_concepts
                WHERE shared_concepts > 0
                RETURN t1.name AS topic1, t2.name AS topic2, shared_concepts
                ORDER BY shared_concepts DESC
                LIMIT 10
            """)
            
            related_topics = [{"topic1": row["topic1"], "topic2": row["topic2"], "shared_concepts": row["shared_concepts"]} for row in result]
            
            return {
                "topic_concept_map": topic_concept_map,
                "related_topics": related_topics
            }

def process_with_llm_batch(processor, ocr_samples, batch_size=2):
    """
    Process OCR samples in batches to avoid overwhelming the LLM API.
    """
    total_samples = len(ocr_samples)
    print(f"Processing {total_samples} samples in batches of {batch_size}...")

    for i in range(0, total_samples, batch_size):
        batch = ocr_samples[i:min(i+batch_size, total_samples)]
        print(f"Processing batch {i//batch_size + 1} ({len(batch)} samples)")
        processor.process_ocr_to_concept_graph(batch)
        print(f"Batch {i//batch_size + 1} completed")

def run_concept_poc():
    """
    Run the proof of concept for concept-oriented knowledge graph extraction.
    """
    print("Starting Concept-Oriented GraphRAG Proof of Concept with LangChain and GPT-4o...")

    # Check for API key
    if not OPENAI_API_KEY:
        print("⚠️ Warning: OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable.")
        return

    # Initialize processor
    try:
        processor = ConceptGraphProcessor(
            sqlite_path=SQLITE_DB_PATH,
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD
        )
        print("Connected to databases successfully")

        # Set up graph schema
        processor.create_graph_schema()
        print("Created Neo4j schema for concept graph")

        # Fetch sample OCR data - getting more samples for better topic coverage
        ocr_samples = processor.fetch_ocr_samples(limit=10)
        print(f"Fetched {len(ocr_samples)} OCR samples for concept extraction")

        # Process data into concept graph in batches
        process_with_llm_batch(processor, ocr_samples, batch_size=2)
        print("Processed OCR data into concept knowledge graph")

        # Get and display concept graph summary
        summary = processor.get_concept_graph_summary()
        print("\nConcept Graph Summary:")
        print(f"Node counts: {json.dumps(summary['node_counts'], indent=2)}")
        print("\nCentral Topics:")
        for topic, degree in summary['central_topics'].items():
            print(f"  - {topic} (connections: {degree})")
        
        print("\nCentral Concepts:")
        for concept, connections in summary['central_concepts'].items():
            print(f"  - {concept} (connections: {connections})")
        
        print("\nWork Contexts:")
        for context, obs in summary['work_contexts'].items():
            print(f"  - {context} (observations: {obs})")
        
        print("\nConcept Relationships:")
        for rel in summary['concept_relationships']:
            print(f"  - {rel['source']} {rel['relationship']} {rel['target']}")

        # Generate and display topic map
        topic_map = processor.generate_topic_map()
        print("\nTopic Map:")
        for topic, concepts in topic_map['topic_concept_map'].items():
            print(f"  Topic: {topic}")
            print(f"    Related concepts: {', '.join(concepts[:5])}" + (", ..." if len(concepts) > 5 else ""))
        
        print("\nRelated Topics:")
        for rel in topic_map['related_topics']:
            print(f"  - {rel['topic1']} and {rel['topic2']} share {rel['shared_concepts']} concept(s)")

        print("\nConcept-oriented knowledge graph proof of concept completed successfully!")

    except Exception as e:
        print(f"Error in proof of concept: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_concept_poc()
