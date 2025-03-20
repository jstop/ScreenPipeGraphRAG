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
def summarize_with_llm(text: str, app_name: str, window_name: str, timestamp: str) -> Dict[str, Any]:
    """
    Use LangChain with GPT-4o to summarize OCR text and extract intentions and actions.
    Returns a dictionary with summary, actions, intentions, and entities.
    """
    # Define output schemas for structured parsing
    summary_schema = ResponseSchema(
        name="summary",
        description="A brief summary of what the user was doing based on the OCR text"
    )

    actions_schema = ResponseSchema(
        name="actions",
        description="List of main actions the user performed"
    )

    intentions_schema = ResponseSchema(
        name="intentions",
        description="List of likely intentions or goals the user had"
    )

    entities_schema = ResponseSchema(
        name="entities",
        description="List of important entities mentioned (people, files, tasks, apps, etc.)"
    )

    # Create the parser with the schemas
    parser = StructuredOutputParser.from_response_schemas([
        summary_schema, actions_schema, intentions_schema, entities_schema
    ])

    # Get the format instructions
    format_instructions = parser.get_format_instructions()

    # Create the prompt template
    template = """
    You are an AI specialized in analyzing OCR text data from user screens to understand user behavior.

    Analyze the following OCR text captured from a user's screen:

    App: {app_name}
    Window: {window_name}
    Time: {timestamp}

    Text:
    {text}

    {format_instructions}

    Focus on being accurate and specific. If the text is ambiguous, provide your best guess but keep it grounded in the evidence.
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

        # Convert string lists to actual lists if needed
        if isinstance(parsed_output["actions"], str):
            parsed_output["actions"] = json.loads(parsed_output["actions"])
        if isinstance(parsed_output["intentions"], str):
            parsed_output["intentions"] = json.loads(parsed_output["intentions"])
        if isinstance(parsed_output["entities"], str):
            parsed_output["entities"] = json.loads(parsed_output["entities"])

        return parsed_output

    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Raw response: {response.content}")

        # Fallback response if parsing fails
        return {
            "summary": f"User was interacting with {app_name}",
            "actions": ["using application"],
            "intentions": ["complete task"],
            "entities": [app_name, window_name]
        }

class GraphRAGProcessor:
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

    def fetch_ocr_samples(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch a few sample OCR text chunks from the SQLite database.
        """
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Query to get OCR text along with related metadata
        query = """
        SELECT o.frame_id, o.text, o.app_name, o.window_name, f.timestamp
        FROM ocr_text o
        JOIN frames f ON o.frame_id = f.id
        WHERE o.text IS NOT NULL AND o.text != ''
        ORDER BY f.timestamp DESC
        LIMIT ?
        """

        cursor.execute(query, (limit,))
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return results

    def create_graph_schema(self):
        """
        Set up the initial Neo4j graph schema with constraints and indexes.
        """
        with self.neo4j_driver.session() as session:
            # Create constraints for unique node types
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Activity) REQUIRE a.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Application) REQUIRE a.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Intention) REQUIRE i.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")

            # Create indexes for performance
            session.run("CREATE INDEX IF NOT EXISTS FOR (a:Activity) ON (a.timestamp)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (a:Application) ON (a.name)")

    def process_ocr_to_graph(self, ocr_data: List[Dict[str, Any]]):
        """
        Process OCR chunks and build the knowledge graph in Neo4j.
        """
        with self.neo4j_driver.session() as session:
            for item in ocr_data:
                # Get LLM summary and extracted information
                llm_result = summarize_with_llm(
                    item['text'],
                    item['app_name'] or "Unknown App",
                    item['window_name'] or "Unknown Window",
                    item['timestamp']
                )

                # Create Activity node
                activity_id = f"activity_{item['frame_id']}"
                session.run("""
                    MERGE (a:Activity {id: $id})
                    SET a.summary = $summary,
                        a.timestamp = $timestamp,
                        a.source_text = $text
                """, {
                    'id': activity_id,
                    'summary': llm_result['summary'],
                    'timestamp': item['timestamp'],
                    'text': item['text'][:200] + ('...' if len(item['text']) > 200 else '')  # Truncate long texts
                })

                # Create Application node and relate to Activity
                session.run("""
                    MERGE (app:Application {name: $name})
                    WITH app
                    MATCH (a:Activity {id: $activity_id})
                    MERGE (a)-[:USED]->(app)
                """, {
                    'name': item['app_name'] or "Unknown App",
                    'activity_id': activity_id
                })

                # Create Intention nodes and relate to Activity
                for intention in llm_result['intentions']:
                    session.run("""
                        MERGE (i:Intention {name: $name})
                        WITH i
                        MATCH (a:Activity {id: $activity_id})
                        MERGE (a)-[:HAS_INTENT]->(i)
                    """, {
                        'name': intention,
                        'activity_id': activity_id
                    })

                # Create Entity nodes and relate to Activity
                for entity in llm_result['entities']:
                    session.run("""
                        MERGE (e:Entity {name: $name})
                        WITH e
                        MATCH (a:Activity {id: $activity_id})
                        MERGE (a)-[:INVOLVES]->(e)
                    """, {
                        'name': entity,
                        'activity_id': activity_id
                    })

                # Connect activities in chronological order (if not the first one)
                if ocr_data.index(item) > 0:
                    prev_item = ocr_data[ocr_data.index(item) - 1]
                    prev_activity_id = f"activity_{prev_item['frame_id']}"

                    session.run("""
                        MATCH (prev:Activity {id: $prev_id})
                        MATCH (current:Activity {id: $current_id})
                        WHERE prev.timestamp < current.timestamp
                        MERGE (prev)-[:FOLLOWED_BY]->(current)
                    """, {
                        'prev_id': prev_activity_id,
                        'current_id': activity_id
                    })

    def get_graph_summary(self) -> Dict[str, Any]:
        """
        Return a summary of what's in the graph database.
        """
        with self.neo4j_driver.session() as session:
            # Count nodes by type
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS nodeType, count(*) AS count
                ORDER BY count DESC
            """)
            node_counts = {row["nodeType"]: row["count"] for row in result}

            # Count relationships by type
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS relType, count(*) AS count
                ORDER BY count DESC
            """)
            rel_counts = {row["relType"]: row["count"] for row in result}

            # Get a sample activity path
            result = session.run("""
                MATCH path = (a:Activity)-[:FOLLOWED_BY*1..3]->(b:Activity)
                RETURN [node in nodes(path) | node.summary] AS activities
                LIMIT 1
            """)

            sample_path = next(iter(result), {}).get("activities", [])

            return {
                "node_counts": node_counts,
                "relationship_counts": rel_counts,
                "sample_activity_path": sample_path
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
        processor.process_ocr_to_graph(batch)
        print(f"Batch {i//batch_size + 1} completed")

def run_poc():
    """
    Run the proof of concept to demonstrate GraphRAG functionality.
    """
    print("Starting GraphRAG Proof of Concept with LangChain and GPT-4o...")

    # Check for API key
    if OPENAI_API_KEY == "your_openai_api_key":
        print("⚠️ Warning: You need to set your actual OpenAI API key in the configuration.")
        print("    Update the OPENAI_API_KEY variable with your API key to use GPT-4o.")
        print("    For this demo run, we'll proceed with limited functionality.")

    # Initialize processor
    try:
        processor = GraphRAGProcessor(
            sqlite_path=SQLITE_DB_PATH,
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD
        )
        print("Connected to databases successfully")

        # Set up graph schema
        processor.create_graph_schema()
        print("Created Neo4j schema")

        # Fetch sample OCR data
        ocr_samples = processor.fetch_ocr_samples(limit=5)
        print(f"Fetched {len(ocr_samples)} OCR samples")

        # Process data into graph in batches
        process_with_llm_batch(processor, ocr_samples, batch_size=2)
        print("Processed OCR data into knowledge graph")

        # Get and display graph summary
        summary = processor.get_graph_summary()
        print("\nGraph Summary:")
        print(f"Node counts: {json.dumps(summary['node_counts'], indent=2)}")
        print(f"Relationship counts: {json.dumps(summary['relationship_counts'], indent=2)}")
        print("\nSample activity path:")
        for activity in summary['sample_activity_path']:
            print(f"  → {activity}")

        print("\nProof of concept completed successfully!")

    except Exception as e:
        print(f"Error in proof of concept: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_poc()
