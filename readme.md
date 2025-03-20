# GraphRAG Proof of Concept

This project demonstrates a simple GraphRAG system that processes OCR text data from a database into a Neo4j knowledge graph using LLM-powered summarization and entity extraction.

## Prerequisites

- Python 3.8+
- Neo4j database (local or cloud instance)
- SQLite database with OCR data (as per the provided schema)

## Installation

1. Clone this repository or download the scripts
2. Install the required dependencies:

```bash
pip install sqlite3 neo4j requests networkx matplotlib
```

3. Make sure Neo4j is running and accessible at the configured URI (default: `bolt://localhost:7687`)

## Configuration

Edit the configuration variables at the top of each script:

```python
SQLITE_DB_PATH = "user_activity.db"  # Path to your SQLite database
NEO4J_URI = "bolt://localhost:7687"  # Neo4j connection URI
NEO4J_USER = "neo4j"                 # Neo4j username
NEO4J_PASSWORD = "password"          # Neo4j password
LLM_API_KEY = "your_api_key"         # Your LLM API key (if using a real API)
```

## Usage

1. Run the main proof of concept script:

```bash
python graphrag_poc.py
```

2. Visualize the resulting graph:

```bash
python visualize_graph.py
```

## How It Works

The system follows these steps:

1. Fetches sample OCR text data from the SQLite database
2. Processes each text chunk with an LLM to extract:
   - Summary of user activity
   - User actions
   - User intentions
   - Relevant entities
3. Builds a knowledge graph in Neo4j with nodes for:
   - Activities (from OCR chunks)
   - Applications (used by the user)
   - Intentions (goals the user is trying to accomplish)
   - Entities (objects, people, concepts mentioned)
4. Creates relationships between these nodes
5. Visualizes the resulting graph

## Graph Schema

The graph consists of:

### Nodes
- `Activity`: OCR text chunks summarized as user activities
- `Application`: Software applications the user interacted with
- `Intention`: Inferred user goals and intentions
- `Entity`: Relevant objects, concepts, or people

### Relationships
- `USED`: Activity → Application (user used this application)
- `HAS_INTENT`: Activity → Intention (user had this intention)
- `INVOLVES`: Activity → Entity (activity involves this entity)
- `FOLLOWED_BY`: Activity → Activity (chronological sequence)

## Extending the POC

To expand this proof of concept:

1. Integrate with a real LLM API (OpenAI, Claude, etc.)
2. Process additional data types (audio transcriptions, UI monitoring)
3. Implement more complex relationship extraction
4. Add temporal reasoning capabilities
5. Create a query interface to explore the graph

## Notes

- The current implementation uses simulated LLM responses for demonstration purposes
- For production use, replace the simulated responses with actual API calls
- Consider more sophisticated entity and relationship extraction techniques for complex activities
