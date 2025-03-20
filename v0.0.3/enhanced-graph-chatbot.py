import os
import json
import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from dateutil import parser as date_parser

# Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class EnhancedGraphChatbot:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """
        Initialize the enhanced concept graph chatbot with Neo4j connection.
        """
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_uri, auth=(neo4j_user, neo4j_password)
        )
        
        # Set up LLM with higher temperature for more creative responses
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.4)
        
        # Set up conversation memory
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Enable/disable debug mode for showing context
        self.show_context = True
        
        # Cache the graph schema information for quicker access
        self._cache_graph_schema()

    def __del__(self):
        """
        Clean up Neo4j connection.
        """
        if hasattr(self, 'neo4j_driver'):
            self.neo4j_driver.close()
    
    def _cache_graph_schema(self):
        """
        Cache the graph schema information for better query planning.
        Also extracts sample data to understand the graph structure.
        """
        with self.neo4j_driver.session() as session:
            # Get node labels
            result = session.run("CALL db.labels()")
            self.node_labels = [record["label"] for record in result]
            
            # Get relationship types
            result = session.run("CALL db.relationshipTypes()")
            self.relationship_types = [record["relationshipType"] for record in result]
            
            # Get property keys
            result = session.run("CALL db.propertyKeys()")
            self.property_keys = [record["propertyKey"] for record in result]
            
            # Cache node counts by label
            self.node_counts = {}
            for label in self.node_labels:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) AS count")
                count = result.single()["count"]
                self.node_counts[label] = count
            
            # Get sample relationships to understand graph structure
            self.relationship_patterns = []
            for rel_type in self.relationship_types:
                try:
                    result = session.run(f"""
                        MATCH (a)-[r:{rel_type}]->(b)
                        RETURN labels(a)[0] AS source_label, labels(b)[0] AS target_label
                        LIMIT 1
                    """)
                    record = result.single()
                    if record:
                        self.relationship_patterns.append({
                            "type": rel_type,
                            "source": record["source_label"],
                            "target": record["target_label"]
                        })
                except:
                    pass  # Skip if there was an error
            
            print(f"Cached graph schema: {len(self.node_labels)} node types, {len(self.relationship_types)} relationship types, {len(self.relationship_patterns)} patterns")

    def dynamically_analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Use the LLM to deeply analyze the user's query and determine the most effective
        search strategy on the knowledge graph.
        """
        # Prepare context about the graph schema
        schema_info = {
            "node_labels": self.node_labels,
            "relationship_types": self.relationship_types,
            "node_counts": self.node_counts
        }
        
        # Prepare relationship pattern information
        pattern_info = ""
        for pattern in self.relationship_patterns:
            pattern_info += f"- ({pattern['source']})-[:{pattern['type']}]->({pattern['target']})\n"
        
        # Create a prompt for the LLM to analyze the query
        analysis_prompt = ChatPromptTemplate.from_template("""
        You're an expert in knowledge graph querying. Analyze this user query to determine the best search strategy.
        
        GRAPH SCHEMA:
        Node types: {node_labels}
        Relationship types: {relationship_types}
        Node counts: {node_counts}
        
        COMMON RELATIONSHIP PATTERNS:
        {pattern_info}
        
        User query: "{query}"
        
        Based on the user query and the graph schema, provide the following:
        
        1. Analysis: What is the user really asking about? What concepts, timeframes, or relationships might be relevant?
        
        2. Query Strategy: What would be the best Neo4j Cypher approach to answer this? Don't write the actual Cypher query, but describe the general approach (e.g., "Find Concept nodes related to X, then explore their connections to WorkContext nodes").
        
        3. Extraction Plan: What specific entities should we extract from this query? Include:
           - concepts: List of concepts to search for - use ONLY concepts that might be in a knowledge graph
           - timeframe: If mentioned, parsing of time references
           - focus: The main subject/action of interest (e.g., "error handling", "integration")
           - relationships_of_interest: What connections between nodes are most relevant
        
        4. Response Framework: How should we structure the response to best answer the query?
        
        Return this as a JSON structure with these fields.
        """)
        
        # Get the LLM's analysis
        response = self.llm.invoke(
            analysis_prompt.format(
                query=query,
                node_labels=self.node_labels,
                relationship_types=self.relationship_types,
                node_counts=json.dumps(self.node_counts),
                pattern_info=pattern_info
            )
        )
        
        # Extract and parse the JSON from the response
        try:
            # Look for a JSON block in the response
            content = response.content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
                
            # Parse the JSON
            analysis = json.loads(json_str)
            return analysis
        except Exception as e:
            print(f"Error parsing query analysis: {e}")
            # Return a default analysis structure
            return {
                "analysis": f"Query about {query}",
                "query_strategy": "General search across concepts and observations",
                "extraction_plan": {
                    "concepts": [],
                    "timeframe": None,
                    "focus": "general information",
                    "relationships_of_interest": []
                },
                "response_framework": "Provide a summary of recent activities and concepts"
            }

    def dynamically_generate_cypher(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Dynamically generate appropriate Cypher queries based on the query analysis.
        Returns a list of query specifications with their purpose.
        """
        # Extract key elements from the analysis
        extraction_plan = analysis.get("extraction_plan", {})
        concepts = extraction_plan.get("concepts", [])
        timeframe = extraction_plan.get("timeframe")
        focus = extraction_plan.get("focus", "")
        relationships = extraction_plan.get("relationships_of_interest", [])
        
        # Prepare relationship pattern information
        pattern_info = ""
        for pattern in self.relationship_patterns:
            pattern_info += f"- ({pattern['source']})-[:{pattern['type']}]->({pattern['target']})\n"
        
        # Create a prompt for the LLM to generate Cypher queries
        cypher_prompt = ChatPromptTemplate.from_template("""
        As a Neo4j and Cypher expert, generate appropriate Cypher queries to answer the user's question.
        
        GRAPH SCHEMA:
        Node types: {node_labels}
        Relationship types: {relationship_types}
        Node counts: {node_counts}
        
        OBSERVED RELATIONSHIP PATTERNS:
        {pattern_info}
        
        QUERY ANALYSIS:
        {analysis}
        
        Extraction Plan:
        - Concepts: {concepts}
        - Timeframe: {timeframe}
        - Focus: {focus}
        - Relationships of interest: {relationships}
        
        IMPORTANT GUIDELINES:
        1. DO NOT use any node labels or relationship types that aren't in the schema above
        2. DO NOT assume a 'User' node exists - focus on Observation, Session, Concept, Topic, and WorkContext nodes
        3. Sessions relate to Observations, not directly to Users
        4. Start with simple queries like "MATCH (o:Observation)" or "MATCH (s:Session)"
        5. Time filtering should be done with Observation or Session timestamp properties
        6. Create 3-5 diverse queries to explore different aspects of the question
        
        For each query:
        1. Provide a purpose explaining what this query will find
        2. Write the complete Cypher query
        3. List parameters that would be needed (if any)
        
        Your response should be a JSON array of objects, each with "purpose", "query", and "params" fields.
        Ensure your Cypher syntax is valid for Neo4j 4.4+.
        Keep queries efficient and limit results to 10-20 per query.
        """)
        
        # Get the LLM's Cypher generation
        response = self.llm.invoke(
            cypher_prompt.format(
                node_labels=self.node_labels,
                relationship_types=self.relationship_types,
                node_counts=json.dumps(self.node_counts),
                pattern_info=pattern_info,
                analysis=analysis.get("analysis", ""),
                concepts=concepts,
                timeframe=timeframe,
                focus=focus,
                relationships=relationships
            )
        )
        
        # Extract and parse the JSON from the response
        try:
            # Look for a JSON block in the response
            content = response.content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
                
            # Parse the JSON
            cypher_queries = json.loads(json_str)
            return cypher_queries
        except Exception as e:
            print(f"Error parsing Cypher generation: {e}")
            # Return a default query for fallback
            return [{
                "purpose": "Get recent activities",
                "query": """
                MATCH (o:Observation)
                RETURN o.summary AS summary, o.timestamp AS timestamp
                ORDER BY o.timestamp DESC
                LIMIT 10
                """,
                "params": {}
            }]

    def execute_cypher_queries(self, query_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute multiple Cypher queries and gather their results.
        Includes fallback queries if main ones fail.
        """
        all_results = {}
        
        with self.neo4j_driver.session() as session:
            for spec in query_specs:
                purpose = spec.get("purpose", "Unnamed query")
                cypher = spec.get("query", "")
                params = spec.get("params", {})
                
                # Process any timeframe parameters to ensure they're in the correct format
                processed_params = {}
                for key, value in params.items():
                    if key in ['start_date', 'end_date', 'date'] and isinstance(value, str):
                        try:
                            # Convert to proper Neo4j datetime format if needed
                            dt = date_parser.parse(value)
                            processed_params[key] = dt.isoformat()
                        except:
                            processed_params[key] = value
                    else:
                        processed_params[key] = value
                
                try:
                    # Execute the query
                    result = session.run(cypher, processed_params)
                    records = [dict(record) for record in result]
                    
                    # Process any neo4j nodes/relationships to simplify JSON serialization
                    processed_records = []
                    for record in records:
                        processed_record = {}
                        for key, value in record.items():
                            if hasattr(value, 'keys') and callable(getattr(value, 'keys')):
                                # This is likely a neo4j node or relationship
                                processed_record[key] = dict(value)
                            elif isinstance(value, list):
                                # Handle lists of nodes
                                processed_items = []
                                for item in value:
                                    if hasattr(item, 'keys') and callable(getattr(item, 'keys')):
                                        processed_items.append(dict(item))
                                    else:
                                        processed_items.append(item)
                                processed_record[key] = processed_items
                            else:
                                processed_record[key] = value
                        processed_records.append(processed_record)
                    
                    all_results[purpose] = processed_records
                except Exception as e:
                    print(f"Error executing query '{purpose}': {e}")
                    all_results[purpose] = [{"error": str(e)}]
                    
                    # Try a fallback query if original fails
                    try:
                        if "Session" in cypher:
                            fallback = """
                                MATCH (s:Session)
                                RETURN s.app_name AS app_name, s.start_time AS start_time,
                                       s.end_time AS end_time, s.entry_count AS entry_count
                                ORDER BY s.start_time DESC
                                LIMIT 5
                            """
                        elif "Topic" in cypher:
                            fallback = """
                                MATCH (t:Topic)<-[:INVOLVES_TOPIC]-(o:Observation)
                                RETURN t.name AS topic, count(o) AS count
                                ORDER BY count DESC
                                LIMIT 10
                            """
                        elif "Concept" in cypher:
                            fallback = """
                                MATCH (c:Concept)<-[:INVOLVES_CONCEPT]-(o:Observation)
                                RETURN c.name AS concept, count(o) AS count
                                ORDER BY count DESC
                                LIMIT 10
                            """
                        elif "WorkContext" in cypher:
                            fallback = """
                                MATCH (w:WorkContext)<-[:PART_OF]-(o:Observation)
                                RETURN w.name AS work_context, count(o) AS count
                                ORDER BY count DESC
                                LIMIT 5
                            """
                        else:
                            fallback = """
                                MATCH (o:Observation)
                                RETURN o.summary AS summary, o.timestamp AS timestamp
                                ORDER BY o.timestamp DESC
                                LIMIT 10
                            """
                        
                        result = session.run(fallback)
                        fallback_records = [dict(record) for record in result]
                        all_results[f"{purpose} (fallback)"] = fallback_records
                        print(f"Used fallback query for '{purpose}'")
                    except Exception as fallback_e:
                        print(f"Fallback query also failed: {fallback_e}")
        
        return all_results

    def synthesize_response(self, query: str, query_analysis: Dict[str, Any], 
                           query_results: Dict[str, Any], 
                           conversation_history: str) -> Tuple[str, str]:
        """
        Use the LLM to synthesize a natural response based on the query results.
        Returns both the response and the context used.
        """
        # Format the results for the prompt
        formatted_results = json.dumps(query_results, indent=2, default=str)
        
        # Check if we have meaningful results
        has_results = False
        result_summary = []
        for purpose, results in query_results.items():
            if results and not any("error" in result for result in results):
                has_results = True
                result_summary.append(f"- {purpose}: {len(results)} results")
        
        result_status = "We have the following results:\n" + "\n".join(result_summary) if has_results else "We have limited or no results from the queries."
        
        # Create a prompt for synthesis
        synthesis_prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant with access to a knowledge graph representing a user's work activities, topics, and concepts.
        Your goal is to answer the user's question based on the query results from this knowledge graph.
        
        The user asked: "{query}"
        
        Previous conversation:
        {conversation_history}
        
        Query analysis:
        {analysis}
        
        Status of results:
        {result_status}
        
        Query results from the knowledge graph:
        {results}
        
        Guidelines for your response:
        1. Be honest about what information is available - don't make up facts.
        2. If results are limited, acknowledge this and explain what information is available.
        3. Provide specific examples from the data where possible.
        4. Make your response conversational and natural - don't reference the queries themselves.
        5. Synthesize the information into a coherent answer that helps the user understand their work patterns.
        6. If the data shows error messages or issues, you can acknowledge those and suggest potential fixes.
        
        Based on the available evidence, provide a helpful response to the user's question.
        """)
        
        # Get the synthesized response
        response = self.llm.invoke(
            synthesis_prompt.format(
                query=query,
                conversation_history=conversation_history,
                analysis=json.dumps(query_analysis, indent=2),
                result_status=result_status,
                results=formatted_results
            )
        )
        
        return response.content, formatted_results

    def get_conversation_context(self) -> str:
        """
        Get formatted conversation context from memory.
        """
        messages = self.memory.chat_memory.messages
        
        # Format the conversation history
        formatted_history = []
        for i, message in enumerate(messages[-6:]):  # Get last 6 messages max
            if isinstance(message, HumanMessage):
                formatted_history.append(f"User: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_history.append(f"Assistant: {message.content}")
        
        return "\n".join(formatted_history)

    def toggle_context_display(self):
        """
        Toggle whether to show context information in the chat.
        """
        self.show_context = not self.show_context
        return f"Context display is now {'enabled' if self.show_context else 'disabled'}"

    def process_query(self, query: str) -> str:
        """
        Process a user query through the full enhanced pipeline.
        """
        # Check for context display commands
        if query.lower() in ["show context", "enable context", "display context"]:
            self.show_context = True
            return "Context display is now enabled. You'll see the graph evidence used for responses."
        
        if query.lower() in ["hide context", "disable context"]:
            self.show_context = False
            return "Context display is now disabled."
        
        try:
            # Step 1: Dynamically analyze the query
            query_analysis = self.dynamically_analyze_query(query)
            
            # Step 2: Generate appropriate Cypher queries
            cypher_queries = self.dynamically_generate_cypher(query_analysis)
            
            # Step 3: Execute the queries
            query_results = self.execute_cypher_queries(cypher_queries)
            
            # Step 4: Get conversation context
            conversation_context = self.get_conversation_context()
            
            # Step 5: Synthesize the response
            response, context = self.synthesize_response(
                query, 
                query_analysis, 
                query_results, 
                conversation_context
            )
            
            # Step 6: Save to memory
            self.memory.chat_memory.add_user_message(query)
            self.memory.chat_memory.add_ai_message(response)
            
            # Final response - either with or without context information
            if self.show_context:
                return f"""📊 **GRAPH ANALYSIS AND RESULTS:**
```
{context}
```

**Response:**
{response}"""
            else:
                return response
            
        except Exception as e:
            print(f"Error processing query: {e}")
            import traceback
            traceback.print_exc()
            return f"I encountered an error while processing your query. Please try again or rephrase your question."

def run_enhanced_chatbot():
    """
    Run the enhanced concept graph chatbot in an interactive loop.
    """
    # Set up the chatbot
    chatbot = EnhancedGraphChatbot(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD
    )
    
    print("\n=== Enhanced Concept Graph Chatbot ===")
    print("This chatbot uses a dynamic LLM-based approach to understand and query your knowledge graph.")
    print("Ask questions about your work or type 'exit' to quit.")
    print("Special commands:")
    print("- 'show context' - Display the graph evidence used in responses")
    print("- 'hide context' - Hide the graph evidence")
    print("\nExample questions:")
    print("- What have I been working on this week?")
    print("- Tell me about my Python and Neo4j projects.")
    print("- How are Docker and Neo4j connected in my work?")
    print("- What error handling approaches have I used in my projects?")
    print("- What are the main topics and concepts in my recent work?\n")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nGoodbye!")
            break
        
        try:
            response = chatbot.process_query(user_input)
            print(f"\nAssistant: {response}")
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Sorry, I encountered an error processing your request. Please try again.")

if __name__ == "__main__":
    run_enhanced_chatbot()
