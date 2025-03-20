import os
import json
import datetime
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase
import langchain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from dateutil import parser as date_parser
import re

# Configuration
NEO4J_URI = "bolt://localhost:7687"  # Neo4j connection URI
NEO4J_USER = "neo4j"                  # Neo4j username
NEO4J_PASSWORD = "password"           # Neo4j password
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Your OpenAI API key

class ConceptGraphChatbot:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """
        Initialize the concept graph chatbot with Neo4j connection.
        """
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_uri, auth=(neo4j_user, neo4j_password)
        )
        
        # Set up LLM
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        
        # Set up conversation memory
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Track the latest mentioned concepts for context
        self.recent_concepts = []
        self.recent_timeframe = None
        
        # Enable/disable debug mode for showing context
        self.show_context = True

    def __del__(self):
        """
        Clean up Neo4j connection.
        """
        if hasattr(self, 'neo4j_driver'):
            self.neo4j_driver.close()

    def extract_query_elements(self, query: str) -> Tuple[List[str], Optional[Dict[str, Any]]]:
        """
        Extract concepts and time frame from a user query.
        Returns a tuple of (concepts, time_frame).
        """
        # First, use the LLM to extract concepts and time references
        extraction_prompt = ChatPromptTemplate.from_template("""
        From the following query, extract:
        1. Key concepts the user is asking about
        2. Any time frames mentioned (e.g., "last week", "yesterday", "March 15")
        
        Query: {query}
        
        Return your answer as a JSON object with two fields:
        - "concepts": A list of strings representing key concepts
        - "timeframe": A dictionary with "start" and "end" dates if specified, or null if no timeframe is mentioned. Use dates in ISO format (YYYY-MM-DD).
        
        Example response:
        {{
          "concepts": ["Docker", "Neo4j"],
          "timeframe": {{"start": "2025-03-10", "end": "2025-03-17"}}
        }}
        
        If no timeframe is specified, return:
        {{
          "concepts": ["Docker", "Neo4j"],
          "timeframe": null
        }}
        
        JSON response:
        """)
        
        response = self.llm.invoke(extraction_prompt.format(query=query))
        
        try:
            # Parse the JSON response
            extracted = json.loads(response.content.strip())
            concepts = extracted.get('concepts', [])
            timeframe = extracted.get('timeframe')
            
            # Save recent concepts and timeframe for context
            if concepts:
                self.recent_concepts = concepts
            if timeframe:
                self.recent_timeframe = timeframe
            
            return concepts, timeframe
        except Exception as e:
            print(f"Error parsing extraction response: {e}")
            # Fallback to simple extraction
            concepts = self._extract_concepts_simple(query)
            timeframe = self._extract_timeframe_simple(query)
            
            if concepts:
                self.recent_concepts = concepts
            if timeframe:
                self.recent_timeframe = timeframe
            
            return concepts, timeframe

    def _extract_concepts_simple(self, query: str) -> List[str]:
        """
        Simple fallback method to extract potential concept terms from a query.
        """
        # Get all existing concepts from graph to match against
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (c:Concept)
                RETURN c.name AS concept
                UNION
                MATCH (t:Topic)
                RETURN t.name AS concept
                UNION
                MATCH (w:WorkContext)
                RETURN w.name AS concept
            """)
            all_concepts = [record["concept"] for record in result]
        
        # Match query against existing concepts
        found_concepts = []
        for concept in all_concepts:
            if concept.lower() in query.lower():
                found_concepts.append(concept)
        
        return found_concepts if found_concepts else []

    def _extract_timeframe_simple(self, query: str) -> Optional[Dict[str, str]]:
        """
        Simple fallback method to extract time-related information.
        """
        # Check for common time phrases
        today = datetime.datetime.now().date()
        
        if "yesterday" in query.lower():
            yesterday = today - datetime.timedelta(days=1)
            return {"start": yesterday.isoformat(), "end": today.isoformat()}
        
        if "last week" in query.lower():
            end = today
            start = end - datetime.timedelta(days=7)
            return {"start": start.isoformat(), "end": end.isoformat()}
        
        if "last month" in query.lower():
            end = today
            start = end - datetime.timedelta(days=30)
            return {"start": start.isoformat(), "end": end.isoformat()}
        
        return None

    def get_graph_evidence(self, concepts: List[str], timeframe: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the graph for evidence related to the concepts and timeframe.
        Returns a dictionary with different types of evidence.
        """
        evidence = {
            "direct_concepts": [],
            "related_concepts": [],
            "work_contexts": [],
            "concept_relationships": [],
            "topic_clusters": []
        }
        
        if not concepts and not self.recent_concepts:
            # If no concepts specified, return general recent work info
            return self._get_general_work_evidence(timeframe)
        
        # Use recent concepts as fallback if none specified in this query
        query_concepts = concepts if concepts else self.recent_concepts
        
        with self.neo4j_driver.session() as session:
            # Get direct concept information
            for concept in query_concepts:
                # Try to find the concept as either a Concept, Topic or WorkContext
                result = session.run("""
                    MATCH (n)
                    WHERE (n:Concept OR n:Topic OR n:WorkContext) AND n.name = $name
                    RETURN labels(n)[0] AS type, n.name AS name
                """, name=concept)
                
                for record in result:
                    evidence["direct_concepts"].append({
                        "name": record["name"],
                        "type": record["type"]
                    })
            
            # If no direct concepts found, try fuzzy matching
            if not evidence["direct_concepts"]:
                result = session.run("""
                    MATCH (n)
                    WHERE (n:Concept OR n:Topic OR n:WorkContext)
                    AND toLower(n.name) CONTAINS toLower($partial)
                    RETURN labels(n)[0] AS type, n.name AS name
                """, partial=query_concepts[0] if query_concepts else "")
                
                for record in result:
                    evidence["direct_concepts"].append({
                        "name": record["name"],
                        "type": record["type"]
                    })
            
            # Get related concepts
            for concept in query_concepts:
                # For Concept nodes, get directly connected concepts
                result = session.run("""
                    MATCH (c1:Concept {name: $name})-[r]->(c2:Concept)
                    RETURN c1.name AS source, type(r) AS relationship, c2.name AS target
                    UNION
                    MATCH (c1:Concept)-[r]->(c2:Concept {name: $name})
                    RETURN c1.name AS source, type(r) AS relationship, c2.name AS target
                """, name=concept)
                
                for record in result:
                    evidence["concept_relationships"].append({
                        "source": record["source"],
                        "relationship": record["relationship"],
                        "target": record["target"]
                    })
                
                # For any node type, get concepts that co-occur in observations
                result = session.run("""
                    MATCH (c1)-[:INVOLVES_CONCEPT|INVOLVES_TOPIC|PART_OF]-(o:Observation)-[:INVOLVES_CONCEPT]->(c2:Concept)
                    WHERE (c1:Concept OR c1:Topic OR c1:WorkContext) AND c1.name = $name AND c1 <> c2
                    RETURN c2.name AS related_concept, count(o) AS strength
                    ORDER BY strength DESC
                    LIMIT 5
                """, name=concept)
                
                for record in result:
                    evidence["related_concepts"].append({
                        "name": record["related_concept"],
                        "strength": record["strength"],
                        "source": concept
                    })
            
            # Get work contexts
            for concept in query_concepts:
                result = session.run("""
                    MATCH (c)-[:INVOLVES_CONCEPT|INVOLVES_TOPIC]-(o:Observation)-[:PART_OF]->(w:WorkContext)
                    WHERE (c:Concept OR c:Topic) AND c.name = $name
                    RETURN w.name AS context, count(o) AS occurrence_count
                    ORDER BY occurrence_count DESC
                """, name=concept)
                
                for record in result:
                    evidence["work_contexts"].append({
                        "name": record["context"],
                        "count": record["occurrence_count"],
                        "concept": concept
                    })
            
            # Get topic clusters for these concepts
            for concept in query_concepts:
                result = session.run("""
                    MATCH (c:Concept {name: $name})<-[:INVOLVES_CONCEPT]-(o:Observation)-[:INVOLVES_TOPIC]->(t:Topic)
                    WITH t, count(o) AS relevance
                    ORDER BY relevance DESC
                    LIMIT 3
                    MATCH (t)<-[:INVOLVES_TOPIC]-(o:Observation)-[:INVOLVES_CONCEPT]->(other:Concept)
                    WHERE other.name <> $name
                    WITH t, collect(DISTINCT other.name) AS related_concepts
                    RETURN t.name AS topic, related_concepts
                    LIMIT 3
                """, name=concept)
                
                for record in result:
                    evidence["topic_clusters"].append({
                        "topic": record["topic"],
                        "concepts": record["related_concepts"][:5]  # Limit to 5 concepts per topic
                    })
            
            # Apply timeframe filtering if specified
            if timeframe:
                # Get observations for the concepts within timeframe
                timeframe_filter = ""
                params = {"concepts": query_concepts}
                
                if timeframe.get("start"):
                    timeframe_filter += "AND o.timestamp >= $start_date "
                    params["start_date"] = timeframe["start"]
                
                if timeframe.get("end"):
                    timeframe_filter += "AND o.timestamp <= $end_date "
                    params["end_date"] = timeframe["end"]
                
                result = session.run(f"""
                    MATCH (c)-[:INVOLVES_CONCEPT|INVOLVES_TOPIC]-(o:Observation)
                    WHERE (c:Concept OR c:Topic) AND c.name IN $concepts {timeframe_filter}
                    RETURN o.summary AS summary, o.timestamp AS timestamp
                    ORDER BY o.timestamp DESC
                    LIMIT 5
                """, **params)
                
                evidence["time_specific_observations"] = [
                    {"summary": record["summary"], "timestamp": record["timestamp"]}
                    for record in result
                ]
        
        return evidence

    def _get_general_work_evidence(self, timeframe: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get general evidence about recent work when no specific concepts are mentioned.
        """
        evidence = {
            "recent_observations": [],
            "main_work_contexts": [],
            "key_topics": [],
            "central_concepts": []
        }
        
        with self.neo4j_driver.session() as session:
            # Get recent observations
            timeframe_filter = ""
            params = {}
            
            if timeframe:
                if timeframe.get("start"):
                    timeframe_filter += "WHERE o.timestamp >= $start_date "
                    params["start_date"] = timeframe["start"]
                
                if timeframe.get("end"):
                    if "WHERE" in timeframe_filter:
                        timeframe_filter += "AND o.timestamp <= $end_date "
                    else:
                        timeframe_filter += "WHERE o.timestamp <= $end_date "
                    params["end_date"] = timeframe["end"]
            
            result = session.run(f"""
                MATCH (o:Observation)
                {timeframe_filter}
                RETURN o.summary AS summary, o.timestamp AS timestamp
                ORDER BY o.timestamp DESC
                LIMIT 10
            """, **params)
            
            evidence["recent_observations"] = [
                {"summary": record["summary"], "timestamp": record["timestamp"]}
                for record in result
            ]
            
            # Get main work contexts for this timeframe
            timeframe_filter_contexts = timeframe_filter.replace("o.", "obs.")
            result = session.run(f"""
                MATCH (w:WorkContext)<-[:PART_OF]-(obs:Observation)
                {timeframe_filter_contexts}
                WITH w, count(obs) AS obs_count
                ORDER BY obs_count DESC
                LIMIT 5
                RETURN w.name AS context, obs_count
            """, **params)
            
            evidence["main_work_contexts"] = [
                {"name": record["context"], "count": record["obs_count"]}
                for record in result
            ]
            
            # Get key topics for this timeframe
            timeframe_filter_topics = timeframe_filter.replace("o.", "obs.")
            result = session.run(f"""
                MATCH (t:Topic)<-[:INVOLVES_TOPIC]-(obs:Observation)
                {timeframe_filter_topics}
                WITH t, count(obs) AS topic_count
                ORDER BY topic_count DESC
                LIMIT 5
                RETURN t.name AS topic, topic_count
            """, **params)
            
            evidence["key_topics"] = [
                {"name": record["topic"], "count": record["topic_count"]}
                for record in result
            ]
            
            # Get central concepts for this timeframe
            timeframe_filter_concepts = timeframe_filter.replace("o.", "obs.")
            result = session.run(f"""
                MATCH (c:Concept)<-[:INVOLVES_CONCEPT]-(obs:Observation)
                {timeframe_filter_concepts}
                WITH c, count(obs) AS concept_count
                ORDER BY concept_count DESC
                LIMIT 10
                RETURN c.name AS concept, concept_count
            """, **params)
            
            evidence["central_concepts"] = [
                {"name": record["concept"], "count": record["concept_count"]}
                for record in result
            ]
        
        return evidence

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

    def create_prompt_with_graph_context(self, query: str, evidence: Dict[str, Any], conversation_history: str) -> str:
        """
        Create a prompt for the LLM that includes graph evidence and conversation context.
        """
        # Format the evidence for the prompt
        formatted_evidence = self._format_evidence_for_prompt(evidence)
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant with access to a knowledge graph representing a user's work activities, topics, and concepts.
        You need to answer the user's question based on evidence from their knowledge graph.
        
        Previous conversation:
        {conversation_history}
        
        User's question: {query}
        
        Evidence from the knowledge graph:
        {evidence}
        
        Based on this evidence from the knowledge graph, provide a helpful response to the user's question.
        
        Focus on giving a direct, informative answer that synthesizes the graph evidence. 
        If the evidence doesn't contain enough information to answer the question, acknowledge that limitation
        and provide the best answer you can based on available evidence.
        
        Make your response conversational and natural, without explicitly referencing the "knowledge graph" or "evidence" in your answer.
        """)
        
        return prompt.format(
            query=query,
            evidence=formatted_evidence,
            conversation_history=conversation_history
        )

    def _format_evidence_for_prompt(self, evidence: Dict[str, Any]) -> str:
        """
        Format the graph evidence into a string for the prompt.
        """
        sections = []
        
        # Direct concepts
        if evidence.get("direct_concepts"):
            section = "DIRECT CONCEPTS:\n"
            for concept in evidence["direct_concepts"]:
                section += f"- {concept['name']} (Type: {concept['type']})\n"
            sections.append(section)
        
        # Concept relationships
        if evidence.get("concept_relationships"):
            section = "CONCEPT RELATIONSHIPS:\n"
            for rel in evidence["concept_relationships"]:
                section += f"- {rel['source']} {rel['relationship'].replace('_', ' ').lower()} {rel['target']}\n"
            sections.append(section)
        
        # Related concepts
        if evidence.get("related_concepts"):
            section = "RELATED CONCEPTS:\n"
            for concept in evidence["related_concepts"]:
                section += f"- {concept['name']} (related to {concept['source']})\n"
            sections.append(section)
        
        # Work contexts
        if evidence.get("work_contexts"):
            section = "WORK CONTEXTS:\n"
            for context in evidence["work_contexts"]:
                section += f"- {context['name']} (related to {context['concept']})\n"
            sections.append(section)
        
        # Topic clusters
        if evidence.get("topic_clusters"):
            section = "TOPIC CLUSTERS:\n"
            for cluster in evidence["topic_clusters"]:
                section += f"- Topic: {cluster['topic']}\n"
                section += f"  Related concepts: {', '.join(cluster['concepts'])}\n"
            sections.append(section)
        
        # Time-specific observations
        if evidence.get("time_specific_observations"):
            section = "RELEVANT OBSERVATIONS:\n"
            for obs in evidence["time_specific_observations"]:
                formatted_date = obs["timestamp"]
                if isinstance(formatted_date, str):
                    try:
                        dt = date_parser.parse(formatted_date)
                        formatted_date = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        pass
                section += f"- [{formatted_date}] {obs['summary']}\n"
            sections.append(section)
        
        # Recent observations (for general queries)
        if evidence.get("recent_observations"):
            section = "RECENT ACTIVITIES:\n"
            for obs in evidence["recent_observations"]:
                formatted_date = obs["timestamp"]
                if isinstance(formatted_date, str):
                    try:
                        dt = date_parser.parse(formatted_date)
                        formatted_date = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        pass
                section += f"- [{formatted_date}] {obs['summary']}\n"
            sections.append(section)
        
        # Main work contexts (for general queries)
        if evidence.get("main_work_contexts"):
            section = "MAIN WORK CONTEXTS:\n"
            for context in evidence["main_work_contexts"]:
                section += f"- {context['name']} ({context['count']} observations)\n"
            sections.append(section)
        
        # Key topics (for general queries)
        if evidence.get("key_topics"):
            section = "KEY TOPICS:\n"
            for topic in evidence["key_topics"]:
                section += f"- {topic['name']} ({topic['count']} occurrences)\n"
            sections.append(section)
        
        # Central concepts (for general queries)
        if evidence.get("central_concepts"):
            section = "CENTRAL CONCEPTS:\n"
            for concept in evidence["central_concepts"]:
                section += f"- {concept['name']} ({concept['count']} occurrences)\n"
            sections.append(section)
        
        return "\n".join(sections)

    def toggle_context_display(self):
        """
        Toggle whether to show context information in the chat.
        """
        self.show_context = not self.show_context
        return f"Context display is now {'enabled' if self.show_context else 'disabled'}"

    def process_query(self, query: str) -> str:
        """
        Process a user query and return a response based on the concept graph.
        Now includes printing the actual context used for the LLM prompt.
        """
        # Check for context display commands
        if query.lower() in ["show context", "enable context", "display context"]:
            self.show_context = True
            return "Context display is now enabled. You'll see the exact evidence used for responses."
        
        if query.lower() in ["hide context", "disable context"]:
            self.show_context = False
            return "Context display is now disabled."
            
        # Extract concepts and timeframe from query
        concepts, timeframe = self.extract_query_elements(query)
        
        # Get evidence from the graph
        evidence = self.get_graph_evidence(concepts, timeframe)
        
        # Get conversation context
        conversation_context = self.get_conversation_context()
        
        # Format the evidence exactly as it will be sent to the LLM
        formatted_evidence_for_prompt = self._format_evidence_for_prompt(evidence)
        
        # Create prompt with graph context
        prompt = self.create_prompt_with_graph_context(query, evidence, conversation_context)
        
        # Generate response
        response = self.llm.invoke(prompt)
        
        # Save to memory
        self.memory.chat_memory.add_user_message(query)
        self.memory.chat_memory.add_ai_message(response.content)
        
        # Final response - either with or without context information
        if self.show_context:
            # Show the exact evidence that was used for the LLM prompt
            extracted_info = []
            if concepts:
                extracted_info.append(f"Extracted Concepts: {', '.join(concepts)}")
            if timeframe:
                timeframe_str = ""
                if timeframe.get("start"):
                    timeframe_str += f"From: {timeframe['start']} "
                if timeframe.get("end"):
                    timeframe_str += f"To: {timeframe['end']}"
                extracted_info.append(f"Extracted Timeframe: {timeframe_str}")
            
            extracted_section = "\n".join(extracted_info) + "\n\n" if extracted_info else ""
            
            return f"""📊 **GRAPH CONTEXT USED FOR RESPONSE:**
{extracted_section}```
{formatted_evidence_for_prompt}
```

**Response:**
{response.content}"""
        else:
            return response.content

    # Removed the _format_evidence_for_human_display method as we're now showing the exact prompt evidence


def run_chatbot():
    """
    Run the concept graph chatbot in an interactive loop.
    """
    # Set up the chatbot
    chatbot = ConceptGraphChatbot(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD
    )
    
    print("\n=== Concept Graph Chatbot ===")
    print("Ask questions about your work or type 'exit' to quit.")
    print("Special commands:")
    print("- 'show context' - Display the exact graph evidence used in prompts")
    print("- 'hide context' - Hide the graph evidence")
    print("\nExample questions:")
    print("- What was I working on last week?")
    print("- Tell me about my Docker-related work.")
    print("- What concepts are related to Neo4j?")
    print("- What was my main focus yesterday?")
    print("- What topics have I been exploring?\n")
    
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
    run_chatbot()
