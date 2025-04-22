from typing import Dict, TypedDict, List, Annotated
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
import logging
from pinecone import Pinecone, ServerlessSpec

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()


pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))


class EmailWorkflowState(TypedDict):
    email_subject: str
    email_body: str
    pdf_content: str
    pdf_name: str
    related_documents: List[Dict]
    writing_style: List[str]
    response: str
    final_response: str
    error: str

def initialize_pinecone_index():
    index_name = "pdf-documents"
    namespace = "email-twin"
    
    try:
        indexes = pc.list_indexes()
        if index_name not in [index.name for index in indexes]:
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info(f"Created index {index_name}")
        index = pc.Index(index_name)
        logger.info(f"Connected to index {index_name}")
        return index, namespace
    except Exception as e:
        logger.error(f"Error initializing Pinecone index: {e}")
        return None, namespace

index, namespace = initialize_pinecone_index()
def extract_pdf_content(state: EmailWorkflowState) -> EmailWorkflowState:
    """Extract content from PDF"""
    try:
        from PyPDF2 import PdfReader
        import io
        
        # For demo purposes, assuming pdf_content is already in state
        # In real application, you'd extract it from the PDF file
        if not state.get("pdf_content"):
            logger.error("No PDF content provided")
            return {"error": "No PDF content provided", **state}
            
        logger.info(f"Extracted content from {state.get('pdf_name', 'PDF')}")
        return state
    except Exception as e:
        logger.error(f"Error extracting PDF content: {e}")
        return {"error": f"Error extracting PDF content: {e}", **state}

def get_embedding(text):
    """Get embedding for text using OpenAI's API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return [0] * 1536 

def search_similar_documents(state: EmailWorkflowState) -> EmailWorkflowState:
    """Search for similar documents in vector store"""
    try:
        if index is None:
            logger.error("Pinecone index not initialized")
            return {"error": "Vector database not initialized", **state}
            
        # Get embedding for query
        query = f"{state.get('email_subject', '')} {state.get('email_body', '')[:200]}"
        query_embedding = get_embedding(query)
        
        # Search
        results = index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        
        # Format results
        similar_docs = []
        for match in results.matches:
            similar_docs.append({
                'id': match.id,
                'file_name': match.metadata.get('file_name', 'Unknown'),
                'content': match.metadata.get('content', ''),
                'email_subject': match.metadata.get('email_subject', ''),
                'email_body': match.metadata.get('email_body', ''),
                'similarity': match.score
            })
        
        state["related_documents"] = similar_docs
        logger.info(f"Found {len(similar_docs)} related documents")
        return state
    except Exception as e:
        logger.error(f"Error searching for similar documents: {e}")
        return {"error": f"Error searching for similar documents: {e}", **state}

def generate_draft_response(state: EmailWorkflowState) -> EmailWorkflowState:
    """Generate initial response draft using LLM"""
    try:
        logger.info("Generating draft response...")
        
        # Create prompt
        style_examples = "\n".join(state.get("writing_style", [
            "I hope this email finds you well. Thanks for sharing the document.",
            "Looking at the data you've provided, I think we can proceed with the next steps.",
            "Let me know if you need any clarification on the points I've raised.",
            "I appreciate your prompt response on this matter."
        ]))
        
        related_content = ""
        for doc in state.get("related_documents", []):
            related_content += f"Document: {doc.get('file_name', 'Unknown')}\n"
            related_content += f"Email Subject: {doc.get('email_subject', '')}\n"
            related_content += f"PDF Content: {doc.get('content', '')[:500]}...\n\n"
       
        prompt = f"""
        You are my digital twin, tasked with responding to an email. Your response should match my writing style based on these examples:
        
        {style_examples}
        
        The email I received contains:
        {state.get('email_body', '')}
        
        The attached PDF contains:
        {state.get('pdf_content', '')[:2000] if len(state.get('pdf_content', '')) > 2000 else state.get('pdf_content', '')}
        
        Previous related documents and emails:
        {related_content}
        
        Please draft a response that:
        1. Addresses the specific points raised in the email
        2. References relevant information from the PDF
        3. Maintains continuity with any previous communications
        4. Matches my writing style
        """
       
        logger.info("Calling LLM with prompt...")
        llm = ChatOpenAI(
            model="gpt-4", 
            api_key=os.getenv('OPENAI_API_KEY'),
            temperature=0.7
        )
        messages = [
            SystemMessage(content="You are a digital twin email assistant that mimics the user's writing style."),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        
        if not response or not hasattr(response, 'content'):
            logger.error("LLM returned empty or invalid response")
            return {
                "error": "Failed to generate response: LLM returned empty response",
                **state
            }
        
        logger.info("Successfully generated draft response")
        return {
            "response": response.content,
            "final_response": response.content, 
            **state
        }
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {"error": f"Error generating response: {e}", **state}

def store_document(state: EmailWorkflowState) -> EmailWorkflowState:
    """Store document in vector store"""
    try:
        if index is None:
            logger.error("Pinecone index not initialized")
            return {"error": "Vector database not initialized", **state}
            
        # Generate unique ID
        import uuid
        from datetime import datetime
        doc_id = str(uuid.uuid4())
        
        # Get embedding for content
        embedding = get_embedding(state.get('pdf_content', ''))
        
        # Prepare metadata
        metadata = {
            "file_name": state.get('pdf_name', 'Unknown'),
            "content": state.get('pdf_content', '')[:8000], 
            "email_subject": state.get('email_subject', ''),
            "email_body": state.get('email_body', '')[:8000], 
            "created_at": datetime.now().isoformat()
        }
        
        # Insert data
        index.upsert(
            vectors=[
                {
                    "id": doc_id,
                    "values": embedding,
                    "metadata": metadata
                }
            ],
            namespace=namespace
        )
        
        logger.info(f"Stored document {state.get('pdf_name', 'Unknown')} with ID {doc_id}")
        return state
    except Exception as e:
        logger.error(f"Error storing document: {e}")
        return {"error": f"Error storing document: {e}", **state}

def finalize_response(state: EmailWorkflowState) -> EmailWorkflowState:
    """Finalize the response"""
    try:
        # Here you might do additional processing on the response
        # For now, just pass it through
        state["final_response"] = state.get("response", "")
        logger.info("Finalized response")
        return state
    except Exception as e:
        logger.error(f"Error finalizing response: {e}")
        return {"error": f"Error finalizing response: {e}", **state}

def router(state: EmailWorkflowState) -> str:
    """Route based on state"""
    if state.get("error"):
        logger.error(f"Workflow ended with error: {state.get('error')}")
        return "handle_error"
    return "continue"

def build_email_workflow_graph():
  
    workflow = StateGraph(EmailWorkflowState)
    workflow.add_node("extract_pdf", extract_pdf_content)
    workflow.add_node("search_documents", search_similar_documents)
    workflow.add_node("generate_response", generate_draft_response)
    workflow.add_node("store_document", store_document)
    workflow.add_node("finalize", finalize_response)
    workflow.add_node("handle_error", lambda state: {"final_response": f"Error: {state.get('error')}", **state})
    workflow.set_entry_point("extract_pdf")
    workflow.add_conditional_edges(
        "extract_pdf",
        router,
        {
            "continue": "search_documents",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "search_documents",
        router,
        {
            "continue": "generate_response",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "generate_response",
        router,
        {
            "continue": "store_document",
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "store_document",
        router,
        {
            "continue": "finalize",
            "handle_error": "handle_error"
        }
    )
    
    # Add final edges
    workflow.add_edge("finalize", END)
    workflow.add_edge("handle_error", END)
    
    # Compile the graph
    return workflow.compile()

# Function to run the workflow
def process_email_with_pdf(email_subject: str, email_body: str, pdf_content: str, pdf_name: str, writing_style: List[str] = None):
    """Process an email with PDF attachment using the LangGraph workflow"""
    try:
        email_processor = build_email_workflow_graph()
        
        # Initialize the state
        initial_state = EmailWorkflowState(
            email_subject=email_subject,
            email_body=email_body,
            pdf_content=pdf_content,
            pdf_name=pdf_name,
            related_documents=[],
            writing_style=writing_style or [
                "I hope this email finds you well. Thanks for sharing the document.",
                "Looking at the data you've provided, I think we can proceed with the next steps.",
                "Let me know if you need any clarification on the points I've raised.",
                "I appreciate your prompt response on this matter."
            ],
            response="",
            final_response="",
            error=""
        )
        
        # Run the workflow
        logger.info("Starting email workflow")
        result = email_processor.invoke(initial_state)
        logger.info(f"Workflow completed. Result has keys: {list(result.keys() if result else [])}")
        
        # Ensure we always return something useful
        if not result:
            return {
                "final_response": "The workflow did not produce any results",
                "error": "Empty workflow result"
            }
        
        if "final_response" not in result or not result.get("final_response"):
            # If we don't have a final_response but have a response, use that
            if "response" in result and result.get("response"):
                result["final_response"] = result["response"]
            # Otherwise set a default error message
            else:
                result["final_response"] = "Failed to generate a response"
                if "error" not in result or not result.get("error"):
                    result["error"] = "Unknown error in workflow execution"
        
        return result
    except Exception as e:
        logger.error(f"Error in workflow execution: {e}")
        return {
            "final_response": f"An error occurred: {str(e)}",
            "error": str(e)
        } 