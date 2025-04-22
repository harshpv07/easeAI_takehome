from typing import Dict, List, Optional
import os
import io
import base64
import PyPDF2
import streamlit as st
import logging
import uuid
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class EmailAssistant:
    def __init__(self):
        """Initialize the Email Assistant with connections to OpenAI and Pinecone"""
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.llm = ChatOpenAI(
            model="gpt-4",
            api_key=os.getenv('OPENAI_API_KEY'),
            temperature=0.7
        )

        self.conversation_history = []

        self.style_examples = [
            "I hope this email finds you well. Thanks for sharing the document.",
            "Looking at the data you've provided, I think we can proceed with the next steps.",
            "Let me know if you need any clarification on the points I've raised.",
            "I appreciate your prompt response on this matter."
        ]

        self.connect_to_pinecone()
        self.initialize_index()
        
    def connect_to_pinecone(self):
        """Connect to Pinecone vector database."""
        try:
            self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            logger.info("Connected to Pinecone")
        except Exception as e:
            logger.error(f"Error connecting to Pinecone: {e}")
            return False
        return True
            
    def initialize_index(self):
        """Initialize Pinecone index if it doesn't exist."""
        self.index_name = "pdf-documents"
        self.namespace = "email-twin"
        
        try:
            indexes = self.pc.list_indexes()
            
            if self.index_name not in [index.name for index in indexes]:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                logger.info(f"Created index {self.index_name}")
            
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to index {self.index_name}")
            return True
        except Exception as e:
            logger.error(f"Error initializing Pinecone index: {e}")
            return False
    
    def extract_text_from_pdf(self, pdf_content):
        """Extract text content from PDF bytes."""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return "Could not extract text from PDF."
    
    def get_embedding(self, text):
        """Get embedding for text using OpenAI's API."""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return [0] * 1536 
    
    def store_document(self, file_name, content, email_subject, email_body):
        """Store document in Pinecone with its embedding."""
        try:
            doc_id = str(uuid.uuid4())
            embedding = self.get_embedding(content)
            
            metadata = {
                "file_name": file_name,
                "content": content[:8000], 
                "email_subject": email_subject,
                "email_body": email_body[:8000],  
                "created_at": datetime.now().isoformat()
            }
            
            self.index.upsert(
                vectors=[
                    {
                        "id": doc_id,
                        "values": embedding,
                        "metadata": metadata
                    }
                ],
                namespace=self.namespace
            )
            logger.info(f"Stored document {file_name} with ID {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            return None
    
    def search_similar_documents(self, query_text, limit=5):
        """Search for similar documents in Pinecone."""
        try:
            query_embedding = self.get_embedding(query_text)
            results = self.index.query(
                namespace=self.namespace,
                vector=query_embedding,
                top_k=limit,
                include_metadata=True
            )
            
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
            
            return similar_docs
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []
    
    def generate_response(self, email_subject, email_body, pdf_content, pdf_name, additional_input=""):
        """Generate a response using LangChain with direct message passing."""
        try:

            query = f"{email_subject} {email_body[:200]}"
            related_documents = self.search_similar_documents(query)
   
            related_content = ""
            for doc in related_documents:
                related_content += f"Document: {doc.get('file_name', 'Unknown')}\n"
                related_content += f"Email Subject: {doc.get('email_subject', '')}\n"
                related_content += f"PDF Content: {doc.get('content', '')[:500]}...\n\n"

            style_examples = "\n".join(self.style_examples)
            truncated_pdf_content = pdf_content[:2000] if len(pdf_content) > 2000 else pdf_content
       
            system_content = f"""
            You are my digital twin, tasked with responding to an email. Your response should match my writing style based on these examples:
            
            {style_examples}
            
            Previous related documents and emails:
            {related_content}
            """
  
            messages = [
                SystemMessage(content=system_content)
            ]

            for message in self.conversation_history:
                messages.append(message)

            human_content = f"""
            The email I received contains:
            Subject: {email_subject}
            Body: {email_body}
            
            The attached PDF contains:
            {truncated_pdf_content}
            
            Additional Input:
            {additional_input}
            
            Please draft a response that:
            1. Addresses the specific points raised in the email
            2. References relevant information from the PDF
            3. Maintains continuity with any previous communications
            4. Matches my writing style
            """
            
            messages.append(HumanMessage(content=human_content))
   
            response = self.llm.invoke(messages)

            self.conversation_history.append(HumanMessage(content=human_content))
            self.conversation_history.append(AIMessage(content=response.content))
   
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
   
            self.store_document(pdf_name, pdf_content, email_subject, email_body)
            
            return response.content
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {e}"

def process_email_with_pdf(email_subject: str, email_body: str, pdf_content: bytes, pdf_name: str, writing_style: List[str] = None, additional_input: str = ""):
    """Process an email with PDF attachment using LangChain"""
    try:
   
        assistant = EmailAssistant()
        if writing_style:
            assistant.style_examples = writing_style
        pdf_text = assistant.extract_text_from_pdf(pdf_content)
        response = assistant.generate_response(
            email_subject=email_subject,
            email_body=email_body,
            pdf_content=pdf_text,
            pdf_name=pdf_name,
            additional_input=additional_input
        )
        
        return {
            "final_response": response,
            "error": ""
        }
        
    except Exception as e:
        logger.error(f"Error in email processing: {e}")
        return {
            "final_response": f"An error occurred: {str(e)}",
            "error": str(e)
        }

def main():
    st.set_page_config(page_title="Digital Twin Email Responder", layout="wide")
    
    st.title("Digital Twin Email Responder")
    st.markdown("""
    This application simulates a digital twin that generates responses to emails with PDF attachments,
    mimicking your writing style and incorporating context from the documents.
    """)

    digital_twin = EmailAssistant()
    with st.sidebar:
        st.header("Customize Your Style")
        st.markdown("Edit these examples to reflect your writing style:")
        
        new_examples = []
        for i, example in enumerate(digital_twin.style_examples):
            new_example = st.text_area(f"Style Example {i+1}", example, height=100)
            new_examples.append(new_example)
        if st.button("Update Style"):
            digital_twin.style_examples = new_examples
            st.success("Style examples updated!")
    
    tab1, tab2 = st.tabs(["Compose & Generate Response", "Search Previous Documents"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Email Input")
            email_subject = st.text_input("Email Subject")
            email_body = st.text_area("Email Body", height=200)
            uploaded_file = st.file_uploader("Upload PDF Attachment", type="pdf")
            
            if uploaded_file is not None:
                st.subheader("PDF Preview")
                base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="300" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
        
        with col2:
            st.header("Generated Response")
            
            if st.button("Generate Response"):
                if not email_body:
                    st.error("Please enter an email body.")
                elif not uploaded_file:
                    st.error("Please upload a PDF file.")
                else:
                    with st.spinner("Processing PDF and generating response..."):
                        pdf_content = digital_twin.extract_text_from_pdf(uploaded_file.getvalue())
                        response = digital_twin.generate_response(
                            email_subject=email_subject,
                            email_body=email_body,
                            pdf_content=pdf_content,
                            pdf_name=uploaded_file.name
                        )
                        
                        st.text_area("Response", response, height=400)
                        
                        response_bytes = response.encode()
                        st.download_button(
                            label="Download Response",
                            data=response_bytes,
                            file_name="response.txt",
                            mime="text/plain"
                        )
    
    with tab2:
        st.header("Search Previous Documents")
        search_query = st.text_input("Search Query")
        
        if st.button("Search"):
            if search_query:
                with st.spinner("Searching..."):
                    results = digital_twin.search_similar_documents(search_query)
                    
                    if results:
                        st.success(f"Found {len(results)} results")
                        for i, result in enumerate(results):
                            with st.expander(f"Result {i+1}: {result['file_name']}"):
                                st.write(f"**Email Subject:** {result['email_subject']}")
                                st.write(f"**Email Body:**\n{result['email_body']}")
                                st.write(f"**PDF Content Preview:**\n{result['content'][:500]}...")
                    else:
                        st.info("No results found")
            else:
                st.error("Please enter a search query")

if __name__ == "__main__":
    main()
