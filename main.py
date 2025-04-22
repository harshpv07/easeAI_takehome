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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


load_dotenv()
print("hello")
class DigitalTwin:
    def __init__(self):
        
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
      
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
            st.error(f"Error connecting to Pinecone: {e}")
    
    def initialize_index(self):
        """Initialize Pinecone index if it doesn't exist."""
        index_name = "pdf-documents"
        self.namespace = "email-twin"
        
        try:
          
            indexes = self.pc.list_indexes()
            
           
            if index_name not in [index.name for index in indexes]:
               
                self.pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                logger.info(f"Created index {index_name}")
            
            
            self.index = self.pc.Index(index_name)
            logger.info(f"Connected to index {index_name}")
        except Exception as e:
            logger.error(f"Error initializing Pinecone index: {e}")
            st.error(f"Error initializing Pinecone index: {e}")
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text content from a PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            st.error(f"Error extracting text from PDF: {e}")
            return "Could not extract text from PDF."
    
    def get_embedding(self, text):
        """Get embedding for text using OpenAI's API."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            st.error(f"Error getting embedding: {e}")
            return [0] * 1536  # Return zero vector as fallback
    
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
            st.error(f"Error storing document: {e}")
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
            st.error(f"Error searching similar documents: {e}")
            return []
    
    def generate_response(self, email_body, pdf_content, related_documents, additional_input=""):
        """Generate a response using OpenAI's GPT model."""
        try:
            related_content = ""
            for doc in related_documents:
                related_content += f"Document: {doc['file_name']}\nEmail Subject: {doc['email_subject']}\nEmail Body: {doc['email_body']}\nPDF Content: {doc['content'][:500]}...\n\n"
            
            style_examples = "\n".join(self.style_examples)
            
            prompt = f"""
            You are my digital twin, tasked with responding to an email. Your response should match my writing style based on these examples:
            
            {style_examples}
            
            The email I received contains:
            {email_body}
            
            The attached PDF contains:
            {pdf_content[:2000] if len(pdf_content) > 2000 else pdf_content}
            
            Previous related documents and emails:
            {related_content}
            
            Additional Input:
            {additional_input}
            
            Please draft a response that:
            1. Addresses the specific points raised in the email
            2. References relevant information from the PDF
            3. Maintains continuity with any previous communications
            4. Matches my writing style
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a digital twin email assistant that mimics the user's writing style."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            st.error(f"Error generating response: {e}")
            return "I couldn't generate a proper response due to an error. Please try again."

def main():
    st.set_page_config(page_title="RespondMail", layout="wide")
    
    st.title("AutoMail")
    st.markdown("""
        Tired of replying to your mails, automate replies using AutoMail
    """)
    digital_twin = DigitalTwin()
    
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

            st.subheader("Additional Input")
            st.write("Provide any additional notes or instructions for your response:")
            additional_input = st.text_area("Additional Input", height=150, 
                                           placeholder="Example: Make the tone more formal, emphasize point X, include specific information about Y...")
        
        with col2:
            st.header("Generated Response")
            
            if st.button("Generate Response"):
                if not email_body:
                    st.error("Please enter an email body.")
                elif not uploaded_file:
                    st.error("Please upload a PDF file.")
                else:
                    with st.spinner("Processing PDF and generating response..."):
                        pdf_content = digital_twin.extract_text_from_pdf(uploaded_file)
                        
                        query = f"{email_subject} {email_body[:200]}"
                        related_documents = digital_twin.search_similar_documents(query)
                        
                        response = digital_twin.generate_response(
                            email_body, 
                            pdf_content, 
                            related_documents,
                            additional_input
                        )
                        
                        
                        doc_id = digital_twin.store_document(
                            uploaded_file.name, 
                            pdf_content, 
                            email_subject, 
                            email_body
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
