import os
import io
import logging
from PyPDF2 import PdfWriter, PdfReader
from main import DigitalTwin
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def create_sample_pdf(filename, content):
    """Create a sample PDF file with the given content."""
    try:
        # Create a text file with content
        with open("temp_content.txt", "w") as f:
            f.write(content)
        
        # Try to convert text to PDF using external tool if available
        os.system(f'echo "{content}" | txt2pdf -o {filename} 2>/dev/null')
        
        # Check if file was created
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            # Fallback: Create a simple PDF using PyPDF2
            writer = PdfWriter()
            writer.add_blank_page(width=612, height=792)  # Letter size
            
            # Write to file
            with open(filename, "wb") as outfile:
                writer.write(outfile)
            
            logger.info(f"Created blank PDF: {filename}")
            
            # Create a text file as backup
            with open(filename.replace('.pdf', '.txt'), 'w') as f:
                f.write(content)
            logger.info(f"Created text file with content: {filename.replace('.pdf', '.txt')}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating sample PDF: {e}")
        # Create a text file as fallback
        with open(filename.replace('.pdf', '.txt'), 'w') as f:
            f.write(content)
        logger.info(f"Created text file as fallback: {filename.replace('.pdf', '.txt')}")
        return False

def demo():
    # Create a sample PDF
    pdf_content = """
    Project Proposal: AI-Enhanced Customer Support System
    
    Executive Summary:
    This proposal outlines the development of an AI-enhanced customer support system 
    designed to improve response times, accuracy, and customer satisfaction. The system 
    will leverage natural language processing and machine learning to automate routine 
    inquiries while escalating complex issues to human agents.
    
    Key Features:
    1. Automated response generation for common inquiries
    2. Sentiment analysis for customer messages
    3. Priority queuing based on issue severity
    4. Integration with existing CRM systems
    5. Multilingual support
    
    Timeline:
    - Phase 1 (Months 1-2): Requirements gathering and system design
    - Phase 2 (Months 3-4): Development of core NLP components
    - Phase 3 (Months 5-6): Integration with existing systems
    - Phase 4 (Months 7-8): Testing and refinement
    - Phase 5 (Month 9): Deployment and training
    
    Budget:
    The estimated budget for this project is $250,000, including development, 
    integration, and training costs.
    
    ROI Analysis:
    Based on current customer support metrics, we project a 35% reduction in 
    response time and a 25% decrease in support costs within the first year 
    of implementation.
    """
    
    pdf_file = "sample_proposal.pdf"
    pdf_created = create_sample_pdf(pdf_file, pdf_content)
    
    # Create sample email data
    sample_email_subject = "Feedback on AI Customer Support Proposal"
    sample_email_body = """
    Hi there,
    
    Thank you for sending over the proposal for the AI-Enhanced Customer Support System. 
    I've had a chance to review it with my team, and we have a few questions:
    
    1. Does the proposed timeline account for potential integration challenges with our legacy systems?
    2. Can you provide more details about the training requirements for our staff?
    3. We're particularly interested in the multilingual support feature - what languages will be supported initially?
    
    We're excited about the potential ROI mentioned in the proposal and would like to schedule a meeting next week to discuss next steps.
    
    Best regards,
    John
    """
    
    # Initialize the digital twin
    logger.info("Initializing Digital Twin...")
    digital_twin = DigitalTwin()
    
    # Process the PDF
    logger.info("Processing PDF...")
    if os.path.exists(pdf_file):
        with open(pdf_file, 'rb') as f:
            pdf_bytes = io.BytesIO(f.read())
            # Extract content from the PDF
            pdf_content = digital_twin.extract_text_from_pdf(pdf_bytes)
    else:
        logger.warning(f"PDF file not found, using text content instead")
    
    # Store document in vector database
    logger.info("Storing document in Milvus...")
    doc_id = digital_twin.store_document(
        "sample_proposal.pdf",
        pdf_content,
        "Initial AI Customer Support Proposal",
        "As discussed in our meeting last week, attached is our proposal for the AI-Enhanced Customer Support System."
    )
    logger.info(f"Document stored with ID: {doc_id}")
    
    # Search for related documents
    logger.info("Searching for related documents...")
    query = f"{sample_email_subject} {sample_email_body[:200]}"
    related_documents = digital_twin.search_similar_documents(query)
    
    # Generate response
    logger.info("Generating response...")
    response = digital_twin.generate_response(sample_email_body, pdf_content, related_documents)
    
    # Print the results
    print("\n" + "="*80)
    print("INCOMING EMAIL:")
    print("="*80)
    print(f"Subject: {sample_email_subject}")
    print("\n" + sample_email_body)
    
    print("\n" + "="*80)
    print("PDF CONTENT SUMMARY:")
    print("="*80)
    print(pdf_content[:500] + "..." if len(pdf_content) > 500 else pdf_content)
    
    if related_documents:
        print("\n" + "="*80)
        print(f"FOUND {len(related_documents)} RELATED DOCUMENTS:")
        print("="*80)
        for i, doc in enumerate(related_documents):
            print(f"Document {i+1}: {doc['file_name']} (Similarity: {doc['similarity']:.4f})")
    
    print("\n" + "="*80)
    print("GENERATED RESPONSE:")
    print("="*80)
    print(response)
    
    # Clean up
    if os.path.exists(pdf_file):
        os.remove(pdf_file)
    if os.path.exists("temp_content.txt"):
        os.remove("temp_content.txt")
    if os.path.exists(pdf_file.replace('.pdf', '.txt')):
        os.remove(pdf_file.replace('.pdf', '.txt'))

if __name__ == "__main__":
    demo() 