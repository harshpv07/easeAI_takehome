# Digital Twin Email Responder

A Streamlit application that simulates a digital twin to generate responses to emails with PDF attachments, mimicking your writing style and incorporating context from previous documents.

## Features

- Extract text from PDF attachments
- Store documents and their contexts in Pinecone vector database
- Search for similar documents using semantic search
- Generate contextual email responses using OpenAI's GPT model
- Customize your writing style examples

## Setup

1. Install dependencies:
```
pip install streamlit python-dotenv openai pinecone-client PyPDF2
```

2. Set up your environment variables by editing the `.env` file:
```
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
```

3. Create a free Pinecone account at https://www.pinecone.io/ to get your API key

## Running the Application

Run the following command:
```
streamlit run main.py
```

## Usage

1. **Customizing Your Style**:
   - Edit the style examples in the sidebar to match your writing style

2. **Composing and Generating Responses**:
   - Enter the email subject and body you received
   - Upload the PDF attachment
   - Click "Generate Response" to create a personalized reply

3. **Searching Previous Documents**:
   - Use the "Search Previous Documents" tab to find similar past documents
   - Enter a search query and click "Search"

