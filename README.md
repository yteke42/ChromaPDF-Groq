# ChromaPDF-Groq

[![GitHub](https://img.shields.io/badge/GitHub-ChromaPDF--Groq-blue)](https://github.com/yteke42/ChromaPDF-Groq)

A PDF chatbot powered by Groq AI and ChromaDB that enables intelligent document Q&A through vector-based search and natural language processing.

This project is a Streamlit-based chatbot application that allows users to upload large PDF documents and ask questions about their content. The application uses Groq's AI API for generating responses and ChromaDB for document storage and retrieval.  
The application uses streamlit_chroma to chunk the PDF based on relevance, then retrieves the most relevant sections and answers your questions using Groq.

## Features

- PDF document upload and text extraction
- Interactive chat interface
- Session management for multiple conversations
- Document-based question answering
- Persistent chat history
- Vector-based document search using ChromaDB

## Prerequisites

- Python 3.8 or higher
- Groq API key
- Required Python packages (see requirements.txt)

## Getting Started

### 1. Get a Groq API Key

1. Visit [Groq's website](https://console.groq.com/)
2. Sign up for an account
3. Navigate to the API keys section
4. Create a new API key
5. Copy your API key

### 2. Set Up Environment Variables

Create a `.env` file in the project root directory and add your Groq API key:

```
GROQ_API_KEY= "your_api_key_here"
```  
Your API Key should be written in quotation marks ("")  

### 3. Installation

1. Clone this repository:
```bash
git clone https://github.com/yteke42/ChromaPDF-Groq.git
cd ChromaPDF-Groq
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

### 4. Running the Application

Run the Streamlit app:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Usage

1. Start a new session using the sidebar
2. Upload a PDF document
3. Wait for the document to be processed
4. Start asking questions about the PDF content
5. The chatbot will provide answers based on the document's content

## How It Works

1. **Document Processing**:
   - PDFs are uploaded and their text is extracted
   - Text is split into chunks for better processing
   - Chunks are embedded and stored in ChromaDB

2. **Question Answering**:
   - User questions are processed to find relevant document chunks
   - The Groq AI model generates responses based on the relevant chunks
   - Chat history is maintained for context

3. **Session Management**:
   - Multiple sessions can be created and managed
   - Each session maintains its own chat history and document context

## Dependencies

The project uses several key libraries:
- Streamlit for the web interface
- Groq for AI-powered responses
- ChromaDB for vector storage
- PyPDF2 for PDF processing
- Sentence Transformers for text embeddings

## Note

This project requires specific versions of libraries to function correctly. Please use the provided `requirements.txt` file to ensure compatibility.  

## Contact  
- Yunus Teke - yunus.teke@metu.edu.tr  
- Atakan Karataş - atakan.karatas@metu.edu.tr
