import os
import streamlit as st
from groq import Groq
from PyPDF2 import PdfReader
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import time
import shutil


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

CHROMA_PATH = "streamlit_chroma/"
COLLECTION_NAME = "streamlit_collection"
EMBEDDING_FUNC_NAME = "multi-qa-MiniLM-L6-cos-v1"


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def split_text_into_chunks(text, chunk_size=512):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


def populate_chroma_collection(chroma_path, collection_name, embedding_func_name, texts, chunk_size=200):
    # Chroma Client
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    time.sleep(0.5)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_func_name
    )
    collection = client.get_or_create_collection(collection_name, embedding_function=embedding_func)

    for idx, text in enumerate(texts):
        chunks = split_text_into_chunks(text, chunk_size)
        embeddings = embedder.encode(chunks, convert_to_tensor=False).tolist()
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": f"text_{idx}"}] * len(chunks),
            ids=[f"text_{idx}_chunk_{i}" for i in range(len(chunks))],
        )


def retrieve_relevant_chunks(chroma_path, collection_name, embedding_func_name, query_text, n_results=10):
    # Chroma Client
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    time.sleep(0.5)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_func_name
    )
    collection = client.get_collection(name=collection_name, embedding_function=embedding_func)
    query_result = collection.query(query_texts=[query_text], n_results=n_results)

    results = []
    for document, metadata, score in zip(
            query_result["documents"], query_result["metadatas"], query_result["distances"]
    ):
        results.append({"chunk": document, "metadata": metadata, "score": score})
    return results


def get_answer_based_on_chunks(query_text, relevant_chunks, chat_history):
    reviews_str = "\n".join([f"{index + 1}. {result['chunk']}" for index, result in enumerate(relevant_chunks)])
    context = "Provide answers based on the following text chunks."

    # Prepare messages, including the system message
    messages = [{"role": "system", "content": context}]
    
    # Extract role and content from HumanMessage and AIMessage objects
    for msg in chat_history.messages:
        # Map msg.type to valid role values
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = "system"  # Default case (could be expanded if needed)

        messages.append({
            "role": role,  # Use mapped role
            "content": msg.content  # Access content directly
        })

    messages.append({
        "role": "user", 
        "content": f"""
            Based on the following text chunks:
            {reviews_str}
            Answer the following question:
            {query_text}
        """
    })

    chat_completion = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        temperature=0
    )
    return chat_completion.choices[0].message.content





# Streamlit 
def main():

    # Delete the old vector database, it was creating conflicts when I run again.
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
            print("The old vector database has deleted")
        except PermissionError as e:
            pass

    st.title("Chatbot with PDF Knowledge Base")
    st.write("Upload a PDF, ask questions, and track chat history across sessions!")



    # Initialize session state
    if "sessions" not in st.session_state:
        st.session_state.sessions = {}
    if "current_session" not in st.session_state:
        st.session_state.current_session = None

    # Sidebar for session management
    st.sidebar.title("Session Management")
    session_names = list(st.session_state.sessions.keys())
    selected_session = st.sidebar.selectbox(
        "Select a Session", options=["None"] + session_names
    )

    if selected_session != "None":
        st.session_state.current_session = selected_session

    if st.sidebar.button("Start New Session"):
        session_id = f"session_{len(st.session_state.sessions) + 1}"
        st.session_state.sessions[session_id] = {
            "messages": ChatMessageHistory(),
            "pdf_text": None,
            "chroma_initialized": False,
            "last_uploaded_file": None,
        }
        st.session_state.current_session = session_id
        st.sidebar.success(f"New session '{session_id}' started!")

    if st.session_state.current_session is None:
        st.info("Please start or select a session to begin.")
        return

    session = st.session_state.sessions[st.session_state.current_session]
    st.sidebar.write(f"Current Session: {st.session_state.current_session}")

    # File upload section
    if not session["chroma_initialized"]:  # Show the upload section only if ChromaDB is not populated
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf", key=st.session_state.current_session)

        # If a PDF is uploaded
        if uploaded_file:
            if uploaded_file.name != session.get("last_uploaded_file"):
                with st.spinner("Processing the PDF..."):
                    pdf_text = extract_text_from_pdf(uploaded_file)
                    if not pdf_text.strip():
                        st.error("The uploaded PDF does not contain any extractable text.")
                        return
                    session["pdf_text"] = pdf_text
                    session["last_uploaded_file"] = uploaded_file.name
                    session["chroma_initialized"] = False

            # Populate ChromaDB collection once for the PDF
            if not session["chroma_initialized"]:
                with st.spinner("Populating the ChromaDB collection..."):
                    session_collection_name = f"{COLLECTION_NAME}_{st.session_state.current_session}"
                    populate_chroma_collection(
                        chroma_path=CHROMA_PATH,
                        collection_name=session_collection_name,
                        embedding_func_name=EMBEDDING_FUNC_NAME,
                        texts=[session["pdf_text"]],
                        chunk_size=512
                    )
                    session["chroma_initialized"] = True
                    session["collection_name"] = session_collection_name  # Save collection name in session
                    st.success("ChromaDB collection is ready!")

                    time.sleep(0.5)
                    session["messages"].add_ai_message("Hey, how can I help you with your PDF?")  # AI bot message


    else:
        st.info("PDF already uploaded. You can ask questions now.")


    # Display chat messages from history
    for message in session["messages"].messages:
        # Determine the role based on message type
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            role = "unknown"  # Default case (could be handled further)

        content = message.content  
        with st.chat_message(role):  
            st.markdown(content)  


    # Input section for the query
    if session["pdf_text"]:
        if user_input := st.chat_input("Ask a question about the uploaded PDF:"):
            # Add user message to chat history
            session["messages"].add_user_message(user_input)
            with st.chat_message("user"):
                st.markdown(user_input)

            # Getting assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    relevant_chunks = retrieve_relevant_chunks(
                        chroma_path=CHROMA_PATH,
                        collection_name=session["collection_name"],
                        embedding_func_name=EMBEDDING_FUNC_NAME,
                        query_text=user_input,
                        n_results=10
                    )

                    response = get_answer_based_on_chunks(user_input, relevant_chunks, session["messages"])
                    session["messages"].add_ai_message(response)
                st.markdown(response)


if __name__ == "__main__":
    main()
