import streamlit as st
import tempfile
import hashlib
import chromadb
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
    .stApp {
        background-color: #F5F7FB;
    }

    h1, h2, h3 {
        color: #172B4D;
    }

    .stButton > button {
        background-color: #176B55;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #105440;
        color: white;
        border: none;
    }

    [data-testid="stChatMessage"] {
        background-color: white;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
    }

    [data-testid="stSidebar"] {
        background-color: #EAF0F6;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODELS ----------------

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b"
    )


embedding_model = load_embedding_model()
llm = load_llm()


# ---------------- PROMPT ----------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant.

        Answer using ONLY the provided context.
        If the answer is not available in the context,
        say: "I could not find the answer in the document."

        Keep your answers clear and informative.
        """
    ),
    (
        "human",
        """
        Context:
        {context}

        Question:
        {question}
        """
    )
])


# ---------------- SESSION STATE ----------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_hash" not in st.session_state:
    st.session_state.processed_hash = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- HEADER ----------------

st.title("📚 PDF RAG Assistant")

st.caption(
    "Upload your PDF and ask questions. "
    "Get answers based on your document."
)

st.divider()


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    if uploaded_file:

        st.success(f"Selected: {uploaded_file.name}")

        if st.button("Process PDF", use_container_width=True):

            with st.spinner("Processing your PDF..."):

                temp_path = None

                try:
                    pdf_bytes = uploaded_file.getvalue()

                    file_hash = hashlib.sha256(
                        pdf_bytes
                    ).hexdigest()[:12]

                    # Save uploaded PDF temporarily
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(pdf_bytes)
                        temp_path = temp_file.name

                    # Load PDF
                    loader = PyPDFLoader(temp_path)
                    docs = loader.load()

                    if not docs:
                        raise ValueError("No content found in PDF.")

                    # Preserve original filename
                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name

                    # Split documents into chunks
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )

                    chunks = splitter.split_documents(docs)

                    if not chunks:
                        raise ValueError("No readable text found in PDF.")

                    # Unique, versioned collection
                    collection_name = (
                        f"pdf_{file_hash}_minilm384_v1"
                    )

                    # Remove old collection if it exists
                    client = chromadb.PersistentClient(
                        path="chroma-db"
                    )

                    try:
                        client.delete_collection(
                            name=collection_name
                        )
                    except Exception:
                        pass

                    # Create vector database
                    vectorstore = Chroma.from_documents(
                        documents=chunks,
                        embedding=embedding_model,
                        collection_name=collection_name,
                        persist_directory="chroma-db"
                    )

                    st.session_state.vectorstore = vectorstore
                    st.session_state.processed_hash = file_hash
                    st.session_state.messages = []

                    st.success("PDF processed successfully!")

                    st.info(
                        f"Pages: {len(docs)} | "
                        f"Chunks: {len(chunks)}"
                    )

                except Exception as e:
                    st.error(f"Processing failed: {e}")

                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

    st.divider()

    if st.session_state.vectorstore is not None:
        st.success("Document ready for chatting")

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    else:
        st.info("Upload and process a PDF to get started.")


# ---------------- CHAT INTERFACE ----------------

st.subheader("💬 Chat with your PDF")

current_hash = (
    hashlib.sha256(uploaded_file.getvalue()).hexdigest()[:12]
    if uploaded_file is not None
    else None
)

ready = (
    st.session_state.vectorstore is not None
    and st.session_state.processed_hash == current_hash
)

if not ready:
    st.info("Please upload and process a PDF before asking questions.")

else:

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    query = st.chat_input("Ask something about your PDF...")

    if query:

        # Display user message
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.markdown(query)

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": 4,
                            "fetch_k": 10,
                            "lambda_mult": 0.5
                        }
                    )

                    docs = retriever.invoke(query)

                    context = "\n\n".join(
                        doc.page_content for doc in docs
                    )

                    final_prompt = prompt.invoke({
                        "context": context,
                        "question": query
                    })

                    response = llm.invoke(final_prompt)

                    answer = response.content

                    st.markdown(answer)

                    # Show source documents
                    with st.expander("📚 View Sources"):

                        for i, doc in enumerate(docs):

                            st.markdown(
                                f"**Source {i+1}:** "
                                f"{doc.metadata.get('source', 'PDF')}"
                            )

                            st.markdown(
                                f"Page: {doc.metadata.get('page', 'N/A') + 1}"
                                if isinstance(doc.metadata.get('page'), int)
                                else "Page: N/A"
                            )

                            st.caption(doc.page_content[:300])

                            st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    st.error(f"Something went wrong: {e}")
