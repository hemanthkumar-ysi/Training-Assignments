import streamlit as st
import os
import shutil
from utils.ingestion import process_file, split_documents, create_vector_store, load_vector_store
from utils.agent import handle_query
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
VECTOR_STORE_DIR = os.path.join(CURRENT_DIR, "vector_store")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

st.set_page_config(page_title="AI Knowledge Assistant", layout="wide")

st.title("🚀 AI Knowledge Assistant")
st.markdown("Upload documents and chat with your internal knowledge base.")

# Initialize session state for vector DB and chat history
if "vector_db" not in st.session_state:
    st.session_state.vector_db = load_vector_store(VECTOR_STORE_DIR)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for file uploads
with st.sidebar:
    st.header("📂 Document Upload")
    uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                all_docs = []
                for uploaded_file in uploaded_files:
                    # Save file temporarily
                    temp_path = os.path.join(DATA_DIR, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process and split
                    docs = process_file(temp_path)
                    all_docs.extend(docs)
                
                chunks = split_documents(all_docs)
                st.session_state.vector_db = create_vector_store(chunks, VECTOR_STORE_DIR)
                st.success(f"Processed {len(uploaded_files)} files!")
        else:
            st.warning("Please upload files first.")

    if st.button("Clear History"):
        st.session_state.chat_history = []
        st.rerun()

# Chat interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, context = handle_query(prompt, st.session_state.vector_db)
            st.markdown(answer)
            
            with st.expander("📚 View Sources"):
                for doc in context:
                    source = getattr(doc, 'metadata', {}).get('source', 'Unknown')
                    st.write(f"- {source}")
                    st.caption(doc.page_content[:200] + "...")

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
