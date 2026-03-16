import os
from dotenv import load_dotenv

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings


DATA_PATH = "data/docs"
VECTOR_PATH = "vector_store"

documents = []

# Load documents
for file in os.listdir(DATA_PATH):

    path = os.path.join(DATA_PATH, file)

    if file.endswith(".pdf"):
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    elif file.endswith(".txt"):
        loader = TextLoader(path)
        documents.extend(loader.load())


print(f"Loaded {len(documents)} documents")


# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=70
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# Local embeddings (no API required)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create vector DB
db = FAISS.from_documents(chunks, embeddings)

db.save_local(VECTOR_PATH)

print("Vector database created!")