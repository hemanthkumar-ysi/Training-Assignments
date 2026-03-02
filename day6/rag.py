from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


loader = PyPDFLoader("AEM_book.pdf")
documents = loader.load()

print(f"Pages loaded: {len(documents)}")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=40,
    separators=["\n\n", "\n", ".", " "]
)

chunks = splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")

# -----------------------------
# STEP 3 — Embeddings
# -----------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# -----------------------------
# STEP 4 — Vector Store (FAISS)
# -----------------------------
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

# -----------------------------
# STEP 5 — Retriever
# -----------------------------
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# -----------------------------
# STEP 6 — LLM
# -----------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# -----------------------------
# STEP 7 — Prompt Template
# -----------------------------
prompt = ChatPromptTemplate.from_template(
"""
You are a helpful assistant.

Use ONLY the provided context to answer the question.

Context:
{context}

Question:
{question}

Answer clearly:
"""
)

# -----------------------------
# STEP 8 — Output Parser
# -----------------------------
parser = StrOutputParser()

# -----------------------------
# STEP 9 — RAG Chain
# -----------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (prompt | model | parser)

# -----------------------------
# STEP 10 — Ask Questions
# -----------------------------
# -----------------------------
# STEP 10 — Ask Questions
# -----------------------------
while True:
    question = input("\nAsk a question (or 'exit'): ")

    if question.lower() == "exit":
        break

    # ===================================
    # STEP 1 — Retrieve relevant chunks
    # ===================================
    retrieved_docs = retriever.invoke(question)

    print("\n🔎 Retrieved Chunks:\n")

    for i, doc in enumerate(retrieved_docs):
        print(f"Chunk {i+1}")
        
        # ---- Metadata ----
        print("Metadata:", doc.metadata)

        # ---- Chunk preview ----
        print(doc.page_content[:300])  # preview text
        print("-" * 70)

    # ===================================
    # STEP 2 — Format context
    # ===================================
    context = "\n\n".join(
    f"(Page {doc.metadata.get('page')})\n{doc.page_content}"
    for doc in retrieved_docs)

    # ===================================
    # STEP 3 — Generate answer
    # ===================================
    answer = rag_chain.invoke({
        "context": context,
        "question": question
    })

    print("\n✅ Answer:\n")
    print(answer)