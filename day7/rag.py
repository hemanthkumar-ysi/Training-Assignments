import os
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
from langchain_core.runnables import RunnablePassthrough


# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()


# -------------------------------------------------
# Step 1 — Load Single PDF
# -------------------------------------------------
PDF_PATH = "AEM_book.pdf"

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Total pages loaded: {len(documents)}")


# -------------------------------------------------
# Step 2 — Chunking
# -------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)

chunks = splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")


# -------------------------------------------------
# Step 3 — Embeddings
# -------------------------------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vectorstore = FAISS.from_documents(chunks, embeddings)


# -------------------------------------------------
# Step 4 — Retriever
# -------------------------------------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# -------------------------------------------------
# Step 5 — LLM
# -------------------------------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# -------------------------------------------------
# Step 6 — Prompt Template
# -------------------------------------------------
prompt = ChatPromptTemplate.from_template(
"""
You are a helpful assistant.

Use ONLY the provided context to answer the question.
If the answer is not in the context, say:
"I don't know based on the provided document."

When answering, cite every claim using this exact format:
(Source: <source>, Page: <page>)

Use the Source and Page values exactly as shown.

Context:
{context}

Question:
{question}

Answer clearly:
"""
)


# -------------------------------------------------
# Step 7 — Context Formatter
# -------------------------------------------------
def format_docs(docs):
    formatted_docs = []

    for i, doc in enumerate(docs):
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", doc.metadata.get("page_number", "Unknown"))

        formatted_docs.append(
            f"""Document {i+1}
Source: {source}
Page: {page}

Content:
{doc.page_content}
"""
        )

    return "\n\n".join(formatted_docs)


# -------------------------------------------------
# Step 8 — RAG Chain
# -------------------------------------------------
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)


# -------------------------------------------------
# Step 9 — Interactive Loop
# -------------------------------------------------
while True:
    question = input("Ask a question (or 'exit'): ")

    if question.lower() == "exit":
        break

    try:
        answer = rag_chain.invoke(question)
        print("\nAnswer:\n")
        print(answer)
        print("\n" + "=" * 80 + "\n")

    except Exception as e:
        print("Error:", str(e))