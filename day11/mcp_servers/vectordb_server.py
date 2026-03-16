from mcp.server.fastmcp import FastMCP
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

mcp = FastMCP("vectordb-server")


# Same embedding model used in ingest.py
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


db = None

if os.path.exists("vector_store"):
    db = FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )


@mcp.tool()
async def search_knowledge(query: str) -> str:

    if db is None:
        return "Vector database not available."

    docs = db.similarity_search(query, k=5)

    results = []
    for d in docs:
        source = d.metadata.get("source", "Unknown")
        page = d.metadata.get("page")
        
        info = f"Content: {d.page_content}\nSource: {source}"
        if page is not None:
            # LangChain/PyPDFLoader uses 0-indexed pages, making it 1-indexed for the user
            info += f", Page: {page + 1}"
            
        results.append(info)

    return "\n\n---\n\n".join(results)


if __name__ == "__main__":
    mcp.run()