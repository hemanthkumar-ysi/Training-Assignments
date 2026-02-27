import os
import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai

# =====================================
# LOAD ENV VARIABLES
# =====================================
load_dotenv()

# =====================================
# CONFIG
# =====================================
DOCS_PATH = "documents"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 100
TOP_K = 3

# =====================================
# GEMINI CLIENT
# =====================================
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# =====================================
# GEMINI EMBEDDING FUNCTION
# (REPLACES SentenceTransformer)
# =====================================
def get_embeddings(texts):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    )

    embeddings = [e.values for e in response.embeddings]
    return np.array(embeddings).astype("float32")


# =====================================
# CHUNKING FUNCTION (same as before)
# =====================================
def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


# =====================================
# LOAD DOCUMENTS
# =====================================
all_chunks = []
chunk_sources = []

print("Reading documents...")

for filename in os.listdir(DOCS_PATH):
    if filename.endswith(".txt"):
        path = os.path.join(DOCS_PATH, filename)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_sources.append(filename)

print(f"Total chunks created: {len(all_chunks)}")


# =====================================
# CREATE EMBEDDINGS (Gemini)
# =====================================
print("Creating Gemini embeddings...")
embeddings = get_embeddings(all_chunks)

dimension = embeddings.shape[1]

# ⭐ Normalize for cosine similarity (recommended)
faiss.normalize_L2(embeddings)

# =====================================
# CREATE FAISS INDEX
# =====================================
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("FAISS index ready!")


# =====================================
# SEARCH LOOP
# =====================================
while True:
    query = input("\nEnter search query (or 'exit'): ")

    if query.lower() == "exit":
        break

    query_embedding = get_embeddings([query])



    scores, indices = index.search(query_embedding, TOP_K)
    print("scores:", scores)
    print("indices:", indices)
    print("\nTop Results:\n")

    for rank, idx in enumerate(indices[0]):
        score = scores[0][rank]

        print(f"Rank {rank+1}")
        print(f"Source File : {chunk_sources[idx]}")
        print(f"Similarity Score : {score:.4f}")
        print(f"Chunk Text : {all_chunks[idx]}")
        print("-" * 60)