
import os
import numpy as np
from dotenv import load_dotenv
from google import genai
print("SCRIPT STARTED")
# Load env
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Hardcoded sentences
sentences = [
    "I love artificial intelligence",
    "Machine learning is fascinating",
    "I enjoy AI technology",
    "Cooking recipes are fun"
]

MODEL_NAME = "gemini-embedding-001"


# Get embeddings
def get_embedding(text):
    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=text
    )
    return response.embeddings[0].values


# Cosine similarity
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )


# Generate embeddings
embeddings = [get_embedding(s) for s in sentences]

# Compare all sentence pairs
max_score = -1
most_similar_pair = ("", "")

print("\nSimilarity Scores:\n")

for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):

        score = cosine_similarity(embeddings[i], embeddings[j])

        print(f"{sentences[i]}  <->  {sentences[j]}  = {score:.4f}")

        if score > max_score:
            max_score = score
            most_similar_pair = (sentences[i], sentences[j])


print("\nMost Similar Sentences:")
print(most_similar_pair, "Score:", max_score)