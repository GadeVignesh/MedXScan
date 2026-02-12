from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model (fast + reliable)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

KNOWLEDGE_PATH = r"E:\MedXScan\xray_ai_backend\database\medical_knowledge.txt"


def load_chunks(path):
    """
    Load medical knowledge as clean, topic-level chunks.
    Each [SECTION] becomes one chunk.
    """
    chunks = []
    current_chunk = ""

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip empty / junk lines
            if not line or line.isdigit() or line.startswith("("):
                continue

            if line.startswith("[") and line.endswith("]"):
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = ""
            else:
                current_chunk += " " + line

        if current_chunk:
            chunks.append(current_chunk.strip())

    return chunks


# Load and index documents
documents = load_chunks(KNOWLEDGE_PATH)

embeddings = embedder.encode(documents, convert_to_numpy=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


def retrieve_context(query):
    """
    Retrieve the single most relevant chunk.
    """
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, 1)

    return documents[indices[0][0]]
