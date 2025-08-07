import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
from data_preparation import movie_docs

# Load the sentence transformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for our documents
documents = [doc['content'] for doc in movie_docs]
document_embeddings = embedder.encode(documents)

# Normalize the embeddings
document_embeddings = document_embeddings / \
    np.linalg.norm(document_embeddings, axis=1, keepdims=True)

# Initialize FAISS index - using IndexFlatIP for inner product similarity (cosine similarity for normalized vectors)
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(document_embeddings.astype('float32'))

# Save the index for later use
faiss.write_index(index, 'movies_faiss_index')

# Save mapping of index positions to movie documents
index_to_doc_mapping = {i: movie_docs[i] for i in range(len(movie_docs))}
with open('index_to_doc_mapping.json', 'w') as f:
    json.dump(index_to_doc_mapping, f)
