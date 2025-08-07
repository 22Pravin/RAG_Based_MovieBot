from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
import json
import os
from data_preparation import movie_docs

# Initialize the embedding model for retrieval
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a list of Langchain Documents
langchain_docs = []
for doc in movie_docs:
    langchain_docs.append(
        Document(
            page_content=doc['content'],
            metadata=doc['metadata']
        )
    )

# Create a FAISS vector store
vector_store = LangchainFAISS.from_documents(langchain_docs, embeddings)

# Initialize the language model
llm = OpenAI(temperature=0.7, max_tokens=512)

# Create the RAG pipeline
moviebot_rag = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 5})
)
