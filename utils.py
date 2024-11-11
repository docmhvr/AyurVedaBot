# utils.py
import os
import streamlit as st
from pymilvus import connections, Collection
from openai import OpenAI  # Access NVIDIA API models for embedding and reranker

# Connect to Milvus on Zilliz Cloud
def connect_to_milvus(URI,USER,PASS):
    connections.connect(
        alias="default",
        uri=URI,
        user=USER,
        password=PASS
    )
    # print("Connected to Milvus.")

    # Load Milvus collection
    collection_name="ayurveda_embeddings"
    return Collection(name=collection_name)

# Generate embeddings for a query text
def generate_embedding(text, embed_api_key):
    client = OpenAI(api_key=embed_api_key, base_url="https://integrate.api.nvidia.com/v1")
    response = client.embeddings.create(
        model="nvidia/nv-embedqa-e5-v5",
        input=text,
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "END"}        
    )
    return response.data[0].embedding

# Rerank results using NVIDIA NIM reranker model via OpenAI client
def rerank_results(query, results, rerank_api_key):
    # Initialize the OpenAI client with the NVIDIA reranker model API
    client = OpenAI(api_key=rerank_api_key, base_url="https://integrate.api.nvidia.com/v1")
    
    # Prepare the passages (text chunks) for reranking
    text_chunks = [result.entity.get("text_chunk") for result in results]
    
    # Call the NVIDIA reranker model using the OpenAI client
    response = client.completions.create(
        model="nvidia/nv-rerankqa-mistral-4b-v3",  # Correct NVIDIA NIM reranker model ID
        input=query,
        documents=text_chunks,  # List of text passages to rerank
    )
    
    # Extract scores from the response and sort the results by score
    reranked_results = sorted(
        zip(results, response["choices"]),
        key=lambda x: x[1]["score"],
        reverse=True
    )
    
    # Return the reranked text chunks in order of relevance
    return [result[0].entity.get("text_chunk") for result in reranked_results]


# Execute query with RAG pipeline
def query_with_rag_pipeline(query_text, collection, embed_api_key, rerank_api_key, top_k=5):
    query_embedding = generate_embedding(query_text, embed_api_key)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(data=[query_embedding], anns_field="embedding", param=search_params, limit=top_k)
    reranked_text_chunks = rerank_results(query_text, results[0], rerank_api_key)
    return " ".join(reranked_text_chunks)