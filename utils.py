# utils.py
from pymilvus import Collection
from openai import OpenAI
import os

# Connect to Milvus collection
def connect_to_milvus():
    from pymilvus import connections
    connections.connect(
        "default",
        uri=os.getenv("ZILLIZ_ENDPOINT"),
        user=os.getenv("ZILLIZ_USER_NAME"),  # If needed
        password=os.getenv("ZILLIZ_PASSWORD")  # If needed
    )
    return Collection("ayurveda_embeddings")

# Retrieve relevant context chunks from Milvus
def retrieve_context(user_query, embed_model_client, collection, top_k=5):
    query_embedding = embed_model_client.embeddings.create(
        model="nvidia/nv-embedqa-e5-v5",  # Specify NV Embed model ID if needed
        input=user_query
    )["data"][0]["embedding"]

    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(query_embedding, "embedding", search_params, limit=top_k)
    context = " ".join([result.entity.get("text_chunk") for result in results])
    return context

# Generate response using NVIDIA Llama3 model
def generate_response(user_query, context, llm_client):
    prompt = f"{context}\nUser Query: {user_query}"
    response_stream = llm_client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        stream=True
    )
    response_text = "".join([chunk["choices"][0]["delta"]["content"] for chunk in response_stream])
    return response_text
