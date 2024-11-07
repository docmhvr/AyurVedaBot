# utils.py
from pymilvus import Collection
from nvidia_nim import NVEmbedModel, LlamaModel

# Connect to Milvus
def connect_to_milvus():
    from pymilvus import connections
    connections.connect(
        "default",
        uri="tcp://<YOUR_ZILLIZ_HOST>:<YOUR_PORT>",
        user="your_username",  # If needed
        password="your_password"  # If needed
    )
    return Collection("ayurveda_embeddings")

# Query Milvus for relevant chunks
def retrieve_context(query_text, embed_model, collection, top_k=5):
    query_embedding = embed_model.encode(query_text)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(query_embedding, "embedding", search_params, limit=top_k)
    context = " ".join([result.entity.get("text_chunk") for result in results])
    return context

# Generate a response using Llama3
def generate_response(user_query, context, llm_model):
    prompt = f"{context}\nUser Query: {user_query}"
    response = llm_model.generate(prompt=prompt)
    return response
