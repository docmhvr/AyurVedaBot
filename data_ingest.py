# data_ingest.py
import os
import gdown
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from openai import OpenAI  # Access NVIDIA NV Embed API

# List of file IDs for each PDF in the folder
file_ids = [
    "1fUhUfHL2Gk7Zi7FPH89ixgrYL4rz6MYD"  # Drive PDF's file ID
    # "1_o8Lqj2MCZsy7EbKBWP-m57YGJGK2QuA",
    # "184DEG5M1nB4AY4fNPFlPPLV6z2NHw55h",
    # "10A-6Zi2P-QZ2n5Q19XytQcXHVGqfd6b4"
    # "1pNLvTz0Z0x9lXKGF32yPiyz3gCPxkKcx",
]

# Download PDFs from Google Drive
def download_pdfs(file_ids):
    pdf_files = []
    for file_id in file_ids:
        url = f"https://drive.google.com/uc?id={file_id}"
        output = f"{file_id}.pdf"  # Name each file by its ID or customize it
        gdown.download(url, output, quiet=False)
        pdf_files.append(output)
    return pdf_files

# Process PDFs into hierarchical nodes
def process_pdfs(pdf_files):
    documents = SimpleDirectoryReader(input_files=pdf_files).load_data()
    merged_text = "\n\n".join([doc.text for doc in documents])
    return Document(text=merged_text)

# Connect to Milvus on Zilliz Cloud
def connect_to_milvus():
    connections.connect(
        alias="default",
        uri=os.getenv("ZILLIZ_ENDPOINT"),
        user=os.getenv("ZILLIZ_USER_NAME"),
        password=os.getenv("ZILLIZ_PASSWORD")
    )

# Set up Milvus schema with primary key
def setup_milvus_collection():
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # Primary key field
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
        FieldSchema(name="text_chunk", dtype=DataType.VARCHAR, max_length=1024)
    ]
    schema = CollectionSchema(fields, description="Ayurveda text embeddings")
    collection = Collection(name="ayurveda_embeddings", schema=schema)
    return collection

# Generate embeddings and store in Milvus
def generate_and_store_embeddings(document, embed_api_key, collection):
    client = OpenAI(api_key=embed_api_key, base_url="https://integrate.api.nvidia.com/v1")
    data_to_insert = []
    
    # Embed each node's text (ensured to be <= 512 tokens by the parser)
    for node in nodes:
        response = client.embeddings.create(
            input=[node.text],
            model="nvidia/nv-embedqa-e5-v5",
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "END"}
        )
        embedding = response.data[0].embedding
        data_to_insert.append({"embedding": embedding, "text_chunk": node.text})
    
    # Insert all data into Milvus collection
    collection.insert(data_to_insert)

if __name__ == "__main__":
    EMBED_API_KEY = os.getenv("NVIDIA_EMBED_API_KEY")

    # Download and process PDF files
    pdf_files = download_pdfs(file_ids)
    print("Downloaded PDF files:", pdf_files)
    document = process_pdfs(pdf_files)
    
    # Parse nodes
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[512, 256, 128])
    nodes = node_parser.get_nodes_from_documents([document])
    
    # Connect to Milvus and set up collection
    connect_to_milvus()
    collection = setup_milvus_collection()
    print("Created and connected with collection on Milvus")

    # Generate and store embeddings in Milvus
    generate_and_store_embeddings(nodes, EMBED_API_KEY, collection)
    print("Data ingestion and storage in Milvus complete!")
