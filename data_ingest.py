# data_ingest.py
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from llama_index import SimpleDirectoryReader, GPTListIndex
from nvidia_nim import NVEmbedModel

# Set up Google Drive authentication
def download_pdfs_from_drive(file_ids):
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    pdf_files = []

    for file_id in file_ids:
        file = drive.CreateFile({'id': file_id})
        file_name = file['title']
        file.GetContentFile(file_name)
        pdf_files.append(file_name)

    return pdf_files

# Load and process PDFs
def process_pdfs(pdf_files):
    documents = SimpleDirectoryReader(input_dir=".").load_data()
    text_chunks = GPTListIndex(documents).build_index()
    return text_chunks

# Connect to Zilliz Cloud Milvus
def connect_to_milvus():
    connections.connect(
        "default",
        uri="tcp://<YOUR_ZILLIZ_HOST>:<YOUR_PORT>",
        user="your_username",  # If needed
        password="your_password"  # If needed
    )

# Set up Milvus schema
def setup_milvus_collection():
    fields = [
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
        FieldSchema(name="text_chunk", dtype=DataType.VARCHAR, max_length=1024)
    ]
    schema = CollectionSchema(fields, description="Ayurveda text embeddings")
    collection = Collection(name="ayurveda_embeddings", schema=schema)
    return collection

# Generate embeddings with NVIDIA NV Embed model
def generate_embeddings(text_chunks):
    embed_model = NVEmbedModel("nv-embed-model-id")  # Replace with the actual model ID
    embeddings = [embed_model.encode(chunk.text) for chunk in text_chunks]
    return embeddings

# Insert embeddings into Milvus
def insert_embeddings_into_milvus(collection, embeddings, text_chunks):
    data = [{"embedding": embedding.tolist(), "text_chunk": chunk.text} for embedding, chunk in zip(embeddings, text_chunks)]
    collection.insert(data)

if __name__ == "__main__":
    # Download and process PDFs
    file_ids = ["file_id1", "file_id2", "file_id3", "file_id4", "file_id5"]
    pdf_files = download_pdfs_from_drive(file_ids)
    text_chunks = process_pdfs(pdf_files)

    # Connect to Milvus and setup collection
    connect_to_milvus()
    collection = setup_milvus_collection()

    # Generate embeddings and insert into Milvus
    embeddings = generate_embeddings(text_chunks)
    insert_embeddings_into_milvus(collection, embeddings, text_chunks)
    print("Data ingestion complete!")
