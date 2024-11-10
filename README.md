# 🌿 AyurVeda AI ChatBot 🌿

Welcome to **AyurVeda AI ChatBot**, a retrieval-augmented generation (RAG) chatbot designed to provide Ayurvedic health solutions and knowledge on natural remedies. Powered by **NVIDIA NIM models** and the **LlamaIndex stack**, this chatbot efficiently retrieves, ranks, and generates natural language responses based on an extensive Ayurveda knowledge base.

The chatbot is hosted on **Streamlit Cloud**, making it easily accessible to users who seek natural and safe health advice grounded in the principles of Ayurveda.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

## ✨ Features

- **Real-time Q&A** on Ayurvedic treatments, herbal remedies, and lifestyle tips.
- **NeMo Guardrails** to ensure the chatbot stays on-topic, avoiding sensitive or irrelevant responses.
- **NVIDIA Embedding and Reranking Models** for advanced retrieval-augmented generation.
- **LlamaIndex** for document ingestion, indexing, and creating a structured database of Ayurvedic knowledge.
- **Milvus Vector Database** hosted on Zilliz Cloud for efficient retrieval of relevant passages.

---

## 🚀 Technology Stack

1. **NVIDIA Models**:
   - **NV Embed Model** (`nvidia/nv-embedqa-e5-v5`): Generates embeddings for Ayurveda-related text.
   - **NV Reranker Model** (`nvidia/nv-rerankqa-mistral-4b-v3`): Reranks retrieved passages to ensure relevance.
   - **Llama 3.1 LLM** (`meta/llama-3.1-405b-instruct`): Generates coherent and informative responses based on retrieved passages.

2. **LlamaIndex**:
   - **Document Parsing**: Parses PDF documents containing Ayurvedic knowledge.
   - **Hierarchical Node Parser**: Organizes text into searchable chunks for optimized retrieval.

3. **Milvus Vector Database** (Hosted on Zilliz Cloud):
   - Stores embeddings for efficient similarity search.
   - Integrates with LlamaIndex and NVIDIA models to retrieve the most relevant passages.

4. **NeMo Guardrails**:
   - Defines rules to ensure safe, relevant, and accurate responses.
   - Topics include scope-limited queries and fallback responses for off-topic inputs.

---

## 🗂️ Project Structure

- `data_ingest.py`: Parses and indexes Ayurvedic knowledge PDFs into Milvus.
- `utils.py`: Utility functions for embedding generation, retrieval, and reranking.
- `app.py`: The main Streamlit application to provide an interactive chat interface.
- `guardrails/`: Folder containing NeMo Guardrails YAML files for topic-specific rules.
  - `ayurveda_queries.yml`: Handles Ayurveda-related questions.
  - `safety_rules.yml`: Provides fallback responses for sensitive topics.
  - `response_relevance.yml`: Ensures relevance in answers.

---

## ⚙️ Setup and Run Instructions

### Prerequisites

- Python 3.8+
- NVIDIA API keys (for embedding, reranking, and LLM models)
- Zilliz Cloud account and endpoint for Milvus vector database

### Step 1: Clone the Repository

```bash
git clone https://github.com/docmhvr/AyurVedaBot.git
cd AyurVedaBot
```

### Step 2: Set Up Environment Variables

Create a `.env` file with your NVIDIA and Zilliz Cloud credentials:

```env
NVIDIA_EMBED_API_KEY=your_embed_api_key
NVIDIA_RERANK_API_KEY=your_rerank_api_key
NVIDIA_LLM_API_KEY=your_llm_api_key
ZILLIZ_ENDPOINT=your_zilliz_endpoint
ZILLIZ_USER_NAME=your_zilliz_username
ZILLIZ_PASSWORD=your_zilliz_password
```

> **Note**: Alternatively, if deploying on Streamlit Cloud, add these as secrets in the app settings.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Data Ingestion

Run `data_ingest.py` to parse and index the Ayurveda knowledge PDFs into Milvus.

```bash
python data_ingest.py
```

### Step 5: Start the Streamlit Application

To launch the chatbot locally:

```bash
streamlit run app.py
```

### Step 6: Check the Hosted App on Streamlit Cloud

If you’ve deployed the app on Streamlit Cloud, visit your app’s URL to interact with the chatbot.

---

## 📖 How It Works

1. **Data Ingestion**:
   - `data_ingest.py` parses Ayurveda-related PDFs, splitting them into searchable chunks using LlamaIndex.
   - Embeddings are generated for each chunk using NVIDIA's **NV Embed Model** and stored in Milvus.

2. **Query Processing**:
   - When a user asks a question, the query is embedded with NVIDIA's **NV Embed Model**.
   - Relevant passages are retrieved from Milvus based on embedding similarity.

3. **Reranking**:
   - Retrieved passages are reranked using NVIDIA's **NV Reranker Model** to ensure relevance.

4. **Response Generation**:
   - The top-ranked passages are passed to **Llama 3.1 LLM** to generate a natural language answer.

5. **NeMo Guardrails**:
   - Guardrails guide the chatbot to stay within Ayurveda-related topics, prevent sensitive responses, and handle off-topic questions gracefully.

---

## 🔧 Configuration for NeMo Guardrails

To customize the guardrails, edit the YAML files in the `guardrails/` folder. Here are the topics:

- **ayurveda_queries.yml**: Defines rules for Ayurveda-focused queries.
- **safety_rules.yml**: Specifies safe response patterns for sensitive topics.
- **response_relevance.yml**: Ensures relevance by restricting the chatbot to Ayurveda-related answers.

---

## 📚 Knowledge Base

The AyurVeda AI ChatBot draws its knowledge from a curated set of **Ayurvedic texts** stored as PDF files. These resources provide insights into herbal remedies, wellness practices, and traditional treatments, enabling the chatbot to offer guidance based on authentic Ayurvedic wisdom.

- **Ayurvedic Textual PDFs**: The core texts for this project are available [here on Google Drive](https://drive.google.com/drive/folders/1yptl_faY4MPnBmuSQnwy2djSaLD459yI).
- **Cover Folder**: The `cover/` folder in the repository contains cover images of these books for easy reference.

These documents are parsed, embedded using **NVIDIA's NV Embed Model**, and stored in a Milvus vector database, creating a responsive and knowledge-rich chatbot experience.

--- 

## 🚀 Future Improvements

- **Expand Knowledge Base**: Add more Ayurvedic texts and resources.
- **Enhanced Fallback Responses**: Refine NeMo Guardrails to handle a broader range of off-topic queries.
- **Additional Models**: Experiment with other NVIDIA models for improved accuracy.

---

## 📝 License

This project is open-source and available under the Apache License.

---

Feel free to explore, contribute, and help make **AyurVeda AI ChatBot** a more knowledgeable and user-friendly application for Ayurvedic health and wellness! 🌱
