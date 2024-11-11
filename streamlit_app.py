import streamlit as st
from openai import OpenAI
from utils import connect_to_milvus, generate_embedding, rerank_results
from nemoguardrails import RailsConfig, LLMRails

# Load the RailsConfig from the 'config' directory
config = RailsConfig.from_path("config")  # Make sure 'rails_config.yml' is in the 'config' folder
rails = LLMRails(config)

# Load API keys from Streamlit secrets
EMBED_API_KEY = st.secrets["NVIDIA_EMBED_API_KEY"]
RERANK_API_KEY = st.secrets["NVIDIA_RERANK_API_KEY"]
LLM_API_KEY = st.secrets["NVIDIA_API_KEY"]

# Connect to Milvus and load collection
collection = connect_to_milvus(URI=st.secrets["ZILLIZ_ENDPOINT"],USER=st.secrets["ZILLIZ_USER_NAME"],PASS=st.secrets["ZILLIZ_PASSWORD"])

client = OpenAI(api_key=LLM_API_KEY, base_url="https://integrate.api.nvidia.com/v1")

# Streamlit app title and intro
st.title("🌿 AyurVeda AI ChatBot 🌿")
st.image("AyurVedaBot.webp", width=720)
st.write("Welcome to AyurVeda AI, bringing you ancient Ayurvedic solutions for health and well-being!")

# Initialize session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="👨🏻‍💻"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant", avatar="🌿"):
            st.markdown(message["content"])

# Capture user input for chat
if user_input := st.chat_input("Tell me about your ailment or ask an Ayurvedic health question..."):

    # Display the user's message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👨🏻‍💻"):
        st.markdown(user_input)

    # Prepare user input for guardrails
    user_message = {"role": "user", "content": user_input}
    
    # Generate response using LLMRails, which includes guardrails processing
    response = rails.generate(messages=[user_message])
    
    if response:  # If guardrails provide a response, use it
        assistant_reply = response["content"]
    else:
        # If guardrails didn't respond, fall back on RAG pipeline for response
        query_embedding = generate_embedding(user_input, EMBED_API_KEY)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(data=[query_embedding], anns_field="embedding", param=search_params, limit=5)

        # Rerank retrieved results
        reranked_text_chunks = rerank_results(user_input, results[0], RERANK_API_KEY)

        # Prepare prompt and call NVIDIA LLM
        context = "\n".join(reranked_text_chunks)
        prompt = f"Question: {user_input}\n\nContext:\n{context}\n\nAnswer:"

        response_placeholder = st.empty()
        full_response = ""

        # Stream response from NVIDIA LLM
        with st.chat_message("assistant", avatar="🌿"):
            for chunk in client.completions.create(
                model="meta/llama-3.1-405b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                top_p=0.7,
                max_tokens=1024,
                stream=True
            ):
                full_response += chunk["choices"][0]["delta"]["content"]
                response_placeholder.markdown(full_response)
        
        assistant_reply = full_response

    # Display the assistant's response
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
