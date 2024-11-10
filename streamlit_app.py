import streamlit as st
from openai import OpenAI
from utils import connect_to_milvus, load_milvus_collection, generate_embedding, rerank_results, generate_final_response
from nemo_guardrails import Guardrails

# Load API keys from Streamlit secrets
EMBED_API_KEY = st.secrets["NVIDIA_EMBED_API_KEY"]
RERANK_API_KEY = st.secrets["NVIDIA_RERANK_API_KEY"]
LLM_API_KEY = st.secrets["NVIDIA_API_KEY"]

# Initialize NeMo Guardrails with specified rule files
guardrails = Guardrails(rules_folder="guardrails")

# Connect to Milvus and load collection
connect_to_milvus()
collection = load_milvus_collection()

# Streamlit app title and intro
st.title("🌿 AyurVeda AI ChatBot 🌿")
st.image("AyurVedaBot.webp", width=500)
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

    # Apply NeMo Guardrails
    user_intent = guardrails.check_intent(user_input)
    
    # Step 1: Check if the input is within the Ayurveda scope
    if user_intent in ["ayurveda_query", "ayurveda_scope_check"]:
        query_embedding = generate_embedding(user_input, EMBED_API_KEY)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(data=[query_embedding], anns_field="embedding", param=search_params, limit=5)

        # Step 2: Rerank the retrieved results
        reranked_text_chunks = rerank_results(user_input, results[0], RERANK_API_KEY)

        # Step 3: Prepare the context for the LLM and set up streaming for the response
        context = "\n".join(reranked_text_chunks)
        prompt = f"Question: {user_input}\n\nContext:\n{context}\n\nAnswer:"

        # Initialize the OpenAI client for NVIDIA LLM with streaming
        client = OpenAI(api_key=LLM_API_KEY, base_url="https://integrate.api.nvidia.com/v1")

        # Placeholder for the assistant's response
        response_placeholder = st.empty()
        full_response = ""

        # Stream the response from the LLM
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

        # Store the full response in session state to preserve chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    else:
        # Apply guardrail fallback for off-topic or sensitive queries
        fallback_response = guardrails.get_fallback_response(user_intent)
        with st.chat_message("assistant", avatar="🌿"):
            st.markdown(fallback_response)
        st.session_state.messages.append({"role": "assistant", "content": fallback_response})
