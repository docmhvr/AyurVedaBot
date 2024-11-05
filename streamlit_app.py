import streamlit as st
import time

# Title
st.title("🌿 AyurVeda AI ChatBot 🌿")

# Introduction text
st.write("Welcome to AyurVeda AI, bringing you ancient Ayurvedic solutions for health and well-being!")

# ChatBot Image
st.image("AyurVedaBot.webp")

# Nvidia NIM TODO

# Create a session state variable to store the chat messages. This ensures that the
# messages persist across reruns.

# Initialize session state to store chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to simulate streaming response by yielding one word at a time
def stream_response(response_text):
    for word in response_text.split():
        yield word + " "  # Yield one word at a time with a space
        time.sleep(0.2)  # Adjust the delay for streaming effect

# Display chat history in sequential order
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="👨🏻‍💻"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant", avatar="🌿"):
            st.markdown(message["content"])

# User input field at the bottom, prompts latest message above it
if user_input := st.chat_input("Tell me about your ailment?"):

    # Append user input to messages and display it immediately
    with st.chat_message("user", avatar="👨🏻‍💻"):
        st.markdown(user_input)
    
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Placeholder for the assistant's streaming response
    response_text = "Based on Ayurveda, here's some guidance for your well-being."
    with st.chat_message("assistant", avatar="🌿"):
        # Stream response by words using `st.write_stream`
        st.write_stream(stream_response(response_text))

    # Store the full response in session state to preserve chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})