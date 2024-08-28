import asyncio
import json
import base64
from io import BytesIO
import os

import openai
import streamlit as st
from openai import AsyncAzureOpenAI
from PIL import Image

from src.tech_bot.configs import (
    API_KEY,
    API_VERSION,
    AZURE_DEPLOYMENT,
    AZURE_ENDPOINT,
    OAI_MODEL,
)
from src.tech_bot.utils import num_tokens_from_messages

st.title(f"Azure Open AI: GPT 4 Tech Bot 🤖")

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = OAI_MODEL
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

def get_image_base64(file):
    if isinstance(file, str):
        with open(file, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('ascii')
    else:
        return base64.b64encode(file.getvalue()).decode('ascii')

async def get_response():
    try:
        response = await client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=st.session_state.messages,
            stream=True,
            temperature=0.8,
        )
    except openai.BadRequestError:
        st.error(
            "The response was filtered due to the prompt triggering Azure OpenAI's \
                  content management policy. Please modify your prompt and retry."
        )
        return
    full_response = ""
    async for chunk in response:
        data = json.loads(chunk.model_dump_json(indent=2))

        if len(data["choices"]) > 0:
            if data["choices"][0]["delta"]["content"]:
                full_response += data["choices"][0]["delta"]["content"]
            if full_response:
                message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

client = AsyncAzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=API_VERSION,
    azure_deployment=AZURE_DEPLOYMENT,
)

# Display previous messages
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            if isinstance(message["content"], list):
                for content in message["content"]:
                    if content["type"] == "image_url":
                        st.image(content["image_url"]["url"], caption="Uploaded Image")
            else:
                st.markdown(message["content"])

# Add system message if not present
if not any(message["role"] == "system" for message in st.session_state.messages):
    st.session_state.messages.append({
        "role": "system",
        "content": "You are an expert assistant. You can answer to anything.",
    })

# Image upload section
st.sidebar.header("Image Upload")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.sidebar.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.session_state.current_image = get_image_base64(uploaded_file)
    st.sidebar.success("Image uploaded successfully. You can now add it to the conversation.")

# Button to add image to conversation
if st.session_state.current_image and st.sidebar.button("Add Image to Conversation"):
    st.session_state.messages.append({"role": "user", "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{st.session_state.current_image}"
          }
        }
      ]
    })
    st.session_state.current_image = None
    st.sidebar.success("Image added to the conversation.")
    st.rerun()  # Changed from st.experimental_rerun()

# Chat input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        asyncio.run(get_response())

# Display token usage
try:
    token_count = num_tokens_from_messages(st.session_state.messages, OAI_MODEL)
    st.sidebar.markdown(
        f"<span style='color:red'>Total tokens used: {token_count}</span>",
        unsafe_allow_html=True,
    )
except Exception as e:
    st.sidebar.error(f"Error calculating token usage: {str(e)}")