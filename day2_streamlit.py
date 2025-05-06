import streamlit as st
from streamlit.logger import get_logger
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os

logger = get_logger(__name__)

# Set Hugging Face token securely
try:
    if os.getenv('USER', "None") == 'appuser':  # Running on Streamlit Cloud
        hf_token = st.secrets["HF_TOKEN"]
    else:  # Local use
        hf_token = os.environ["MY_HF_API_TOKEN"]
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
except KeyError as e:
    st.error(f"Missing Hugging Face token: {e}")
    st.stop()

# App title
st.title("My Gen AI App")

# Model configuration
repo_id = "microsoft/Phi-3-mini-4k-instruct"
temperature = 1.0
logger.info(f"Using model: {repo_id} with temperature={temperature}")

# Text input form
with st.form("sample_app"):
    txt = st.text_area("Enter text:", "What does GPT stand for?")
    submitted = st.form_submit_button("Submit")

    if submitted:
        try:
            # Create LLM endpoint
            llm = HuggingFaceEndpoint(
                repo_id=repo_id,
                task="text-generation",
                temperature=temperature
            )

            chat = ChatHuggingFace(llm=llm, verbose=True)
            logger.info("Invoking model...")

            # Get response
            ans = chat.invoke(txt)
            st.info(ans.content)
            logger.info("Response displayed.")

        except Exception as e:
            logger.exception("Failed to get response from model.")
            st.error(f"Error: {e}")
