import streamlit as st
from streamlit.logger import get_logger
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

logger = get_logger(__name__)

import os
if os.getenv('USER', "None") == 'appuser': # streamlit
    hf_token = st.secrets['HF_TOKEN']
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
else:
    # ALSO ADD HERE YOUR PROXY VARS
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ["MY_HF_API_TOKEN"]

st.title("my gen AI app")
repo_id = "microsoft/Phi-3-mini-4k-instruct"
temp = 1
print(repo_id, temp)
logger.info(f"{temp=}")

with st.form("sample_app"):
    txt = st.text_area("Enter text:", "what GPT stands for?")
    sub = st.form_submit_button("submit")
    if sub:
        llm = HuggingFaceEndpoint(
                repo_id=repo_id,  #3.8B model
                task="text-generation",
                temperature=temp
                )
        chat = ChatHuggingFace(llm=llm, verbose=True)
        logger.info("invoking")
        ans = chat.invoke(txt)
        st.info(ans.content)
        logger.info("Done")

