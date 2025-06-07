import time
import streamlit as st
from utils.AgentHandler import AgentHandler
import os
st.set_page_config(page_title="Login", page_icon="🔐")


def is_token_valid(token: str) -> bool:
    try:
        handler = AgentHandler(api_key=token, model_keys=["llama-3-8b"])
        answer = handler.generate("Hello", context=None, model_key="llama-3-8b")
        return answer and not answer.startswith("Error")
    except Exception as e:
        print(f"Token validation error: {e}")
        return False

st.title("🔐 Wprowadź Hugging Face Token")

token = st.text_input("Hugging Face Token:", type="password")

if st.button("Zaloguj się"):
    if not token:
        st.error("Token nie może być pusty.")
    elif not is_token_valid(token):
        st.error("❌ Niepoprawny token Hugging Face.")
    else:
        st.session_state["hf_token"] = token
        st.success("✅ Token poprawny!")
        st.page_link("pages/app.py", label="👉 Przejdź do aplikacji")