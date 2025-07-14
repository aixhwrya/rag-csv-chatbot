import streamlit as st
from rag_pipeline import load_csv_docs, get_vectordb
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os

load_dotenv()  # Optional: if you're using HuggingFaceHub token

st.set_page_config(page_title="CSV Q&A Chatbot", page_icon="🧠")
st.title("🧠 CSV RAG Q&A Chatbot")

uploaded_file = st.file_uploader("📄 Upload your CSV file", type=["csv"])
query = st.text_input("🔍 Ask a question based on your data:")

if uploaded_file and query:
    with open("temp.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("🔎 Searching for the answer..."):
        docs = load_csv_docs("temp.csv")
        vectordb = get_vectordb(docs)
        retriever = vectordb.as_retriever()

        # ✅ Fixed: Specify the repo_id (e.g., flan-t5-base, mistralai/Mistral-7B, etc.)
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",  # You can change to any available model
            model_kwargs={"temperature": 0.2, "max_new_tokens": 256}
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever
        )

        result = qa.run(query)
        st.success("✅ Answer:")
        st.write(result)
