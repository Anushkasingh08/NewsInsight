import os
import streamlit as st
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.vectorstores import FAISS

# --- Configuration ---
GOOGLE_API_KEY = "YOUR-GOOGLE-API-KEY"
MODEL_NAME = "gemini-1.5-flash"
VECTORSTORE_DIR = "faiss_store_gemini"

st.set_page_config(page_title="RockyBot - News Research", layout="centered")
st.title("📰 NewsInsight ")
st.sidebar.title("🔗 News Article URLs")

# --- URL Input ---
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    if url.strip():
        urls.append(url.strip())

process_clicked = st.sidebar.button("⚙️ Process URLs")
placeholder = st.empty()

# --- Model Setup ---
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.9,
    google_api_key=GOOGLE_API_KEY
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# --- Process URLs ---
if process_clicked:
    if not urls:
        st.warning("Please enter at least one valid URL.")
    else:
        placeholder.info("🔄 Loading and processing articles...")
        loader = SeleniumURLLoader(urls=urls)
        data = loader.load()

        if not data:
            st.error("❌ Failed to load any content from the URLs.")
        else:
            st.subheader("🧾 Preview of Extracted Articles")
            for i, doc in enumerate(data):
                st.markdown(f"**Article {i+1}:**")
                st.write(doc.page_content[:500] + "...")

            placeholder.info("✂️ Splitting Text...")
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", ","],
                chunk_size=1000,
                chunk_overlap=100
            )
            docs = splitter.split_documents(data)

            if not docs:
                st.error("❌ No documents created after splitting.")
            else:
                placeholder.info("🔍 Generating Embeddings...")
                vectorstore = FAISS.from_documents(docs, embeddings)
                vectorstore.save_local(VECTORSTORE_DIR)
                placeholder.success("✅ Embeddings saved successfully!")

# --- Query Input ---
query = st.text_input("💬 Ask a question about the articles:")

# --- Answer Query ---
if query:
    if not os.path.exists(VECTORSTORE_DIR):
        st.warning("⚠️ Please process URLs before asking questions.")
    else:
        try:
            vectorstore = FAISS.load_local(
                VECTORSTORE_DIR,
                embeddings,
                allow_dangerous_deserialization=True  # ✅ Safe only if you trust your own files
            )

            qa_chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )

            response = qa_chain({"question": query}, return_only_outputs=True)

            st.subheader("🧠 Answer")
            st.write(response.get("answer", "No answer generated."))

            sources = response.get("sources", "")
            if sources:
                st.subheader("🔗 Sources")
                for src in sources.split("\n"):
                    if src.strip():
                        st.write(f"• {src.strip()}")

        except Exception as e:
            st.error(f"❌ Error occurred:\n```\n{str(e)}\n```")
