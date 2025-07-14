from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  # 👈 No OpenAI needed

def load_csv_docs(path):
    with open(path, "r", encoding="latin1", errors="ignore") as f:
        content = f.read()
    return [Document(page_content=content)]

def get_vectordb(docs):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # ✅ free, fast, local
    return FAISS.from_documents(chunks, embeddings)
