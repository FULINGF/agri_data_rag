import os
import streamlit as st

# ================= 路径配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
WAREHOUSE_DIR = os.path.join(DATA_DIR, "warehouse_exports")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")

# ================= 支持的文件格式 =================
SUPPORTED_FORMATS = {
    ".pdf": "PyPDFLoader",
    ".docx": "Docx2txtLoader",
    ".csv": "CSVLoader",  # 保留但实际 etl 中已自定义加载
    ".txt": "TextLoader",
    ".md": "TextLoader",
}

# ================= RAG 参数 =================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
TOP_K = 3

# ================= DeepSeek API 配置 =================
# 优先从 streamlit secrets 读取，其次环境变量
try:
    DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except (AttributeError, KeyError, FileNotFoundError):
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
TEMPERATURE = 0.2
MAX_TOKENS = 1024
TOP_P = 0.9