import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from dotenv import load_dotenv

load_dotenv()

# ================= 网络与环境 =================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 国内加速，防下载卡死

# ================= API 与认证 =================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk??????????")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# ================= 模型参数（集中管理，体现工程规范） =================
DEEPSEEK_MODEL = "deepseek-chat"
TEMPERATURE = 0.2          # 低温度保证回答稳定，适合知识问答
MAX_TOKENS = 1024          # 限制输出长度，控制成本与延迟
TOP_P = 0.9                # 核采样，平衡多样性与准确性

# ================= 嵌入与检索 =================
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
TOP_K = 3                  # 检索返回片段数，平衡上下文长度与噪声

# ================= 路径配置 =================
PDF_PATH = "./data/pest_control.pdf"
FAISS_INDEX_PATH = "./data/faiss_index"

# ================= RAG 切分参数 =================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50