import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import hashlib
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # 消掉警告

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import PDF_PATH, FAISS_INDEX_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

def compute_md5(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def dqc_check(text: str) -> tuple[bool, str]:
    if len(text.strip()) < 20:
        return False, "长度不足"
    special_ratio = len([c for c in text if not c.isalnum() and not c.isspace()]) / len(text)
    if special_ratio > 0.3:
        return False, "乱码/特殊字符过多"
    return True, "PASS"

def run_etl():
    try:
        print("[1/4] 加载PDF...")
        if not os.path.exists(PDF_PATH):
            raise FileNotFoundError(f"❌ 找不到文件: {PDF_PATH}")
        docs = PyPDFLoader(PDF_PATH).load()

        print("[2/4] 切分 & DQC & MD5去重...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(docs)

        seen_md5, valid_chunks = set(), []
        dqc_report = {"total": len(chunks), "pass": 0, "fail": 0, "reasons": {}}
        for c in chunks:
            is_valid, reason = dqc_check(c.page_content)
            if not is_valid:
                dqc_report["fail"] += 1
                dqc_report["reasons"][reason] = dqc_report["reasons"].get(reason, 0) + 1
                continue
            chunk_md5 = compute_md5(c.page_content)
            if chunk_md5 in seen_md5:
                continue
            seen_md5.add(chunk_md5)
            valid_chunks.append(c)
            dqc_report["pass"] += 1

        print(f"     DQC报告: 总{dqc_report['total']} | 合格{dqc_report['pass']} | 拦截{dqc_report['fail']} | 原因{dqc_report['reasons']}")

        print("[3/4] 向量化...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
        vectorstore = FAISS.from_documents(valid_chunks, embeddings)

        print("[4/4] 安全持久化...")
        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
        vectorstore.save_local(FAISS_INDEX_PATH)
        print("✅ ETL流水线执行完成！")
    except Exception as e:
        print(f"❌ ETL失败: {str(e)}")
        raise

# ⚠️ 关键：必须要有下面这两行，程序才会真的跑！
if __name__ == "__main__":
    run_etl()