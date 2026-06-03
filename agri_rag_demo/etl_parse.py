import os
# 设置 Hugging Face 国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# (可选) 增加下载超时时间，防止大文件下载中断
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")

import json
import hashlib
import ssl
from datetime import datetime
import pandas as pd

# 可选：禁用 SSL 验证（解决 huggingface 证书问题，仅开发环境）
if os.getenv("DISABLE_SSL_VERIFY", "0") == "1":
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
# CSVLoader 不再直接使用，改用 pandas 方式
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config import (
    DOCUMENTS_DIR,
    WAREHOUSE_DIR,
    FAISS_INDEX_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    SUPPORTED_FORMATS
)

# ================= 文件元数据管理 =================
METADATA_FILE = os.path.join(FAISS_INDEX_PATH, "file_metadata.json")

def load_file_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed_files": {}}

def save_file_metadata(metadata):
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def compute_file_md5(file_path, chunk_size=8192):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_loader(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    loader_class = SUPPORTED_FORMATS.get(ext)

    if loader_class == "PyPDFLoader":
        return PyPDFLoader(file_path)
    elif loader_class == "Docx2txtLoader":
        return Docx2txtLoader(file_path)
    elif loader_class == "TextLoader":
        return TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"不支持的文件格式：{ext}")

def load_csv_as_documents(file_path, filename):
    """使用 pandas 读取 CSV 并转换为 Document 列表（每行一个文档）"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')  # 自动处理 BOM
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='gbk')  # 后备编码
    docs = []
    for idx, row in df.iterrows():
        # 将整行转换为文本，保留列名
        content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        metadata = {
            "source": filename,
            "row_idx": idx,
            "folder": os.path.basename(WAREHOUSE_DIR)
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

def scan_documents(folder_path, metadata):
    new_docs = []
    files_to_process = []

    if not os.path.exists(folder_path):
        print(f"⚠️ 文件夹不存在：{folder_path}")
        return new_docs, files_to_process

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext not in SUPPORTED_FORMATS:
            print(f"⚠️ 跳过不支持的格式：{filename}")
            continue

        file_md5 = compute_file_md5(file_path)
        processed = metadata["processed_files"].get(filename, {})

        if filename not in processed or processed.get("md5") != file_md5:
            files_to_process.append({
                "path": file_path,
                "filename": filename,
                "md5": file_md5,
                "folder": folder_path
            })
        else:
            print(f"✅ 文件未变化，跳过：{filename}")

    return new_docs, files_to_process

def load_documents(files_to_process):
    all_docs = []
    for file_info in files_to_process:
        file_path = file_info["path"]
        filename = file_info["filename"]
        ext = os.path.splitext(filename)[1].lower()

        try:
            print(f"📄 加载文件：{filename}")
            if ext == ".csv":
                docs = load_csv_as_documents(file_path, filename)
            else:
                loader = get_loader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["folder"] = os.path.basename(file_info["folder"])
            all_docs.extend(docs)
            print(f"     加载 {len(docs)} 个片段")
        except Exception as e:
            print(f"❌ 加载失败 {filename}: {e}")
    return all_docs

def run_etl(incremental=True):
    print("=" * 50)
    print(" 农业知识库 ETL 流水线")
    print("=" * 50)

    if incremental and os.path.exists(FAISS_INDEX_PATH):
        metadata = load_file_metadata()
        print("📋 检测到现有索引，使用增量更新模式")
    else:
        metadata = {"processed_files": {}}
        print("📋 未检测到现有索引，使用全量重建模式")
        incremental = False

    print("\n[1/5] 扫描文档文件夹...")
    _, docs_to_process = scan_documents(DOCUMENTS_DIR, metadata)
    _, warehouse_to_process = scan_documents(WAREHOUSE_DIR, metadata)
    all_files = docs_to_process + warehouse_to_process

    if not all_files:
        print("✅ 没有新文件或变更文件，无需更新")
        return

    print(f"     发现 {len(all_files)} 个文件需要处理")

    print("\n[2/5] 加载并解析文档...")
    new_docs = load_documents(all_files)

    if not new_docs:
        print("❌ 没有成功加载任何文档")
        return

    print("\n[3/5] 文本切分...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(new_docs)
    print(f"     切分为 {len(chunks)} 个片段")

    print("\n[4/5] 向量化...")
    # 如果 SSL 问题依然存在，可以尝试设置环境变量后再运行
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

    if incremental and os.path.exists(FAISS_INDEX_PATH):
        print("     检测到现有索引，追加新向量...")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
    else:
        print("     创建新索引...")
        vectorstore = FAISS.from_documents(chunks, embeddings)

    print("\n[5/5] 保存索引...")
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)

    for file_info in all_files:
        metadata["processed_files"][file_info["filename"]] = {
            "md5": file_info["md5"],
            "processed_at": datetime.now().isoformat(),
            "folder": os.path.basename(file_info["folder"])
        }
    save_file_metadata(metadata)

    print("\n" + "=" * 50)
    print("✅ ETL 流水线执行完成！")
    print("=" * 50)

if __name__ == "__main__":
    import sys
    incremental = "--full" not in sys.argv
    # 如果你想临时禁用 SSL 验证，可以设置环境变量后运行：
    # set DISABLE_SSL_VERIFY=1 && python etl_parse.py
    run_etl(incremental=incremental)