import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time
import csv
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import (
    FAISS_INDEX_PATH,
    EMBEDDING_MODEL,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    TOP_K,
    TEMPERATURE,
    MAX_TOKENS,
    TOP_P,
)

# ================= 日志初始化 =================
LOG_FILE = "./data/query_log.csv"
if not os.path.exists(LOG_FILE):
    os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["time", "query", "show_refs", "chunk_id", "chunk_preview", "latency_ms", "answer"])


def log_query(query: str, show_refs: bool, chunk_id: int, chunk_preview: str, latency_ms: int, answer: str) -> None:
    """问答日志回流：结构化写入 CSV（模拟数仓 ODS 层采集）"""
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            query,
            show_refs,
            chunk_id,
            chunk_preview,
            latency_ms,
            answer
        ])


# ================= 组件初始化 =================
print("🔧 正在加载模型与向量索引...")

try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
except Exception as e:
    print(f"❌ Embedding 模型加载失败: {e}")
    exit(1)

try:
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
except Exception as e:
    print(f"❌ FAISS 索引加载失败: {e}")
    print("   请先运行 `python etl_parse.py` 生成向量索引")
    exit(1)

try:
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
    )
except Exception as e:
    print(f"❌ LLM 客户端初始化失败: {e}")
    print("   请检查: 1. API Key 是否正确 2. base_url 是否可访问")
    exit(1)

print("✅ 模型与索引加载完成\n")

# ================= Prompt 模板 =================
SYSTEM_PROMPT = """你是一个严谨的农业技术助手。请严格基于以下【参考资料】回答问题。

【强制规则】
1. 仅使用参考资料中的信息作答，禁止编造、推测或引入外部知识。
2. 若参考资料未提及或信息不足，必须统一回复："根据现有资料无法回答该问题。"
3. 回答需简洁专业。

【参考资料】
{context}

【用户问题】
{question}

【回答】"""

prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)


# ================= 辅助函数 =================
def format_docs(docs) -> str:
    """格式化检索文档，标注内部 Chunk 编号"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "未知页码")
        formatted.append(f"[Chunk-{i} | 来源: {source}, 第{page}页]\n{doc.page_content}")
    return "\n\n".join(formatted)


def format_refs_only(docs) -> str:
    """仅展示参考片段，用于单独展示"""
    refs = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "未知页码")
        preview = doc.page_content[:100].replace("\n", " ")
        refs.append(f"[Chunk-{i}] {source} (第{page}页): {preview}...")
    return "\n".join(refs)


# ================= RAG 链 =================
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# ================= 对外接口 =================
def ask_question_stream(query: str):
    """流式版本：逐 token 返回，不输出参考片段"""
    start = time.time()
    docs = retriever.invoke(query)
    full_answer = ""
    try:
        for chunk in rag_chain.stream(query):
            full_answer += chunk
            yield chunk
        # 日志记录：选择第一个 Chunk 作为代表
        first_chunk_id = 1 if docs else 0
        first_chunk_preview = docs[0].page_content[:80] if docs else ""
        log_query(query, False, first_chunk_id, first_chunk_preview, int((time.time() - start) * 1000), full_answer)
    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        err_msg = f"❌ 服务异常: {str(e)}"
        log_query(query, False, 0, "", elapsed, err_msg)
        yield err_msg


# ================= 命令行入口 =================
if __name__ == "__main__":
    print("🌽 农业 RAG 问答服务已就绪")
    print("💡 提示：输入 'exit' 退出 | 输入 `ref:你的问题` 可在回答后显示参考片段\n")

    while True:
        try:
            raw_input = input("👤 请输入问题: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 再见！")
            break

        if raw_input.lower() in ("exit", "quit"):
            print("👋 再见！")
            break
        if not raw_input:
            continue

        # 判断是否需要展示参考片段
        show_refs = False
        if raw_input.startswith("ref:"):
            show_refs = True
            query = raw_input[4:].strip()
        else:
            query = raw_input

        if not query:
            continue

        # 先检索
        docs = retriever.invoke(query)

        # 流式输出回答
        print("🤖 ", end="", flush=True)
        for chunk in ask_question_stream(query):
            print(chunk, end="", flush=True)
        print()

        # 按需展示参考片段
        if show_refs and docs:
            print("\n📚 参考片段：")
            print(format_refs_only(docs))