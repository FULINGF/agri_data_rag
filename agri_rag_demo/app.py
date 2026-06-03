import streamlit as st
import os
import time
from rag_query import ask_question_stream, retriever, format_refs_only
from config import FAISS_INDEX_PATH, DOCUMENTS_DIR, WAREHOUSE_DIR

# ================= 页面配置 =================
st.set_page_config(
    page_title="农业 RAG 智能助手",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= 自定义 CSS 美化 =================
st.markdown("""
<style>
    /* 全局 */
    html, body {
        font-family: 'Microsoft YaHei', sans-serif;
        background-color: #F9F7F0;
    }

    /* 侧边栏：浅绿+干净白字 */
    [data-testid="stSidebar"] {
        background-color: #2F4F40;
        padding: 1rem;
    }
    [data-testid="stSidebar"] * {
        color: #F0F5E9 !important;
    }

    [data-testid="stSidebar"] h3 {
        font-size: 0.95rem !important;
        font-weight: 500;
        margin: 0.5rem 0 0.2rem;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMetric-value {
        font-size: 0.78rem !important;
    }

    [data-testid="stSidebar"] .stMetric {
        background: rgba(255,255,255,0.08);
        border-radius: 6px;
        padding: 0.4rem;
        margin: 0.2rem 0;
    }

    [data-testid="stSidebar"] hr {
        border-color: #D4C9A6;
        opacity: 0.2;
        margin: 0.7rem 0;
    }

    [data-testid="stSidebar"] .stButton button {
        background: rgba(255,255,255,0.12);
        color: #F0F5E9 !important;
        border: none;
        border-radius: 12px;
        font-size: 0.7rem;
        padding: 0.25rem 0.6rem;
    }

    /* 主聊天区 */
    .stChatMessage {
        background: #fff;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    [data-testid="stChatMessage"][data-role="user"] { border-left: 3px solid #2E7D32; }
    [data-testid="stChatMessage"][data-role="assistant"] { border-left: 3px solid #FFC107; }

    .stChatInput textarea {
        border-radius: 20px;
        border: 1px solid #E0D6B0;
        font-size: 0.9rem;
    }

    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ================= 侧边栏：知识库状态 + 功能选项 =================
with st.sidebar:
    st.title("📚 知识库状态")
    st.markdown("---")

    # 统计文件数量
    doc_files = 0
    if os.path.exists(DOCUMENTS_DIR):
        doc_files = len([f for f in os.listdir(DOCUMENTS_DIR) if os.path.isfile(os.path.join(DOCUMENTS_DIR, f))])
    warehouse_files = 0
    if os.path.exists(WAREHOUSE_DIR):
        warehouse_files = len([f for f in os.listdir(WAREHOUSE_DIR) if os.path.isfile(os.path.join(WAREHOUSE_DIR, f))])

    st.metric("📄 知识库文档数", doc_files)
    st.metric("📊 数仓导出文件数", warehouse_files)

    # 向量库状态
    if os.path.exists(FAISS_INDEX_PATH):
        st.metric("✅ 向量库状态", "已就绪")
    else:
        st.metric("⚠️ 向量库状态", "未构建")
        st.warning("请先运行 `python etl_parse.py` 构建索引")

    st.markdown("---")
    st.subheader("🔧 功能选项")
    show_refs = st.toggle("📚 显示参考片段", value=False)
    auto_scroll = st.checkbox("自动滚动", value=True)

    st.markdown("---")
    st.caption("MaxCompute + DeepSeek + FAISS")
    st.link_button("🔗 GitHub 源码", "https://github.com/FULINGF/agri_data_rag")

# ================= 主聊天区域 =================
st.title("🌽 农业知识智能问答系统")
st.caption("支持多文件检索 | 数仓数据联动 | 增量更新")

# 初始化会话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 接收用户输入
if prompt := st.chat_input("请输入农业相关问题..."):
    # 1. 记录用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 生成回答
    with st.chat_message("assistant"):
        with st.spinner("🤖 正在检索与思考..."):
            start_time = time.time()
            full_response = ""

            placeholder = st.empty()

            try:
                for chunk in ask_question_stream(prompt):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)
                latency_ms = int((time.time() - start_time) * 1000)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="⏱️ 响应耗时", value=f"{latency_ms}ms")
                with col2:
                    st.metric(label="📝 Token 消耗", value="估算~150")

                # 显示参考片段（如果开启）
                if show_refs:
                    docs = retriever.invoke(prompt)
                    if docs:
                        st.divider()
                        with st.expander(f"📚 查看参考片段 ({len(docs)} 个)", expanded=False):
                            for i, doc in enumerate(docs, 1):
                                source = doc.metadata.get("source", "未知来源")
                                page = doc.metadata.get("page", "?")
                                preview = doc.page_content[:150].replace("\n", " ")
                                st.markdown(f"**Chunk-{i}** (来源：{source}, 第{page}页)")
                                st.text(preview + "...")
                                st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })

            except Exception as e:
                error_msg = f"❌ 服务异常：{str(e)}"
                placeholder.markdown(error_msg)
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# 自动滚动到底部（仅当开启时）
if auto_scroll:
    st.markdown("<br><br>", unsafe_allow_html=True)