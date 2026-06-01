import streamlit as st
import time
from rag_query import ask_question_stream, retriever, format_refs_only

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

    /* 侧边栏：浅绿+干净白字，不暗沉 */
    [data-testid="stSidebar"] {
        background-color: #2F4F40;
        padding: 1rem;
    }
    [data-testid="stSidebar"] * {
        color: #F0F5E9 !important;
    }

    /* 侧边栏标题：小、细、不笨重 */
    [data-testid="stSidebar"] h3 {
        font-size: 0.95rem !important;
        font-weight: 500;
        margin: 0.5rem 0 0.2rem;
    }

    /* 侧边栏正文：超小、干净 */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMetric-value {
        font-size: 0.78rem !important;
    }

    /* 指标卡片：圆角、半透、不突兀 */
    [data-testid="stSidebar"] .stMetric {
        background: rgba(255,255,255,0.08);
        border-radius: 6px;
        padding: 0.4rem;
        margin: 0.2rem 0;
    }

    /* 分割线：细、淡、几乎看不见 */
    [data-testid="stSidebar"] hr {
        border-color: #D4C9A6;
        opacity: 0.2;
        margin: 0.7rem 0;
    }

    /* 侧边栏按钮：小、圆角、浅白 */
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

    /* 输入框 */
    .stChatInput textarea {
        border-radius: 20px;
        border: 1px solid #E0D6B0;
        font-size: 0.9rem;
    }

    /* 隐藏默认元素 */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ================= 侧边栏：极简排版（一眼清爽） =================
with st.sidebar:
    st.markdown("**⚙️ 系统设置**")
    st.divider()

    # 系统状态（一行一个、无多余标题）
    st.markdown("**📊 系统状态**")
    st.metric("文档数", "动态加载")
    st.metric("向量库", "已就绪")
    st.metric("API延迟", "~3.8s")

    st.divider()

    # 功能选项（极简文字）
    st.markdown("**🔧 功能**")
    show_refs = st.toggle("显示参考片段", False)
    auto_scroll = st.checkbox("自动滚动", True)

    st.divider()
    st.caption("MaxCompute + DeepSeek + FAISS")
    st.link_button("GitHub 源码", "https://github.com/FULINGF/agri_data_rag")

# ================= 主聊天区域 =================
st.title("🌽 农业知识智能问答系统")
st.caption("基于本地向量库与大模型检索增强生成 | 数据工程演示项目")

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

            # 流式输出占位符
            placeholder = st.empty()

            try:
                # 调用核心函数
                for chunk in ask_question_stream(prompt):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")

                # 完成输出
                placeholder.markdown(full_response)

                # 计算耗时
                latency_ms = int((time.time() - start_time) * 1000)

                # 显示性能指标
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="⏱️ 响应耗时", value=f"{latency_ms}ms")
                with col2:
                    st.metric(label="📝 Token 消耗", value="估算~150")  # 标注为估算

                # 显示参考片段（如果开启）
                if show_refs:
                    # ⚠️ 性能优化点：这里会再次检索，未来可优化为复用 ask_question_stream 中的 docs
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

                # 记录助手消息
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

# 自动滚动到底部
if auto_scroll:
    st.markdown("<br><br>", unsafe_allow_html=True)