import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import streamlit as st
from rag_query import ask_question_stream, retriever, format_refs_only

st.set_page_config(page_title="农业RAG助手", page_icon="🌽")
st.title("🌽 农业知识智能问答")
st.caption("基于本地向量库与大模型检索增强生成 | 数据工程演示项目")

# 侧边栏：参考片段开关
show_refs = st.sidebar.toggle("📚 显示参考片段", value=False)

if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 接收用户输入
if prompt := st.chat_input("请输入农业相关问题..."):
    # 记录用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("检索与生成中..."):
            # 流式输出回答
            full_response = ""
            placeholder = st.empty()
            for chunk in ask_question_stream(prompt):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

            # 按需展示参考片段
            if show_refs:
                docs = retriever.invoke(prompt)
                if docs:
                    with st.expander("📚 参考片段", expanded=False):
                        st.text(format_refs_only(docs))

    # 记录助手消息
    st.session_state.messages.append({"role": "assistant", "content": full_response})