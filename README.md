# 🌾 农业知识智能问答系统 (RAG + 数仓联动)
**作者：赵飞宏**
基于 **检索增强生成（RAG）** 的农业知识问答系统，支持多格式文档检索、增量更新，并提供**数仓联动**能力，自动读取 ADS 层高频病虫害指标并生成防治报告。

---

## 📌 项目背景
本项目是**数据工程 + AI 应用**个人实战项目，展示全链路落地流程：
- 使用 **ETL 流水线** 处理多源文档（PDF、DOCX、CSV、TXT）并构建向量知识库。
- 通过 **DeepSeek 大模型** + **FAISS 向量检索** 实现智能问答。
- 与**数据仓库**联动：从数仓导出的 CSV 指标自动生成防治建议，形成“数据→知识→决策”闭环。

项目落地聚焦吉林地区玉米病虫害防治场景，代码可快速复用拓展至其他行业知识库。

## 🚀 核心功能
| 功能模块 | 说明 |
|---------|------|
| **多文件知识库** | 支持 `.pdf` `.docx` `.csv` `.txt` `.md` 等格式，自动解析并建索引 |
| **增量 ETL** | 只处理新增/变更文件，MD5 校验，避免重复计算 |
| **RAG 问答** | 基于检索到的文档片段生成回答，严格避免幻觉 |
| **流式对话 Web UI** | Streamlit 界面，支持显示引用片段、自动滚动 |
| **数仓联动** | 读取 ADS 层 CSV（如高频病虫害），批量生成防治报告 |
| **问答日志** | 自动记录用户问题、答案、耗时、引用片段，可回流数仓分析 |

## 🧱 技术架构
```text
[文档文件夹] ──┐
[数仓导出文件夹] ─┼─→ etl_parse.py (切分+向量化) ──→ FAISS 索引
               │
               └─→ 增量更新 (MD5 变化检测)

用户提问 ──→ rag_query.py (检索+LLM生成) ──→ 流式回答
                │
                └─→ 记录日志 query_log.csv

数仓 CSV ──→ integration.py ──→ 批量调用 RAG ──→ 防治报告.txt
```
### 主要依赖
- LangChain：文档加载、切分、RAG 链构建
- FAISS：向量存储与相似度检索
- HuggingFace Embeddings：BAAI/bge-small-zh-v1.5 (中文轻量)
- DeepSeek API：大模型生成
- Streamlit：前端界面

## 📁 项目结构
```
agri_rag_demo/
├── data/
│   ├── documents/               # 知识库源文件（PDF, DOCX, TXT...）
│   ├── warehouse_exports/       # 数仓导出的 CSV（如高频病虫害表）
│   ├── reports/                 # integration.py 生成的报告
│   ├── faiss_index/             # FAISS 索引 + file_metadata.json
│   └── query_log.csv            # 问答日志
├── config.py                    # 全局配置（路径、模型参数）
├── etl_parse.py                 # ETL 流水线（扫描、加载、切分、向量化）
├── rag_query.py                 # 问答核心引擎（检索 + LLM）
├── integration.py               # 数仓联动脚本（批量生成报告）
├── app.py                       # Streamlit Web 界面
├── requirements.txt             # Python 依赖
└── README.md
```

## 🔧 安装与本地运行
### 1. 克隆仓库
```bash
git clone https://github.com/你的用户名/agri_rag_demo.git
cd agri_rag_demo
```

### 2. 创建虚拟环境（推荐）
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 配置环境变量
项目根目录新建 `.env` 文件填写密钥：
```env
DEEPSEEK_API_KEY=sk-xxxxxxxx
```

### 5. 准备知识库文档
将 PDF、DOCX 等源文件放入 `data/documents/`。

### 6. 运行 ETL 构建索引
```bash
# 首次全量构建索引
python etl_parse.py --full
# 新增文件执行增量更新
python etl_parse.py
```

### 7. 启动 Web 界面
```bash
streamlit run app.py
```
浏览器访问 `http://localhost:8501` 即可使用问答。

### 8. 测试数仓联动
数仓CSV放入 `data/warehouse_exports/ads_jilin_corn_pest_top.csv`，执行：
```bash
python integration.py
```
生成报告保存至 `data/reports/pest_control_report.txt`。

## 📦 数仓联动 CSV 格式示例
```csv
pest_name,freq,dt
玉米螟,120,2026-05-27
大斑病,85,2026-05-27
粘虫,60,2026-05-27
```

## 🧪 测试示例
用户提问：
> 玉米螟如何防治？

系统回答：
> 根据资料，防治玉米螟可采用以下方法：1. 选用抗虫品种；2. 秸秆处理降低越冬虫源；3. 生物防治释放赤眼蜂；4. 化学防治在幼虫低龄期使用氯虫苯甲酰胺等。

知识库无相关内容统一回复：
> 根据现有资料无法回答该问题。

## 📝 主要文件说明
| 文件 | 作用 |
| ---- | ---- |
| config.py | 统一管理路径、切分参数、嵌入模型、DeepSeek配置 |
| etl_parse.py | 扫描源文件，文档加载→切分→向量化→保存索引，支持全量/增量构建 |
| rag_query.py | 加载向量库与大模型，封装流式问答接口，自动记录问答日志 |
| integration.py | 读取数仓CSV数据，批量调用RAG生成病虫害防治报告 |
| app.py | Streamlit前端页面，实现人机交互、知识库状态展示 |