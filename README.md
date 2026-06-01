# 🌽 农业 RAG 知识问答系统
> 基于结构化 + 非结构化数据的完整 AI 数据工程流水线






---

## 🎯 项目定位
本项目是一个面向农业知识问答的 RAG（检索增强生成）系统，覆盖从 PDF 文档解析、向量化存储到流式问答的完整 AI 工程链路。

同时，通过 `integration.py` 与数仓项目联动，实现**结构化数据（ADS 层指标）驱动非结构化知识检索**的业务闭环。

> 关联项目：吉林玉米产量分析数据仓库

---

## 📁 项目结构
```text
agri_rag_demo/
├── etl_parse.py           # 离线 ETL：PDF→文本切分→DQC 校验→MD5 去重→FAISS 向量入库
├── rag_query.py           # 在线推理：向量检索→Prompt 组装→LLM 流式生成→日志回流
├── app.py                 # Streamlit UI：多轮对话 + 可选参考片段
├── integration.py         # 数仓联动：从 MaxCompute/CSV 读取高频病虫害，自动生成防治建议
├── config.py              # 配置中心：模型参数、路径、检索参数集中管理
├── .gitignore             # 忽略敏感文件、数据目录与缓存
├── .env                   # 密钥与环境变量（不提交到 Git）
└── data/
    ├── pest_control.pdf   # 原始农业病虫害知识库文档
    ├── pest_list.csv      # 从 MaxCompute ADS 层导出的高频病虫害列表
    ├── faiss_index/       # FAISS 向量索引（etl_parse.py 生成）
    └── query_log.csv      # 结构化问答日志（模拟数仓 ODS 层采集）
🔄 数据流向
text
PDF 文档
   ↓ etl_parse.py
文本切分 → DQC 质量过滤 → MD5 去重 → BGE 向量化 → FAISS 索引存储
   ↓
用户提问 → FAISS 检索 → 可选参考片段 → Prompt 组装 → DeepSeek 流式生成 → 日志回流 CSV
   ↓
MaxCompute ADS 高频病虫害 → integration.py → 自动触发 RAG → 防治建议输出
🛠️ 功能亮点
表格
模块	功能	技术说明
离线 ETL	PDF 解析 + DQC 质量监控 + MD5 幂等去重	过滤长度不足、乱码片段，仅保留高质量 Chunk
在线推理	流式输出 + 可选参考片段	用户体验友好，调试时可溯源
抗幻觉机制	Prompt 强制约束 + 统一兜底话术	无资料时回复：根据现有资料无法回答该问题
日志回流	Chunk 级结构化日志	记录命中片段、延迟、回答内容，可对接数仓
数仓联动	ADS 指标自动驱动 RAG 检索	打通结构化数据与非结构化知识的业务闭环
🚀 快速启动
1. 安装依赖
bash
运行
pip install -r requirements.txt
2. 配置 API Key
在系统环境变量中设置（或在项目根目录创建 .env 文件）：
bash
运行
DEEPSEEK_API_KEY=sk-你的DeepSeek密钥
3. 放入文档
将农业 PDF 放入 data/ 目录，确保 config.py 中 PDF_PATH 指向该文件。
4. 构建离线索引
bash
运行
python etl_parse.py
5. 启动交互界面
bash
运行
streamlit run app.py
浏览器打开 http://localhost:8501，即可进行农业知识问答。
6.（可选）命令行模式
bash
运行
python rag_query.py
# 提问示例：
# 如何防治玉米螟
# 查看参考片段：ref:如何防治玉米螟
7.数仓联动演示
bash
运行
python integration.py
从 data/pest_list.csv 读取高频病虫害，批量生成防治建议。
🔗 与数仓项目的结合
本项目是 吉林玉米产量分析数据仓库 的 AI 能力扩展：
表格
模块	处理数据类型	技术栈	产出
数仓	结构化业务数据（产量、气象、土壤）	DataWorks + MaxCompute	五层数仓 + ADS 指标
RAG	非结构化知识数据（农技文档）	LangChain + FAISS + DeepSeek	向量检索 + 智能问答
联动	ADS 指标 → RAG 自动推理	integration.py	“指标监控 → 知识检索 → 决策建议” 闭环
👨‍💻 开发者
赵飞宏
吉林农业大学・数据科学与大数据技术