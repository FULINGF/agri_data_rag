"""
本地数仓 + RAG 联动脚本（从本地 CSV 读取 ADS 数据，调用本地 RAG 生成答案）
"""
import csv
import time
import os
from datetime import datetime
from rag_query import ask_question_stream, retriever, format_refs_only

# ==================== 配置 ====================
ADS_CSV_PATH = "./data/pest_list.csv"        # 从 DataWorks 下载的数据文件
LOG_CSV_PATH = "./rag_ads_log.csv"      # 联动日志输出


def get_top_pests_from_csv(csv_path, limit=3):
    """直接从 CSV 读取前三行有效数据（忽略多余列和空行）"""
    pests = []
    if not os.path.exists(csv_path):
        print(f"⚠️ 未找到 {csv_path}，使用默认测试数据")
        return ["玉米螟", "大斑病", "粘虫"][:limit]

    with open(csv_path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig 自动去除 BOM
        lines = f.readlines()
        # 跳过第一行表头，取后续非空行
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.strip().split(',')
            if len(parts) >= 2:
                pest_name = parts[0].strip('"')
                freq_str = parts[1].strip('"')
                if pest_name and freq_str and freq_str.isdigit():
                    pests.append((pest_name, int(freq_str)))
            if len(pests) >= limit:
                break
    return [name for name, _ in pests]

def write_log_to_csv(query, answer, latency_ms, ref_info):
    """将问答日志写入 CSV"""
    file_exists = os.path.exists(LOG_CSV_PATH)
    with open(LOG_CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "query", "answer", "latency_ms", "ref_info"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            query,
            answer.replace("\n", " "),
            latency_ms,
            ref_info.replace("\n", " ")
        ])

def main():
    print("=" * 60)
    print("🌽 本地数仓 + RAG 联动任务")
    print("=" * 60)

    # 1. 从本地 CSV 读取高频病虫害
    pests = get_top_pests_from_csv(ADS_CSV_PATH, limit=3)
    print(f"✅ 读取到高发病虫害：{pests}")

    if not pests:
        print("⚠️ 无病虫害数据，退出")
        return

    # 2. 对每个病虫害生成防治方案
    for idx, pest in enumerate(pests, 1):
        query = f"吉林玉米种植中，{pest} 如何防治？"
        print(f"\n[{idx}/{len(pests)}] 正在处理：{query}")

        # 调用你已有的 RAG 流式接口，收集完整答案
        start_time = time.time()
        full_answer = ""
        for chunk in ask_question_stream(query):
            full_answer += chunk
        latency_ms = int((time.time() - start_time) * 1000)

        # 获取检索到的参考资料（可选）
        docs = retriever.invoke(query)
        ref_info = format_refs_only(docs) if docs else "无参考资料"

        print(f"🤖 生成答案（{latency_ms}ms）：{full_answer[:100]}...")

        # 3. 将日志写入本地 CSV
        write_log_to_csv(query, full_answer, latency_ms, ref_info)
        print(f"📝 日志已写入 {LOG_CSV_PATH}")

    print("\n✅ 本地联动任务完成")

if __name__ == "__main__":
    main()