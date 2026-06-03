"""
数仓联动 - 批量防治建议生成器
从数仓 ADS 层读取高频病虫害指标，自动生成防治建议报告
"""
import os
import csv
from datetime import datetime
from rag_query import ask_question_stream, retriever

# ================= 配置 =================
# 数仓导出文件路径（从 MaxCompute 导出的 CSV）
WAREHOUSE_CSV = "./data/warehouse_exports/ads_jilin_corn_pest_top.csv"
# 报告输出路径
REPORT_OUTPUT = "./data/reports/pest_control_report.txt"


def read_pest_top_from_csv():
    """从数仓导出的 CSV 读取高频病虫害 Top N"""
    pests = []
    if os.path.exists(WAREHOUSE_CSV):
        with open(WAREHOUSE_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pests.append({
                    "pest_name": row["pest_name"],
                    "freq": row["freq"],
                    "dt": row["dt"]
                })
    else:
        # 兜底：硬编码示例数据
        pests = [
            {"pest_name": "玉米螟", "freq": 120, "dt": "2026-05-27"},
            {"pest_name": "大斑病", "freq": 85, "dt": "2026-05-27"},
            {"pest_name": "粘虫", "freq": 60, "dt": "2026-05-27"},
        ]
    return pests


def generate_report(pests):
    """生成防治建议报告"""
    os.makedirs(os.path.dirname(REPORT_OUTPUT), exist_ok=True)

    with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  吉林省玉米病虫害防治建议报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据日期：{pests[0]['dt'] if pests else 'N/A'}\n")
        f.write(f"高频病虫害数量：{len(pests)}\n")
        f.write("=" * 60 + "\n\n")

        for i, pest in enumerate(pests, 1):
            f.write(f"[{i}/{len(pests)}] {pest['pest_name']}（发生频次：{pest['freq']}）\n")
            f.write("-" * 60 + "\n")

            # 调用 RAG 生成防治建议
            query = f"如何防治{pest['pest_name']}？"
            print(f"🔍 检索 {pest['pest_name']} 的防治方案...")

            answer = ""
            for chunk in ask_question_stream(query):
                answer += chunk

            f.write(f"防治建议：\n{answer}\n\n")
            f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("报告生成完成\n")
        f.write("=" * 60 + "\n")

    print(f"\n✅ 报告已生成：{REPORT_OUTPUT}")


def main():
    print("=" * 60)
    print("  数仓联动 - 批量防治建议生成器")
    print("=" * 60)

    # 读取数仓数据
    pests = read_pest_top_from_csv()
    print(f"📊 读取到 {len(pests)} 个高频病虫害")

    # 生成报告
    generate_report(pests)

    print("\n" + "=" * 60)
    print("  批处理完成")
    print("=" * 60)


if __name__ == "__main__":
    main()