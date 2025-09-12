import pymupdf  # pip install pymupdf
import re
import json
import csv
import os

# ---------- 配置 ----------
PDF_PATH = r"D:\HayStack\English.pdf"          # PDF 文件路径，注意用原始字符串 r""
OUTPUT_JSON = "rules.json"      # 输出 JSON
OUTPUT_CSV = "rules.csv"        # 输出 CSV
OCR_LANG = "chi_sim+jpn"        # 中文+日文 OCR 语言包（Tesseract）

# ---------- 打开 PDF ----------
doc = pymupdf.open(PDF_PATH)
all_text = ""

for i, page in enumerate(doc, 1):
    text = page.get_text()
    if not text.strip():
        # OCR 扫描件 PDF
        tp = page.get_textpage_ocr(language=OCR_LANG)
        text = page.get_text(textpage=tp)
    # 每页之间加一个空行，保持分页可读性
    all_text += f"--- Page {i} ---\n{text}\n\n"

# ---------- 匹配角色及规则 ----------
# 假设每条规则以 【角色名】 开头，匹配到下一条或文档末尾
pattern = r"【(.*?)】\n(.*?)(?=(\n【|$))"
matches = re.findall(pattern, all_text, re.DOTALL)

data = []
for char_name, rule_text, _ in matches:
    cleaned_rule = rule_text.strip().replace("\n", " ")
    data.append({
        "character": char_name.strip(),
        "rule": cleaned_rule
    })

# ---------- 输出 JSON ----------
with open(OUTPUT_JSON, "w", encoding="utf-8") as f_json:
    json.dump(data, f_json, ensure_ascii=False, indent=2)

# ---------- 输出 CSV ----------
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_csv:
    writer = csv.DictWriter(f_csv, fieldnames=["character", "rule"])
    writer.writeheader()
    for item in data:
        writer.writerow(item)

print(f"✅ 完成！共提取 {len(data)} 条规则")
print(f"JSON 文件: {os.path.abspath(OUTPUT_JSON)}")
print(f"CSV 文件: {os.path.abspath(OUTPUT_CSV)}")
