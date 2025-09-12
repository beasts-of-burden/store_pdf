import json
from haystack import Document

# ---------- 1. 读取 JSON ----------
with open("D:\\HayStack\haystack\\rules_character.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

# ---------- 2. 转换成 Haystack Document ----------
docs = []

for item in json_data:
    character = item.get("character", "")
    rules = item.get("rules", [])  # 注意 JSON 中可能是列表
    for rule in rules:
        docs.append(Document(content=rule, meta={"character": character}))

# ---------- 3. 查看结果 ----------
print(f"✅ 转换完成，共 {len(docs)} 条 Document")
# print(docs[:3])  # 打印前三条示例
