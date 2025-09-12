import json

# ---------- 1. 版本兼容导入 ----------
try:
    # 新版本 haystack >= 2.x
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.nodes import DensePassageRetriever
    from haystack.pipelines import DocumentSearchPipeline
    print("✅ 使用 Haystack 新版本 (>=2.x)")
except ImportError:
    # 老版本 farm-haystack <=1.x
    from haystack.document_stores import InMemoryDocumentStore
    from haystack.retriever.dense import DensePassageRetriever
    from haystack.pipeline import DocumentSearchPipeline
    print("✅ 使用 Haystack 老版本 (farm-haystack)")

from haystack import Document

# ---------- 2. 读取 JSON ----------
JSON_PATH = r"D:\HayStack\project\rules_character.json"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    json_data = json.load(f)

# ---------- 3. 转换成 Document ----------
docs = []
for item in json_data:
    character = item.get("character", "")
    rules = item.get("rules", [])
    for rule in rules:
        docs.append(Document(content=rule, meta={"character": character}))

print(f"✅ 转换完成，共 {len(docs)} 条 Document")

# ---------- 4. 创建 InMemory DocumentStore ----------
document_store = InMemoryDocumentStore()
document_store.write_documents(docs)

# ---------- 5. 创建 Retriever ----------
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    use_gpu=True  # 如果有 GPU，可设置 True，否则 False
)

# ---------- 6. 更新索引 ----------
document_store.update_embeddings(retriever)

# ---------- 7. 创建检索管道 ----------
pipe = DocumentSearchPipeline(retriever)

# ---------- 8. 测试查询 ----------
query_character = "炭治郎"
results = pipe.run(query=query_character, params={"Retriever": {"top_k": 5}})

print(f"\n角色 {query_character} 的规则：")
for r in results["documents"]:
    print("-", r.content)
