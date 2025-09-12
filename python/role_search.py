import json
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.nodes import DensePassageRetriever
from haystack.pipelines import DocumentSearchPipeline


# ---------- 1. 读取 JSON ----------
with open(r"D:\HayStack\project\rules_character.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

# ---------- 2. 转换成 Haystack Document ----------
docs = []
for item in json_data:
    character = item.get("character", "")
    rules = item.get("rules", [])
    for rule in rules:
        docs.append(Document(content=rule, meta={"character": character}))

print(f"✅ 转换完成，共 {len(docs)} 条 Document")

# ---------- 3. 创建 InMemory DocumentStore ----------
document_store = InMemoryDocumentStore()
document_store.write_documents(docs)

# ---------- 4. 创建 Retriever ----------
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    use_gpu=True  # 如果有 GPU，可以开，否则 False
)

# ---------- 5. 更新索引 ----------
document_store.update_embeddings(retriever)

# ---------- 6. 创建管道 ----------
pipe = DocumentSearchPipeline(retriever)

# ---------- 7. 测试查询 ----------
query_character = "炭治郎"
results = pipe.run(query=query_character, params={"Retriever": {"top_k": 5}})

print(f"\n角色 {query_character} 的规则：")
for r in results["documents"]:
    print("-", r.content)
