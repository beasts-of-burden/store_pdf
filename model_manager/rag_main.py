import os
import csv
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.preprocessors import DocumentSplitter
from component.generator import LocalLLMGenerator
from haystack import Document

# ================== 主流程 ==================
# 1. 加载 CSV 文件
csv_path = "/work/zxx/demon/model_manager/sample_rules.csv" # ✅ 修改为你的 CSV 文件路径
if not os.path.exists(csv_path):
    # 如果找不到原始路径，尝试使用示例路径
    example_csv_path = "./sample_rules.csv"
    if os.path.exists(example_csv_path):
        csv_path = example_csv_path
        print(f"⚠️ 使用示例规则文件: {csv_path}")
    else:
        print(f"❌ 未找到文件: {csv_path}")
        print("请创建一个包含规则的CSV文件。")
        exit(1)

documents = []
try:
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        # ✅ 使用 DictReader，它会将第一行作为列名
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            # ✅ 假设 CSV 有 'character' 和 'rule' 列
            character = row.get('character', '').strip()
            rule = row.get('rule', '').strip()
            
            # ✅ 检查必要字段是否为空
            if not character or not rule:
                print(f"⚠️ 警告: CSV 文件第 {i+2} 行缺少 'character' 或 'rule' 字段，已跳过。行内容: {row}")
                continue

            # ✅ 构造文档内容
            content = f"{character}的设计规则：{rule}"
            
            # ✅ 构造元数据，包含 CSV 中的所有列（方便后续使用）
            meta = {k.strip(): v.strip() for k, v in row.items() if k and v} # 清理键和值的空格
            
            documents.append(Document(content=content, meta=meta))

    print(f"✅ 成功从 CSV 加载 {len(documents)} 条规则。")

except Exception as e:
    print(f"❌ 读取或解析 CSV 文件时出错: {e}")
    exit(1)

# 2. 初始化 Document Store
document_store = InMemoryDocumentStore()

# 3. 分块和嵌入
splitter = DocumentSplitter(
    split_by="word",           
    split_length=300,          
    split_overlap=50,          
    language="zh"              
)
splitter.warm_up()
split_docs = splitter.run(documents=documents)["documents"]

print(f"✅ 文档分块完成，原始文档数: {len(documents)}，分块后文档数: {len(split_docs)}")

# 打印分块后的前几个文档作为示例
print("\n" + "="*80)
print("📋 分块后文档示例：")
print("="*80)
for i, doc in enumerate(split_docs[:3]):  # 只打印前3个示例
    print(f"文档 {i+1}:")
    print(f"内容: {doc.content[:100]}...")  # 只打印前100个字符
    print(f"元数据: {doc.meta}")
    print("-"*80)

# 生成文档嵌入
doc_embedder = SentenceTransformersDocumentEmbedder(
    model="/work/models/sentence-transformers/all-MiniLM-L6-v2"
)
doc_embedder.warm_up()
embedded_docs = doc_embedder.run(split_docs)

print(f"✅ 文档嵌入完成，共 {len(embedded_docs['documents'])} 个文档嵌入")

# 将嵌入后的文档写入 document_store
document_store.write_documents(embedded_docs["documents"])
print(f"✅ 文档已成功写入数据库")

# 5. 初始化检索器
retriever = InMemoryEmbeddingRetriever(document_store=document_store)

# 6. 初始化生成器
generator = LocalLLMGenerator(
    model_path="/work/models/Qwen/Qwen3-8B",  # 根据实际路径调整
    device="cuda"
)

# 7. 构建 Prompt
template = """
根据以下规则回答问题：

{% for doc in documents %}
- {{ doc.content|string }}
{% endfor %}

问题：{{ query|string }}
请根据上述规则回答问题，如果规则中没有相关信息，请说明无法根据提供的规则回答该问题。
"""

prompt_builder = PromptBuilder(
    template=template,
    required_variables=["documents", "query"]
)

# 8. 构建 Pipeline
pipe = Pipeline()

# 初始化组件
text_embedder = SentenceTransformersTextEmbedder(model="/work/models/sentence-transformers/all-MiniLM-L6-v2")
text_embedder.warm_up()

retriever = InMemoryEmbeddingRetriever(document_store=document_store)
prompt_builder = PromptBuilder(template=template)
generator = LocalLLMGenerator(
    model_path="/work/models/Qwen/Qwen3-8B",  # 根据实际路径调整
    device="cuda"
)

# 添加组件
pipe.add_component("text_embedder", text_embedder)
pipe.add_component("retriever", retriever)
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("generator", generator)

# 连接组件
pipe.connect("text_embedder.embedding", "retriever.query_embedding")
pipe.connect("retriever", "prompt_builder.documents")
# ✅ 不再从 text_embedder 传 query，而是从外部直接传入 query 文本
pipe.connect("prompt_builder", "generator.prompt")

# 9. 提问测试
queries = [
    "炭治郎的耳环在中国大陆使用时有什么限制？",
    "祢豆子的竹筒设计要注意什么？",
]
for query_text in queries:
    print("\n" + "=" * 80)
    print("📌 问题:", query_text)
    print("=" * 80)
    try:
        # 先获取嵌入的查询向量
        query_embedding = text_embedder.run(text=query_text)["embedding"]
        
        # 使用检索器获取相关文档
        retrieved_docs = retriever.run(query_embedding=query_embedding, top_k=5)["documents"]
        
        # 打印检索到的文档到醒目美观的框中
        print("\n" + "🔍" + "#" * 78 + "🔍")
        print(f"{'🔍 数据库检索结果':^78}")
        print("🔍" + "#" * 78 + "🔍")
        
        for i, doc in enumerate(retrieved_docs):
            print(f"🔍 检索结果 {i+1}:")
            print(f"🔍 内容: {doc.content}")
            print(f"🔍 元数据: {doc.meta}")
            print("🔍" + "-" * 78 + "🔍")
        
        print("🔍" + "#" * 78 + "🔍\n")
        
        # 运行完整的管道获取答案
        response = pipe.run(
            data={
                "text_embedder": {"text": query_text},
                "retriever": {"top_k": 150},
                "prompt_builder": {"query": query_text},  # ✅ 关键：手动传 query
                "generator": {}
            }
        )
        
        answer = response["generator"]["replies"][0]
        
        # 打印生成的答案到醒目美观的框中
        print("\n" + "📝" + "=" * 78 + "📝")
        print(f"{'📝 生成答案':^78}")
        print("📝" + "=" * 78 + "📝")
        print(f"{answer}")
        print("📝" + "=" * 78 + "📝\n")
        
    except Exception as e:
        print("❌ 错误:", str(e))