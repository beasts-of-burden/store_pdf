from flask import Flask, render_template, request, jsonify
import torch
import os
import json
import csv
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline
from component.utils import ModelUtils
from component.template import template_dict
from component.splitter import DocumentSplitter
from sentence_transformers import SentenceTransformer
import copy
from werkzeug.utils import secure_filename
from component.generator import LocalLLMGenerator

# 定义Document类用于表示文档
class Document:
    def __init__(self, content, meta=None):
        self.content = content
        self.meta = meta if meta else {}

# 定义DocumentStore类用于存储文档
class InMemoryDocumentStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.doc_to_idx = {}
        self.next_idx = 0
    
    def write_documents(self, documents):
        for doc in documents:
            if isinstance(doc, dict):
                content = doc.get('content', '')
                embedding = doc.get('embedding')
                meta = doc.get('meta', {})
            else:
                content = getattr(doc, 'content', '')
                embedding = getattr(doc, 'embedding', None)
                meta = getattr(doc, 'meta', {})
            
            self.documents.append(content)
            self.embeddings.append(embedding)
            self.metadata.append(meta)
            self.doc_to_idx[self.next_idx] = len(self.documents) - 1
            self.next_idx += 1

# 定义EmbeddingRetriever类用于检索
class InMemoryEmbeddingRetriever:
    def __init__(self, document_store):
        self.document_store = document_store
    
    def run(self, query_embedding, top_k=3):
        if not self.document_store.embeddings or all(emb is None for emb in self.document_store.embeddings):
            return {'documents': []}
            
        similarities = []
        for idx, embedding in enumerate(self.document_store.embeddings):
            if embedding is not None:
                similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
                similarities.append((similarity, idx))
                
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_indices = similarities[:top_k]
        
        retrieved_docs = []
        for sim, idx in top_indices:
            retrieved_docs.append({
                'content': self.document_store.documents[idx],
                'embedding': self.document_store.embeddings[idx],
                'meta': self.document_store.metadata[idx],
                'similarity': float(sim)
            })
            
        return {'documents': retrieved_docs}

# 定义PromptBuilder类用于构建提示
class PromptBuilder:
    def __init__(self, template, required_variables):
        self.template = template
        self.required_variables = required_variables
    
    def run(self, documents, query):
        # 格式化模板，替换变量
        prompt = self.template
        prompt = prompt.replace('{{ documents }}', str(documents))
        prompt = prompt.replace('{{ query }}', str(query))
        return {'prompt': prompt}
# 添加llama2和mistral的模板配置
template_dict.update({
    'llama2': {
        'template_name': 'llama2',
        'system': None,
        'system_format': "[INST] <<SYS>>\n{content}\n<</SYS>>\n\n",
        'user_format': "{content} [/INST]",
        'assistant_format': "{content} </s>",
        'stop_word': "</s>"
    },
    'mistral': {
        'template_name': 'mistral',
        'system': None,
        'system_format': "<s>[INST] {content} [/INST]",
        'user_format': "[INST] {content} [/INST]",
        'assistant_format': "{content} </s>",
        'stop_word': "</s>"
    }
})

app = Flask(__name__)

# 全局变量
model = None
tokenizer = None
template_name = None

# RAG相关全局变量
rules_data = None
embeddings_model = None
vector_db = None
cached_original_prediction = None

def get_device(gpu_id=5):
    # 使用指定的GPU ID
    return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_model', methods=['POST'])
def load_model():
    global model, tokenizer, template_name

    data = request.get_json()
    model_path = data.get('model_path')
    gpu_id     = data.get('gpu', 5)
    
    # 保存GPU ID到全局变量
    global gpu_id_global
    gpu_id_global = gpu_id
    print(f"设置使用GPU {gpu_id_global}")

    # 校验路径
    if not model_path or not os.path.isdir(model_path):
        return jsonify({'status': 'error', 'message': '模型路径无效'})

    # 清理旧模型
    if model is not None:
        print(f'开始卸载模型...')
        model_name = os.path.basename(model_path) if model_path else '当前模型'
        return jsonify({
            'status': 'unloading',
            'message': f'{model_name}正在卸载中...',
            'log': f'{model_name}正在卸载中...'
        })
        
        del model
        del tokenizer
        torch.cuda.empty_cache()
        print(f'{model_name}卸载成功，显存已释放')
        return jsonify({
            'status': 'success',
            'message': f'{model_name}卸载成功',
            'log': f'{model_name}已成功卸载，显存已释放'
        })

    # 1. 加载 tokenizer
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # 根据模型类型选择模板
    if config.model_type == 'llama':
        template_name = 'llama2'
    elif config.model_type == 'mistral':
        template_name = 'mistral'
    elif hasattr(config, 'chat_template') and 'qwen' in config.chat_template.lower():
        template_name = 'qwen3'
    elif 'qwen' in os.path.basename(model_path).lower():
        template_name = 'qwen3'  # 如果模型路径包含qwen，也使用qwen3模板
    else:
        template_name = 'llama2'  # 默认使用llama2模板
    
    # llama和mistral模型不支持fast tokenizer
    use_fast = False if config.model_type in ['llama', 'mistral'] else True
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=use_fast
    )
    
    # 如果是Qwen3模板，应用自定义的chat_template
    if template_name == 'qwen3' and 'qwen3' in template_dict:
        qwen3_template = template_dict['qwen3']
        if 'chat_template' in qwen3_template:
            tokenizer.chat_template = qwen3_template['chat_template']
            print(f"已应用Qwen3自定义chat_template")

    # 2. 加载模型：使用指定的GPU
    device = get_device(gpu_id)
    torch.backends.cudnn.benchmark = True

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,# FP16 精度
        low_cpu_mem_usage=True,   # 减少 CPU 内存占用
        trust_remote_code=True,
        device_map={"": gpu_id}  # 映射到指定的GPU
    )
    model.to(device)
    model.eval()
    # 强制全模型半精度，避免生成概率出现 NaN/Inf
    model.half()

    print(f'模型加载成功，路径: {model_path}, 设备: {device}')
    return jsonify({
        'status': 'success',
        'message': f'模型已加载到 {device}',
        'log': f'模型加载成功，路径: {model_path}, 设备: {device}'
    })

def build_prompt(tokenizer, template, query, history, system=None):
    global template_name  # 添加全局变量声明
    messages = []
    if system:
        messages.append({'role': 'system', 'content': str(system)})
    for h in history:
        messages.append({'role': h['role'], 'content': str(h['message'])})
    messages.append({'role': 'user', 'content': str(query)})
    
    # 如果是Qwen3模板且tokenizer有chat_template属性，使用原生chat_template
    if template_name == 'qwen3' and hasattr(tokenizer, 'chat_template'):
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"使用Qwen3 chat_template时出错: {e}")
            # 回退到原始方法
            prompt = ''
            for msg in messages:
                if msg['role'] == 'system' and template['system_format']:
                    prompt += template['system_format'].format(content=str(msg['content']))
                elif msg['role'] == 'user':
                    prompt += template['user_format'].format(content=str(msg['content']))
                elif msg['role'] == 'assistant':
                    prompt += template['assistant_format'].format(content=str(msg['content']))
    else:
        # 原始方法构建prompt
        prompt = ''
        for msg in messages:
            if msg['role'] == 'system' and template['system_format']:
                prompt += template['system_format'].format(content=str(msg['content']))
            elif msg['role'] == 'user':
                prompt += template['user_format'].format(content=str(msg['content']))
            elif msg['role'] == 'assistant':
                prompt += template['assistant_format'].format(content=str(msg['content']))
    
    return tokenizer(prompt, return_tensors='pt').input_ids

import csv
from werkzeug.utils import secure_filename

@app.route('/load_rules', methods=['POST'])
def load_rules():
    global rules_data, embeddings_model, vector_db
    
    data = request.get_json()
    rules_path = data.get('rules_path')
    
    if not rules_path or not os.path.isfile(rules_path):
        return jsonify({'status': 'error', 'message': '规则文件路径无效'})
    
    try:
        # 初始化嵌入模型
        if embeddings_model is None:
            # 添加参数避免连接Hugging Face
            embeddings_model = SentenceTransformer(
                '/work/models/sentence-transformers/all-MiniLM-L6-v2',
                local_files_only=True,  # 只使用本地文件
                use_auth_token=False    # 不使用auth token
            )
        
        # 为规则生成向量嵌入
        vector_db = []
        rules_data = []
        
        # 初始化文档分块器
        splitter = DocumentSplitter(
            split_by="word",
            split_length=300,
            split_overlap=50,
            language="zh"
        )
        splitter.warm_up()
        
        # 只支持CSV文件格式
        file_ext = os.path.splitext(rules_path)[1].lower()
        if file_ext != '.csv':
            raise ValueError(f"只支持CSV文件格式，不支持{file_ext}格式")
        
        # 处理CSV文件
        print(f"加载CSV文件: {rules_path}")
        documents = []
        with open(rules_path, "r", encoding="utf-8") as csvfile:
            # 使用DictReader，它会将第一行作为列名
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                # 假设CSV有'character'和'rule'列
                character = row.get('character', '').strip()
                rule = row.get('rule', '').strip()
                
                # 检查必要字段是否为空
                if not character or not rule:
                    print(f"⚠️ 警告: CSV文件第{i+2}行缺少'character'或'rule'字段，已跳过。行内容: {row}")
                    continue

                # 构造文档内容
                content = f"{character}的设计规则：{rule}"
                
                # 构造元数据，包含CSV中的所有列（方便后续使用）
                meta = {k.strip(): v.strip() for k, v in row.items() if k and v} # 清理键和值的空格
                
                documents.append(Document(content=content, meta=meta))
                rules_data.append(row)

        print(f"✅ 成功从CSV加载 {len(documents)} 条规则。")
        
        # 对文档进行分块
        if documents:
            print(f"开始对CSV文档进行分块，原始文档数: {len(documents)}")
            split_docs = []
            for doc in documents:
                # 模拟Document对象的分块
                chunks = splitter.split_text(doc.content, chunk_size=300, chunk_overlap=50)
                for i, chunk in enumerate(chunks):
                    chunk_meta = doc.meta.copy()
                    chunk_meta['chunk_index'] = i
                    chunk_meta['original_length'] = len(doc.content)
                    split_docs.append({'text': chunk, 'meta': chunk_meta})
                
            print(f"分块完成，分块后文档数: {len(split_docs)}")
            
            # 为分块后的文档生成嵌入
            for doc in split_docs:
                embedding = embeddings_model.encode(doc['text'])
                vector_db.append({'text': doc['text'], 'embedding': embedding, 'meta': doc['meta']})
        
        print(f'成功加载规则文件，共{len(vector_db)}条规则')
        return jsonify({
            'status': 'success',
            'message': f'成功加载{len(vector_db)}条规则',
            'log': f'规则文件加载成功: {rules_path}, 共{len(vector_db)}条规则'
        })
    except Exception as e:
        print('加载规则文件时出错:', str(e))
        return jsonify({'status': 'error', 'message': '加载规则文件时出错: ' + str(e)})

@app.route('/unload_model', methods=['POST'])
def unload_model():
    global model, tokenizer, template_name, llm_generator
    
    if model is None:
        return jsonify({'status': 'error', 'message': '没有加载的模型可卸载'})
        
    try:
        model_name = os.path.basename(model_path) if 'model_path' in globals() else '当前模型'
        
        # 释放模型资源
        del model
        del tokenizer
        torch.cuda.empty_cache()
        model = None
        tokenizer = None
        template_name = None
        
        # 清理LocalLLMGenerator资源
        if llm_generator is not None:
            llm_generator.model = None
            llm_generator.tokenizer = None
            llm_generator.pipe = None
            llm_generator = None
        
        print(f'{model_name}卸载成功，显存已释放')
        return jsonify({
            'status': 'success',
            'message': f'{model_name}卸载成功',
            'log': f'{model_name}已成功卸载，显存已释放'
        })
    except Exception as e:
        return jsonify({
            'status': 'error', 
            'message': f'卸载过程中出现异常: {str(e)}'
        })




# 初始化全局变量
llm_generator = None

@app.route('/predict', methods=['POST'])
def predict():
    global model, tokenizer, template_name, llm_generator
    if model is None or tokenizer is None:
        return jsonify({'status': 'error', 'message': '请先加载模型'})

    data = request.get_json()
    user_input = data.get('input_text', '').strip()
    print('收到预测请求，输入内容:', user_input)
    if not user_input:
        return jsonify({'status': 'error', 'message': '请输入内容'})

    # 使用模板构建prompt
    history = data.get('history', [])
    system_prompt = data.get('system_prompt', None)
    
    # 存储原始用户输入用于对比
    original_user_input = user_input
    
    # 检索相关规则
    retrieved_rules = []
    user_input_with_context = original_user_input
    
    if vector_db and len(vector_db) > 0:
            # 初始化DocumentStore和Retriever
            document_store = InMemoryDocumentStore()
            
            # 将vector_db中的数据添加到document_store
            docs_to_store = []
            for item in vector_db:
                docs_to_store.append({
                    'content': item['text'],
                    'embedding': item['embedding'],
                    'meta': item['meta']
                })
            document_store.write_documents(docs_to_store)
            
            retriever = InMemoryEmbeddingRetriever(document_store=document_store)
            
            # 生成查询向量
            query_embedding = embeddings_model.encode(original_user_input)
            
            # 检索相似文档
            retrieval_result = retriever.run(query_embedding, top_k=3)
            top_rules = retrieval_result['documents']
            
            # 保存检索到的规则，符合前端显示格式要求
            retrieved_rules = []
            for i, item in enumerate(top_rules):
                # 构建每条规则信息，包含所有必要字段
                rule_info = {
                    'id': i + 1,
                    'text': item['content'], 
                    'similarity': float(item['similarity']),
                    'rank': i + 1,
                    'content': item['content'],  # 为前端显示添加content字段
                    'score': float(item['similarity'])  # 为前端显示添加score字段
                }
                # 如果有元数据，也添加到规则信息中
                if 'meta' in item and item['meta']:
                    meta_info = " | ".join([f"{str(k)}: {str(v)}" for k, v in item['meta'].items()])
                    rule_info['meta'] = meta_info
                retrieved_rules.append(rule_info)
                
            # 打印检索结果到醒目美观的框中，符合用户要求的格式
            print("\n" + "=" * 60)
            print("📌 问题:", original_user_input)
            print("🔍 检索到的规则：")
            for i, rule in enumerate(retrieved_rules):
                print(f"\n💡 规则 #{rule['rank']} (相似度: {rule['similarity']:.4f}):")
                print(f"{rule['text']}")
                if 'meta' in rule:
                    print(f"📝 元数据: {rule['meta']}")
            print("=" * 60 + "\n")
            
            # 构建上下文，使用更专业的模板
            template = """
根据以下规则回答问题：

{% for doc in documents %}
- 角色: {{ doc.meta['character'] }}
  规则: {{ doc.content.split('：', 1)[1] if '：' in doc.content else doc.content }} 
  {% if 'category' in doc.meta and doc.meta['category'] %}分类: {{ doc.meta['category'] }} {% endif %}
{% endfor %}

问题：{{ query }}
请根据上述规则回答问题，如果规则中没有相关信息，请说明无法根据提供的规则回答该问题。
"""
            
            # 手动格式化模板
            formatted_rules = []
            for doc in top_rules:
                character = doc.get('meta', {}).get('character', 'Unknown')
                rule_content = doc['content'].split('：', 1)[1] if '：' in doc['content'] else doc['content']
                category = doc.get('meta', {}).get('category', '')
                
                rule_str = f"- 角色: {character}\n  规则: {rule_content}"
                if category:
                    rule_str += f"\n  分类: {category}"
                formatted_rules.append(rule_str)
            
            formatted_rules_str = "\n".join(formatted_rules)
            user_input_with_context = f"根据以下规则回答问题：\n\n{formatted_rules_str}\n\n问题：{original_user_input}\n请根据上述规则回答问题，如果规则中没有相关信息，请说明无法根据提供的规则回答该问题。"
            
            # 打印检索结果到醒目美观的框中
            print("\n" + "=" * 80)
            print("🔍 数据库检索内容")
            print("=" * 80)
            for i, item in enumerate(top_rules):
                char = item.get('meta', {}).get('character', 'Unknown')
                # 提取规则部分用于显示
                rule_part = item['content'].split('：', 1)[1] if '：' in item['content'] else item['content']
                print(f"\n💡 检索结果 #{i+1} (相似度: {item['similarity']:.4f}):")
                print(f"  角色: {char}")
                print(f"  规则: {rule_part[:200]}...")
                # 可以打印其他元数据
                if 'category' in item.get('meta', {}):
                    print(f"  分类: {item['meta']['category']}")
            print("=" * 80 + "\n")
    
    try:
        # 初始化LocalLLMGenerator
        if llm_generator is None:
            print("初始化LocalLLMGenerator...")
            # 从全局变量获取模型路径
            model_path_val = globals().get('model_path', '')
            # 从全局变量获取GPU ID，如果没有则使用默认值5
            gpu_id_val = globals().get('gpu_id_global', 5)
            device = f'cuda:{gpu_id_val}' if torch.cuda.is_available() else 'cpu'
            llm_generator = LocalLLMGenerator(model_path=model_path_val, device=device)
            # 传递tokenizer，避免重复加载
            llm_generator.tokenizer = tokenizer
            llm_generator.model = model
            # 当模型通过accelerate加载时，不需要在pipeline中指定device参数
            llm_generator.pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer
            )
            llm_generator._is_warmed_up = True
            print("✅ LocalLLMGenerator初始化完成")
        
        global cached_original_prediction
        prediction_original = None
        prediction_with_context = ""
        
        # 始终确保without rag框有内容（模型回答）
        if not cached_original_prediction:
            print("🔄 生成原始回答并缓存（without rag框内容）")
            response_original = llm_generator.run(prompt=original_user_input)
            cached_original_prediction = response_original["replies"][0]
        prediction_original = cached_original_prediction
        
        # 只有在加载了规则文件时，才生成带规则的回答
        if vector_db and len(vector_db) > 0:
            # 已加载规则的情况：在with rag框显示结合规则的回答
            print("🔄 已加载规则文件，生成带规则的回答（with rag框内容）...")
            response_with_context = llm_generator.run(prompt=user_input_with_context)
            prediction_with_context = response_with_context["replies"][0]
        else:
            # 未加载规则的情况：with rag框显示空字符串（什么都没有）
            print("🔄 未加载规则文件，with rag框内容为空")
            prediction_with_context = ""
        
        print(f'生成完成，输入: {original_user_input}, 带规则输出: {str(prediction_with_context)[:50]}...')
    except Exception as e:
        print('生成过程中出现异常:', str(e))
        return jsonify({'status': 'error', 'message': '生成过程中出现异常:' + str(e)})

    # 确保返回结果中包含原始回答，无论是否加载规则
    if prediction_original:
        # 确定使用哪个回答更新历史记录
        if prediction_with_context and vector_db and len(vector_db) > 0:
            # 有规则文件且生成了带规则的回答
            history_answer = prediction_with_context
            log_content = f'输入: {original_user_input}\n带规则输出: {str(prediction_with_context)}\n原始输出: {str(prediction_original)}'
        else:
            # 没有规则文件或带规则的回答为空
            history_answer = prediction_original
            log_content = f'输入: {original_user_input}\n原始输出: {str(prediction_original)}'
        
        # 更新历史记录
        updated_history = copy.deepcopy(history)
        updated_history.append({'role': 'user', 'message': original_user_input})
        updated_history.append({'role': 'assistant', 'message': history_answer})
        
        # 构建返回结果
        result = {
            'status': 'success', 
            'prediction': prediction_with_context,  # 带规则的回答
            'original_prediction': prediction_original,  # 不带规则的回答
            'history': updated_history,
            'retrieved_rules': retrieved_rules,  # 只有在已加载规则时才会有内容
            'log': log_content
        }
        return jsonify(result)
    else:
        return jsonify({'status':'error','message':'生成内容为空'})
        


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
    