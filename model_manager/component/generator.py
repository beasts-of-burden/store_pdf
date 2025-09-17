import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List

class LocalLLMGenerator:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self._is_warmed_up = False
        
        # 检查设备可用性
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️ CUDA不可用，将使用CPU")
            self.device = "cpu"
            
    def _load_model(self):
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True  # 必须启用以支持Qwen3等模型的特殊token
        )
        
        # 尝试从device字符串中提取GPU ID，如果是cuda:X格式
        gpu_id = 0  # 默认值
        if self.device.startswith('cuda:'):
            try:
                gpu_id = int(self.device.split(':')[1])
            except (IndexError, ValueError):
                pass
        
        # 加载模型，使用指定的GPU ID
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map={"": gpu_id} if torch.cuda.is_available() else "auto",  # 映射到指定的GPU
            trust_remote_code=True  # ✅ 必须启用
        )
        
        # 当模型通过accelerate加载时，不需要在pipeline中指定device参数
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
        
        self._is_warmed_up = True
        print("✅ Local LLM Generator ready.")
    
    def run(self, prompt: str):
        if not self._is_warmed_up:
            self._load_model()

        print(f"🎯 收到提示词: {str(prompt)[:100]}...")

        # ✅ 构造符合 Qwen3 模板的消息结构
        messages = [
            {"role": "system", "content": "你是一个角色设计合规助手，根据提供的规则回答问题。"},
            {"role": "user", "content": str(prompt)}  # 确保prompt是字符串
        ]

        # ✅ 直接使用手动构建的模板格式，避免tokenizer.apply_chat_template可能带来的问题
        print(f"🔄 直接使用手动构建的模板格式，跳过tokenizer.apply_chat_template")
        
        # 对于Qwen2模型，使用兼容的模板格式
        formatted_prompt = "<|im_start|>system\n" + messages[0]["content"] + "<|im_end|>\n<|im_start|>user\n" + messages[1]["content"] + "<|im_end|>\n<|im_start|>assistant\n"

        # ✅ 调试打印，使用更安全的方式
        print("📝 Prompt sent to model:")
        try:
            # 确保formatted_prompt是字符串格式
            print(str(formatted_prompt))
        except Exception as e:
            print(f"打印prompt时出错: {e}")
        print("-" * 50)

        outputs = self.pipe(
            formatted_prompt,
            max_new_tokens=1024,  # 增加token限制到1024
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            return_full_text=False  # ✅ 不返回输入部分
        )

        generated_text = outputs[0]["generated_text"].strip()
        return {"replies": [generated_text]}