import re
import jieba
from typing import List, Dict, Any

class DocumentSplitter:
    def __init__(self, split_by: str = "word", split_length: int = 300, 
                 split_overlap: int = 50, language: str = "zh"):
        self.split_by = split_by
        self.split_length = split_length
        self.split_overlap = split_overlap
        self.language = language
        
        # 确保overlap小于split_length
        if split_overlap >= split_length:
            self.split_overlap = max(0, split_length - 1)
            print(f"警告: split_overlap不能大于等于split_length，已调整为{self.split_overlap}")
    
    def warm_up(self):
        # 预热方法，确保jieba分词器已加载
        if self.language == "zh" and self.split_by == "word":
            try:
                # 测试分词
                jieba.lcut("测试分词")
                print("DocumentSplitter预热完成")
            except Exception as e:
                print(f"DocumentSplitter预热时出错: {e}")
    
    def split_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """将文本按照指定的方式分块"""
        if not text:
            return []
        
        # 使用指定的chunk_size和chunk_overlap，如果没有提供则使用类初始化时的值
        current_chunk_size = chunk_size or self.split_length
        current_chunk_overlap = chunk_overlap or self.split_overlap
        
        if self.split_by == "word":
            return self._split_by_word(text, current_chunk_size, current_chunk_overlap)
        elif self.split_by == "char":
            return self._split_by_char(text, current_chunk_size, current_chunk_overlap)
        else:
            # 默认按word分割
            return self._split_by_word(text, current_chunk_size, current_chunk_overlap)
    
    def _split_by_char(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """按字符分块"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            # 如果已经是最后一个块，则退出循环
            if end >= len(text):
                break
            # 否则，移动start位置
            start = end - chunk_overlap
        
        return chunks
    
    def _split_by_word(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """按词分块"""
        if self.language == "zh":
            # 中文文本使用jieba分词
            words = jieba.lcut(text)
        else:
            # 英文等其他语言按空格分词
            words = text.split()
            
        # 如果没有分词结果或chunk_size小于等于0，返回原文本
        if not words or chunk_size <= 0:
            return [text]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        # 构建分块
        for word in words:
            word_length = len(word)
            
            # 如果当前块加上新词的长度超过chunk_size，则完成当前块
            if current_length + word_length > chunk_size:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                    # 处理重叠
                    overlap_words = []
                    overlap_length = 0
                    # 从当前块的末尾开始，收集不超过chunk_overlap长度的词
                    for i in range(len(current_chunk)-1, -1, -1):
                        overlap_length += len(current_chunk[i])
                        if overlap_length > chunk_overlap:
                            break
                        overlap_words.insert(0, current_chunk[i])
                    # 开始新的块，包含重叠部分
                    current_chunk = overlap_words.copy()
                    current_length = sum(len(w) for w in current_chunk)
            
            # 添加新词到当前块
            current_chunk.append(word)
            current_length += word_length
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        return chunks
    
    def run(self, documents: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """处理文档列表，返回分块后的文档"""
        split_docs = []
        
        for doc in documents:
            text = doc.get('content', '') or doc.get('text', '')
            if not text:
                continue
            
            # 获取文档的元数据
            meta = doc.get('meta', {})
            
            # 对文本进行分块
            chunks = self.split_text(text)
            
            # 为每个分块创建新的文档对象
            for i, chunk in enumerate(chunks):
                chunk_meta = meta.copy()
                chunk_meta['chunk_index'] = i
                chunk_meta['total_chunks'] = len(chunks)
                chunk_meta['original_length'] = len(text)
                
                split_docs.append({
                    'content': chunk,  # 兼容Haystack的Document对象
                    'text': chunk,      # 兼容我们自己的文档格式
                    'meta': chunk_meta
                })
        
        return {'documents': split_docs}

# 确保导入jieba库
try:
    import jieba
    print("成功导入jieba分词库")
except ImportError:
    print("警告: 未找到jieba分词库，中文分词功能可能受限")
    # 定义一个简单的替代函数
    def jieba_lcut(text):
        return list(text)
    
    class MockJieba:
        @staticmethod
        def lcut(text):
            return jieba_lcut(text)
    
    jieba = MockJieba()