import os, sys
sys.path.append("..")
import logging, ollama
import numpy as np
from graph_rag.graphrag import GraphRAG
from graph_rag.parameters import QueryParam
from graph_rag.base import BaseKVStorage
from graph_rag.utils import compute_args_hash, wrap_embedding_func_with_attrs
from time import time

logging.basicConfig(level=logging.warning)
logging.getLogger('graph_rag').setLevel(logging.INFO)
# LLM模型设置
MODEL = "llama3:8b"
# embedding模型设置
EMBEDDING_MODEL = "nomic-embed-text:latest" # 目的是将“纯文字”变成“向量”
EMBEDDING_MODEL_DIM = 768   # 输出的向量维度
EMBEDDING_MODEL_MAX_TOKENS = 8192   # 单次输出的最大分词量

async def ollama_model_if_cache(prompt, system_prompt=None, history_messages=[], **kwargs):
    """
    在请求ollama之前先查看缓存，如果发现相同的问题且LLM已经回答过了，就秒回历史答案；如果没有，再去调用LLM
    :param prompt: 提问关键词
    :param system_prompt: 系统提示词
    :param history_messages: 历史聊天记录
    """
    # 移除ollama不支持的参数
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)
    # 初始化ollama客户端：实例化一个本地的ollama异步客户端
    ollama_client = ollama.AsyncClient()
    messages = []
    # 如果有LLM有人设，那么先将人设塞进对话历史
    if system_prompt:
        messages.append({'role': "system", "content": system_prompt})
    # 获取缓存
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    # 将之前的聊天历史和用户现在提问的问题全部追加到对话中
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    # 如果缓存存在，则对prompt进行“指纹比对”
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        # 将“指纹”拿到缓存中查
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        # 如果缓存命中，直接返回历史答案
        if if_cache_return is not None:
            return if_cache_return["return"]
    # 缓存没命中，直接调用本地LLM推理
    response = await ollama_client.chat(model=MODEL, messages=messages, **kwargs)
    result = response["message"]["content"]
    # 更新缓存
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": MODEL}})
    return result

def remove_if_exist(file):
    """
    删除文件
    """
    if os.path.exists(file):
        os.remove(file)

# 定义装饰器标签
@wrap_embedding_func_with_attrs(
    embedding_dim=EMBEDDING_MODEL_DIM,
    max_token_size=EMBEDDING_MODEL_MAX_TOKENS,
)

async def ollama_embedding(texts):
    """
    将文本向量化
    """
    embed_text = []
    for text in texts:
        data = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        embed_text.append(data["embedding"])
    return embed_text

def insert(work_path, data_path):
    """
    构造知识图谱
    """
    with open(data_path, encoding='utf-8-sig') as f:
        FAKE_TEXT = f.read()
    remove_if_exist(f"{work_path}/vdb_entities.json")
    remove_if_exist(f"{work_path}/kv_store_full_docs.json")
    remove_if_exist(f"{work_path}/kv_store_text_chunks.json")
    remove_if_exist(f"{work_path}/kv_store_community_reports.json")
    remove_if_exist(f"{work_path}/graph_chunk_entity_relation.graphml")
    # 实例化知识图谱
    rag = GraphRAG(
        work_path=work_path,
        enable_llm_cache=True,
        best_model_func=ollama_model_if_cache,
        cheap_model_func=ollama_model_if_cache,
        embedding_func=ollama_embedding,
    )
    start_time = time()
    # 根据读取到的文档构建知识图谱
    rag.insert(FAKE_TEXT)
    print('insert time is: ', time() - start_time)

def query(work_path):
    """
    查询过程
    """
    rag = GraphRAG(
        work_path=work_path,
        best_model_func=ollama_model_if_cache,
        cheap_model_func=ollama_model_if_cache,
        embedding_func=ollama_embedding
    )
    print(
        rag.query(
            "What are thr top themes in this story?", param=QueryParam(model="global")
        )
    )


def main():
    work_path = "./graphrag_ollama_test"
    data_path = r"test.txt"
    insert(work_path=work_path, data_path=data_path)
    query(work_path=work_path)

if __name__ == '__main__':
    main()