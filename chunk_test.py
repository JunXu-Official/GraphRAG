import os
import sys

sys.path.append("..")
import logging
import ollama
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

WORKING_DIR = "./chunk_graphrag_cache_ollama_TEST"

MODEL = "llama3:8b"
# Assumed embedding model settings
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_MODEL_DIM = 768
EMBEDDING_MODEL_MAX_TOKENS = 8192
async def ollama_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # remove kwargs that are not supported by ollama
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)

    ollama_client = ollama.AsyncClient()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------
    response = await ollama_client.chat(model=MODEL, messages=messages, **kwargs)

    result = response["message"]["content"]
    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": MODEL}})
    # -----------------------------------------------------
    return result

@wrap_embedding_func_with_attrs(
    embedding_dim=EMBEDDING_MODEL_DIM,
    max_token_size=EMBEDDING_MODEL_MAX_TOKENS,
)
async def ollama_embedding(texts: list[str]) -> np.ndarray:
    embed_text = []
    for text in texts:
        data = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        embed_text.append(data["embedding"])

    return embed_text

def insert():
    from time import time
    with open(r"C:\Users\Lenovo\Desktop\nano-graphrag-main\examples\mock_data.txt", encoding="utf-8-sig") as f:
        texts = f.read()
    start = time()
    rag = GraphRAG(
    working_dir=WORKING_DIR,
    enable_llm_cache=True,
    best_model_func=ollama_model_if_cache,
    cheap_model_func=ollama_model_if_cache,
    embedding_func=ollama_embedding,
)
    rag.insert(texts)
    print('time', time() - start)


def query():
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=ollama_model_if_cache,
        cheap_model_func=ollama_model_if_cache,
        embedding_func=ollama_embedding,
    )
    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="global")
        )
    )

from nano_graphrag._utils import compute_mdhash_id

def MD5(string_or_strings):
    # 将传进来的文档装进列表统一处理
    if isinstance(string_or_strings, str):
        string_or_strings = [string_or_strings]
    # 对文档内容进行MD5哈希计算得到唯一ID,并给哈希值前面加上前缀doc-
    # 同一段文字的哈希值永远相同，也是去重的一部分
    new_docs = {
        compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
        for c in string_or_strings
    }

if __name__ == '__main__':
    # insert()
    # query()
    for i in range(0, 10, 2):
        print(i)
    