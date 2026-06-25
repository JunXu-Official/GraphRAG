from openai import AsyncOpenAI
import numpy as np
from graph_rag.base import BaseKVStorage
from graph_rag.utils import compute_args_hash

global_openai_async_client = None



def get_openai_async_client_instance():
    global global_openai_async_client
    if global_openai_async_client is None:
        global_openai_async_client = AsyncOpenAI()
    return global_openai_async_client


async def openai_complete_if_cache(model, prompt, system_prompt=None, history_messages=[], **kwargs):
    """
        进行缓存命中匹配
    """
    openai_async_client = get_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return['return']
    
    response = await openai_async_client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )
    if hashing_kv is not None:
        await hashing_kv.upsert({
            args_hash: {"return": response.choice[0].message.content,"model": model}
        })
        await hashing_kv.index_done_callback()
    
    return response.choice[0].message.content
        

async def openai_embedding(texts):
    openai_async_client = get_openai_async_client_instance()
    response = await openai_async_client.embedding.create(
        model='text-embedding-3-small',
        input=texts,
        encoding_format='float'
    )
    return np.array([dp.embedding for dp in response.data])


async def gpt_4o_complete(prompt, system_prompt=None, history_messages=[], **kwargs):

    return await openai_complete_if_cache(
        "gpt-4o",
        prompt, 
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )


async def gpt_4o_mini_complete(prompt, system_prompt=None, history_messages=[], **kwargs):

    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt, 
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )