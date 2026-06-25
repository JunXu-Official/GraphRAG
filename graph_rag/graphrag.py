from dataclasses import dataclass, field, asdict
from datetime import datetime
from graph_rag.utils import TokenizerWrapper, extract_entities, always_get_an_event_loop
from graph_rag.llm import openai_embedding, gpt_4o_complete, gpt_4o_mini_complete
from graph_rag.base import JsonKVStorage, VectorDBStorage, NetworkXStorage, StorageNameSpace
from graph_rag.utils import convert_response_to_json
from graph_rag.base import EmbeddingFunc, logger
import os
from graph_rag.utils import limit_async_func_call, compute_mdhash_id
from functools import partial
from typing import cast
import asyncio
from graph_rag.base import BaseGraphStorage, BaseKVStorage, CommunitySchema, SingleCommunitySchema, NanoVectorDBStorage
from graph_rag.prompt import PROMPTS
from graph_rag.utils import truncate_list_by_token_size
from graph_rag.utils import list_of_list_to_csv, _community_report_json_to_str
from graph_rag.base import QueryParam
from graph_rag.utils import global_query









def chunk_by_token_size(tokens_list, doc_keys, tokenizer_wrapper: TokenizerWrapper, 
                        overlap_token_size=128, max_token_size=1024):
    """
    分块的核心函数
        tokens_list: text进行encode后的东西
        doc_keys: 对应token的hash-id
    """
    
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        # 切分
        for start in range(0, len(tokens), max_token_size - overlap_token_size):
            chunk_token.append(tokens[start: start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))
        # 对切块的token解码
        chunk_texts = tokenizer_wrapper.decode_batch(chunk_token)
        # 保存
        for i, chunk in enumerate(chunk_texts):
            results.append({
                "tokens": lengths[i],   # 对应text进行encode编码后的token长度
                "content": chunk.strip(),   # 对应token进行decode解码后的实际text内容
                "chunk_order_index": i,     # 块编号
                "full_doc_id": doc_keys[index]  # 对应文档的hash-id即doc-xxx
            })
    
    return results


def get_chunks(new_docs, chunk_func=chunk_by_token_size, tokenizer_wrapper: TokenizerWrapper=None, **chunk_func_params):
    """
    """
    inserting_chunks = {}
    # new_docs_list = [('doc-id', {'content': 'xxxxxxxxxxxxxxx'})]
    new_docs_list = list(new_docs.items())
    # 取text内容
    docs = [new_doc[1]["content"] for new_doc in new_docs_list]
    # 取hash-id
    doc_keys = [new_doc[0] for new_doc in new_docs_list]
    # 对text内容encode编码
    tokens = [tokenizer_wrapper.encode(doc) for doc in docs]
    # 分块
    chunks = chunk_func(
        tokens, doc_keys, tokenizer_wrapper
    )
    # chunks = [{'tokens': 147, 'content': 'xxxx', 'chunk_order_index': 0, 'full-doc-id': 'doc-xxx'}]
    for chunk in chunks:
        inserting_chunks.update({
            compute_mdhash_id(chunk["content"], prefix="chunk-"):chunk
        })
    # inserting_chunks: {'chunk-xxx': {'tokens': 147, 'content': 'xxxx', 'chunk_order_index': 0, 'full-doc-id': 'doc-xxx'}}
    return inserting_chunks

async def generate_community_report(
        community_report_kv: BaseKVStorage[CommunitySchema],    # 存放最终社区报告的键值对
        knwoledge_graph_inst: BaseGraphStorage, #  leiden算法之后图
        tokenizer_wrapper: TokenizerWrapper,    
        global_config: dict,    # 全局配置
):
    """
    生成社区报告的核心函数
    """
    # llm的参数
    llm_extra_kwargs = global_config["special_community_report_llm_kwargs"]
    # llm模型
    use_llm_func: callable = global_config["best_model_func"]
    # string转json函数
    use_string_json_convert_func: callable = global_config["convert_response_to_json_func"]
    # 
    communitites_schema = await knwoledge_graph_inst.community_schema() 
    #
    communtity_keys, community_values = list(communitites_schema.keys()), list(communitites_schema.values())
    already_processed = 0
    # 获取报告的prompt模板
    prompt_template = PROMPTS["community_report"]
    # 获取报告头长度
    prompt_overhead = len(tokenizer_wrapper.encode(prompt_template.format(input_text="")))
    async def _form_single_community_report(
            community: SingleCommunitySchema,
            already_reports: dict[str, CommunitySchema]
    ):
        """
        生成单个社区报告的核心函数
        """
        nonlocal already_processed
        describe = await _pack_single_community_describe(
            knwoledge_graph_inst,
            community,
            tokenizer_wrapper=tokenizer_wrapper,
            max_token_size=global_config["best_model_max_token_size"] - prompt_overhead - 200,
            already_reports=already_reports,
            global_config=global_config
        )
        prompt = prompt_template.format(input_text=describe)
        response = await use_llm_func(prompt, **llm_extra_kwargs)
        data = use_string_json_convert_func(response)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]
        print(f"{now_ticks} Processed {already_processed} communities\r", end="", flush=True)
        return data

    levels = sorted(set([c["level"] for c in community_values]), reverse=True)
    logger.info(f"Generating by levels: {levels}")
    community_datas = {}
    for level in levels:
        this_level_community_keys, this_level_community_values = zip(
            *[
                (k, v)
                for k, v in zip(communtity_keys, community_values)
                if v['level'] == level
            ]
        )
        this_level_community_reports = await asyncio.gather(
            *[
                _form_single_community_report(c, community_datas)
                for c in this_level_community_values
            ]
        )
        community_datas.update(
            {
                k: {
                    "report_string": _community_report_json_to_str(r),
                    "report_json": r,
                    **v,
                }
                for k, r, v in zip(
                    this_level_community_keys,
                    this_level_community_reports, 
                    this_level_community_values
                )
            }
        )
    print()
    await community_report_kv.upsert(community_datas)




async def _pack_single_community_describe(
        knowledge_graph_inst: BaseGraphStorage,
        community: SingleCommunitySchema,
        tokenizer_wrapper: "TokenizerWrapper",
        max_token_size: int=12000,
        already_reports: dict[str, CommunitySchema] = {},
        global_config: dict = {}
):
    """
    将社区的所有信息（报告，实体，关系）等打包成结构化的文本描述，用于LLM调用生成社区报告
    """
    # 排序节点和边
    nodes_in_order = sorted(community["nodes"])
    edges_in_order = sorted(community["edges"], key=lambda x: x[0] + x[1])
    # 获取节点数据
    nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.node(node) for node  in nodes_in_order]
    )
    # 获取边数据
    edges_data = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(src, tgt) for src, tgt in edges_in_order]
    )
    # 定义最终生成的输出模板
    final_template = """---Reports---
    
    ```csv
{reports}
```
-----Entities-----
```csv
{entities}
```
-----Relationships-----
```csv
{relationships}
```"""
    # 模板占用的token
    base_template_tokens = len(tokenizer_wrapper.encode(final_template.format(reports="", entities="", relationships="")))
    # 剩余可用的token
    remaining_budget = max_token_size - base_template_tokens
    report_describe = ""
    contain_nodes = set()
    contain_edges = set()
    # 判断是否使用社区报告，目的是对于大规模社区，用子社区报告替代原始数据，节省token
    # 判断条件：社区规模过大 && 已经有子社区 && 已经有子社区的报告
    truncated = len(nodes_in_order) > 100 or len(edges_in_order) > 100
    need_to_use_sub_communities = (
        truncated and
        community["sub_communities"] and
        already_reports
    )
    force_to_use_sub_communities = global_config["addon_params"].get("force_to_use_sub_communities", False)
    # 如果满足条件，获取子社区报告
    if need_to_use_sub_communities or force_to_use_sub_communities:
        logger.debug(f'Community {community["title"]} using sub-communities')
        result = _pack_single_community_by_sub_communitites(
            community,
            remaining_budget,
            already_reports,
            tokenizer_wrapper
        )
        report_describe, report_size, contain_nods, contain_edges = result
        remaining_budget = max(0, remaining_budget - report_size)


    def format_row(row):
        """
        """
        return ','.join('"{}"'.format(str(item).replace('"', '""')) for item in row)
    
    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "source", "target", "description", "rank"]
    # 获取节点度数
    node_degrees = await knowledge_graph_inst.node_degrees_batch(nodes_in_order)
    # 获取边度数
    edge_degrees = await knowledge_graph_inst.edge_degrees_batch(edges_in_order)
    # 构建实体列表，过滤掉已经在子社区报告占用的节点
    ndoes_list_data = [
        [i, name, data.get("entity_type", "UNKNOWN"),
         data.get("description", "UNKNOWN"), node_degrees[i]]
         for i, (name, data) in enumerate(zip(nodes_in_order, nodes_data))
         if name not in contain_nodes
    ]
    # 构建关系列表
    edges_list_data = [
        [i, edge[0], edge[1], data.get("description", "UNKNOWN"), edge_degrees[i]]
        for i, (edge, data) in enumerate(zip(edges_in_order, edges_data))
        if (edge[0], edge[1]) not in contain_edges
    ]
    # 依据度数或权重进行重要性排序
    ndoes_list_data.sort(key=lambda x: x[-1], reverse=True)
    edges_list_data.sort(key=lambda x: x[-1], reverse=True)
    header_tokens = len(tokenizer_wrapper.encode(list_of_list_to_csv([node_fields]) + "\n" + list_of_list_to_csv([edge_fields])))
    # 动态分配token。逻辑就是实体和关系谁多给随分配更多的token
    data_budget = max(0, remaining_budget - header_tokens)
    total_items = len(ndoes_list_data) + len(edges_list_data)
    node_ratio = len(ndoes_list_data) / max(1, total_items)
    edge_ratio = 1 - node_ratio
    nodes_final = truncate_list_by_token_size(
        ndoes_list_data, key=format_row,
        max_token_size=int(data_budget * node_ratio),
        tokenizer_wrapper=tokenizer_wrapper
    )
    edges_final = truncate_list_by_token_size(
    edges_list_data, key=format_row,
    max_token_size= int(data_budget * edge_ratio),
    tokenizer_wrapper=tokenizer_wrapper
    )

    nodes_describe = list_of_list_to_csv([node_fields] + nodes_final)
    edges_describe = list_of_list_to_csv([edge_fields] + edges_final)
    final_output = final_template.format(
        reports=report_describe,
        entities=nodes_describe,
        relationships=edges_describe
    )

    return final_output



def _pack_single_community_by_sub_communitites(
        community: SingleCommunitySchema,
        max_token_size: int, 
        already_reports: dict[str, CommunitySchema],
        tokenizer_wrapper: TokenizerWrapper
):
    """
    获取子社区报告的狠心函数
    """
    all_sub_communities = [already_reports[k] for k in community['sub_communities'] if k in already_reports]
    all_sub_communities = sorted(all_sub_communities, key=lambda x: x['occurrence'], reverse=True)
    may_trun_all_sub_communities = truncate_list_by_token_size(
        all_sub_communities,
        key=lambda x: x["report_string"],
        max_token_size=max_token_size,
        tokenizer_wrapper=tokenizer_wrapper
    )
    sub_fields = ["id", "report", "rating", "importance"]
    sub_communities_describe = list_of_list_to_csv(
        [sub_fields]
        + [
            [
                i,
                c["report_string"],
                c["report_json"].get("rating", -1),
                c["occurrence"],
            ]
            for i, c in enumerate(may_trun_all_sub_communities)
        ]
    )
    already_nodes = []
    already_edges = []
    for c in may_trun_all_sub_communities:
        already_nodes.extend(c["nods"])
        already_edges.extend(tuple(e) for e in c["edges"])
    
    return (
        sub_communities_describe,
        len(tokenizer_wrapper.encode(sub_communities_describe)),
        set(already_nodes),
        set(already_edges)
    )




@dataclass
class GraphRAG:
    # 设置目录，用来存放切片数据、图数据库等
    work_path: str = field(
        default_factory = lambda: f"./graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # 是否开启相应缓存
    enable_llm_cache: bool = True
    # 设置rag模式
    enable_local = True
    enable_native_rag = False
    # 设置token类型
    tokenizer_type = 'tiktoken'
    tiktoken_model_name = 'gpt-4o'
    huggingface_model_name = "bert-base-uncased"
    # 切分函数
    chunk_func: callable = chunk_by_token_size
    chunk_token_size = 1200
    chunk_overlap_token_size = 100
    # llm提取时候回头看的次数
    entity_extract_max_gleaning: int = 1
    # 合并和生成单个实体摘要的token上限
    entity_summary_to_max_tokens: int = 500
    graph_cluster_algorithm = 'leiden'
    # 每个底层社区最多包含10个实体
    max_graph_cluster_size: int = 10
    graph_cluster_seed = 0xDEABEEF
    # node2vec将图的拓扑结构翻译成“向量”
    node_embedding_algorithm = 'node2vec'
    node2vec_params: dict = field(
        default_factory = lambda: {
            'dimensions': 1536, # 将一个实体节点在图谱中的位置压缩成1536个浮点数组成的数组
            'num_walks': 10, # 从某个实体出发，进行10次随机探索
            'walk_length': 40,  # 每次探索的最大步长
            'window_size': 2,   # 只分析前后紧挨着的2个邻居
            'iteration': 3, # 迭代次数
            'random_seed': 3,
        })
    # 强制输出为json格式
    special_community_report_llm_kwargs : dict = field(
        default_factory= lambda : {"response_format": {"type": "json+_obeject"}}
    )
    # openai的脚本进行批量向量化
    # field: 告诉dataclass，embedding_func不是类的方法，而是数据属性
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16   # 设置16个异步进程同时开工
    query_better_than_threshold: int = 0.2   # 相似度阈值
    # 模型配置
    best_model_id = ""
    cheap_model_id = ""
    best_model_func: callable = gpt_4o_complete
    best_model_max_token_size: int = 32768
    best_model_max_aysnc: int = 16
    cheap_model_func: callable = gpt_4o_mini_complete
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 16
    # 构造知识图谱的核心函数
    entity_extraction_func: callable = extract_entities
    # 存放原始文档
    key_string_value_json_storage_cls = JsonKVStorage
    # 存放向量数据库
    vector_db_storage_cls = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    # 存放图结构
    graph_storage_cls = NetworkXStorage
    # 创建工作目录
    always_create_work_path= True
    addon_params: dict = field(default_factory=dict)
    # LLM输出转json
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self):
        _print_config = ",\n".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GraphRAG init with param: \n\n {_print_config}")
        # 分词器初始化
        self.tokenizer_wrapper = TokenizerWrapper(
            tokenizer_type=self.tokenizer_type,
            model_name=self.tiktoken_model_name if self.tokenizer_type == 'tiktoken' else self.huggingface_model_name
        )
        if not os.path.exists(self.work_path) and self.always_create_work_path:
            logger.info(f"Creating work path {self.work_path}")
            os.makedirs(self.work_path)
        # 保存原始文档
        self.full_docs = self.key_string_value_json_storage_cls(namespace="full_docs", global_config=asdict(self))
        # 保存文本块
        self.text_chunks = self.key_string_value_json_storage_cls(namespace='text_chunks', global_config=asdict(self))
        # 保存LLM回答
        self.llm_response_cache = (self.key_string_value_json_storage_cls(namespace='llm_response_cache', global_config=asdict(self)) if self.enable_llm_cache else None)
        # 保存社区报告
        self.community_reports = self.key_string_value_json_storage_cls(namespace='community_reports', global_config=asdict(self))
        # 保存知识图谱
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace='chunk_entity_relation', global_config=asdict(self)
        )
        # 限制“流水线”上限
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(self.embedding_func)
        # 
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"}
            )
            if self.enable_local else None
        )
        # 
        self.chunks_vdb = (self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        ) if self.enable_native_rag else None)
        #
        self.best_model_func = limit_async_func_call(self.best_model_max_aysnc)(
            partial(self.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            partial(self.cheap_model_func, hashing_kv=self.llm_response_cache)
        )
    
    def insert(self, string_or_strings):
        # 调用ainsert方法
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    async def ainsert(self, string_or_strings):
        """
        构造graph的核心函数入口
        """
        await self._insert_start()
        try:
            # 统一处理成list
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]
                # 生成文档唯一id
                # new_doc = {'doc-id': {'content': 'xxxxxxxx'}}
                new_docs = {
                    compute_mdhash_id(c.strip(), prefix='doc-'): {"content": c.strip()} for c in string_or_strings
                }
                # 去重：根据doc-id查询是否已存在， 返回去重后{'doc-id'}
                _add_docs_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
                new_docs = {k: v for k, v in new_docs.items() if k in _add_docs_keys}
                if not len(new_docs):
                    logger.warning(f"该文档在数据库中已存在！！！")
                    return
                logger.info(f"[New docs] inserting {len(new_docs)} docs")
                # 对不在数据库中的新文档切块
                # inserting_chunks: {'chunk-xxx': {'tokens': 147, 'content': 'xxxx', 'chunk_order_index': 0, 'full-doc-id': 'doc-xxx'}}
                inserting_chunks = get_chunks(new_docs=new_docs, chunk_func=self.chunk_func, 
                                              overlap_token_size=self.chunk_overlap_token_size,
                                              max_token_size=self.chunk_token_size,
                                              tokenizer_wrapper=self.tokenizer_wrapper)
                # 再对inserting_chunks去重
                _add_chunk_keys = await self.text_chunks.filter_keys(list(inserting_chunks.keys()))
                inserting_chunks = {k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys}
                if not len(inserting_chunks):
                    logger.warning(f"所有划分的块均已在数据库中存在！！！")
                    return 
                if self.enable_native_rag:
                    logger.info("Insert chnks for native rag")
                    await self.chunks_vdb.upsert(inserting_chunks)
                # 把旧的报告全部丢了--暂不支持增量更新
                await self.community_reports.drop()
                logger.info(f"【开始进行实体关系提取...】")
                # 得到新的实体关系
                maybe_new_kg = await self.entity_extraction_func(
                    inserting_chunks,
                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                    entity_vdb=self.entities_vdb,
                    tokenizer_wrapper=self.tokenizer_wrapper,
                    global_config=asdict(self),
                )
                if maybe_new_kg is None:
                    logger.warning(f"【未发现新的实体关系！】")
                    return
                self.chunk_entity_relation_graph = maybe_new_kg
                logger.info(f"【leiden算法聚类，生成社区报告...】")
                await self.chunk_entity_relation_graph.clustering(
                    self.graph_cluster_algorithm
                )
                # 生成社区报告
                await generate_community_report(
                    self.community_reports, self.chunk_entity_relation_graph,
                    self.tokenizer_wrapper, asdict(self)
                )
                await self.full_docs.upsert(new_docs)
                await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()

                
    
    
    async def _insert_start(self):
        """
        用同步接口作为入口，用asyncio做内部并发pipeline，通过gather并行初始化多个storage
        """
        tasks = []
        for storage_inst in [self.chunk_entity_relation_graph]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        
        await asyncio.gather(*tasks)

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.community_reports,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)


    def query(self, query, param: QueryParam = QueryParam()):
        """
        查询的入口函数
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))
    
    async def aquery(self, query, param: QueryParam = QueryParam()):
        """
        查询的核心函数
        """
        if param.mode == "local" and not self.enable_local:
            raise ValueError("无法使用简单查询！")
        if param.mode == "native" and not self.enable_native_rag:
            raise ValueError("无法使用传统RAG查询！")
        if param.mode == "global":
            response = await global_query(
                query,  # 要查询的问题
                self.chunk_entity_relation_graph,   # 图实例
                self.entities_vdb,  # 关系表
                self.community_reports, # 社区报告
                self.text_chunks,  # 文本块
                param,  # 参数
                self.tokenizer_wrapper,
                asdict(self)
            )
        else:
            raise ValueError("Can't support other mode now!")
        await self._query_done(self)
        return response


    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)