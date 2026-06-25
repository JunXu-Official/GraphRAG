from dataclasses import dataclass
from hashlib import md5
from typing import Literal
from collections import defaultdict,Counter
import logging
import tiktoken
from transformers import AutoTokenizer
from graph_rag.base import BaseGraphStorage, BaseVectorStorage, EmbeddingFunc
from graph_rag.prompt import PROMPTS, GRAPH_FIELD_SEP
import re, html, asyncio
import json
from functools import wraps
import numbers
from graph_rag.base import CommunitySchema, TextChunkSchema, QueryParam, BaseKVStorage


logger = logging.getLogger('graphrag')
logging.getLogger("neo4j").setLevel(logging.ERROR)


def _community_report_json_to_str(parsed_output):
    """
    """
    title = parsed_output.get("title", "Report")
    summary = parsed_output.get("summary", "")
    findings = parsed_output.get("findings", [])
    def finding_summary(finding):
        if isinstance(finding, str):
            return finding
        return finding.get("summary")
    
    def finding_explanation(finding):
        if isinstance(finding, str):
            return ""
        return finding.get("explanation")

    report_sections = "\n\n".join(
        f'## {finding_summary(f)}\n\n{finding_explanation(f)}' for f in findings
    )
    return f"# {title}\n\n{summary}\n\n{report_sections}"


def list_of_list_to_csv(data):
    return "\n".join(
        [
            ",\t".join([f"{enclose_string_with_quotes(data_dd)}" for data_dd in data_d]) for data_d in data
        ]
    )

def enclose_string_with_quotes(content):
    """
    """
    if isinstance(content, numbers.Number):
        return str(content)
    content = str(content)
    content = content.strip().strip("'").strip('"')
    return f'"{content}"'




def truncate_list_by_token_size(
        list_data, key, max_token_size, tokenizer_wrapper
):
    """
    """
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(tokenizer_wrapper.encode(key(data))) + 1
        if tokens > max_token_size:
            return list_data[:i]
    
    return list_data


def compute_mdhash_id(content, prefix: str=''):
    return prefix +md5(content.encode()).hexdigest()



def always_get_an_event_loop():
    """
    确保当前线程能拿到一个可用的事件循环
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        logger.info("Creating a new event loop in a sub_thread")
        loop = asyncio.new_event_loop() # 创建
        asyncio.set_event_loop(loop)    # 绑定
    return loop


def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))

def compute_args_hash(*args):
    """
    通过MD5算法，压缩成唯一的固定32位的数字指纹字符串
    :param *args: 不定长位置参数
    """
    return md5(str(args).encode()).hexdigest()


def wrap_embedding_func_with_attrs(**kwargs):
    """装饰函数"""
    def final_decro(func):
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func
    return final_decro


def limit_async_func_call(max_size, waitting_time=0.0001):
    """
    """
    def final_decro(func):
        
        __current_size = 0
        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func
    
    return final_decro



def extract_first_complete_json(s):
    stack = []
    first_json_start = None
    for i, char in enumerate(s):
        if char == '{':
            stack.append(i)
            if first_json_start is None:
                first_json_start = i
        elif char == '}':
            if stack:
                start = stack.pop()
                if not stack:
                    first_json_str = s[first_json_start: i+1]
                    try:
                        return json.loads(first_json_str.replace("\n", ""))
                    except json.JSONDecodeError as e:
                        logger.info(f"JSON decoding failed")
                        return None
                    finally:
                        first_json_str = None
    logger.info(f"No complete JSON object found in the input string")
    return None

def parse_value(value: str):
    """Convert a string value to its appropriate type (int, float, bool, None, or keep as string). Work as a more broad 'eval()'"""
    value = value.strip()

    if value == "null":
        return None
    elif value == "true":
        return True
    elif value == "false":
        return False
    else:
        # Try to convert to int or float
        try:
            if '.' in value:  # If there's a dot, it might be a float
                return float(value)
            else:
                return int(value)
        except ValueError:
            # If conversion fails, return the value as-is (likely a string)
            return value.strip('"')  # Remove surrounding quotes if they exist

def extract_values_from_json(json_string, keys=['reasoning', 'answer', 'data'], allow_no_quotes=False):
    extracted_values = {}
    # Enhanced pattern to match both quoted and unquoted values, as well as nested objects
    regex_pattern = r'(?P<key>"?\w+"?)\s*:\s*(?P<value>{[^}]*}|".*?"|[^,}]+)'
    
    for match in re.finditer(regex_pattern, json_string, re.DOTALL):
        key = match.group('key').strip('"')  # Strip quotes from key
        value = match.group('value').strip()
        # If the value is another nested JSON (starts with '{' and ends with '}'), recursively parse it
        if value.startswith('{') and value.endswith('}'):
            extracted_values[key] = extract_values_from_json(value)
        else:
            # Parse the value into the appropriate type (int, float, bool, etc.)
            extracted_values[key] = parse_value(value)
    if not extracted_values:
        logger.warning("No values could be extracted from the string.")
    
    return extracted_values




def convert_response_to_json(response):
    prediction_json = extract_first_complete_json(response)
    if prediction_json is None:
        logger.info(f"Attempting to extract values from a non-standard JSON string...")
        prediction_json = extract_values_from_json(response, allow_no_quotes=True)
    
    if not prediction_json:
        logger.error("Unable to extract meaningful data from the response")
    else:
        logger.info("JSON data successfully extracted")
    
    return prediction_json




def pack_user_ass_to_openai_messages(prompt, generated_content):
    """
    """
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": generated_content}
    ]


def split_string_by_multi_markers(content, markers):
    """
    切割句子
    :param content: 输入文本
    :param markers: 切割标志
    """
    if not markers:
        return [content]
    # escape给符号加上\语义。即原来的“，”变成“\，”
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]

def clean_str(input):
    """
    对于输入字符串去除HTML标签，特殊字符等任何不想要的字符
    """
    if not isinstance(input, str):
        return input
    result = html.unescape(input.strip())
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


async def _handle_single_entity_extraction(record_attributes, chunk_key):
    """
    判断AI返回的一行属性到底是不是一个合格的实体
    """
    # 一个标准的实体记录至少包括：标志位、名称、类型、描述，如果少传直接丢弃。
    # 标志位就是"entity"
    if len(record_attributes) < 4 or record_attributes[0] != '"entity':
        return None
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id
    )

async def _handle_single_relationship_extraction(record_attributes, chunk_key):
    """
    识别实体之间的逻辑关联
    """
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # 关系至少需要5个字段：标志位relationship, 源实体source， 目标实体target, 关系描述 description, 权重weight/importance
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    # 标记来源
    edge_source_id = chunk_key
    # 判断最后一个字段是否为有效数字
    weight = float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0

    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id
    )


async def _handle_entity_relation_summary(entity_or_relation_name, description, global_config, tokenizer_wrapper):
    """
    """
    use_llm_func = global_config['cheap_model_func']
    llm_max_tokens = global_config['cheap_model_max_token_size']
    summary_max_tokens = global_config['entity_summary_to_max_tokens']
    tokens = tokenizer_wrapper.encode(description)
    if len(tokens) < summary_max_tokens:
        return description
    prompt_template = PROMPTS['summarize_entity_description']
    use_description = tokenizer_wrapper.decode(tokens[:llm_max_tokens])
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP)
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Input Summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _merge_nodes_then_upsert(entity_name, nodes_data, knowledge_graph_inst: BaseGraphStorage, global_config, tokenizer_wrapper):
    """
    向图中插入节点
    """
    already_entity_types = []
    already_source_ids = []
    already_description = []
    # 检查图中是否已经有该节点
    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node['entity_type'])
        already_source_ids.append(split_string_by_multi_markers(already_node['source_id'], [GRAPH_FIELD_SEP]))
        already_description.append(already_node['description'])
    entity_type = sorted(Counter([dp['entity_type'] for dp in nodes_data] + already_entity_types).items(),
                         key=lambda x: x[1],
                         reverse=True)[0][0]
    description = GRAPH_FIELD_SEP.join(sorted(set([dp['description'] for dp in nodes_data] + already_description)))
    source_id = GRAPH_FIELD_SEP.join(set([dp['source_id'] for dp in nodes_data] + already_source_ids))
    description = await _handle_entity_relation_summary(entity_name, description, global_config, tokenizer_wrapper)
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data
    )
    node_data['entity_name'] = entity_name

    return node_data


async def _merge_edges_then_upsert(src_id, tgt_id, edges_data, knowledge_graph_inst: BaseGraphStorage, global_config, tokenizer_wrapper):
    """
    """
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge['weight'])
        already_source_ids.extend(split_string_by_multi_markers(already_edge['source_id'], [GRAPH_FIELD_SEP]))
        already_description.append(already_edge['description'])
        already_order.append(already_edge.get('order', 1))

    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum([dp['weight'] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(sorted(set([dp['description'] for dp in edges_data] + already_description)))
    source_id = GRAPH_FIELD_SEP.join(set([dp['source_id'] for dp in edges_data] + already_source_ids))
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id, 
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": "UNKOWN"
                }
            )
    description = await _handle_entity_relation_summary((src_id, tgt_id), description, global_config, tokenizer_wrapper)

    await knowledge_graph_inst.upsert_edge(
    src_id,
    tgt_id,
    edge_data=dict(
        weight=weight, description=description, source_id=source_id, order=order
    ),
)


class TokenizerWrapper:
    """
    token编码-解码类
    """
    def __init__(self, tokenizer_type: Literal['tiktoken', 'huggingface']='tiktoken',
                 model_name = 'gpt-4o'):
        self.tokenizer_type = tokenizer_type
        self.model_name = model_name
        self._tokenizer = None
        self._lazy_load_tokenizer()

    def _lazy_load_tokenizer(self):
        if self._tokenizer is not None:
            return
        logger.info(f"Loading tokenizer: type='{self.tokenizer_type}, name='{self.model_name}")
        if self.tokenizer_type == 'tiktoken':
            self._tokenizer = tiktoken.encoding_for_model(self.model_name)
        elif self.tokenizer_type == 'huggingface':
            if AutoTokenizer is None:
                raise ImportError('transformers is not installed')
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        else:
            raise ValueError(f'Unknown tokenizer_type: {self.tokenizer_type}')
    
    def get_tokenizer(self):
        """
            提供对底层tokenizer对象的访问
        """
        self._lazy_load_tokenizer()
        return self._tokenizer
    
    def encode(self, text):
        """
        对text编码向量化
        """
        self._lazy_load_tokenizer()
        return self._tokenizer.encode(text)

    def decode(self, tokens):
        """
        token解码成text
        """
        self._lazy_load_tokenizer()
        return self._tokenizer.decode(tokens)

    def decode_batch(self, tokens_list):
        """
        解码：要注意一下，huggingface的tiktoken有decode_batch但是tiktoken没有，因此要列表推导
        """
        self._lazy_load_tokenizer()
        if self.tokenizer_type == 'tiktoken':
            return [self._tokenizer.decode(tokens) for tokens in tokens_list]
        elif self.tokenizer_type == 'huggingface':
            return self._tokenizer.batch_decode(tokens_list, slip_special_tokens=True)
        else:
            raise ValueError(f"Unknown tokenizer_type: {self.tokenizer_type}")

async def extract_entities(chunks, knowledge_graph_inst: BaseGraphStorage, entity_vdb: BaseVectorStorage,
                           tokenizer_wrapper, global_config: dict):
    """
    实体关系提取的核心函数：接收文本块，返回图空间
    :param chunks: 切分好的块
    :param knowledge_graph_inst: 图数据库
    :param entity_vdb: 存储实体描述的向量数据库
    :param tokenizer_wrapper: 分词器
    :param global_config: 全局配置
    """
    use_llm_func: callable = global_config['best_model_func']
    entity_extract_max_gleaning = global_config['entity_extract_max_gleaning']
    # 字典转为列表
    ordered_chunks = list(chunks.items())
    # 提示词
    entity_extract_prompt = PROMPTS['entity_extraction']
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"], #关系三元组的分隔符，如<|>
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],   # 每条记录的分隔符，如##
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],   # 输出结束的分隔符
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]), # 想要提取的类型
    )
    # 是否继续
    continue_prompt = PROMPTS['entiti_continue_extraction']
    # 是否需要循环提示词
    # 运行主提取-->if_loop_prompt(你觉得提取干净了吗)-->continue_prompt(没提取干净，继续提取)
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]
    # 已经处理的文本块数量
    already_processed = 0
    # 累积提取出的实体数量
    already_entities = 0
    # 累积提取出的关系/边的数量
    already_relations = 0
    async def _process_single_content(chunk_key_dp):
        """
        将单个chunk块喂给LLM，多次“榨取”知识
        :param chunk_key_dp: md5值+内容
        """
        # chunk_key_dp = ('chunk-xxx', {'tokens': xx, 'content': 'xxxx', 'chunk_order_index': xx, 'full_doc_id': 'doc-xxx'})
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0] # 获取chunk的md5值
        chunk_dp = chunk_key_dp[1]  # 获取chunk内容
        content = chunk_dp['content']   # chunk的文本块内容
        #将文本分隔符context_base和当前的文本内容input_text填入prompt
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        # 调用LLM，根据输入输出结果
        final_result = await use_llm_func(hint_prompt)
        if isinstance(final_result, list):
            final_result = final_result[0]["text"]
        # 将第一轮的prompt和LLM的回答打包成OpenAI标准的对话格式
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            # 代码会发送一个continue_prompt询问是否仍有漏掉的尚未提取
            # history_message进行上下文衔接，告诉LLM这是刚才提取到的内容，避免重复
            glean_result = await use_llm_func(continue_prompt, history_messages=history)
            # 将新发现的“漏网之鱼”glean_result 添加到 history中
            # 通过多次榨取提高知识图谱的密度
            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break
             # 给AI发送if_loop_prompt，AI回复yes或者no
            if_loop_result = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != 'yes':
                break
        ####-------------------------------------------------------------------------------
        """
        举个例子，LLM最终输出的final_result是
        ("entity"<|>"张三"<|>"person"<|> "zhang san is a person who hold the title of huawei's boss") ##
        ("relationship"<|>"JunXu"<|>"Elon"<|>"JunXu reports directly to Elon"<|>9)<|COMPLETE>
        """
        ####--------------------------------------------------------------------------------
        # records = ['("entity"<|>"张三"<|>"person"<|> "zhang san is a person who hold the title of huawei\'s boss")']
        # records = ['("relationship"<|>"JunXu"<|>"Elon"<|>"JunXu reports directly to Elon"<|>9)']
        records = split_string_by_multi_markers(
            final_result, [context_base['record_delimiter'], context_base['completion_delimiter']]
        )
        maybe_nodes = defaultdict(list) # 构造节点
        maybe_edges = defaultdict(list) # 构造关系/边
        # 正则化表达式清洗
        # 如果没有匹配到括号说明不对，直接continue跳过
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            """
            record = '"entity"<|>"张三"<|>"person"<|> "zhang san is a person who hold the 
                        title of huawei\'s boss"'
            record = '"relationship"<|>"JunXu"<|>"Elon"<|>"JunXu reports directly to Elon"<|>9'
            """
            record = record.group(1)
            """
            record_attributes = [
                    '"entity"',                                    # [0] 标志位
                    '"张三"',                                      # [1] 实体名称
                    '"person"',                                    # [2] 实体类型
                    '"zhang san is a person who hold the title of huawei\'s boss"'  # [3] 描述]
            record_attributes = [
                    '"relationship"',                          # [0] 标志位
                    '"JunXu"',                                # [1] 源实体
                    '"Elon"',                                 # [2] 目标实体
                    '"JunXu reports directly to Elon"',       # [3] 关系描述
                    '9'                                       # [4] 权重]   
            """
            record_attributes = split_string_by_multi_markers(
                record, [context_base['tuple_delimiter']]
            )
            """
            if_entities = {
                            'entity_name': '张三',
                            'entity_type': 'PERSON',
                            'description': "zhang san is a person who hold the title of huawei's boss",
                            'source_id': 'chunk-xxx'}
            """
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            """
            maybe_nodes = {
                    '张三': [{
                    'entity_name': '张三',
                    'entity_type': 'PERSON',
                    'description': "zhang san is a person who hold the title of huawei's boss",
                    'source_id': 'chunk-xxx'}]}
            """
            if if_entities is not None:
                maybe_nodes[if_entities['entity_name']].append(if_entities)
                continue
            """
            if_relation = {
                    'src_id': 'JUNXU',
                    'tgt_id': 'ELON',
                    'weight': 9.0,
                    'description': 'JunXu reports directly to Elon',
                    'source_id': 'chunk-xxx'}
            """
            if_relation = await _handle_single_relationship_extraction(record_attributes, chunk_key)
            """
            maybe_edges = {
            ('JUNXU', 'ELON'): [{
                    'src_id': 'JUNXU',
                    'tgt_id': 'ELON',
                    'weight': 9.0,
                    'description': 'JunXu reports directly to Elon',
                    'source_id': 'chunk-xxx'}]}
            """
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(if_relation)
        
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])]
        print(f"{now_ticks} Processed {already_processed}({already_processed*100//len(ordered_chunks)}%) chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
              end="", flush=True)
        
        return dict(maybe_nodes), dict(maybe_edges)
    # c: ('chunk-xxx', {'tokens': xx, 'content': 'xxxx', 'chunk_order_index': xx, 'full_doc_id': 'doc-xxx'})

    """
    results = {'Elon': [{'entity_name': 'Elon','entity_type': 'PERSON','description': "Elon is the boss of spacex",'source_id': 'chunk-xxx'}]}, 
              {'JUNXU', 'ELON'): [{'src_id': 'JUNXU','tgt_id': 'ELON','weight': 9.0,'description': 'JunXu reports directly to Elon','source_id': 'chunk-xxx'}]}
    """
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)
    """
    maybe_nodes = defaultdict(list, {
    'Elon': [{'entity_name': 'Elon', 'entity_type': 'PERSON', 'description': "Elon is the boss of spacex", 'source_id': 'chunk-xxx'}]})
    """
    
    """
    maybe_edges = defaultdict(list, {
    ('ELON', 'JUNXU'): [{'src_id': 'JUNXU', 'tgt_id': 'ELON', 'weight': 9.0, 'description': 'JunXu reports directly to Elon', 'source_id': 'chunk-xxx'}]})
    """

    #########################################################
    # 节点融合插入图
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config, tokenizer_wrapper)
            for k,  in maybe_nodes.items()
        ]
    )
    await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config, tokenizer_wrapper)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("DON'T extract any entities, maybe llm is not working!")
        return None
    # 边融合插入图
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)
    
    return knowledge_graph_inst



#-------------------------查询--------------------------
async def global_query(
        query,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        community_reports: BaseKVStorage[CommunitySchema],
        text_chunks_db: BaseKVStorage[TextChunkSchema],
        query_param: QueryParam, 
        tokenizer_wrapper,
        global_config: dict
):
    """
    查询的入口函数
    """
    community_schema = await knowledge_graph_inst.community_schema()
    community_schema = {k: v for k, v in community_schema.items() if v["level"] <= query_param.level}
    if not len(community_schema):
        return PROMPTS["fail_response"]
    use_model_func = global_config["best_model_func"]
    sorted_community_schema = sorted(
        community_schema.items(),
        key=lambda x: x[1]["occurrence"],
        reverse=True
    )
    sorted_community_schema = sorted_community_schema[:query_param.global_max_consider_community]
    community_datas = await community_reports.get_by_ids([k[0] for k in sorted_community_schema])
    community_datas = [c for c in community_datas if c is not None]
    community_datas = [
        c
        for c in community_datas
        if c["report_json"].get("rating", 0) >= query_param.global_min_community_rating
    ]
    community_datas = sorted(
        community_datas,
        key=lambda x: (x["occurrence"], x["report_json"].get("rating", 0)),
        reverse=True
    )
    logger.info(f"Revtrieved {len(community_datas)} communities")
    map_communities_points = await _map_global_communities(
        query, community_datas, query_param, global_config, tokenizer_wrapper
    )
    final_support_points = []
    for i, mc in enumerate(map_communities_points):
        for point in mc:
            if "description" not in point:
                continue
            final_support_points.append(
                {
                    "analyst": i,
                    "answer": point["description"],
                    "score": point.get("score", 1),
                }
            )
    final_support_points = [p for p in final_support_points if p["score"] > 0]
    if not len(final_support_points):
        return PROMPTS["fail_response"]

    final_support_points = sorted(
        final_support_points, key=lambda x: x["score"], reverse=True
    )
    final_support_points = truncate_list_by_token_size(
        final_support_points,
        key=lambda x: x["answer"],
        max_token_size=query_param.global_max_token_for_community_report,
        tokenizer_wrapper=tokenizer_wrapper
    )
    points_context = []
    for dp in final_support_points:
        points_context.append(
            f"""----Analyst {dp['analyst']}----
Importance Score: {dp['score']}
{dp['answer']}
"""
        )
    points_context = "\n".join(points_context)
    if query_param.only_need_context:
        return points_context
    sys_prompt_temp = PROMPTS["global_reduce_rag_response"]
    response = await use_model_func(
        query,
        sys_prompt_temp.format(report_data=points_context, response_type=query_param.response_type)
    )

    return response

    


async def _map_global_communities(
        query,
        communities_data,
        query_param,
        global_config,
        tokenizer_wrapper
):
    use_string_json_convert_func = global_config["convert_response_to_json_func"]
    use_model_func = global_config["best_model_func"]
    community_groups = []
    while len(communities_data):
        this_group = truncate_list_by_token_size(
            communities_data,
            key=lambda x: x["report_string"],
            max_token_size=query_param.global_max_toekn_for_community_report,
            tokenizer_wrapper=tokenizer_wrapper
        )
        community_groups.append(this_group)
        communities_data = communities_data[len(this_group):]
    
    async def _process(community_truncated_datas):
        """
        """
        communities_section_list = [["id", "content", "rating", "importance"]]
        for i, c in enumerate(community_truncated_datas):
            communities_section_list.append(
                [
                    i,
                    c["report_string"],
                    c["report_json"].get("rating", 0),
                    c["occurrence"]
                ]
            )
        community_context = list_of_list_to_csv(communities_section_list)
        sys_prompt_temp = PROMPTS["global_map_rag_points"]
        sys_prompt = sys_prompt_temp.format(context_data=community_context)
        response = await use_model_func(
            query,
            system_prompt=sys_prompt,
            **query_param.global_special_community_map_llm_kwargs,
        )
        data = use_string_json_convert_func(response)
        return data.get("points", [])
    
    logger.info(f"Grouping to {len(community_groups)} groups for global search!")
    responses= await asyncio.gather(*[_process(c) for c in community_groups])
    return responses
