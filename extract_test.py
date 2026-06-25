from typing  import Any
import html
from nano_graphrag.prompt import PROMPTS
from nano_graphrag._utils import split_string_by_multi_markers, is_float_regex
import re

def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    """
        判断AI返回的一行属性到底是不是一个合格的实体
    """
    # 一个标准的实体记录至少包括：标志位、名称、类型、描述，如果少传直接丢弃。
    # 标志位就是"entity"
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # 强制大写，避免Elon和elon识被识别为2个不同的实体
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key    # 给实体打上标签
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )

def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )


if __name__ == '__main__':
     from collections import Counter, defaultdict
     maybe_nodes = defaultdict(list)
     context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"], #关系三元组的分隔符，如<|>
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],   # 每条记录的分隔符，如##
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],   # 输出结束的分隔符
    )
     final_result = 'Based on the provided text and entity types, here is the output:\n\n**##**\n("entity"<|>"Project Gutenberg"<|>"organization"<|>"Project Gutenberg is a digital library that provides free access to classic literature.")##\n("entity"<|>"Charles Dickens"<|>"person"<|>"Charles Dickens was an author who wrote the classic novel A Christmas Carol.")##\n("entity"<|>"Arthur Rackham"<|>"person"<|>"Arthur Rackham was an illustrator who contributed to the edition of A Christmas Carol.")##\n("entity"<|>"Suzanne Shell"<|>"person"<|>"Suzanne Shell was a producer involved in digitizing A Christmas Carol for Project Gutenberg.")##\n("entity"<|>"Janet Blenkinship"<|>"person"<|>"Janet Blenkinship was another producer involved in digitizing A Christmas Carol for Project Gutenberg.")##\n("entity"<|>"Online Distributed Proofreading Team"<|>"organization"<|>"The Online Distributed Proofreading Team contributed to the digitization of A Christmas Carol.")##\n("relationship"<|>"Suzanne Shell"<|>"Janet Blenkinship"<|>"Suzanne Shell and Janet Blenkinship collaborated on producing A Christmas Carol for Project Gutenberg."<|>8)##\n("relationship"<|>"Online Distributed Proofreading Team"<|>"Project Gutenberg"<|>"The Online Distributed Proofreading Team contributed to the digitization of A Christmas Carol, which is part of Project Gutenberg."<|>7)<|COMPLETE|>\n**##**\n\nNote that there are no specific relationships between entities in this text, as it primarily consists of information about the book and its creators.'
     final_result = '("entity"<|>"张三"<|>"人名"<|>"华为老板")##("entity"<|>"马斯克"<|>"人名"<|>"spacex老板")'
     records = split_string_by_multi_markers(
     final_result,
     [context_base["record_delimiter"], context_base["completion_delimiter"]],
)
     print('11111',records)
     for record in records:
         record = re.search(r"\((.*)\)", record)
         if record is None:
             continue
         record = record.group(1)
         print('222222', record)
        # 切分 张三<|>人名<|>华为的老板 ---->  ["张三", "人名", "技术总监"]
         record_attributes = split_string_by_multi_markers(
             record, [context_base["tuple_delimiter"]]
         )
         print('3333', record_attributes)
        #  检查属性的数量和内容
         chunk_key = "0ab0ab"
         if_entities = _handle_single_entity_extraction(
             record_attributes, chunk_key
         )
         print('4444444', if_entities)
         if if_entities is not None:
             maybe_nodes[if_entities["entity_name"]].append(if_entities)
             continue
         if_relation = _handle_single_relationship_extraction(
             record_attributes, chunk_key
             )
         print('6666666', if_relation)
        
     print('55555555', maybe_nodes)