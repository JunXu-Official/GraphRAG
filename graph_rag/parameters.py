from dataclasses import dataclass, field
from typing import Literal

@dataclass
class QueryParam:
    mode: Literal["local", "naive", "global"] = "global"
    only_need_text: bool = False
    response_type: str = "Multiple Paragraphs"
    level: int = 2
    top_k: int = 20
    # naive search: 不依赖知识图谱，纯粹靠相似度去数据库中比对搜索
    naive_max_token_for_text_unit = 12000   # 定义最长文本token
    # local search: 按照知识图谱顺藤摸瓜找
    local_max_token_for_text_unit = 4000    # 单个文本的最大token
    local_max_token_for_local_context = 4800    # p拼接起来的总上下文最大输入token
    local_max_token_for_community_report = 3200 # prompt中允许塞进来的社区报告最大token
    # True只匹配最精准命中的那个；False: 上下文一起打包
    local_community_single_one = False  
    # global search: 查看社区报告，摘要等
    global_min_community_rating = 0 # 进入全局搜索的社区报告的最低AI评分门槛
    # 查询时候，最多允许调用并考虑的社区报告总数量
    global_max_consider_community = 512
    # 在map阶段分组，以及在reduce阶段汇总时，塞给单个上下文LLM的token长度
    global_max_token_for_community_report = 16384
    # map阶段的LLM的控制参数，强制json格式
    global_special_community_map_llm_kwargs: dict = field(
        default_factory = lambda: {"response_format": {"type": "json_object"}}
    )
    