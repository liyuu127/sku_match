# 使用langgraph接入大模型，使用大模型判断是否相似
# 输入商品名称和top3商品名称，输入相似的结果
import asyncio
# 设置langsmith环境配置
import os

# os.environ["LANGSMITH_API_KEY"] = "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"
# os.environ["LANGSMITH_PROJECT"] = "sku_match"
# os.environ["LANGSMITH_TRACING"] = "true"
from langchain_openai import ChatOpenAI

qwen3_8B = ChatOpenAI(
    model="Qwen/Qwen3-8B",
    base_url="https://api.siliconflow.cn/v1",
    api_key="sk-idaudysppselrwglygkbtatkregsbxhaxypaeulbfpavrals",

)

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class RankSelect(BaseModel):
    """Model for match top rank. Provide json constraints."""

    rank_index: int = Field(
        description="Returns the index of the candidate product information that matches the description. If it is top1_candidate, return 1; if it is top2_candidate, return 2; if it is top3_candidate, return 3; otherwise, return 0.",
    )


rank_prompt = """
你是一个电商运营专家，请你根据输入的商品描述信息在多个候选商品描述中选择描述一致的编号值。
你需要按照下面步骤进行处理和返回：
- 需要先理解输入的商品信息 origin_product，可以从商品品牌、商品描述、商品名称、商品数量规格等多个角度进行整理
- 对输入的候选商品信息 top1_candidate,top2_candidate,top3_candidate 依次进行整理，按照上面四个角度进行匹配，如所有候选商品信息和origin_product商品信息都不一致，请直接返回0。
- 选择描述一致候选商品，返回索引id,范围为1，2，3，没有描述一致的商品返回0

以下是输入信息：
<products_info>
origin_product：{origin_product}
top1_candidate：{top1_candidate}
top2_candidate：{top2_candidate}
top3_candidate：{top3_candidate}
</products_info>

以下是origin_product和候选商品不匹配的举例情况：
1. candidate信息为空
2. 商品品牌不一致，origin_product识别品牌为可口可乐，candidate识别为品牌为百事可乐
3. 商品数量规格不一致，origin_product识别规格为250ml,candidate识别规格为300ml；规格可包括L、kg、千克、盒、件、片、条等市面计量单位，如没有说明默认数量为一
4. 商品名称语义不符合，origin_product识别为识字卡，candidate识别为挪车卡
5. 商品描述有冲突，origin_product识别描述冬季保暖，candidate识别描述为夏季防晒
6. 商品信息整体语义冲突，origin_product商品信息为小型便携式冲牙器，candidate商品信息为家用取暖器暖风机

请使用以下键值以有效的 JSON 格式进行响应：
"rank_index": int.Returns the index of the candidate product information that matches the description. If it is top1_candidate, return 1; if it is top2_candidate, return 2; if it is top3_candidate, return 3; otherwise, return 0.,
"""

model = qwen3_8B.with_structured_output(RankSelect).with_retry(stop_after_attempt=2)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)


REQUEST_INTERVAL = 1

async def llm_rank(product_name, top1, top2, top3) -> int:
    await asyncio.sleep(REQUEST_INTERVAL)  # 限速控制
    print("正在处理：", product_name)
    prompt_format = rank_prompt.format(origin_product=product_name, top1_candidate=top1,
                                       top2_candidate=top2, top3_candidate=top3)

    try:
        rankSelect = await asyncio.wait_for(model.ainvoke([HumanMessage(content=prompt_format)]), timeout=60)

        index = rankSelect.rank_index
        # 判断index是否是0，1，2，3，  如果不是返回-1
        if index not in [0, 1, 2, 3]:
            return -1
    # 捕获所有错误，打印信息
    except Exception as e:
        print(f"商品：{product_name} 发生异常: {e}")
        return -1
    return index


async def process_llm_row(i, row, top1_list, top2_list, top3_list):
    idx = await llm_rank(row.商品名称, top1_list[i], top2_list[i], top3_list[i])
    if idx == 1:
        return i, top1_list[i]
    elif idx == 2:
        return i, top2_list[i]
    elif idx == 3:
        return i, top3_list[i]
    elif idx == 0:
        return i, None
    return i, "error"
