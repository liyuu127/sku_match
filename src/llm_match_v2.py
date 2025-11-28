# 使用langgraph接入大模型，使用大模型判断是否相似
# 输入商品名称和top3商品名称，输入相似的结果
import asyncio
from typing import Tuple, Any

from opik import configure
from opik.integrations.langchain import OpikTracer

from LLM import qwen3_8B, qwen3_30b_instruct, GLM4_9B, qwen3_235b_instruct
from Limter import RateLimiter
from utils import pandas_str_to_series

# configure(api_key="dHhF5jDOpifAMNJH9UalSjR0o", workspace="yu-li")
# opik_tracer = OpikTracer(project_name="sku_match")

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class MatchSelect(BaseModel):
    """Model for match top rank. Provide json constraints."""

    match_index: int = Field(
        description="返回匹配 origin_product 商品的候选商品编号；不匹配或其他情况返回0。",
    )
    match_reason: str = Field(
        description="返回匹配结果过程的思考原因，需要体现对输入信息的处理、比对、分析等过程，不要重复和冗余分析。",
    )


rank_prompt_v1 = """
你是一个电商运营专家，请你根据输入的商品描述信息在多个候选商品中选择描述为同一款商品类型的索引值。

<Task>
你需要按照下面步骤进行思考处理和返回：
1. 接收 origin_product、top1_candidate、top2_candidate、top3_candidate 四个商品信息描述，origin_product 为原始商品信息、top*_candidate 为候选匹配商品信息。
2. 每个商品信息包含品牌、商品名、规格数量3部分信息，分析时先提取 origin_product 输入商品的品牌、商品名、规格数量信息。
3. 依次提取并对比origin_product与候选商品top1_candidate,top2_candidate,top3_candidate的品牌、商品名、规格数量信息，判断是否为同款商品。
4. 全面分析思考，根据匹配规则选出与origin_product描述为同款商品的候选商品编号（如，1，2，3），需要特别注意，如果候选商品与origin_product都不匹配，请返回0。
</Task>

<Information Extraction Rules>
以下为信息提取规则：
1.商品品牌：通常位于信息前部，使用空格分离，部分商品可能不含品牌名。如：旺仔 精选进口乳源原味小馒头 14g_袋，品牌提取为旺仔；部分品牌和商品名重叠，如 元气森林外星人电解质水 0糖0卡青柠口味电解质饮料 500ml_瓶，品牌为元气森林；
2.商品名称：通常位于信息中部。如：元气森林外星人电解质水 0糖0卡青柠口味电解质饮料 500ml_瓶，名称提取为青柠口味电解质饮料；
3.商品规格数量：1）通常位于信息尾部，由规格和数量两部分组成。如：喜力Heineken 11.4°P啤酒 500ml3罐_包，规格为500ml，数量为3罐_包；2）部分数量信息位于尾部只有数字，如：乌苏 11°P乌苏啤酒 500ml_听3（新老包装随机发货），规格数量部分为500ml_听3，规格为500ml_听，数量为3；3）如果没有数量部分默认为1，如：旺仔 精选进口乳源原味小馒头 14g_袋，规格为14g_袋，数量为1；
</Information Extraction Rules>

<Match rules>
以下是origin_product和候选商品的匹配规则：
1. 商品品牌不一致时视为不匹配，例如origin_product识别品牌为可口，candidate识别为品牌为百事；评判时由于提取品牌可能包含商品信息，因此商品部分词汇重叠也视为一致，例如乌苏与乌苏啤酒视为品牌一致。
2. 如果origin_product和top_candidate任意一方没有品牌时视为不匹配.
3. 商品数量和规格都一致才视为匹配，数据和规格不一致标明不是同款商品，匹配度为0。1）规格可包括L、kg、千克、盒、件、片、条等市面计量单位；2）评判时注意规格单位换算，如1L和1000ml为一致；3）个、袋、盒、瓶等单个包装规格在整体语义下视为同等规格；4）数量不一致如： 1000ml_瓶和1000ml_瓶2数量不匹配；5）规格中没有体现数量，视为数量为1。
4. 商品名语义或分类不一时致视为不匹配，origin_product识别为识字卡，candidate识别为挪车卡。
5. 候选top_candidate都不匹配时不需要寻找最接近的商品，视为没有商品匹配，返回0。
</Match rules>

<Examples>
示例1：
origin_product：卫龙魔芋爽 微辣麻酱素毛肚 （15+3）g_袋
top1_candidate：卫龙 魔芋爽 香辣味素毛肚 15克_袋
分析：origin_product和top1_candidate品牌、商品名都为卫龙和魔芋爽，但是由于规格数量不同，origin_product为18g/袋,top1_candidate为15g/袋，所以不是同款商品。

示例2：
origin_product：【毛绒长款睡袍】卡皮巴拉冬季新款加厚睡袍卡通可爱风珊瑚绒睡衣女家居服_件
top1_candidate：珊瑚绒睡袍女冬季甜美加厚新款长款睡裙法兰绒睡衣女可外穿家居服_件（没有裤子）
分析：origin_product和top1_candidate品牌、商品无法提取，商品描述未高度重叠，所有判断不是同款商品，需要返回0，match_index=0。
</Examples>

以下是输入信息：
<products_info>
origin_product：{origin_product}
top1_candidate：{top1_candidate}
top2_candidate：{top2_candidate}
top3_candidate：{top3_candidate}
</products_info>

<Output Format>
请使用以下键值以有效的 JSON 格式进行响应：
"match_index": int. 返回origin_product描述一致的候选商品索引。 如果是top1_candidate返回1; 如果是top2_candidate返回2; 如果是top3_candidate返回3；没有候选商品匹配或其他情况返回0。
"match_reason":str. 返回匹配结果过程的思考原因，需要体现对输入信息的处理、比对、分析等过程。
</Output Format>

<Critical Reminder>
1. 规格数量未必须条件，需满足完全一致，基本一致视为不匹配。
2. 当商品提取不到品牌、数量规格信息时，商品名称和商品描述需要高度一致才视为同一款商品。
3. 特别注意，如果候选商品信息都不是同款，不需要考虑寻找最接近的候选商品，说明没有同款商品，直接返回0。
4. 必须保证分析结果编号或索引与返回的match_index字段值相同，分析得出没有匹配结果时返回0。
</<Critical Reminder>
"""

rank_prompt_v2 = """
你是一个电商运营专家，请你根据输入的商品描述信息在多个候选商品中选择描述为同一款商品的索引值。

<Task>
你需要按照下面步骤进行思考处理和返回：
1. 接收 origin_product、top1_candidate、top2_candidate、top3_candidate 四个商品信息描述，origin_product 为原始商品信息、top*_candidate 为候选匹配商品信息。
2. 每个商品信息包含品牌、商品名、规格数量3部分信息，分析时先提取 origin_product 输入商品的品牌、商品名、规格数量信息。
3. 提取并对比origin_product与候选商品top1_candidate,top2_candidate,top3_candidate的品牌、商品名、规格数量信息，判断是否为同款商品。
4. 根据匹配规则选出与origin_product描述为同款商品的候选商品编号（如，1，2，3），需要特别注意，如果候选商品与origin_product都不匹配，请返回0。
</Task>

<Information Extraction Rules>
以下为信息提取规则：
1.商品品牌：通常位于信息前部，一般使用空格分离，部分商品可能不含品牌名。如：旺仔 精选进口乳源原味小馒头 14g_袋，品牌提取为旺仔；部分品牌和商品名重叠，如 元气森林外星人电解质水 0糖0卡青柠口味电解质饮料 500ml_瓶，品牌为元气森林；
2.商品名称：通常位于信息中部，需要去除描述部分。如：元气森林外星人电解质水 0糖0卡青柠口味电解质饮料 500ml_瓶，名称提取为青柠口味电解质饮料；
3.商品规格数量：
    1）通常位于信息尾部，由规格和数量两部分组成。如：喜力Heineken 11.4°P啤酒 500ml3罐_包，规格为500ml3罐_包，数量为1；
    2）部分商品数量信息位于尾部且只有数字，如：乌苏 11°P乌苏啤酒 500ml_听3（新老包装随机发货），规格数量部分为500ml_听3，规格为500ml_听，数量为3；
    3）如果没有数量部分默认为1，如：旺仔 精选进口乳源原味小馒头 14g_袋，规格为14g_袋，数量为1；
</Information Extraction Rules>

<Match rules>
以下是origin_product和候选商品的匹配规则：
1. 商品品牌不一致时视为不匹配，例如origin_product识别品牌为可口，candidate识别为品牌为百事；特别注意，评判时由于提取品牌可能包含商品信息，因此商品部分词汇重叠也视为一致，例如乌苏与乌苏啤酒视为品牌一致。
2. 如果origin_product和top_candidate任意一方没有品牌时视为不匹配.
3. 商品数量和规格严格一致才视为匹配，数据和规格不一致标明不是同款商品，匹配度为0。以下是规格数据匹配时处理注意事项：
    1）规格可包括L、kg、千克、盒、件、片、条等市面计量单位，评判时注意规格单位换算，如1L和1000ml为一致；
    2）个、袋、盒、瓶等单个包装规格在整体语义下视为同等规格；
    3）规格中没有体现数量，视为数量为1。
4. 商品名称语义不一致视为不匹配，origin_product识别为识字卡，candidate识别为挪车卡。
5. 候选top_candidate都不匹配时不需要寻找最接近的商品，视为没有商品匹配，返回0。
</Match rules>

<Examples>
示例1：
origin_product：卫龙魔芋爽 微辣麻酱素毛肚 （15+3）g_袋
top1_candidate：卫龙 魔芋爽 香辣味素毛肚 15克_袋
分析：origin_product和top1_candidate品牌、商品名都为卫龙和魔芋爽，但是由于规格数量不同，origin_product为18g/袋,top1_candidate为15g/袋，所以不是同款商品。

示例2：
origin_product：【毛绒长款睡袍】卡皮巴拉冬季新款加厚睡袍卡通可爱风珊瑚绒睡衣女家居服_件
top1_candidate：珊瑚绒睡袍女冬季甜美加厚新款长款睡裙法兰绒睡衣女可外穿家居服_件（没有裤子）
分析：origin_product和top1_candidate品牌、商品无法提取，商品描述未高度重叠，所有判断不是同款商品，需要返回0，match_index=0。
</Examples>

以下是输入信息：
<products_info>
origin_product：{origin_product}
top1_candidate：{top1_candidate}
top2_candidate：{top2_candidate}
top3_candidate：{top3_candidate}
</products_info>

<Output Format>
请使用以下键值以有效的 JSON 格式进行响应：
"match_index": int. 返回origin_product描述一致的候选商品索引；没有候选商品匹配或其他情况返回0。
"match_reason":str. 返回匹配结果过程的思考原因，需要体现对输入信息的处理、比对、分析等过程。
</Output Format>

<Critical Reminder>
1. 规格数量严格匹配为必须条件，基本一致视为不匹配。
2. 当商品提取不到品牌、数量规格信息时，商品名称和商品描述需要高度一致才视为同一款商品。
3. 特别注意，如果候选商品信息都不是同款，不需要考虑寻找最接近的候选商品，说明没有同款商品，直接返回0。
4. 必须保证分析结果编号或索引与返回的match_index字段值相同，分析得出没有匹配结果时返回0。
</Critical Reminder>
"""

rank_prompt_v3 = """
你是一个电商运营专家，请你根据输入的商品描述信息在多个候选商品中选择描述为同一款商品的索引值。

<Task>
你需要按照下面步骤进行思考处理和返回：
1. 接收 origin_product、top1_candidate、top2_candidate、top3_candidate 四个商品信息描述，origin_product 为原始商品信息、top*_candidate 为候选匹配商品信息。
2. 每个商品信息包含品牌、商品名、规格数量3部分信息，分析时先提取 origin_product 输入商品的品牌、商品名、规格数量信息。
3. 提取并对比origin_product与候选商品top1_candidate,top2_candidate,top3_candidate的品牌、商品名、规格数量信息，判断是否为同款商品。
4. 根据匹配规则选出与origin_product描述为同款商品的候选商品编号（如，1，2，3），需要特别注意，如果候选商品与origin_product都不匹配，请返回0。
</Task>

<Information Extraction Rules>
以下为信息提取规则：
1.商品品牌：通常位于信息前部，一般使用空格分离，部分商品可能不含品牌名。如：旺仔 精选进口乳源原味小馒头 14g_袋，品牌提取为旺仔；部分品牌和商品名重叠，如 元气森林外星人电解质水 0糖0卡青柠口味电解质饮料 500ml_瓶，品牌为元气森林；
2.商品名称：通常位于信息中部，需要去除描述部分。如：元气森林外星人电解质水 0糖0卡青柠口味电解质饮料 500ml_瓶，名称提取为青柠口味电解质饮料；
3.商品规格数量：
    1）通常位于信息尾部，由规格和数量两部分组成。如：喜力Heineken 11.4°P啤酒 500ml3罐_包，规格为500ml3罐_包，数量为1；
    2）部分商品数量信息位于尾部且只有数字，如：乌苏 11°P乌苏啤酒 500ml_听3（新老包装随机发货），规格数量部分为500ml_听3，规格为500ml_听，数量为3；
    3）如果没有数量部分默认为1，如：旺仔 精选进口乳源原味小馒头 14g_袋，规格为14g_袋，数量为1；
</Information Extraction Rules>

<Match rules>
以下是origin_product和候选商品的匹配规则：
1. 商品品牌不一致时视为不匹配，例如origin_product识别品牌为可口，candidate识别为品牌为百事；特别注意，评判时由于提取品牌可能包含商品信息，因此商品部分词汇重叠也视为一致，例如乌苏与乌苏啤酒视为品牌一致。
2. 商品数量和规格严格一致才视为匹配，数据和规格不一致标明不是同款商品，匹配度为0。以下是规格数据匹配时处理注意事项：
    1）规格可包括L、kg、千克、盒、件、片、条等市面计量单位，评判时注意规格单位换算，如1L和1000ml为一致；
    2）个、袋、盒、瓶等单个包装规格在整体语义下视为同等规格；
    3）规格中没有体现数量，视为数量为1。
    4）比较商品双方规格数量都不存在视为一致。
3. 商品名称语义不一致视为不匹配，origin_product识别为识字卡，candidate识别为挪车卡。
4. 候选top_candidate都不匹配时不需要寻找最接近的商品，视为没有商品匹配，返回0。
</Match rules>

<products_info>
以下是输入商品信息：
origin_product：{origin_product}
top1_candidate：{top1_candidate}
top2_candidate：{top2_candidate}
top3_candidate：{top3_candidate}
</products_info>

<Output Format>
请使用以下键值以有效的 JSON 格式进行响应：
"match_index": int. 返回origin_product描述一致的候选商品索引；没有候选商品匹配或其他情况返回0。
"match_reason":str. 返回匹配结果过程的思考原因，需要体现对输入信息的处理、比对、分析等过程。
</Output Format>

<Critical Reminder>
1. 特别注意，如果候选商品信息都不是同款，不需要考虑寻找最接近的候选商品，说明没有同款商品，直接返回0。
2. 必须保证分析结果编号或索引与返回的match_index字段值相同，分析得出没有匹配结果时返回0。
</Critical Reminder>
"""
# model = qwen3_8B.with_structured_output(MatchSelect).with_retry(stop_after_attempt=2)
model = qwen3_30b_instruct.with_structured_output(MatchSelect).with_retry(stop_after_attempt=2)
# model = GLM4_9B.with_structured_output(MatchSelect).with_retry(stop_after_attempt=2)
# model = qwen3_235b_instruct.with_structured_output(MatchSelect).with_retry(stop_after_attempt=2)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

REQUEST_INTERVAL = 1


async def llm_rank(product_name, top1, top2, top3) -> tuple[int, str]:
    await asyncio.sleep(REQUEST_INTERVAL)  # 限速控制
    print(f"正在处理：product_name:{product_name}")
    prompt_format = rank_prompt_v3.format(origin_product=product_name, top1_candidate=top1,
                                          top2_candidate=top2, top3_candidate=top3)

    try:
        rankSelect = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt_format)],
                          # config={"callbacks": [opik_tracer]}
                          ),
            timeout=60)

        index, reason = rankSelect.match_index, rankSelect.match_reason
        # 判断index是否是0，1，2，3，  如果不是返回-1
        if index not in [0, 1, 2, 3]:
            return -1, reason
    # 捕获所有错误，打印信息
    except Exception as e:
        print(f"商品：{product_name} 发生异常: {e}")
        return -1, ""
    return index, reason


async def process_llm_row(i, row, top1_list, top2_list, top3_list) -> tuple[Any, int, str] | tuple[Any, Any, str] | \
                                                                      tuple[Any, None, str] | tuple[Any, str, str]:
    # # 判断是否top1条码相同，相同忽略
    # if top1_list[i] is not None and str(row.条码).strip() not in ('', 'nan', 'None') and str(top1_list[i].条码) == str(
    #         row.条码):
    #     return i, 1, "条码一致"

    idx, reason = await llm_rank(row.商品名称,
                                 top1_list[i].商品名称 if top1_list[i] is not None else "",
                                 top2_list[i].商品名称 if top2_list[i] is not None else "",
                                 top3_list[i].商品名称 if top3_list[i] is not None else "")
    if idx == 1:
        return i, top1_list[i], reason
    elif idx == 2:
        return i, top2_list[i], reason
    elif idx == 3:
        return i, top3_list[i], reason
    elif idx == 0:
        return i, None, reason
    return i, "error", reason


async def llm_match_fill(owner_df, top1_list, top2_list, top3_list):
    print("开始LLM匹配...")
    limiter = RateLimiter(rpm_limit=800, tpm_limit=40000)
    owner_subset = owner_df  # 这里可以改成 owner_df.iloc[:100] 测试
    batch_size = 10  # 每批次处理 10 条，可根据速率限制调整
    delay_seconds = 5  # 每批之间等待 3 秒，可根据 API 限速调整
    all_results = []
    for start_idx in range(0, len(owner_subset), batch_size):
        est_tokens = batch_size * 700
        await limiter.record_call(est_tokens)

        end_idx = min(start_idx + batch_size, len(owner_subset))
        batch = owner_subset.iloc[start_idx:end_idx]
        print(f"正在处理第 {start_idx}~{end_idx - 1} 条（共 {len(owner_subset)} 条）...")

        # 按批次创建任务
        tasks = [
            process_llm_row(i, row, top1_list, top2_list, top3_list)
            for i, row in batch.iterrows()
        ]

        # 阻塞等待本批结果（而不是一次性 gather 所有）
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)

        # 每批之间延迟，防止速率限制
        if end_idx < len(owner_subset):
            print(f"等待 {delay_seconds} 秒以避免触发频率限制...")
            await asyncio.sleep(delay_seconds)
    # 更新 DataFrame
    for i, val, reason in all_results:
        owner_df.at[i, '相似商品'] = val
        owner_df.at[i, '相似商品原因'] = reason
    # 尝试重试
    await retry_error_rows(owner_df, limiter, 10, 10)


async def handle_error_row(index, row, limiter, df):
    """重新处理错误匹配行"""

    await limiter.record_call(800)

    similar_product1 = pandas_str_to_series(row['相似商品1'])
    similar_product2 = pandas_str_to_series(row['相似商品2'])
    similar_product3 = pandas_str_to_series(row['相似商品3'])

    top1 = similar_product1.商品名称 if similar_product1 is not None else ""
    top2 = similar_product2.商品名称 if similar_product2 is not None else ""
    top3 = similar_product3.商品名称 if similar_product3 is not None else ""

    # 模型排序
    idx, reason = await llm_rank(row.商品名称, top1, top2, top3)
    print(f"  匹配结果：{idx}")

    if idx == 1:
        df.at[index, '相似商品'] = row['相似商品1']
    elif idx == 2:
        df.at[index, '相似商品'] = row['相似商品2']
    elif idx == 3:
        df.at[index, '相似商品'] = row['相似商品3']
    elif idx == 0:
        df.at[index, '相似商品'] = None
    else:
        df.at[index, '相似商品'] = 'error'  # 仍然是错误，等待下一轮 retry
    df.at[index, '相似商品原因'] = reason


async def run_retry_batch(df, limiter, batch_size=10):
    """批次重试"""
    tasks = []
    for index, row in df.iterrows():
        # 判断是否是error记录，如果不是跳过这条记录
        if row['相似商品'] != 'error':
            continue
        tasks.append(handle_error_row(index, row, limiter, df))
        if len(tasks) >= batch_size:
            await asyncio.gather(*tasks)
            tasks = []
    if tasks:
        await asyncio.gather(*tasks)


async def retry_error_rows(df, limiter, max_rounds=5, batch_size=10):
    """
    df: 原始 DataFrame，包含 '相似商品' 字段
    max_rounds: 最多重试几轮
    batch_size: 每次协程批处理大小
    """

    for round_num in range(1, max_rounds + 1):
        # 过滤需要 retry 的记录
        error_rows = df[df['相似商品'] == 'error']

        retry_count = len(error_rows)
        print(f"\n====== 第 {round_num} 次遍历，需重试记录：{retry_count} 条 ======")

        if retry_count == 0:
            print("所有错误记录已处理完成！")
            break

        await run_retry_batch(df, limiter, batch_size=batch_size)

    else:
        print("\n⚠ 达到最大重试次数，但仍有未修复的错误记录")
