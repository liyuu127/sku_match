import re
import asyncio
import uuid
from uuid import UUID

import pandas as pd
from typing import List, Set, Tuple
import jieba
from concurrent.futures import ThreadPoolExecutor
import time
import asyncio
import functools

# # Python 3.8 兼容 asyncio.to_thread
# if not hasattr(asyncio, "to_thread"):
#     async def to_thread(func, *args, **kwargs):
#         loop = asyncio.get_event_loop()
#         return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
#
#
#     asyncio.to_thread = to_thread

# ====================================================
# 全局配置
# ====================================================
MAX_CONCURRENCY = 100  # 控制最大并发任务数
jieba.load_userdict("../data/jieba_dict.txt")

# 设置langsmith环境配置
import os
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "sku_match"
os.environ["LANGSMITH_TRACING"] = "true"

# ====================================================
# 工具函数
# ====================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def chinese_tokenize_sync(text: str) -> set:
    """jieba 同步分词"""
    if not text or text.isspace():
        return set()
    return set(jieba.cut(text, cut_all=False))


async def chinese_tokenize(text: str) -> set:
    """异步分词包装"""
    return await asyncio.to_thread(chinese_tokenize_sync, text)


def extract_full_spec_and_clean(text):
    if not isinstance(text, str):
        return ""
    units = ["盒", "箱", "件", "个", "袋", "包", "瓶", "罐", "卷", "片", "只",
             "ml", "l", "g", "kg", "mm", "cm", "m"]
    pack_units = ["盒", "箱", "件", "个", "袋", "包", "瓶", "罐", "卷", "片", "只", "条", "支"]
    pattern = re.compile(
        rf'((?:\d+(?:\.\d+)?\s?(?:{"|".join(units)}|{"|".join(pack_units)}))'
        rf'(?:[*_×xX/\\-]?\s?\d*(?:\.\d+)?\s?(?:{"|".join(units)}|{"|".join(pack_units)}))*)',
        flags=re.IGNORECASE
    )
    matches = pattern.findall(text)
    if not matches:
        return ""
    specs = sorted(matches, key=len, reverse=True)
    return specs[0].strip()


def extract_brand_from_tokens(text_to_search: str, brand_dictionary: Set[str]) -> str:
    if not text_to_search or text_to_search.isspace() or not brand_dictionary:
        return ""
    for brand_info in brand_dictionary:
        if brand_info in text_to_search:
            return brand_info
    return ""


# 读取brand.txt解析为列表
BRAND_DICTIONARY = set(line.strip() for line in open("../data/brand.txt", encoding="utf-8"))
print("已加载品牌列表：", BRAND_DICTIONARY)


def calculate_brand_similarity(brand1: str, brand2: str) -> float:
    brand1 = brand1.strip().lower()
    brand2 = brand2.strip().lower()
    if not brand1 or not brand2:
        return 0.5
    if brand1 == brand2:
        return 1.0
    return 0.0


def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0.0


# ====================================================
# 相似度计算
# ====================================================
async def calculate_similarity_for_single_product(
        p_name: str, p_tokenize: set, p_brand: str,
        candidate_row: pd.Series, candidate_tokens: set
) -> Tuple[float, str]:
    candidate_id = candidate_row.商品ID
    candidate_name = candidate_row.商品名称
    candidate_brand = extract_brand_from_tokens(candidate_name, BRAND_DICTIONARY)
    similarity_tokens = jaccard_similarity(p_tokenize, candidate_tokens)
    similarity_brand = calculate_brand_similarity(p_brand, candidate_brand)
    similarity = similarity_tokens * 0.7 + similarity_brand * 0.3
    result_str = f"{candidate_id}-{candidate_name}--{similarity:.4f}--{similarity_tokens:.4f}--{similarity_brand:.4f}"
    return similarity, result_str


async def find_top3_similar_combined_async(product_name: str, candidate_df: pd.DataFrame,
                                           candidate_tokens_list: List[set],
                                           sem: asyncio.Semaphore) -> Tuple[str, str, str]:
    async with sem:
        p_name = clean_text(product_name)
        p_tokenize = await chinese_tokenize(p_name)
        p_brand = extract_brand_from_tokens(p_name, BRAND_DICTIONARY)

        tasks = []
        for idx, row in enumerate(candidate_df.itertuples(index=False)):
            candidate_tokens = candidate_tokens_list[idx]
            tasks.append(calculate_similarity_for_single_product(
                p_name, p_tokenize, p_brand, row, candidate_tokens
            ))

        results = await asyncio.gather(*tasks)
        sorted_results = sorted(results, key=lambda x: x[0], reverse=True)

        top3_combined = [r[1] for r in sorted_results[:3]]
        while len(top3_combined) < 3:
            top3_combined.append("")
        return tuple(top3_combined)


async def preprocess_candidate_tokens(candidate: pd.DataFrame) -> List[set]:
    """异步批量预处理分词"""
    tasks = [chinese_tokenize(clean_text(name)) for name in candidate['商品名称']]
    return await asyncio.gather(*tasks)


def load_excel(file_path: str) -> pd.DataFrame:
    """
    读取Excel文件，验证必要列并处理空值
    要求文件必须包含"商品ID"和"商品名称"列（可修改required_cols适配实际列名）
    """
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        required_cols = ['商品ID', '商品名称']
        # 检查必要列是否存在
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列：{', '.join(missing_cols)}，需包含{required_cols}")
        # 去除商品ID或名称为空的无效行
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        # 强制转为字符串类型，避免数字ID/名称拼接出错（如科学计数法、格式丢失）
        df['商品ID'] = df['商品ID'].astype(str).str.strip()
        df['商品名称'] = df['商品名称'].astype(str).str.strip()
        return df
    except Exception as e:
        print(f"读取文件{file_path}失败：{str(e)}")
        raise


async def save_excel_async(df: pd.DataFrame, file_path: str):
    df.to_excel(file_path, index=False, engine='openpyxl')


# ====================================================
# 批处理逻辑
# ====================================================
async def process_batch(owner_df: pd.DataFrame, ele_df: pd.DataFrame, tokens: List[set],
                        start_idx: int, batch_size: int, sem: asyncio.Semaphore):
    top1_list, top2_list, top3_list = [], [], []
    end_idx = min(start_idx + batch_size, len(owner_df))
    batch_df = owner_df.iloc[start_idx:end_idx]

    tasks = []
    for row in batch_df.itertuples(index=False):
        tasks.append(find_top3_similar_combined_async(row.商品名称, ele_df, tokens, sem))

    batch_results = await asyncio.gather(*tasks)
    for t1, t2, t3 in batch_results:
        top1_list.append(t1)
        top2_list.append(t2)
        top3_list.append(t3)

    print(f"已完成批次：{start_idx + 1}~{end_idx}")
    return top1_list, top2_list, top3_list


async def process_owner_data_async(owner_df: pd.DataFrame, ele_df: pd.DataFrame, tokens: List[set],
                                   batch_size: int = 100):
    total_items = len(owner_df)
    num_batches = (total_items + batch_size - 1) // batch_size
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    print(f"开始处理数据，共{num_batches}批次，每批{batch_size}条。")

    tasks = [
        process_batch(owner_df, ele_df, tokens, i * batch_size, batch_size, sem)
        for i in range(num_batches)
    ]

    batch_results = await asyncio.gather(*tasks)
    top1_list, top2_list, top3_list = [], [], []
    for b1, b2, b3 in batch_results:
        top1_list.extend(b1)
        top2_list.extend(b2)
        top3_list.extend(b3)
    return top1_list, top2_list, top3_list


# 使用langgraph接入大模型，使用大模型判断是否相似
# 输入商品名称和top3商品名称，输入相似的结果
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
        description="Returns the index of the closest candidate product information. If it is top1_candidate, return 1; if it is top1_candidate, return 2; if it is top1_candidate, return 3; otherwise, return 0.",
    )


rank_prompt = """
你是一个电商运营专家，请你根据输入的商品描述信息在多个候选商品描述中选择最符合输入商品的编号值。
你需要按照下面步骤进行处理和返回：
- 需要先理解输入商品的描述信息origin_product，可以从商品品牌、商品描述、商品名称、商品规格等多个角度进行整理
- 对输入的候选商品信息top1_candidate,top2_candidate,top3_candidate依次进行整理，按照上面四个角度进行匹配，如果无候选商品或候选商品规格、商品描述等明显不符合，请直接返回0。
- 选择最接近的候选商品，返回索引id,范围为1，2，3

以下是输入信息：
<products_info>
origin_product：{origin_product}
top1_candidate：{top1_candidate}
top2_candidate：{top2_candidate}
top3_candidate：{top3_candidate}
</products_info>

请使用以下键值以有效的 JSON 格式进行响应,不要带有额外回车或标识符：
"rank_index": int，Returns the index of the closest candidate product information. If it is top1_candidate, return 1; if it is top1_candidate, return 2; if it is top1_candidate, return 3; otherwise, return 0.,
"""

model = qwen3_8B.with_structured_output(RankSelect).with_retry(stop_after_attempt=3)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

REQUEST_INTERVAL=0.2
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
        import traceback
        print(f"发生异常: {e}")
        traceback.print_exc()
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
    return i, None


# ====================================================
# 主函数入口
# ====================================================
async def main():
    print("开始加载数据...")
    start_time = time.time()
    # ele_df = pd.read_csv("../data/elme_sku_small.csv")
    # owner_df = pd.read_csv("../data/meituan_sku_small.csv")

    owner_df = load_excel("../output/附件1_补充Top3相似商品（ID-名称组合）6888d2b1-c459-46e3-acd1-18727e8b35dd.xlsx")
    # 打印前几行数据
    print(owner_df.head())

    owner_subset = owner_df.iloc[:200]

    # 从匹配结果中提取top1_list, top2_list, top3_list
    top1_list, top2_list, top3_list = owner_subset['相似商品1（ID-名称）'], owner_subset['相似商品2（ID-名称）'], owner_subset['相似商品3（ID-名称）']
    # 创建任务列表
    tasks = [
        process_llm_row(i, row, top1_list, top2_list, top3_list)
        for i, row in owner_subset.iterrows()
    ]
    # 并发执行
    results = await asyncio.gather(*tasks)

    # 更新 DataFrame
    for i, val in results:
        owner_df.at[i, '相似商品'] = val


    output_path = "../output/附件1_补充Top3相似商品（llm）" + str(uuid.uuid4()) + ".xlsx"
    await save_excel_async(owner_df, output_path)

    end_time = time.time()
    print(f"\n处理完成！结果已保存到：{output_path}")
    print(f"总耗时：{end_time - start_time:.2f}秒")


if __name__ == '__main__':
    asyncio.run(main())
