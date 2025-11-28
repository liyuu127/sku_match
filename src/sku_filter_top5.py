# sku第一阶段过滤
import asyncio
from typing import List

import pandas as pd
from pandas import Series

from utils import BRAND_DICTIONARY, extract_brand, clean_text, chinese_tokenize

MAX_CONCURRENCY = 100


def calculate_brand_similarity(brand1: str, brand2: str) -> float:
    """品牌相似度"""
    brand1 = brand1.strip().lower()
    brand2 = brand2.strip().lower()
    if not brand1 or not brand2:
        return 0.5
    if brand1 == brand2:
        return 1.0
    return 0.0


def jaccard_similarity(set1: set, set2: set) -> float:
    """分词相似度"""
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0.0


def calculate_barcode_similarity(barcode1: str, barcode2: str) -> float:
    """条码相似度"""
    # 判断是否为空
    if not barcode1 or not barcode2:
        return 0.0
    return 1.0 if barcode1 == barcode2 else 0.0


async def calculate_similarity_for_single_product(
        row: Series, p_tokenize: set, p_brand: str,
        candidate_row: pd.Series, candidate_tokens: set
) -> tuple[float, Series]:
    """计算单个商品间相似度"""
    candidate_id = candidate_row.商品ID
    candidate_name = candidate_row.商品名称
    # p_barcode = row.条码
    # candidate_barcode = candidate_row.条码
    # # 条码是否匹配
    # similarity_barcode = calculate_barcode_similarity(p_barcode, candidate_barcode)
    # if similarity_barcode == 1.0:
    #     return 1.0, candidate_row
    candidate_brand = extract_brand(candidate_name, BRAND_DICTIONARY)
    similarity_tokens = jaccard_similarity(p_tokenize, candidate_tokens)
    similarity_brand = calculate_brand_similarity(p_brand, candidate_brand)

    # 品牌不匹配直接过滤
    similarity = similarity_tokens
    if similarity_brand == 0.0:
        similarity = 0.0
    return similarity, candidate_row


async def find_top5_similar_combined_async(row: pd.Series, candidate_df: pd.DataFrame,
                                           candidate_tokens_list: List[set],
                                           sem: asyncio.Semaphore,
                                           top_n=5):
    """"批量返回多个最相似的商品"""
    async with sem:
        product_name = row.商品名称
        price = row.原价
        p_name = clean_text(product_name)
        p_tokenize = await chinese_tokenize(p_name)
        p_brand = extract_brand(p_name, BRAND_DICTIONARY)

        tasks = []
        for idx, candidate_row in enumerate(candidate_df.itertuples(index=False)):
            candidate_tokens = candidate_tokens_list[idx]
            tasks.append(calculate_similarity_for_single_product(
                row, p_tokenize, p_brand, candidate_row, candidate_tokens
            ))

        results = await asyncio.gather(*tasks)
        # 如果相似度为1.0，直接返回第一个相似度为1.0的记录，top2 top2填充为空
        if any(r[0] == 1.0 for r in results):
            top3_combined = [r[1] for r in results if r[0] == 1.0]
        else:
            # 过滤相似度==0.0的记录
            results = [r for r in results if r[0] > 0.0]
            sorted_results = sorted(results, key=lambda x: x[0], reverse=True)

            top3_combined = [r[1] for r in sorted_results[:top_n]]
            # 过滤top3价格上下超出price 9倍以上的项
            top3_combined = [r for r in top3_combined if abs(price - r.原价) < price * 9]

        while len(top3_combined) < top_n:
            top3_combined.append(None)

        return tuple(top3_combined)


async def preprocess_candidate_tokens(candidate: pd.DataFrame) -> List[set]:
    """异步批量预处理分词"""
    tasks = [chinese_tokenize(clean_text(name)) for name in candidate['商品名称']]
    return await asyncio.gather(*tasks)


async def process_batch(owner_df: pd.DataFrame, ele_df: pd.DataFrame, tokens: List[set],
                        start_idx: int, batch_size: int, sem: asyncio.Semaphore):
    """批量处理匹配"""
    top1_list, top2_list, top3_list, top4_list, top5_list = [], [], [], [], []
    end_idx = min(start_idx + batch_size, len(owner_df))
    batch_df = owner_df.iloc[start_idx:end_idx]

    tasks = []
    for row in batch_df.itertuples(index=False):
        tasks.append(find_top5_similar_combined_async(row, ele_df, tokens, sem))

    batch_results = await asyncio.gather(*tasks)
    for t1, t2, t3, t4, t5 in batch_results:
        top1_list.append(t1)
        top2_list.append(t2)
        top3_list.append(t3)
        top4_list.append(t4)
        top5_list.append(t5)

    print(f"已完成批次：{start_idx + 1}~{end_idx}")
    return top1_list, top2_list, top3_list, top4_list, top5_list


async def process_owner_data_async(owner_df: pd.DataFrame, ele_df: pd.DataFrame, tokens: List[set],
                                   batch_size: int = 100):
    """分批次处理任务"""
    total_items = len(owner_df)
    num_batches = (total_items + batch_size - 1) // batch_size
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    print(f"开始处理数据，共{num_batches}批次，每批{batch_size}条。")

    tasks = [
        process_batch(owner_df, ele_df, tokens, i * batch_size, batch_size, sem)
        for i in range(num_batches)
    ]

    batch_results = await asyncio.gather(*tasks)
    top1_list, top2_list, top3_list, top4_list, top5_list = [], [], [], [], []
    for b1, b2, b3, b4, b5 in batch_results:
        top1_list.extend(b1)
        top2_list.extend(b2)
        top3_list.extend(b3)
        top4_list.extend(b4)
        top5_list.extend(b5)
    return top1_list, top2_list, top3_list, top4_list, top5_list
