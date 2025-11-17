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

from pandas import Series

from Limter import RateLimiter
from llm_match import process_llm_row

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
LLM_MATCH = True


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


# 计算条码字段是否一致，如果一致相似度为1
def calculate_barcode_similarity(barcode1: str, barcode2: str) -> float:
    # 判断是否为空
    if not barcode1 or not barcode2:
        return 0.0
    return 1.0 if barcode1 == barcode2 else 0.0


# ====================================================
# 相似度计算
# ====================================================
async def calculate_similarity_for_single_product(
        row, p_tokenize: set, p_brand: str,
        candidate_row: pd.Series, candidate_tokens: set
) -> tuple[float, Series]:
    candidate_id = candidate_row.商品ID
    candidate_name = candidate_row.商品名称
    p_barcode = row.条码
    candidate_barcode = candidate_row.条码
    similarity_barcode = calculate_barcode_similarity(p_barcode, candidate_barcode)
    if similarity_barcode == 1.0:
        return 1.0, candidate_row
    candidate_brand = extract_brand_from_tokens(candidate_name, BRAND_DICTIONARY)
    similarity_tokens = jaccard_similarity(p_tokenize, candidate_tokens)
    similarity_brand = calculate_brand_similarity(p_brand, candidate_brand)

    # similarity = similarity_tokens * 0.7 + similarity_brand * 0.3
    similarity = similarity_tokens
    if similarity_brand == 0.0:
        similarity = 0.0

    result_str = f"{candidate_id}-{candidate_name}--{similarity:.4f}--{similarity_tokens:.4f}--{similarity_brand:.4f}"
    return similarity, candidate_row


async def find_top3_similar_combined_async(row, candidate_df: pd.DataFrame,
                                           candidate_tokens_list: List[set],
                                           sem: asyncio.Semaphore):
    async with sem:
        product_name = row.商品名称
        price = row.原价
        p_name = clean_text(product_name)
        p_tokenize = await chinese_tokenize(p_name)
        p_brand = extract_brand_from_tokens(p_name, BRAND_DICTIONARY)

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

            top3_combined = [r[1] for r in sorted_results[:3]]
            # 过滤top3价格上下超出price 5倍以上的项
            top3_combined = [r for r in top3_combined if abs(price - r.原价) < price * 5]

        while len(top3_combined) < 3:
            top3_combined.append(None)

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
        tasks.append(find_top3_similar_combined_async(row, ele_df, tokens, sem))

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


# ====================================================
# 主函数入口
# ====================================================
async def main():
    print("开始加载数据...")
    start_time = time.time()
    # ele_df = pd.read_csv("../data/elme_sku_small.csv")
    # owner_df = pd.read_csv("../data/meituan_sku_small.csv")

    # owner_df = load_excel("../data/美团-快驿点特价超市(虹桥店)全量商品信息20251109.xlsx").iloc[:1000]
    # target_df = load_excel("../data/美团-邻侣超市（虹桥中心店）全量商品信息20251109.xlsx").iloc[:1000]
    # owner_df = load_excel("../data/美团-快驿点特价超市(虹桥店)全量商品信息20251109.xlsx")
    # target_df = load_excel("../data/美团-邻侣超市（虹桥中心店）全量商品信息20251109.xlsx")
    owner_df = load_excel("../data/sku_kyd_sampled.xlsx")
    target_df = load_excel("../data/sku_ll_dedup.xlsx")
    # 打印前几行数据
    print(owner_df.head())
    print(f"待匹配数据加载完成，共{len(owner_df)}条记录")
    print(target_df.head())
    print(f"匹配目标数据加载完成，共{len(target_df)}条记录")

    tokens = await preprocess_candidate_tokens(target_df)
    print("饿了么数据预处理完成")

    print("开始异步处理匹配任务...")
    top1_list, top2_list, top3_list = await process_owner_data_async(owner_df, target_df, tokens)

    owner_df['相似商品1（ID-名称）'] = top1_list
    owner_df['相似商品2（ID-名称）'] = top2_list
    owner_df['相似商品3（ID-名称）'] = top3_list

    if LLM_MATCH:
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
        for i, val in all_results:
            owner_df.at[i, '相似商品'] = val

    output_path = "../output/补充Top3相似商品" + str(uuid.uuid4()) + ".xlsx"
    await save_excel_async(owner_df, output_path)

    end_time = time.time()
    print(f"\n处理完成！结果已保存到：{output_path}")
    print(f"总耗时：{end_time - start_time:.2f}秒")


if __name__ == '__main__':
    asyncio.run(main())
