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
from llm_match_v2 import llm_match_fill
from sku_filter import preprocess_candidate_tokens, process_owner_data_async
from utils import load_excel, save_excel_async

MAX_CONCURRENCY = 100  # 控制最大并发任务数
LLM_MATCH = True


async def main():
    print("开始加载数据...")
    start_time = time.time()
    # ele_df = pd.read_csv("../data/elme_sku_small.csv")
    # owner_df = pd.read_csv("../data/meituan_sku_small.csv")

    # owner_df = load_excel("../data/美团-快驿点特价超市(虹桥店)全量商品信息20251109.xlsx").iloc[:1000]
    # target_df = load_excel("../data/美团-邻侣超市（虹桥中心店）全量商品信息20251109.xlsx").iloc[:1000]
    # owner_df = load_excel("../data/美团-快驿点特价超市(虹桥店)全量商品信息20251109.xlsx")
    # target_df = load_excel("../data/美团-邻侣超市（虹桥中心店）全量商品信息20251109.xlsx")
    # owner_df = load_excel("../data/sku_kyd_sampled.xlsx").iloc[:100]
    owner_df = load_excel("../output/门店美团相同的品.xlsx")
    target_df = load_excel("../data/美团-邻侣超市（虹桥中心店）全量商品信息20251109.xlsx")
    target_df = target_df.drop_duplicates(subset=['商品ID'])
    # 打印前几行数据
    print(owner_df.head())
    print(f"待匹配数据加载完成，共{len(owner_df)}条记录")
    print(target_df.head())
    print(f"匹配目标数据加载完成，共{len(target_df)}条记录")

    tokens = await preprocess_candidate_tokens(target_df)
    print("饿了么数据预处理完成")

    print("开始异步处理匹配任务...")
    top1_list, top2_list, top3_list = await process_owner_data_async(owner_df, target_df, tokens)

    owner_df['相似商品1'] = top1_list
    owner_df['相似商品2'] = top2_list
    owner_df['相似商品3'] = top3_list

    if LLM_MATCH:
        await llm_match_fill(owner_df, top1_list, top2_list, top3_list)

    output_path = "../output/top3相似_GLM4_9B" + str(uuid.uuid4()) + ".xlsx"
    await save_excel_async(owner_df, output_path)

    end_time = time.time()
    print(f"\n处理完成！结果已保存到：{output_path}")
    print(f"总耗时：{end_time - start_time:.2f}秒")


if __name__ == '__main__':
    asyncio.run(main())
