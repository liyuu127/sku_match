import re
import asyncio
import pandas as pd
from typing import List, Set, Tuple
import jieba
from concurrent.futures import ThreadPoolExecutor
import time

# 全局线程池，用于CPU密集型任务如jieba分词
thread_pool = ThreadPoolExecutor()

def clean_text(text):
    """
    清洗文本，去除特殊字符和多余空格
    """
    if not isinstance(text, str):
        return ""
    # 移除非中文字符、字母和数字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    # 将多个空格合并为一个
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


import jieba

jieba.load_userdict("./jieba_dict.txt")

def chinese_tokenize(text: str) -> set:
    """中文精确分词，返回去重后的分词集合（适配Jaccard相似度计算）"""
    if not text or text.isspace():
        return set()
    return set(jieba.cut(text, cut_all=False))


def extract_full_spec_and_clean(text):
    """
    从商品名称中提取完整规格串，并返回去除规格后的文本。

    返回:
        spec, cleaned_text
    示例:
        输入:  "加热鼠标垫...8033cm 饼干熊款一个"
        输出:  ("8033cm", "加热鼠标垫... 饼干熊款一个")
    """
    if not isinstance(text, str):
        return ""

    # 单位字典（小写匹配）
    units = [
        "盒", "箱", "件", "个", "袋", "包", "瓶", "罐", "卷", "片", "只",
        "ml", "l", "g", "kg", "mm", "cm", "m"
    ]
    # 包装单位（可扩展）
    pack_units = ["盒", "箱", "件", "个", "袋", "包", "瓶", "罐", "卷", "片", "只", "条", "支"]
    units_pattern = "|".join(re.escape(u) for u in units)
    pack_pattern = "|".join(re.escape(p) for p in pack_units)

    pattern = re.compile(
        rf'((?:\d+(?:\.\d+)?\s?(?:{units_pattern}|{pack_pattern}))'
        rf'(?:[*_×xX/\\-]?\s?\d*(?:\.\d+)?\s?(?:{units_pattern}|{pack_pattern}))*)',
        flags=re.IGNORECASE
    )

    matches = pattern.findall(text)
    if not matches:
        return ""

    # 取最长匹配作为规格串
    specs = sorted(matches, key=len, reverse=True)
    spec = specs[0].strip()

    return spec


from typing import List, Set


# 提取品牌 如输入特仑苏 有机纯牛奶 250ml10盒_箱 返回特仑苏
def extract_brand_from_tokens(text_to_search: str, brand_dictionary: Set[str]) -> str:
    """
    从jieba分词后的词语列表中，根据品牌词典提取品牌名。

    该算法遵循“最长匹配”原则，优先匹配由更多词组成的品牌名。

    参数:
    - tokenized_words (List[str]): jieba.lcut() 分词后得到的词语列表。
    - brand_dictionary (Set[str]): 包含所有已知品牌名的集合，便于快速查找。

    返回:
    - str: 匹配到的品牌名。如果没有匹配到，则返回 "未知"。
    """
    if not text_to_search or text_to_search.isspace() or not brand_dictionary:
        return ""

    for brand_info in brand_dictionary:
        # 直接在拼接后的字符串中查找，这比复杂的列表滑动窗口更高效简洁
        if brand_info in text_to_search:
            return brand_info  # 找到第一个（也是最长的）匹配项，立即返回

    return ""


BRAND_DICTIONARY = {
    "海氏海诺",
    "爱斐堡",
    "汤达人",
    "康师傅",
    "上好佳",
    "海天",
    "老干妈",
    "可口可乐",
    "特仑苏",
    "华为",
    "华为荣耀",
    "味全"
}


# 获取商品名称字段的品牌/规格/分词三元组
def get_product_info(product_name: str) -> tuple:
    spec = extract_full_spec_and_clean(product_name)
    brand = extract_brand_from_tokens(clean_text(product_name), BRAND_DICTIONARY)
    tokenize = chinese_tokenize(clean_text(product_name))
    return brand, spec, tokenize


def calculate_brand_similarity(brand1: str, brand2: str) -> float:
    """
    计算两个品牌字符串的相似度。
    - 完全相同: 1.0
    - 任意一个未知: 0.5
    - 完全不同或双方: 0.0
    """
    brand1 = brand1.strip().lower()
    brand2 = brand2.strip().lower()

    if not brand1 or not brand2:
        return 0.5
    if brand1 == brand2:
        return 1.0

    return 0.0


def jaccard_similarity(set1: set, set2: set) -> float:
    """计算Jaccard相似度：交集大小 / 并集大小（取值0-1，越大越相似）"""
    if not set1 and not set2:
        return 1.0  # 两个空字符串视为完全相似
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0.0


from typing import Tuple
import pandas as pd

WEIGHTS = {
    'similarity_tokens': 0.7,  # 品牌权重占 60%
    'similarity_brand': 0.3,  # 商品名权重占 40%
}


def calculate_similarity_for_single_product(p_name: str, p_tokenize: set, p_brand: str,
                                                  candidate_row: pd.Series, candidate_tokens: set) -> Tuple[float, str]:
    """异步计算单个商品的相似度"""
    candidate_id = candidate_row['商品ID']
    candidate_name = candidate_row['商品名称']
    candidate_brand = extract_brand_from_tokens(candidate_name, BRAND_DICTIONARY)

    similarity_tokens = jaccard_similarity(p_tokenize, candidate_tokens)
    similarity_brand = calculate_brand_similarity(p_brand, candidate_brand)
    similarity = similarity_tokens * WEIGHTS['similarity_tokens'] + similarity_brand * WEIGHTS['similarity_brand']

    result_str = f"{candidate_id}-{candidate_name}--{similarity}--{similarity_tokens}--{similarity_brand}"
    return similarity, result_str


async def find_top3_similar_combined_async(product_name: str, candidate_df: pd.DataFrame,
                                           candidate_tokens_list: List[set]) -> Tuple[str, str, str]:
    """异步版本的相似度匹配函数"""
    p_name = clean_text(product_name)
    # 使用线程池执行CPU密集型分词任务
    p_tokenize = await asyncio.get_event_loop().run_in_executor(thread_pool, chinese_tokenize, p_name)
    p_brand = extract_brand_from_tokens(p_name, BRAND_DICTIONARY)
    loop = asyncio.get_event_loop()
    # 创建任务列表
    tasks = []
    for idx in range(len(candidate_df)):
        candidate_row = candidate_df.iloc[idx]
        candidate_tokens = candidate_tokens_list[idx]
        tasks.append(loop.run_in_executor(
            thread_pool, calculate_similarity_for_single_product,
            p_name, p_tokenize, p_brand, candidate_row, candidate_tokens
        ))
    # 并发执行所有相似度计算任务
    results = await asyncio.gather(*tasks)

    # 按相似度降序排序
    sorted_results = sorted(results, key=lambda x: x[0], reverse=True)

    # 提取Top3组合，不足3个时用空字符串填充
    top3_combined = [result[1] for result in sorted_results[:3]]
    while len(top3_combined) < 3:
        top3_combined.append("")  # 补空确保始终返回3个元素

    return top3_combined[0], top3_combined[1], top3_combined[2]


def preprocess_candidate_tokens(candidate: pd.DataFrame) -> List[set]:
    return [chinese_tokenize(clean_text(name)) for name in candidate['商品名称']]


# %%
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


async def process_batch(owner_df: pd.DataFrame, ele_df: pd.DataFrame, tokens: List[set], start_idx: int,
                        batch_size: int) -> Tuple[List[str], List[str], List[str]]:
    """处理一批商品的异步函数"""
    top1_list = []
    top2_list = []
    top3_list = []

    end_idx = min(start_idx + batch_size, len(owner_df))
    batch_df = owner_df.iloc[start_idx:end_idx]
    for idx, row in batch_df.iterrows():
        product_name = row['商品名称']
        # 异步获取3个"ID-名称"组合（分别对应Top1、Top2、Top3）
        top1, top2, top3 = await find_top3_similar_combined_async(product_name, ele_df, tokens)
        top1_list.append(top1)
        top2_list.append(top2)
        top3_list.append(top3)

        print("正在处理：", product_name)
    print(f"已处理商品：{start_idx + 1}~{end_idx}")


    return top1_list, top2_list, top3_list


async def process_owner_data_async(owner_df: pd.DataFrame, ele_df: pd.DataFrame, tokens: List[set],
                                   batch_size: int = 50) -> Tuple[List[str], List[str], List[str]]:
    """异步处理所有商品数据"""
    total_items = len(owner_df)
    num_batches = (total_items + batch_size - 1) // batch_size  # 向上取整

    print(f"开始处理数据，共{num_batches}批次")
    print(f"每批次处理{batch_size}个商品")
    # 创建所有批次的异步任务
    tasks = []
    for i in range(num_batches):
        start_idx = i * batch_size
        print(f"处理批次{i + 1}/{num_batches}")
        task = process_batch(owner_df, ele_df, tokens, start_idx, batch_size)
        tasks.append(task)

    print("开始处理批次数据...")
    # 并发处理所有批次
    batch_results = await asyncio.gather(*tasks)

    # 合并所有批次的结果
    top1_list = []
    top2_list = []
    top3_list = []

    for batch_top1, batch_top2, batch_top3 in batch_results:
        top1_list.extend(batch_top1)
        top2_list.extend(batch_top2)
        top3_list.extend(batch_top3)

    return top1_list, top2_list, top3_list

if __name__ == '__main__':
    print("开始加载数据...")
    start_time = time.time()

    # 加载数据
    ele_df = load_excel("./饿了么-京东便利店（虹桥中心店）全量商品数据20251110.xlsx")
    print(f"饿了么数据加载完成，共{len(ele_df)}条记录")

    tokens = preprocess_candidate_tokens(ele_df)
    print("饿了么数据预处理完成")

    owner_df = load_excel("./美团-快驿点特价超市(虹桥店)全量商品信息20251109.xlsx")
    print(f"美团数据加载完成，共{len(owner_df)}条记录")

    # 异步处理数据
    print("开始异步处理商品匹配...")
    # top1_list, top2_list, top3_list = asyncio.run(process_owner_data_async(owner_df, ele_df, tokens))
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    top1_list, top2_list, top3_list = loop.run_until_complete(
        process_owner_data_async(owner_df, ele_df, tokens)
    )
    loop.close()
    # 将结果添加到DataFrame
    owner_df['相似商品1（ID-名称）'] = top1_list  # 第三列：最相似
    owner_df['相似商品2（ID-名称）'] = top2_list  # 第四列：次相似
    owner_df['相似商品3（ID-名称）'] = top3_list  # 第五列：第三相似

    # 保存结果
    output_path = "liyu附件1_补充Top3相似商品（ID-名称组合）.xlsx"
    owner_df.to_excel(output_path, index=False, engine='openpyxl')

    end_time = time.time()
    print(f"\n处理完成！结果已保存到：{output_path}")
    print(f"总耗时：{end_time - start_time:.2f}秒")
    print("列说明：")
    print(" - 第三列：相似商品1（ID-名称）→ 最相似商品")
    print(" - 第四列：相似商品2（ID-名称）→ 次相似商品")
    print(" - 第五列：相似商品3（ID-名称）→ 第三相似商品")
    print("注：不足3个相似商品时，对应列为空字符串")

    # 关闭线程池
    thread_pool.shutdown(wait=True)
