import math

import pandas as pd
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

# 读取brand.txt解析为列表
BRAND_DICTIONARY = set(line.strip() for line in open("../data/brand.txt", encoding="utf-8"))
print("已加载品牌列表：", BRAND_DICTIONARY)
jieba.load_userdict("../data/jieba_dict.txt")
print("已加载自定义分词词典：", jieba.get_dict_file())


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


def clean_text(text) -> str:
    """清理文本，只保留中文、英文、数字和常用符号"""
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


def extract_full_spec(text) -> str:
    """"提取规格信息"""
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


def extract_brand(text_to_search: str, brand_dictionary: Set[str]) -> str:
    """提取品牌信息"""
    if not text_to_search or text_to_search.isspace() or not brand_dictionary:
        return ""
    for brand_info in brand_dictionary:
        if brand_info in text_to_search:
            return brand_info
    return ""


def pandas_str_to_series(s) -> pd.Series:
    """字符串转为Series"""
    # 判断是否已经是 Series
    if not isinstance(s, str):
        return s
    # s为None或""或nan
    if s is None or s == "" or (isinstance(s, float) and math.isnan(s)):
        return None

    inner = s[s.find("(") + 1: s.rfind(")")]

    pattern = re.compile(r"(\w+)=('[^']*'|[^,]*)")
    data = {k: v.strip("'") if v.strip() != "nan" else None for k, v in pattern.findall(inner)}

    # 3️⃣ 转为 DataFrame
    df = pd.DataFrame([data])
    return df.iloc[0]
