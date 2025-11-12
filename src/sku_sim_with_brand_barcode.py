import pandas as pd
import jieba
from typing import List, Tuple, Dict
import numpy as np

def load_brands(brand_path: str) -> List[str]:
    """
    加载品牌名称库，去除每行末尾的英文逗号并去重
    :param brand_path: 品牌文件路径
    :return: 排序后的品牌列表（按长度降序，避免短品牌被长品牌覆盖）
    """
    with open(brand_path, 'r', encoding='utf-8') as f:
        # 读取并处理：去除换行符、末尾英文逗号，过滤空行
        brands = [line.strip().rstrip(',') for line in f if line.strip()]
    # 去重后按品牌长度降序排序（优先匹配长品牌，避免"可乐"覆盖"可口可乐"）
    unique_brands = list(set(brands))
    return sorted(unique_brands, key=lambda x: len(x), reverse=True)

def split_product_name(name: str, brands: List[str]) -> Tuple[set, str]:
    """
    商品名称分词（优先保留品牌，不拆分品牌名词）
    :param name: 商品名称
    :param brands: 品牌列表
    :return: (分词集合, 提取到的品牌名称)
    """
    if pd.isna(name) or name.strip() == "":
        return set(), ""
    
    name_clean = name.strip()
    extracted_brand = ""
    
    # 步骤1：提取商品名称中的品牌（精确匹配）
    for brand in brands:
        if brand in name_clean:
            extracted_brand = brand
            # 移除品牌后的剩余文本（避免重复分词）
            name_clean = name_clean.replace(brand, "").strip()
            break
    
    # 步骤2：对剩余文本进行分词（使用jieba默认分词）
    remaining_words = jieba.lcut(name_clean) if name_clean else []
    
    # 步骤3：合并品牌和分词结果，去重（品牌作为独立分词）
    final_words = set(remaining_words)
    if extracted_brand:
        final_words.add(extracted_brand)
    
    return final_words, extracted_brand

def calculate_jaccard_similarity(
    barcode1: str, words1: set, brand1: str,
    barcode2: str, words2: set, brand2: str
) -> float:
    """
    计算Jaccard相似度，增加条码匹配和品牌一致性双重校验
    :param barcode1: 商品1的条码
    :param words1: 商品1的分词集合
    :param brand1: 商品1的品牌
    :param barcode2: 商品2的条码
    :param words2: 商品2的分词集合
    :param brand2: 商品2的品牌
    :return: 相似度（0-1）
    """
    # 规则1：两个商品条码相同 → 相似度直接为1（最高优先级）
    if pd.notna(barcode1) and pd.notna(barcode2):
        # 处理条码可能的格式差异（如字符串前后空格、数字转字符串）
        barcode1_clean = str(barcode1).strip()
        barcode2_clean = str(barcode2).strip()
        if barcode1_clean and barcode2_clean and barcode1_clean == barcode2_clean:
            return 1.0
    
    # 规则2：两个商品都有品牌且品牌不同 → 相似度直接为0（次高优先级）
    if brand1 and brand2 and brand1 != brand2:
        return 0.0
    
    # 规则3：计算Jaccard相似度（基础规则）
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union != 0 else 0.0

def find_top3_similar_products(
    target_product: Dict, 
    candidate_products: List[Dict],
    brands: List[str]
) -> List[Dict]:
    """
    为目标商品找到候选商品中相似度Top3的商品
    :param target_product: 目标商品（来自sku_store1）
    :param candidate_products: 候选商品列表（来自sku_store2）
    :param brands: 品牌列表
    :return: Top3相似商品（按相似度降序排列，不足3个则补空）
    """
    # 目标商品的核心信息
    target_barcode = target_product["条码"]
    target_words = target_product["words"]
    target_brand = target_product["brand"]
    
    # 计算与所有候选商品的相似度
    similarity_scores = []
    for candidate in candidate_products:
        score = calculate_jaccard_similarity(
            target_barcode, target_words, target_brand,
            candidate["条码"], candidate["words"], candidate["brand"]
        )
        similarity_scores.append((score, candidate))
    
    # 按相似度降序排序（分数相同则按商品ID升序，保证稳定性；条码相同的商品优先）
    similarity_scores.sort(
        key=lambda x: (-x[0], 
                      # 条码相同的商品在分数相同时排序更靠前
                      1 if (pd.notna(x[1]["条码"]) and pd.notna(target_barcode) and 
                            str(x[1]["条码"]).strip() == str(target_barcode).strip()) else 0,
                      x[1]["商品ID"])
    )
    
    # 取Top3，不足3个则用空字典填充
    top3 = []
    for i in range(3):
        if i < len(similarity_scores):
            top3.append(similarity_scores[i][1])
        else:
            # 补空（字段与商品结构一致）
            top3.append({
                "商品ID": "", "商品名称": "", "条码": "",
                "店内一级分类": "", "图片": ""
            })
    
    return top3

def main():
    # ---------------------- 配置参数 ----------------------
    store1_path = "sku_store1.xlsx"       # 源文件1路径
    store2_path = "sku_store2.xlsx"       # 源文件2路径
    brand_path = "brand.txt"              # 品牌文件路径
    output_path = "sku_store1_with_top3.xlsx"  # 输出文件路径
    # ------------------------------------------------------
    
    # 步骤1：加载数据和品牌库
    print("正在加载数据...")
    df1 = pd.read_excel(store1_path)
    df2 = pd.read_excel(store2_path)
    brands = load_brands(brand_path)
    
    # 校验Excel字段是否完整（修正：原需求是4个字段，但实际列出5个，以实际字段为准）
    required_columns = ["商品ID", "商品名称", "条码", "店内一级分类", "图片"]
    for df, df_name in [(df1, "sku_store1"), (df2, "sku_store2")]:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"{df_name}.xlsx 缺少必要字段：{','.join(missing_cols)}")
    
    # 步骤2：预处理商品名称（分词+提取品牌）
    print("正在预处理商品名称...")
    # 处理sku_store1：添加分词和品牌列
    df1["words"] = ""
    df1["brand"] = ""
    for idx, row in df1.iterrows():
        words, brand = split_product_name(row["商品名称"], brands)
        df1.at[idx, "words"] = words
        df1.at[idx, "brand"] = brand
    
    # 处理sku_store2：转换为字典列表（包含分词和品牌），方便遍历
    store2_products = []
    for idx, row in df2.iterrows():
        words, brand = split_product_name(row["商品名称"], brands)
        store2_products.append({
            "商品ID": row["商品ID"],
            "商品名称": row["商品名称"],
            "条码": row["条码"],
            "店内一级分类": row["店内一级分类"],
            "图片": row["图片"],
            "words": words,
            "brand": brand
        })
    
    # 步骤3：为每个sku_store1商品匹配Top3相似商品
    print("正在匹配相似商品...")
    # 定义输出列名（原5列 + Top1-3各5列，共20列）
    output_columns = required_columns.copy()
    for i in range(1, 4):
        output_columns.extend([
            f"Top{i}商品ID", f"Top{i}商品名称", f"Top{i}条码",
            f"Top{i}店内一级分类", f"Top{i}图片"
        ])
    
    # 创建结果DataFrame，初始化Top1-3列为空
    result_df = df1[required_columns].copy()
    for col in output_columns[len(required_columns):]:
        result_df[col] = ""
    
    # 逐个商品匹配
    total_products = len(result_df)
    for idx, row in result_df.iterrows():
        # 构建目标商品信息（包含条码、分词、品牌）
        target_product = {
            "条码": row["条码"],
            "words": df1.at[idx, "words"],
            "brand": df1.at[idx, "brand"]
        }
        
        # 找到Top3相似商品
        top3_products = find_top3_similar_products(
            target_product, store2_products, brands
        )
        
        # 将Top3商品字段写入结果
        for i, product in enumerate(top3_products, 1):
            prefix = f"Top{i}"
            result_df.at[idx, f"{prefix}商品ID"] = product["商品ID"]
            result_df.at[idx, f"{prefix}商品名称"] = product["商品名称"]
            result_df.at[idx, f"{prefix}条码"] = product["条码"]
            result_df.at[idx, f"{prefix}店内一级分类"] = product["店内一级分类"]
            result_df.at[idx, f"{prefix}图片"] = product["图片"]
        
        # 打印进度（每100个商品输出一次）
        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{total_products} 个商品")
    
    # 步骤4：保存结果到Excel
    result_df.to_excel(output_path, index=False, engine="openpyxl")
    print(f"\n处理完成！结果已保存到：{output_path}")
    print(f"最终文件字段数：{len(result_df.columns)}（预期20列，实际：{len(result_df.columns)}）")
    print(f"处理商品总数：{total_products}")

if __name__ == "__main__":
    # 安装依赖提示（如果运行报错，先执行以下命令）
    # pip install pandas jieba openpyxl numpy
    try:
        main()
    except Exception as e:
        print(f"程序运行出错：{str(e)}")