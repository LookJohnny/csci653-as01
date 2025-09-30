from datasets import load_dataset
import pandas as pd, pyarrow as pa, pyarrow.dataset as ds
import json, pathlib, numpy as np, hashlib

# 1) 载入 parent_asin -> main_category 映射（一次性内存字典，或分块广播）
asin2cat = json.load(open("asin2category.json"))  # 官方文件
def get_cat(pid): return asin2cat.get(pid, "Unknown")

# 2) 遍历所有 review 配置（不要硬编码三类）
# Amazon Reviews 2023 dataset categories
ALL_CONFIGS = [
    "raw_review_All_Beauty",
    "raw_review_Amazon_Fashion",
    "raw_review_Appliances",
    "raw_review_Arts_Crafts_and_Sewing",
    "raw_review_Automotive",
    "raw_review_Baby_Products",
    "raw_review_Beauty_and_Personal_Care",
    "raw_review_Books",
    "raw_review_CDs_and_Vinyl",
    "raw_review_Cell_Phones_and_Accessories",
    "raw_review_Clothing_Shoes_and_Jewelry",
    "raw_review_Digital_Music",
    "raw_review_Electronics",
    "raw_review_Gift_Cards",
    "raw_review_Grocery_and_Gourmet_Food",
    "raw_review_Handmade_Products",
    "raw_review_Health_and_Household",
    "raw_review_Health_and_Personal_Care",
    "raw_review_Home_and_Kitchen",
    "raw_review_Industrial_and_Scientific",
    "raw_review_Kindle_Store",
    "raw_review_Magazine_Subscriptions",
    "raw_review_Movies_and_TV",
    "raw_review_Musical_Instruments",
    "raw_review_Office_Products",
    "raw_review_Patio_Lawn_and_Garden",
    "raw_review_Pet_Supplies",
    "raw_review_Software",
    "raw_review_Sports_and_Outdoors",
    "raw_review_Subscription_Boxes",
    "raw_review_Tools_and_Home_Improvement",
    "raw_review_Toys_and_Games",
    "raw_review_Video_Games",
]
OUT = pathlib.Path("/scratch/$USER/darkhorse/bronze/reviews")  # Hadoop/并行FS更佳
S = 64  # 哈希分片数

def shard_id(pid):  # 稳定哈希
    return int(hashlib.md5(pid.encode()).hexdigest(), 16) % S

for cfg in ALL_CONFIGS:
    ds_cfg = load_dataset("McAuley-Lab/Amazon-Reviews-2023", cfg,
                          split="full", trust_remote_code=True)
    # 建议批处理/流式map，避免一次性落入内存
    for batch in ds_cfg.to_iterable_dataset(batch_size=200_000):
        df = batch.to_pandas()[["parent_asin","user_id","timestamp",
                                "verified_purchase","helpful_vote"]]
        # 统一到秒；并取周起始（UTC 周一）
        is_ms = (df["timestamp"].astype("int64") > 10**12).all()
        ts = pd.to_datetime(df["timestamp"].astype("int64")//(1000 if is_ms else 1),
                            unit="s", utc=True)
        df["week"] = ts.dt.to_period("W-MON").dt.start_time
        # 映射类目（不用在输入上分类，后面再groupby）
        df["main_category"] = df["parent_asin"].map(get_cat).fillna("Unknown")
        # 哈希分片
        df["shard"] = df["parent_asin"].map(lambda x: shard_id(x))
        # 分区落盘：week + shard（便于下游选择性读取）
        for sid, dsub in df.groupby("shard"):
            outdir = OUT / f"week={dsub['week'].iloc[0].date()}" / f"shard={sid:02d}"
            outdir.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pandas(dsub, preserve_index=False)
            ds.write_dataset(table, outdir, format="parquet", existing_data_behavior="overwrite_or_ignore")
