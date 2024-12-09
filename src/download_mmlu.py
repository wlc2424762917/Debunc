from datasets import load_dataset

ds = load_dataset("cais/mmlu", "all", cache_dir="/data/hf_models/mmlu/")
print(ds)

import json

# 假设 `ds` 是已经加载好的数据集
# 获取 `train` 集的总数据条数
total_rows = len(ds['test'])

# 计算前 1/10 的行数
subset_size = max(1, total_rows // 10)

# 打乱数据集
shuffled_ds = ds['test'].shuffle(seed=42)  # 设置 seed 保证结果可复现

# 提取前 1/10 的数据
subset_data = shuffled_ds['train'][:subset_size]

question_prefix = "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering."


# 保存为目标格式的 JSON 文件
formatted_data = [
    {
        "question": question_prefix + example["question"] + "\n\n" + example["choices"],
        "answer": example["answer"]
    }
    for example in subset_data
]

# 保存到文件
output_file = "subset_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, indent=4, ensure_ascii=False)

print(f"前 1/10 的数据已保存至 {output_file}")


# 将训练集转为 Pandas DataFrame
# train_df = pd.DataFrame(ds['test'][:100])  # 取前 100 行数据
# # print(train_df[0])  # 查看前几行
#
#
# # from datasets import load_dataset
# #
# # ds_ab = load_dataset("cais/mmlu", "abstract_algebra", cache_dir="/data/hf_models/mmlu_abstract_alg/")
# # print(ds_ab)