from datasets import load_dataset

ds = load_dataset("cais/mmlu", "all", cache_dir="/data/hf_models/mmlu/")
print(ds)

import json

# 假设 `ds` 是已经加载好的数据集
# 获取 `train` 集的总数据条数
total_rows = len(ds['test'])
print(len(ds['test']))
# 计算前 1/10 的行数
subset_size = max(1, total_rows // 10)

# 打乱数据集
shuffled_ds = ds['test'].shuffle(seed=42)  # 设置 seed 保证结果可复现

# 提取前 1/10 的数据
subset_data = shuffled_ds[:subset_size]
# print(subset_data)

question_prefix = "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering. \n\n"

question_answer_pairs = []
for i in range(total_rows // 10):
    question_answer_pair_new = {}
    cur_question = subset_data["question"][i]
    cur_choices = subset_data["choices"][i]
    cur_answer = subset_data["answer"][i]
    # print(cur_question)
    # print(cur_choices)
    # print(cur_answer)
    # print(question_prefix + cur_question + "\n")
    new_choices = ""
    for idx, choice in enumerate(cur_choices):
        new_choices += f"{chr(65 + idx)}. {choice}\n"
    # print(new_choices)
    question_new = question_prefix + cur_question + "\n" + new_choices
    # print(question_new)
    answer_new = f"{chr(65 + cur_answer)}"
    # print(answer_new)
    question_answer_pair_new["question"] = question_new
    question_answer_pair_new["answer"] = answer_new
    print(question_answer_pair_new)
    question_answer_pairs.append(question_answer_pair_new)
    # quit()

output_file = "subset_data_mmlu.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(question_answer_pairs, f, indent=4, ensure_ascii=False)

print(f"前 1/10 的数据已保存至 {output_file}")


