from datasets import load_dataset

ds = load_dataset("cais/mmlu", "all", cache_dir="/data/hf_models/mmlu/")
print(ds)

from datasets import load_dataset

ds_ab = load_dataset("cais/mmlu", "abstract_algebra", cache_dir="/data/hf_models/mmlu_abstract_alg/")
print(ds_ab)