## Llama-3.1-8B-Instruct
python mmlu_attention_all.py --model_name /data/hf_models/Llama-3.1-8B-Instruct --agents 4 --rounds 3

## Llama-3-8B-Instruct
python mmlu_attention_all.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --agents 3 --rounds 3

## Mistral-7B-Instruct
python mmlu_attention_all.py --model_name /data/hf_models/Mistral-7B-Instruct --agents 4 --rounds 3


#### 2024_12_10 ####
CUDA_VISIBLE_DEVICES=0,1,2 python mmlu_attention_all.py --model_name meta-llama/Meta-Llama-3-8B-Instruct --agents 3 --rounds 3 --output_dir ./reimplementation_results --data_path ./data/qas_0_shot.json

CUDA_VISIBLE_DEVICES=3,4,5 python mmlu_attention_all.py --model_name /data/hf_models/Llama-3.1-8B-Instruct --agents 3 --rounds 3 --output_dir ./reimplementation_results --data_path ./data/qas_0_shot.json
