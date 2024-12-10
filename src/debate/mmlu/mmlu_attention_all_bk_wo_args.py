import json
import numpy as np
import torch
from debate.gen_utils import (
    Debate,
    RWJSONEncoder,
    construct_assistant_message,
    generate_answer_uncertainty,
)
from debate.mmlu.common import (
    construct_message_attention_all,
)
from lm_polygraph.estimators import MeanTokenEntropy, TokenSAR
from models.model import WhiteboxModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from lm_polygraph.utils.generation_parameters import GenerationParameters
from tqdm import trange
import os
import time
os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
# model_name = "/data/hf_models/Mistral-7B-Instruct"
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name = "/data/hf_models/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Loading model {model_name}")

config = AutoConfig.from_pretrained(model_name)
config.max_position_embeddings = 10000  # 修改最大位置嵌入

model = WhiteboxModel.from_pretrained(
    model_name,
    device_map="auto",
    # torch_dtype=torch.bfloat16,
    config=config,
)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
# model_llama = LlamaForCausalLM.from_pretrained(model_name, device_map="auto")
# print(model_llama)
#
# tokenizer = AutoTokenizer.from_pretrained(
#             model_name,
#             device_map="auto"
#         )
#
# model_llama.eval()
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
# generation_params = GenerationParameters({})
# model = WhiteboxModel(
#     model_llama, tokenizer, model_name, "CausalLM", generation_params
# )

# quit()
model_name_sim = model_name.split("/")[-1]

ue_method = MeanTokenEntropy()

if __name__ == "__main__":
    agents = 4
    rounds = 3
    trials = 1
    print(f"saving to results/{os.path.basename(__file__)[:-3]}_model_name_sim_{model_name_sim}_{agents}_{rounds}_{trials}_{ue_method.__class__.__name__}.json")
    for num_shots in [0]:
        questions = json.load(open("/home/wanglichao/debunc/src/subset_data_mmlu.json"))
        filename = f"results/{os.path.basename(__file__)[:-3]}_model_name_sim_{model_name_sim}_{agents}_{rounds}_{trials}_{num_shots}_{ue_method.__class__.__name__}.json"

        print(f"start agent: {agents}, rounds: {rounds}, trials: {trials}, num_shots: {num_shots}")
        all_trial_data = []
        current_trial = 0
        start_time = time.time()
        dump_time = 0
        for trial in trange(trials):
            current_question = 0
            response_dict = {}
            all_trial_data.append(response_dict)

            for q_i in trange(
                current_question,
                len(questions),
                initial=current_question,
                total=len(questions),
            ):
                q_data = questions[q_i]
                question = q_data["question"]
                answer = q_data["answer"]
                agent_contexts: Debate = [
                    [{"role": "user", "content": question}] for agent in range(agents)
                ]

                for round in range(rounds):
                    torch.cuda.empty_cache()
                    confidences = None
                    if round != 0:
                        uncertainties = []
                        for agent in agent_contexts:
                            agent = agent[-1]
                            uncertainties.append(agent["uncertainty"])
                        confidences = 1 / np.array(uncertainties)
                    for i, agent_context in enumerate(agent_contexts):
                        if confidences is not None:
                            agent_contexts_other = (
                                agent_contexts[:i] + agent_contexts[i + 1 :]
                            )
                            other_confidences = np.concatenate(
                                (confidences[:i], confidences[i + 1 :])
                            )
                            message = construct_message_attention_all(
                                this_agent=agent_context,
                                this_confidence=confidences[i],
                                other_agents=agent_contexts_other,
                                other_confidences=other_confidences,
                                conv_idx=2 * round - 1,
                                tokenizer=tokenizer,
                            )
                            agent_context.append(message)

                        completion, uncertainty = generate_answer_uncertainty(
                            agent_context, model, tokenizer, ue_method
                        )

                        assistant_message = construct_assistant_message(completion)
                        assistant_message["uncertainty"] = uncertainty
                        agent_context.append(assistant_message)

                    tmp_dump_time_start = time.time()
                    response_dict[question] = (agent_contexts, answer)
                    all_trial_data[-1] = response_dict
                    json.dump(
                        all_trial_data,
                        open(filename, "w"),
                        cls=RWJSONEncoder,
                    )
                    tmp_dump_time_end = time.time()
                    tmp_dump_time = tmp_dump_time_end - tmp_dump_time_start
                    dump_time += tmp_dump_time

        end_time = time.time()
        print(f"Time taken: {end_time - start_time}")
        ## hours, minutes, seconds
        print(f"Time taken: {(end_time - start_time) // 3600}h {((end_time - start_time) % 3600) // 60}m {((end_time - start_time) % 3600) % 60}s")
        print(f"Dump time: {dump_time}")
        print(f"Total inference time: {end_time - start_time - dump_time}")
        print(f"Total inference time: {(end_time - start_time - dump_time) // 3600}h {((end_time - start_time - dump_time) % 3600) // 60}m {((end_time - start_time - dump_time) % 3600) % 60}s")