import json
import os
import torch
from debate.gen_utils import (
    Debate,
    construct_assistant_message,
    generate_answer_standard,
)
from debate.mmlu.common import (
    construct_message_standard,
)
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name = "/data/hf_models/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)


if __name__ == "__main__":
    agents = 3
    rounds = 3
    trials = 5
    model_name_sim = model_name.split("/")[-1]
    for num_shots in [0]:
        questions = json.load(open(f"data/qas_{num_shots}_shot.json"))
        filename = f"reimplementation_results/{os.path.basename(__file__)[:-3]}_model_name_sim_{model_name_sim}_{agents}_{rounds}_{trials}_{num_shots}_transformer_442.json"
        print(f"saving to {filename}")
        all_trial_data = []
        current_trial = 0

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
                    for i, agent_context in enumerate(agent_contexts):
                        if round != 0:
                            agent_contexts_other = (
                                agent_contexts[:i] + agent_contexts[i + 1 :]
                            )

                            message = construct_message_standard(
                                other_agents=agent_contexts_other,
                                conv_idx=2 * round - 1,
                            )
                            agent_context.append(message)

                        completion = generate_answer_standard(
                            agent_context, model, tokenizer
                        )

                        assistant_message = construct_assistant_message(completion)
                        agent_context.append(assistant_message)

                    response_dict[question] = (agent_contexts, answer)
                    all_trial_data[-1] = response_dict
                    json.dump(
                        all_trial_data,
                        open(filename, "w"),
                    )
