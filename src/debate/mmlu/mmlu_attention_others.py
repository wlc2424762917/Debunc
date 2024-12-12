import argparse
import json
import os
import numpy as np
import torch
from debate.gen_utils import (
    Debate,
    RWJSONEncoder,
    construct_assistant_message,
    generate_answer_uncertainty,
)
from debate.mmlu.common import (
    construct_message_attention_others,
)
from lm_polygraph.estimators import MeanTokenEntropy
from models.model import WhiteboxModel
from tqdm import trange
from transformers import AutoTokenizer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a debate simulation with specified parameters.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Path or name of the model to use (default: meta-llama/Meta-Llama-3-8B-Instruct)",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=3,
        help="Number of agents in the debate (default: 3)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of rounds in the debate (default: 3)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of trials to run (default: 5)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing the question files (default: data)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save the result JSON files (default: results)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    model_name = args.model_name
    agents = args.agents
    rounds = args.rounds
    trials = args.trials
    data_dir = args.data_dir
    output_dir = args.output_dir

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = WhiteboxModel.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    ue_method = MeanTokenEntropy()

    for num_shots in [0, 5]:
        # Load the question file
        question_file = f"{data_dir}/qas_{num_shots}_shot.json"
        if not os.path.exists(question_file):
            raise FileNotFoundError(f"Question file not found: {question_file}")
        questions = json.load(open(question_file))

        # Construct the output filename
        filename = f"{output_dir}/{os.path.basename(__file__)[:-3]}_{agents}_{rounds}_{trials}_{num_shots}_{ue_method.__class__.__name__}.json"

        all_trial_data = []
        current_trial = 0

        for trial in trange(trials, desc="Trials"):
            current_question = 0
            response_dict = {}
            all_trial_data.append(response_dict)

            for q_i in trange(
                current_question,
                len(questions),
                initial=current_question,
                total=len(questions),
                desc="Questions",
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
                            message = construct_message_attention_others(
                                this_agent=agent_context,
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

                    response_dict[question] = (agent_contexts, answer)
                    all_trial_data[-1] = response_dict

                    # Save the results incrementally
                    json.dump(
                        all_trial_data,
                        open(filename, "w"),
                        cls=RWJSONEncoder,
                    )

"""
    CUDA_VISIBLE_DEVICES=3,4,5 python ./mmlu_attention_others.py --model_name=/data/hf_models/Llama-3.1-8B-Instruct --agents=3 --rounds=3 --trials=5 --data_dir=data --output_dir=reimplementation_results
"""