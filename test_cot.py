#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

from src.model import CODI, DataArguments, ModelArguments, TrainingArguments


@dataclass
class COTEvalArguments:
    max_new_tokens: int = field(default=256, metadata={"help": "Maximum generated CoT plus answer tokens."})
    temperature: float = field(default=0.1, metadata={"help": "Sampling temperature."})
    top_k: int = field(default=40, metadata={"help": "Top-k sampling cutoff. Ignored for greedy decoding."})
    top_p: float = field(default=0.95, metadata={"help": "Top-p sampling cutoff. Ignored for greedy decoding."})
    print_outputs: bool = field(default=True, metadata={"help": "Print each decoded CoT output."})
    prediction_path: Optional[str] = field(default=None, metadata={"help": "Optional JSONL path for predictions."})
    append_cot_prompt: bool = field(
        default=False,
        metadata={"help": "Append an explicit step-by-step instruction to each question before generation."},
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def build_lora_config(model_args: ModelArguments) -> LoraConfig:
    task_type = TaskType.CAUSAL_LM
    model_name = model_args.model_name_or_path.lower()
    if any(name in model_name for name in ["llama", "mistral", "falcon", "qwen"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    elif "phi" in model_name:
        target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    elif "gpt2" in model_name:
        target_modules = ["c_attn", "c_proj", "c_fc"]
    else:
        raise ValueError(f"Unsupported model for LoRA target modules: {model_args.model_name_or_path}")

    return LoraConfig(
        task_type=task_type,
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights=True,
    )


def load_codi_model(model_args: ModelArguments, training_args: TrainingArguments) -> CODI:
    if not model_args.lora_init:
        raise NotImplementedError("Evaluation currently expects --lora_init, matching test.py.")
    if not model_args.ckpt_dir:
        raise ValueError("--ckpt_dir is required for explicit CoT evaluation.")

    model = CODI(model_args, training_args, build_lora_config(model_args))
    safetensor_path = os.path.join(model_args.ckpt_dir, "model.safetensors")
    bin_path = os.path.join(model_args.ckpt_dir, "pytorch_model.bin")
    if os.path.exists(safetensor_path):
        state_dict = load_file(safetensor_path, device="cpu")
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {model_args.ckpt_dir}")

    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()

    if device.type == "cuda" and training_args.bf16:
        model = model.to(device=device, dtype=torch.bfloat16)
    elif device.type == "cuda" and training_args.fp16:
        model = model.to(device=device, dtype=torch.float16)
    else:
        model = model.to(device)
    model.eval()
    return model


def load_tokenizer(model_args: ModelArguments, training_args: TrainingArguments, model: CODI):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    return tokenizer


def load_eval_examples(data_name: str) -> Tuple[List[str], List[object]]:
    question_name = "question"
    answer_name = "answer"
    if data_name == "gsm-hard":
        dataset = load_dataset("juyoung-trl/gsm-hard")
        test_set = dataset["train"]
        question_name = "instruction"
        answer_name = "response"
    elif data_name == "multi-arith":
        dataset = load_dataset("ChilleD/MultiArith")
        test_set = dataset["test"]
        answer_name = "final_ans"
    elif data_name == "svamp":
        dataset = load_dataset("ChilleD/SVAMP")
        test_set = concatenate_datasets([dataset["train"], dataset["test"]])
        question_name = "question_concat"
        answer_name = "Answer"
    elif data_name == "commonsense":
        dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")
        test_set = dataset["validation"]
    elif data_name == "gsm8k":
        dataset = load_dataset("gsm8k", "main")
        test_set = dataset["test"]
    else:
        raise NotImplementedError(f"Dataset {data_name} is not supported.")

    questions = [f"{example[question_name].strip().replace('  ', ' ')}" for example in test_set]
    answers = [normalize_gold_answer(example[answer_name]) for example in test_set]
    return questions, answers


def normalize_gold_answer(answer):
    if isinstance(answer, bool):
        return answer
    if answer is None:
        return float("inf")
    if isinstance(answer, (int, float)):
        return float(answer)
    answer = str(answer).strip()
    if answer in ["True", "False"]:
        return answer == "True"
    if answer in "ABCDE":
        return answer
    if "####" in answer:
        answer = answer.split("####")[-1]
    answer = answer.replace(",", "")
    try:
        return float(answer)
    except ValueError:
        return float("inf")


def format_cot_prompt(question: str, eos_token: Optional[str], remove_eos: bool, append_cot_prompt: bool) -> str:
    prompt = question
    if append_cot_prompt:
        prompt += "\nAnswer the above question. First think step by step and then answer the final number.\n"
    if not remove_eos and eos_token is not None:
        prompt += eos_token
    return prompt


def extract_answer(sentence: str, data_name: str):
    sentence = sentence.replace(",", "")
    if "commonsense" in data_name:
        tail = sentence.split("The answer is:")[-1].strip()
        match = re.search(r"\b([ABCDE])\b", tail)
        return match.group(1) if match else "C"

    if "strategy" in data_name or "prontoqa" in data_name.lower():
        if "True" in sentence:
            return True
        if "False" in sentence:
            return False
        return None

    numbers = re.findall(r"-?\d+\.?\d*", sentence)
    if not numbers:
        return float("inf")
    return float(numbers[-1])


def compute_accuracy(gold: List[object], pred: List[object]) -> float:
    correct = 0.0
    for pred_answer, gold_answer in zip(pred, gold):
        if pred_answer == gold_answer:
            correct += 1
    return correct / len(gold)


def generation_kwargs(tokenizer, model: CODI, cot_args: COTEvalArguments, training_args: TrainingArguments) -> Dict:
    kwargs = {
        "max_new_tokens": cot_args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bad_words_ids": [[model.pad_token_id], [model.bot_id], [model.eot_id]],
    }
    if training_args.greedy:
        kwargs["do_sample"] = False
    else:
        kwargs.update(
            {
                "do_sample": True,
                "temperature": cot_args.temperature,
                "top_k": cot_args.top_k,
                "top_p": cot_args.top_p,
            }
        )
    return kwargs


def evaluate(model_args, data_args, training_args, cot_args):
    transformers.set_seed(training_args.seed)
    model = load_codi_model(model_args, training_args)
    tokenizer = load_tokenizer(model_args, training_args, model)

    logging.warning("Downloading Data")
    questions, gold_answers = load_eval_examples(data_args.data_name)
    prompts = [
        format_cot_prompt(q, tokenizer.eos_token, training_args.remove_eos, cot_args.append_cot_prompt)
        for q in questions
    ]

    eval_step = math.ceil(len(prompts) / data_args.batch_size)
    logging.warning(
        f"Total example: {len(prompts)} | eval batch size: {data_args.batch_size} | eval steps: {eval_step}"
    )

    pred_answers = []
    output_lengths = []
    prediction_file = None
    if cot_args.prediction_path:
        os.makedirs(os.path.dirname(cot_args.prediction_path) or ".", exist_ok=True)
        prediction_file = open(cot_args.prediction_path, "w", encoding="utf-8")

    try:
        for step in range(eval_step):
            start = step * data_args.batch_size
            end = min((step + 1) * data_args.batch_size, len(prompts))
            batch_prompts = prompts[start:end]
            batch = tokenizer(batch_prompts, return_tensors="pt", padding="longest").to(device)

            with torch.no_grad():
                generated = model.codi.generate(
                    **batch,
                    **generation_kwargs(tokenizer, model, cot_args, training_args),
                )

            prompt_len = batch["input_ids"].shape[1]
            generated_only = generated[:, prompt_len:]
            decoded_outputs = tokenizer.batch_decode(generated_only, skip_special_tokens=True)

            for offset, decoded in enumerate(decoded_outputs):
                idx = start + offset
                pred_answer = extract_answer(decoded, data_args.data_name)
                pred_answers.append(pred_answer)
                generated_tokens = generated_only[offset]
                eos_positions = (generated_tokens == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                output_len = eos_positions[0].item() + 1 if len(eos_positions) > 0 else len(generated_tokens)
                output_lengths.append(output_len)

                if cot_args.print_outputs:
                    print(f"Question {idx} Starts...")
                    print(f"Q: {questions[idx]}")
                    print(decoded)
                    print(f"Question {idx} Ends")
                    print(f"Prediction={pred_answer}; Groundtruth={gold_answers[idx]}")
                    print("")

                if prediction_file is not None:
                    prediction_file.write(
                        json.dumps(
                            {
                                "index": idx,
                                "question": questions[idx],
                                "generation": decoded,
                                "prediction": pred_answer,
                                "gold": gold_answers[idx],
                                "correct": pred_answer == gold_answers[idx],
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
    finally:
        if prediction_file is not None:
            prediction_file.close()

    accuracy = compute_accuracy(gold_answers, pred_answers)
    print(f"adapter: {model_args.adapter_name_or_path} | {data_args.data_name} explicit CoT accuracy: {100 * accuracy:.2f}% |")
    print(f"average generated length: {sum(output_lengths) / len(output_lengths):.2f}")
    return 100 * accuracy


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, COTEvalArguments))
    model_args, data_args, training_args, cot_args = parser.parse_args_into_dataclasses()

    accuracy_list = []
    for _ in range(training_args.inf_num_iterations):
        accuracy_list.append(evaluate(model_args, data_args, training_args, cot_args))
    print(f"Average accuracy over {training_args.inf_num_iterations} sampling: {sum(accuracy_list) / len(accuracy_list)}")
