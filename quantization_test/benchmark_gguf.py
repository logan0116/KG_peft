# LLM benchmark ARC HellaSwag MMLU TruthfulQA

import json
import os
import time
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Tuple
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import tensor_parallel as tp
from auto_gptq import AutoGPTQForCausalLM
from awq import AutoAWQForCausalLM
import accelerate
from llama_cpp import Llama

# logging
import logging

# logging config
logging.basicConfig(
    format="Logan233: %(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

TASKS = [
    'abstract_algebra',
    'anatomy',
    'astronomy',
    'business_ethics',
    'clinical_knowledge',
    'college_biology',
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_medicine',
    'college_physics',
    'computer_security',
    'conceptual_physics',
    'econometrics',
    'electrical_engineering',
    'elementary_mathematics',
    'formal_logic',
    'global_facts',
    'high_school_biology',
    'high_school_chemistry',
    'high_school_computer_science',
    'high_school_european_history',
    'high_school_geography',
    'high_school_government_and_politics',
    'high_school_macroeconomics',
    'high_school_mathematics',
    'high_school_microeconomics',
    'high_school_physics',
    'high_school_psychology',
    'high_school_statistics',
    'high_school_us_history',
    'high_school_world_history',
    'human_aging',
    'human_sexuality',
    'international_law',
    'jurisprudence',
    'logical_fallacies',
    'machine_learning',
    'management',
    'marketing',
    'medical_genetics',
    'miscellaneous',
    'moral_disputes',
    'moral_scenarios',
    'nutrition',
    'philosophy',
    'prehistory',
    'professional_accounting',
    'professional_law',
    'professional_medicine',
    'professional_psychology',
    'public_relations',
    'security_studies',
    'sociology',
    'us_foreign_policy',
    'virology',
    'world_religions']

choices = ["A", "B", "C", "D"]


def compute_metric(output_filename):
    """
    en:Compute the accuracy for each task and the overall accuracy
    zh:计算每个任务的准确率和总体准确率
    """
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    total_acc = 0
    total_num = 0
    for task in run_results:
        acc = 0
        pred_answers = run_results[task]['pred_answers']
        gold_answers = run_results[task]['gold_answers']
        for pred, gold in zip(pred_answers, gold_answers):
            if pred == gold: acc += 1
        print("ACC-%s: %.4f" % (task, acc / len(gold_answers)))
        total_acc += acc
        total_num += len(gold_answers)
    print("ACC-all: %.4f" % (total_acc / total_num))


def format_subject(subject):
    """
    en:Format the subject name
    zh:格式化科目名称
    """
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    """
    en:Format the example
    zh:格式化例子
    """
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    """
    en:Generate the prompt
    zh:生成提示
    """
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


# def custom_stopping_criteria(input_ids, score, **kwargs):
#     stop_ids = [29871, 13, 13] # \n\n
#     return input_ids[-len(stop_ids)]

def prepare_input(tokenizer, prompts):
    """
    en:Prepare the input
    zh:准备输入
    """
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')

    return input_tokens


def load(ckpt_dir):
    """
    en:Load the model
    zh:加载模型
    """

    model = Llama(
        model_path=ckpt_dir,
        n_gpu_layers=100,  # Uncomment to use GPU acceleration
        n_ctx=2304,  # Uncomment to increase the context window
    )

    return model


def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts


def batch_infer(model, prompts):
    """
    en:Batch inference
    zh:批量推理
    :param model:
    :param prompts:


    gguf mode example

    output = llm(
          "Q: Name the planets in the solar system? A: ", # Prompt
          max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
          stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
          echo=True # Echo the prompt back in the output
    ) # Generate a completion, can also call create_completion

    output

    {
      "id": "cmpl-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
      "object": "text_completion",
      "created": 1679561337,
      "model": "./models/7B/llama-model.gguf",
      "choices": [
        {
          "text": "Q: Name the planets in the solar system? A: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune and Pluto.",
          "index": 0,
          "logprobs": None,
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 14,
        "completion_tokens": 28,
        "total_tokens": 42
      }
    }

    """

    batch_size = 1
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        for prompt in batch_input:
            output = model(prompt, max_tokens=32, stop=["Q:", "\n"])
            output = output['choices'][0]['text'].strip()
            if output == '':
                output = ' '
            answers.append(output)
    answers = [answer[-1] for answer in answers]
    return answers


def main(ckpt_dir: str, param_size: str):
    run_results = {}
    output_filename = 'run_results_qwen_1.5_gguf_%sb.json' % (param_size)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat", padding_side="left")

    model = load(ckpt_dir)
    start_time = time.time()
    for task in TASKS:
        print('Testing %s ...' % task)
        records = []
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", task + "_dev.csv"), header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", task + "_test.csv"), header=None)
        for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            k = args.ntrain
            prompt_end = format_example(test_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, task, k)
            prompt = train_prompt + prompt_end
            while len(tokenizer.tokenize(prompt)) + 1 > 2048:  # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = '\n\n'.join(prompt_split)
            label = test_df.iloc[i, test_df.shape[1] - 1]
            records.append({'prompt': prompt, 'answer': label})

        pred_answers = batch_infer(model, [record['prompt'] for record in records])
        gold_answers = [record['answer'] for record in records]
        run_results[task] = {'pred_answers': pred_answers, 'gold_answers': gold_answers}
    with open(output_filename, 'w') as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)

    compute_metric(output_filename)
    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--param_size', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--ntrain', type=int, default=5)
    args = parser.parse_args()

    main(args.ckpt_dir, args.param_size)
    # compute_metric('run_results_qwen_1.5_gguf_{}b.json'.format(args.param_size))