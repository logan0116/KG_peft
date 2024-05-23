# gguf将会是我们后续测试的主要版本
# 这里形成一个针对gguf的简化版本


import torch
from llama_cpp import Llama

from tqdm import tqdm
import json
import time
import os

# logging
import logging

# acc llm
from qa_eval import acc_llm

# args
from parser import parameter_parser

# logging config
logging.basicConfig(
    format="Logan233: %(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)


def model_load(model_type, model_path):
    """
    en:Load the model
    zh:加载模型

    :param model_type: str, model type
        llama3-gguf-q4_0
        llama3-gguf-q8_0
        qwen1.5-gguf-q4_0
        qwen1.5-gguf-q8_0


    :param model_path: str, model path

    """
    # model
    if model_type == 'llama3-gguf-q4_0':
        ckpt_dir = os.path.join(model_path, 'Meta-Llama-3-8B-Instruct.Q4_0.gguf')
    elif model_type == 'llama3-gguf-q8_0':
        ckpt_dir = os.path.join(model_path, 'Meta-Llama-3-8B-Instruct.Q8_0.gguf')
    elif model_type == 'qwen1.5-gguf-q4_0':
        ckpt_dir = os.path.join(model_path, 'qwen1_5-7b-chat-q4_0.gguf')
    elif model_type == 'qwen1.5-gguf-lora-q4_0':
        ckpt_dir = os.path.join(model_path, 'qwen1_5-7b-chat-lora-q4_0.gguf')
    else:
        raise ValueError("model_type not currently supported")
    model = Llama(
        model_path=ckpt_dir,
        n_gpu_layers=32,  # Uncomment to use GPU acceleration
        n_ctx=3072,  # Uncomment to increase the context window
        verbose=False
    )

    return model


def data_process(data_set):
    """
    en:Data processing
    zh:数据处理
    """
    # load data
    # save_path = 'data/{}_qa.json'.format(data_set)
    # with open(save_path, 'w', encoding='utf-8') as f:
    #     json.dump(data, f, ensure_ascii=False, indent=4)

    with open('data/{}_qa.json'.format(data_set), 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    data_list = data_list[:1000]

    prompt_list = []
    label_list = []

    # for eval
    question_list = []
    context_list = []

    for data in data_list:
        prompt_list.append('Question: ' + data['question'] +
                           '\nContent: ' + data['context'])
        label_list.append(data['answer'])
        # for eval
        question_list.append(data['question'])
        context_list.append(data['context'])

    return prompt_list, label_list, question_list, context_list


def qa_eval(model_type, model_path, data_set):
    """
    en:Evaluation
    zh:评估
    """
    # load data
    prompts, labels, question_list, context_list = data_process(data_set)
    # load model
    model = model_load(model_type=model_type, model_path=model_path)

    logging.info("Start evaluating..." + model_type)

    predicts = []
    start_time = time.time()
    for prompt in tqdm(prompts):
        history = [
            {"role": "system",
             "content": 'For "Question", please answer according to "Content".' +
                        ' (Note: please keep answers simple and clear.)\n'
             },
            {"role": "user",
             "content": prompt
             }
        ]
        output = model.create_chat_completion(messages=history, max_tokens=1024, temperature=0.7)
        output = output['choices'][0]['message']['content'].strip()
        predicts.append(output)
    end_time = time.time()

    logging.info("Time: %s" % (end_time - start_time))
    # save label and predict
    with open('data/qa_result_{}.json'.format(model_type), 'w', encoding='utf-8') as f:
        data_list = [{'question': question, 'context': context, 'label': label, 'predict': predict}
                     for question, context, label, predict in zip(question_list, context_list, labels, predicts)]
        json.dump(data_list, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # qa_eval("qwen1.5-gguf-q4_0", "../model", "test")
    acc_llm(model_type="qwen1.5-gguf-q4_0")
    # qa_eval("qwen1.5-gguf-q8_0", "../model", "test")
    # acc_llm(model_type="qwen1.5-gguf-q8_0")
    # qa_eval("llama3-gguf-q4_0", "../model", "test")
    # acc_llm(model_type="llama3-gguf-q4_0")
    # qa_eval("qwen1.5-gguf-lora-q4_0", "../model", "test")
    acc_llm(model_type="qwen1.5-gguf-lora-q4_0")
