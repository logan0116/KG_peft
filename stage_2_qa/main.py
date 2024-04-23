import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM
from awq import AutoAWQForCausalLM
from llama_cpp import Llama

from tqdm import tqdm
import json
import time

# logging
import logging

# args
from parser import parameter_parser

# logging config
logging.basicConfig(
    format="Logan233: %(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)


def load(ckpt_dir, model_type):
    """
    en:Load the model
    zh:加载模型
    """
    # Setting `pad_token_id` to `eos_token_id`:151645 for open-end generation.
    # A decoder-only architecture is being used, but right-padding was detected!
    # For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat", padding_side="left")

    if model_type == 'gptq':
        model = AutoGPTQForCausalLM.from_quantized(ckpt_dir)
        logging.info("GPTQ Model loaded from %s" % ckpt_dir)
    elif model_type == 'awq':
        model = AutoAWQForCausalLM.from_pretrained(ckpt_dir)
        logging.info("AWQ Model loaded from %s" % ckpt_dir)
    elif model_type == 'gguf':
        model = Llama(
            model_path=ckpt_dir,
            n_gpu_layers=32,  # Uncomment to use GPU acceleration
            n_ctx=3072,  # Uncomment to increase the context window
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir,
                                                     torch_dtype=torch.float16,
                                                     trust_remote_code=True)

    if model_type != 'gguf':
        # cuda
        model = model.to('cuda')
        model.eval()

    return model, tokenizer


def data_process(data_set, part=None):
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

    if part == 0:
        data_list = data_list[:1000]
    elif part == 1:
        data_list = data_list[1000:2000]

    prompt_list = []
    label_list = []

    # for eval
    question_list = []
    context_list = []

    for data in data_list:
        prompt_list.append('For [question], please answer according to [context]. ' +
                           '(Note: please keep answers simple and clear.)\n' +
                           '[question] ' + data['question'] + '\n' +
                           '[context] ' + data['context'] + '\n')
        label_list.append(data['answer'])
        # for eval
        question_list.append(data['question'])
        context_list.append(data['context'])

    return prompt_list, label_list, question_list, context_list


def qa_eval(ckpt_dir, model_type, data_set, part=None):
    """
    en:Evaluation
    zh:评估
    """
    # load data
    prompts, labels, question_list, context_list = data_process(data_set, part=part)
    # load model
    model, tokenizer = load(ckpt_dir, model_type)

    logging.info("Start evaluating..." + ckpt_dir)

    predicts = []
    start_time = time.time()
    for prompt in tqdm(prompts):
        if model_type == 'gguf':
            output = model(prompt)
            output = output['choices'][0]['text'].strip()
            print(output)
            predicts.append(output)
        else:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {name: tensor.to('cuda') for name, tensor in inputs.items()}
            output = model.generate(**inputs, max_length=512, pad_token_id=tokenizer.eos_token_id)
            output = tokenizer.decode(output[0], skip_special_tokens=True)
            predicts.append(output)

    end_time = time.time()

    # calculate accuracy
    # correct_count = 0
    # for label, predict in zip(labels, predicts):
    #     if label in predict:
    #         correct_count += 1
    #
    # accuracy = correct_count / len(labels)
    # logging.info("Accuracy: %s" % accuracy)
    logging.info("Time: %s" % (end_time - start_time))
    # save label and predict
    with open('data/qa_{}_result.json'.format(model_type), 'w', encoding='utf-8') as f:
        data_list = [{'question': question, 'context': context, 'label': label, 'predict': predict}
                     for question, context, label, predict in zip(question_list, context_list, labels, predicts)]
        json.dump(data_list, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    args = parameter_parser()
    # qa_eval('../quantization_test/Qwen1.5-7B-Chat-gptq-4bit-128gr-desc-act/',
    #         'gptq', 'test', part=args.part)

    # qa_eval('../quantization_test/Qwen1.5-7B-Chat-gptq-4bit-128gr-no-desc-act/',
    #         'gptq', 'test')
    #
    # qa_eval('../quantization_test/Qwen1.5-7B-Chat-awq-4bit-128gr/',
    #         'awq', 'test')
    #
    # qa_eval('../quantization_test/Qwen1.5-7B-Chat-q4_0.gguf',
    #         'gguf', 'test', part=args.part)
    #
    qa_eval('../quantization_test/Qwen1.5-7B-Chat-awq-q4_0.gguf',
            'gguf', 'test', part=args.part)
