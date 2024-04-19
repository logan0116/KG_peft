#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/1/31 上午8:39
# @Author  : liu yuhan
# @FileName: utils.py
# @Software: PyCharm

import torch
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

import os
import json
from tqdm import tqdm
from transformers import BertTokenizer, BertForQuestionAnswering


def data_process4qa(data_set="train"):
    """
    SQuAD 2.0
    :param data_set:
    :param tokenizer_model:
    :return:
    """
    if data_set == "train":
        data_path = 'data/train-v2.0.json'
    elif data_set == "test":
        data_path = 'data/dev-v2.0.json'
    else:
        raise ValueError("data_set should be 'train' or 'test'")

    print('start data process4qa:', data_set)

    # load data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data_list = data['data']

    question_list = []
    context_list = []
    answer_list = []

    for data in tqdm(data_list):
        paragraphs = data['paragraphs']
        for paragraph in paragraphs:
            qas = paragraph['qas']
            context = paragraph['context']
            for qa in qas:
                question = qa['question']
                answers = qa['answers']

                for answer in answers:
                    question_list.append(question)
                    context_list.append(context)
                    answer_list.append(answer['text'])

    data = [{"question": question, "context": context, "answer": answer}
            for question, context, answer in zip(question_list, context_list, answer_list)]

    print("data length:", len(data))

    # save data
    save_path = 'data/{}_qa.json'.format(data_set)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    data_process4qa("train")
    data_process4qa("test")


