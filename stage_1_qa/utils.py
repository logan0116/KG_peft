#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/1/31 上午8:39
# @Author  : liu yuhan
# @FileName: utils.py
# @Software: PyCharm

import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

import os
import json
from tqdm import tqdm
from transformers import BertTokenizer, BertForQuestionAnswering


def get_dataset_size(dataset_size, cut_rate):
    train_size = int(cut_rate * dataset_size)
    test_size = dataset_size - train_size
    return train_size, test_size


def data_process4qa(data_set="train", tokenizer_model='bert-base-cased'):
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

    # load data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data_list = data['data']

    question_list = []
    text_list = []
    target_start_index_list = []
    target_end_index_list = []

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
                    text_list.append(context)
                    target_start_index_list.append(answer['answer_start'])
                    target_end_index_list.append(answer['answer_start'] + len(answer['text']) - 1)

    data_list = zip(question_list, text_list, target_start_index_list, target_end_index_list)
    # 保证每一项非空
    data_list = [data for data in data_list if data[0] and data[1] and data[2] and data[3]]

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_model)
    model = BertForQuestionAnswering.from_pretrained(tokenizer_model)
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    start_positions_list = []
    end_positions_list = []

    bad_count = 0

    for question, text, start_index, end_index in tqdm(data_list):
        inputs = tokenizer(question, text, padding='max_length', truncation=True, max_length=512)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        # get answer from text
        answer = text[start_index: end_index + 1]
        answer_input_ids = tokenizer.encode(answer, add_special_tokens=False)
        # get start_positions and end_positions by trans to text
        input_ids_str = ' ' + ' '.join([str(i) for i in input_ids]) + ' '
        answer_str = ' ' + ' '.join([str(i) for i in answer_input_ids]) + ' '
        # start_positions = num of space before answer
        input_ids_str_before_answer = input_ids_str.split(answer_str)[0]
        start_positions = input_ids_str_before_answer.count(' ')
        end_positions = start_positions + len(answer_input_ids) - 1
        # max_length check
        if end_positions >= 512:
            continue
        # test
        answer_decode = tokenizer.decode(input_ids[start_positions:end_positions + 1])
        answer = answer.replace(' ', '')
        answer_decode = answer_decode.replace(' ', '')
        if answer != answer_decode:
            print('*****************************error:{}*********************************'.format(bad_count))
            bad_count += 1
            print(answer, ' | ', answer_decode)
            continue
        # add
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        token_type_ids_list.append(token_type_ids)
        start_positions_list.append(start_positions)
        end_positions_list.append(end_positions)

    # trans 2 tensor
    input_ids_list = torch.tensor(input_ids_list, dtype=torch.int)
    attention_mask_list = torch.tensor(attention_mask_list, dtype=torch.int)
    token_type_ids_list = torch.tensor(token_type_ids_list, dtype=torch.int)
    start_positions_list = torch.tensor(start_positions_list, dtype=torch.int)
    end_positions_list = torch.tensor(end_positions_list, dtype=torch.int)

    # save data
    torch.save(input_ids_list, 'data/' + data_set + '_input_ids_list.pth')
    torch.save(attention_mask_list, 'data/' + data_set + '_attention_mask_list.pth')
    torch.save(token_type_ids_list, 'data/' + data_set + '_token_type_ids_list.pth')
    torch.save(start_positions_list, 'data/' + data_set + '_start_positions_list.pth')
    torch.save(end_positions_list, 'data/' + data_set + '_end_positions_list.pth')

    print(data_set + " data process done!")


def data_load(data_set="train"):
    """
    data load
    """
    input_ids_list = torch.load('data/' + data_set + '_input_ids_list.pth')
    attention_mask_list = torch.load('data/' + data_set + '_attention_mask_list.pth')
    token_type_ids_list = torch.load('data/' + data_set + '_token_type_ids_list.pth')
    start_positions_list = torch.load('data/' + data_set + '_start_positions_list.pth')
    end_positions_list = torch.load('data/' + data_set + '_end_positions_list.pth')

    return MyDataSet(input_ids_list, attention_mask_list, token_type_ids_list,
                     start_positions_list, end_positions_list)


# loss曲线绘制
# def loss_draw(epochs, loss_list):
#     plt.plot([i + 1 for i in range(epochs)], loss_list)


class MyDataSet(Data.Dataset):
    """
    data load
    """

    def __init__(self, input_ids_list, attention_mask_list, token_type_ids_list,
                 start_positions_list, end_positions_list):
        self.input_ids_list = input_ids_list
        self.attention_mask = attention_mask_list
        self.token_type_ids = token_type_ids_list
        self.start_positions = start_positions_list
        self.end_positions = end_positions_list

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return (self.input_ids_list[idx], self.attention_mask[idx], self.token_type_ids[idx],
                self.start_positions[idx], self.end_positions[idx])


# data_process4qa(data_set="train")
# data_process4qa(data_set="test")
