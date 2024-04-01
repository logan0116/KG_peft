#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/1/31 上午8:39
# @Author  : liu yuhan
# @FileName: utils.py
# @Software: PyCharm
import json

import numpy as np
import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

import os
from tqdm import tqdm
from transformers import BertTokenizer


def get_dataset_size(dataset_size, cut_rate):
    train_size = int(cut_rate * dataset_size)
    test_size = dataset_size - train_size
    return train_size, test_size


def entity_inputs_ids_match(input_ids, entity_inputs_ids):
    """
    return entity_inputs_ids start_list and end_list in input_ids
    :param input_ids:
    :param entity_inputs_ids:
    :return:
        start_list, end_list
    """
    start_list = []
    end_list = []
    for i in range(len(input_ids) - len(entity_inputs_ids) + 1):
        if input_ids[i: i + len(entity_inputs_ids)] == entity_inputs_ids:
            start_list.append(i)
            end_list.append(i + len(entity_inputs_ids))
    return start_list, end_list


def data_process(data_set="train", tokenizer_model='hfl/chinese-roberta-wwm-ext-large'):
    """
    load CLUENER2020
    label

    地址（address），
    书名（book），
    公司（company），
    游戏（game），
    政府（goverment），
    电影（movie），
    姓名（name），
    组织机构（organization），
    职位（position），
    景点（scene）

    进行了Length的统计，最大length为64。

    """
    max_length = 64

    category2id = {
        "address": 0,
        "book": 1,
        "company": 2,
        "game": 3,
        "government": 4,
        "movie": 5,
        "name": 6,
        "organization": 7,
        "position": 8,
        "scene": 9
    }

    tokenizer = BertTokenizer.from_pretrained(tokenizer_model)

    data_path = 'data/cluener_public/' + data_set + '.json'
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for l in f:
            data.append(json.loads(l))

    input_ids_list = []
    attention_mask_list = []
    label_list = []

    # length check
    # length_list = []
    # for d in data:
    #     # text
    #     inputs = tokenizer(d['text'])
    #     input_ids = inputs['input_ids']
    #     length_list.append(len(input_ids))
    #
    # # count
    # length_count = {}
    # for l in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300]:
    #     length_count[l] = len([i for i in length_list if i <= l])
    # print(length_count)

    for d in tqdm(data):
        # text
        inputs = tokenizer(d['text'], max_length=max_length, truncation=True, padding='max_length')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        # label
        label = np.zeros((len(category2id), max_length, max_length))
        for c, entity_info in d['label'].items():
            for entity in entity_info:
                entity_inputs_ids = tokenizer.encode(entity, add_special_tokens=False)
                start_list, end_list = entity_inputs_ids_match(input_ids, entity_inputs_ids)
                if start_list and end_list:
                    for start, end in zip(start_list, end_list):
                        label[category2id[c]][start][end] = 1
        label_list.append(label)
    # trans to np
    label_list = np.array(label_list)
    # trans to tensor
    input_ids_list = torch.tensor(input_ids_list, dtype=torch.int)
    attention_mask_list = torch.tensor(attention_mask_list, dtype=torch.int)
    label_list = torch.tensor(label_list, dtype=torch.int)

    # save
    torch.save(input_ids_list, 'data/' + data_set + '_input_ids_list.pth')
    torch.save(attention_mask_list, 'data/' + data_set + '_attention_mask_list.pth')
    torch.save(label_list, 'data/' + data_set + '_label_list.pth')


def data_load(data_set="train"):
    """
    data load
    """
    input_ids_list = torch.load('data/' + data_set + '_input_ids_list.pth')
    attention_mask_list = torch.load('data/' + data_set + '_attention_mask_list.pth')
    label_list = torch.load('data/' + data_set + '_label_list.pth')
    label_list = label_list.long()

    return MyDataSet(input_ids_list, attention_mask_list, label_list)


# loss曲线绘制
def loss_draw(epochs, loss_list):
    plt.plot([i + 1 for i in range(epochs)], loss_list)


class MyDataSet(Data.Dataset):
    """
    data load
    """

    def __init__(self, input_ids_list, attention_mask_list, label_list):
        self.input_ids_list = input_ids_list
        self.attention_mask = attention_mask_list
        self.label = label_list

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return self.input_ids_list[idx], self.attention_mask[idx], self.label[idx]


if __name__ == '__main__':
    data_process(data_set="train")
    data_process(data_set="dev")
