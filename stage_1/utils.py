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
from tqdm import tqdm
from transformers import BertTokenizer


def get_dataset_size(dataset_size, cut_rate):
    train_size = int(cut_rate * dataset_size)
    test_size = dataset_size - train_size
    return train_size, test_size


def data_process(data_set="train", tokenizer_model='bert-base-uncased'):
    """
    IMDb数据加载
    """
    data_path = 'data/aclImdb/' + data_set
    text = []
    label_list = []

    for label in ['pos', 'neg']:
        for file in os.listdir(data_path + '/' + label):
            with open(data_path + '/' + label + '/' + file, 'r', encoding='utf-8') as f:
                text.append(f.read().lower())
                label_list.append(1 if label == 'pos' else 0)

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_model)
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    for t in tqdm(text):
        inputs = tokenizer(t, padding='max_length', truncation=True, max_length=512)
        input_ids_list.append(inputs['input_ids'])
        attention_mask_list.append(inputs['attention_mask'])
        token_type_ids_list.append(inputs['token_type_ids'])

    # length_count = [len(i) for i in input_ids_list]
    # length_info = {}
    # for length in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
    #     length_info[length] = sum([1 for i in length_count if (length - 50) <= i < length])
    # print(length_info)

    # trans 2 tensor
    input_ids_list = torch.tensor(input_ids_list, dtype=torch.int)
    attention_mask_list = torch.tensor(attention_mask_list, dtype=torch.int)
    token_type_ids_list = torch.tensor(token_type_ids_list, dtype=torch.int)
    label_list = torch.tensor(label_list, dtype=torch.int)

    # save data
    torch.save(input_ids_list, 'data/' + data_set + '_input_ids_list.pth')
    torch.save(attention_mask_list, 'data/' + data_set + '_attention_mask_list.pth')
    torch.save(token_type_ids_list, 'data/' + data_set + '_token_type_ids_list.pth')
    torch.save(label_list, 'data/' + data_set + '_label_list.pth')

    print(data_set + " data process done!")


def data_load(data_set="train"):
    """
    data load
    """
    input_ids_list = torch.load('data/' + data_set + '_input_ids_list.pth')
    attention_mask_list = torch.load('data/' + data_set + '_attention_mask_list.pth')
    token_type_ids_list = torch.load('data/' + data_set + '_token_type_ids_list.pth')
    label_list = torch.load('data/' + data_set + '_label_list.pth')
    label_list = label_list.long()

    return MyDataSet(input_ids_list, attention_mask_list, token_type_ids_list, label_list)


# loss曲线绘制
def loss_draw(epochs, loss_list):
    plt.plot([i + 1 for i in range(epochs)], loss_list)


class MyDataSet(Data.Dataset):
    """
    data load
    """

    def __init__(self, input_ids_list, attention_mask_list, token_type_ids_list, label_list):
        self.input_ids_list = input_ids_list
        self.attention_mask = attention_mask_list
        self.token_type_ids = token_type_ids_list
        self.label = label_list

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return self.input_ids_list[idx], self.attention_mask[idx], self.token_type_ids[idx], self.label[idx]


# data_process(data_set="train")
# data_process(data_set="test")
