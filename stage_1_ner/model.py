#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/5/10 下午4:55
# @Author  : liu yuhan
# @FileName: model.py
# @Software: PyCharm


import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertConfig

from peft import LoraConfig, get_peft_model, TaskType


class GlobalPointer(nn.Module):
    def __init__(self, hidden_size, num_heads, head_size, device, if_rope):
        super().__init__()
        """
        这里参考原文的实现，使用了两个全连接层，用于构建qi和kj
        在苏神的代码中这两个全连接层被组合了
        参数：
        head: 实体种类
        head_size: 每个index的映射长度
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.device = device
        self.dense = nn.Linear(hidden_size, num_heads * head_size * 2, bias=True)
        if if_rope:
            # PoRE
            self.if_rope = if_rope
            indices = torch.arange(0, head_size // 2, dtype=torch.float)
            indices = torch.pow(torch.tensor(10000, dtype=torch.float), -2 * indices / head_size)
            emb_cos = torch.cos(indices)
            self.emb_cos = torch.repeat_interleave(emb_cos, 2, dim=-1).to(device)
            emb_sin = torch.sin(indices)
            self.emb_sin = torch.repeat_interleave(emb_sin, 2, dim=-1).to(device)
            # [1, 1, 1, 1]->[-1, 1, -1, 1]
            self.trans4tensor = torch.Tensor([-1, 1] * (self.head_size // 2)).to(device)

    def transpose_for_scores(self, x):
        # x:[batch_size, seq_len, head * head_size * 2]
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size * 2)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_size]

    def scores_mask(self, scores, attention_mask):
        # scores: [batch_size, num_heads, seq_len, seq_len]
        # attention_mask: [batch_size, seq_len]
        # low_tri_mask: [seq_len, seq_len]
        # 1. attention_mask的mask
        attention_mask = attention_mask[:, None, None, :]  # [batch_size, 1, 1, seq_len]
        # 2. low_tri_mask的mask
        low_tri_mask = (1 - torch.tril(torch.ones(scores.size()[-2:]), diagonal=0)).to(self.device)  # [seq_len,seq_len]
        # 3. mask combine
        mask = attention_mask + low_tri_mask - 1
        mask = torch.clamp(mask, min=0)
        mask = (1.0 - mask) * -1e12  # [batch_size, 1, seq_len, seq_len]
        return scores + mask

    def get_rope(self, tenser):
        return tenser * self.emb_cos + tenser * self.trans4tensor * self.emb_sin

    def forward(self, inputs, attention_mask):
        # inputs: [batch_size, seq_len, hidden_size]
        inputs = self.dense(inputs)  # [batch_size, seq_len, head * head_size * 2]
        inputs = self.transpose_for_scores(inputs)  # [batch_size, head, seq_len, head_size * 2]

        q, v = inputs[:, :, :, :self.head_size], inputs[:, :, :, self.head_size:]
        # PoRE
        if self.if_rope:
            q = self.get_rope(q)
            v = self.get_rope(v)

        # attention_scores
        scores = torch.matmul(q, v.transpose(-1, -2))  # [batch_size, num_heads, seq_len, seq_len]
        scores = scores / np.sqrt(self.head_size)
        # mask
        return self.scores_mask(scores, attention_mask)


class EfficientGlobalPointer(nn.Module):
    def __init__(self, hidden_size, num_heads, head_size, device, if_rope):
        super().__init__()
        """
        这里参考原文的实现，使用了两个全连接层，用于构建qi和kj
        在苏神的代码中这两个全连接层被组合了
        参数：
        head: 实体种类
        head_size: 每个index的映射长度
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.device = device
        self.dense = nn.Linear(hidden_size, head_size * 2, bias=True)
        self.dense4head = nn.Linear(head_size * 2, num_heads * 2, bias=True)
        self.if_rope = if_rope
        if self.if_rope:
            # PoRE
            indices = torch.arange(0, head_size // 2, dtype=torch.float)
            indices = torch.pow(torch.tensor(10000, dtype=torch.float), -2 * indices / head_size)
            emb_cos = torch.cos(indices)
            self.emb_cos = torch.repeat_interleave(emb_cos, 2, dim=-1).to(device)
            emb_sin = torch.sin(indices)
            self.emb_sin = torch.repeat_interleave(emb_sin, 2, dim=-1).to(device)
            # [1, 1, 1, 1]->[-1, 1, -1, 1]
            self.trans4tensor = torch.Tensor([-1, 1] * (self.head_size // 2)).to(device)

    def scores_mask(self, scores, attention_mask):
        # scores: [batch_size, num_heads, seq_len, seq_len]
        # attention_mask: [batch_size, seq_len]
        # low_tri_mask: [seq_len, seq_len]
        # 1. attention_mask的mask
        attention_mask = attention_mask[:, None, None, :]  # [batch_size, 1, 1, seq_len]
        # 2. low_tri_mask的mask
        low_tri_mask = (1 - torch.tril(torch.ones(scores.size()[-2:]), diagonal=0)).to(self.device)  # [seq_len,seq_len]
        # 3. mask combine
        mask = attention_mask + low_tri_mask - 1
        mask = torch.clamp(mask, min=0)
        mask = (1.0 - mask) * -1e12  # [batch_size, 1, seq_len, seq_len]
        return scores + mask

    def get_rope(self, tenser):
        return tenser * self.emb_cos + tenser * self.trans4tensor * self.emb_sin

    def forward(self, inputs, attention_mask):
        # inputs: [batch_size, seq_len, hidden_size]
        inputs = self.dense(inputs)  # [batch_size, seq_len, head_size * 2]
        q, v = inputs[:, :, :self.head_size], inputs[:, :, self.head_size:]  # [batch_size, seq_len, head_size]
        # PoRE
        if self.if_rope:
            q = self.get_rope(q)
            v = self.get_rope(v)
        # attention_scores
        scores = torch.matmul(q, v.transpose(-1, -2))  # [batch_size, num_heads, seq_len, seq_len]
        scores = scores / np.sqrt(self.head_size)
        # 以上应该没有问题

        bias = self.dense4head(inputs).permute(0, 2, 1) / 2  # [batch_size, num_heads * 2, seq_len]
        scores = scores[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]
        # mask
        return self.scores_mask(scores, attention_mask)


class MyNER(nn.Module):
    def __init__(self, bert_model_path, ner_num_heads, ner_head_size, device,
                 if_lora=False, lora_r=1,
                 if_rope=False, if_efficientnet=False):
        super().__init__()
        self.config = BertConfig.from_pretrained(bert_model_path)
        self.bert = BertModel.from_pretrained(bert_model_path)
        if if_lora:
            lora_config = LoraConfig(r=lora_r,
                                     task_type=TaskType.FEATURE_EXTRACTION,
                                     lora_dropout=0.01)
            self.bert = get_peft_model(self.bert, lora_config)

        if if_efficientnet:
            self.ner_score = EfficientGlobalPointer(hidden_size=self.config.hidden_size,
                                                    num_heads=ner_num_heads,
                                                    head_size=ner_head_size,
                                                    device=device,
                                                    if_rope=if_rope)

        else:
            self.ner_score = GlobalPointer(hidden_size=self.config.hidden_size,
                                           num_heads=ner_num_heads,
                                           head_size=ner_head_size,
                                           device=device,
                                           if_rope=if_rope)

    def forward(self, input_ids, attention_mask):
        hidden = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        scores = self.ner_score(hidden, attention_mask)
        return scores

    def fun_loss(self, score, label):
        # score: [batch_size, num_heads, seq_len, seq_len]
        # label: [batch_size, num_heads, seq_len, seq_len]
        # [batch_size, num_heads, seq_len, seq_len] -> [batch_size * num_heads, seq_len * seq_len]
        score = score.contiguous().view(-1, score.size(-2) * score.size(-1))
        label = label.view(-1, label.size(-2) * label.size(-1))

        score = (1 - 2 * label) * score
        score_neg = score - label * 1e12
        score_pos = score - (1 - label) * 1e12
        zeros = torch.zeros_like(score[..., :1])
        score_neg = torch.cat([score_neg, zeros], dim=-1)
        score_pos = torch.cat([score_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(score_neg, dim=-1)
        pos_loss = torch.logsumexp(score_pos, dim=-1)
        return torch.mean(neg_loss + pos_loss)

    def fun_loss2(self, score, label):
        """
        这是另外一种loss的计算方式
        :param score:
        :param label:
        :return:
        """
        score = score.view(-1, score.size(-2) * score.size(-1))
        label = label.view(-1, label.size(-2) * label.size(-1))
        pos_mask = torch.eq(label, 1)
        score_pos = torch.where(pos_mask, -score, -1e12)
        neg_mask = torch.eq(label, 0)
        score_neg = torch.where(neg_mask, score, -1e12)
        zeros = torch.zeros_like(score[..., :1])
        score_neg = torch.cat([score_neg, zeros], dim=-1)
        score_pos = torch.cat([score_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(score_neg, dim=-1)
        pos_loss = torch.logsumexp(score_pos, dim=-1)
        return torch.mean(neg_loss + pos_loss)

    def fun_evaluate(self, score, labels):
        # score: [batch_size, num_heads, seq_len, seq_len]
        # label: [batch_size, num_heads, seq_len, seq_len]
        score = torch.where(score > 0, 1, 0)
        f1 = 2 * torch.sum(score * labels) / torch.sum(score + labels)
        return f1
