import torch
import torch.nn as nn
from transformers import BertForSequenceClassification


class MySequenceClassificationModel(nn.Module):
    def __init__(self, model, num_labels):
        """
        :param model:
            bert base uncased for IMDb
        :param num_labels:
        """
        super(MySequenceClassificationModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model,
                                                                  num_labels=num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        :param input_ids:
        :param attention_mask:
        :param token_type_ids:
        :return:
        """
        return self.bert(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids).logits

    def loss(self, input_ids, attention_mask, token_type_ids, labels):
        """
        :param input_ids:
        :param attention_mask:
        :param token_type_ids:
        :param labels:
        :return:
        """
        return self.bert(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids,
                         labels=labels).loss