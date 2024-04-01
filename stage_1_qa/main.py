"""
train a SequenceClassification model by using the IMDb dataset
"""

import torch.optim as optim
import numpy as np

from transformers import BertTokenizer, BertForQuestionAnswering
from peft import LoraConfig, get_peft_model, TaskType

from utils import *
from parser import parameter_parser

import logging
import time

logging.basicConfig(level=logging.INFO)


def train(args):
    # device
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() and args.use_gpu else 'cpu')
    logging.info('device: {}'.format(device))
    # data
    dataset_train = data_load(data_set='train', test_mode=args.test_mode)
    dataset_test = data_load(data_set='test', test_mode=args.test_mode)
    dataloader_train = Data.DataLoader(dataset_train, args.batch_size, True)
    dataloader_test = Data.DataLoader(dataset_test, args.batch_size, True)
    logging.info('data load done!')
    # model
    model = BertForQuestionAnswering.from_pretrained(pretrained_model_name_or_path='bert-base-cased')
    model.to(device)
    logging.info('model load done!')

    # args check: freeze and peft
    if args.freeze and args.peft:
        raise ValueError('freeze and peft cannot be set at the same time!')

    # freeze
    if args.freeze:
        for param in model.bert.parameters():
            param.requires_grad = False
        logging.info('freeze done!')

    # peft
    if args.peft is not None:
        if args.peft == "lora":
            config = LoraConfig(r=args.LoRA_r,
                                task_type=TaskType.QUESTION_ANS,
                                lora_dropout=0.01)

            model = get_peft_model(model, config)
            logging.info('peft done!')
            model.print_trainable_parameters()
        elif args.peft == "dora":
            config = LoraConfig(r=args.LoRA_r,
                                task_type=TaskType.QUESTION_ANS,
                                lora_dropout=0.01,
                                use_dora=True)

            model = get_peft_model(model, config)
            logging.info('peft done!')
            model.print_trainable_parameters()

    # # print model & model parameters requires_grad == True
    # print(model)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    # optimizer
    if args.freeze:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.init_lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    logging.info('optimizer load done!')

    # 计算平均的loss
    loss_list = []
    precision_list = []
    recall_list = []
    accuracy_list = []
    f1_list = []
    best_f1 = 0
    logging.info('start training!')
    # log all config
    logging.info('config: batch_size: {}, epochs: {}, init_lr: {}, freeze: {}, peft: {}, LoRA_r: {}'.format(
        args.batch_size, args.epochs, args.init_lr, args.freeze, args.peft, args.LoRA_r))

    for epoch in range(args.epochs):
        # train
        model.train()
        loss_collector = []
        with tqdm(total=len(dataloader_train), desc='train---epoch:{}'.format(epoch)) as bar:
            for input_ids, attention_mask, token_type_ids, start_positions, end_positions in dataloader_train:
                # to device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                start_positions = start_positions.to(device)
                end_positions = end_positions.to(device)
                # loss
                optimizer.zero_grad()
                loss_sc = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                start_positions=start_positions,
                                end_positions=end_positions).loss
                loss_sc.backward()
                optimizer.step()
                loss_collector.append(loss_sc.item())
                bar.update(1)
                bar.set_postfix(loss=loss_sc.item(), lr=optimizer.param_groups[0]['lr'])
        # 计算平均的loss
        loss_list.append(np.mean(loss_collector))

        # evaluate
        model.eval()
        precision_collector = []
        recall_collector = []
        accuracy_collector = []
        f1_collector = []
        with tqdm(total=len(dataloader_test), desc='test') as bar:
            for input_ids, attention_mask, token_type_ids, start_positions, end_positions in dataloader_test:
                # to device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                # predict
                with torch.no_grad():
                    outputs = model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)

                    answer_start_index = outputs.start_logits.argmax(dim=1).cpu()
                    answer_end_index = outputs.end_logits.argmax(dim=1).cpu()
                # evaluate
                precision, recall, accuracy, f1 = metric4qa(answer_start_index, answer_end_index,
                                                            start_positions, end_positions)
                precision_collector.append(precision)
                recall_collector.append(recall)
                accuracy_collector.append(accuracy)
                f1_collector.append(f1)
                bar.update(1)
                bar.set_postfix(precision=np.mean(precision_collector),
                                recall=np.mean(recall_collector),
                                accuracy=np.mean(accuracy_collector),
                                f1=np.mean(f1_collector))

        # 计算平均的precision, recall, accuracy, f1
        precision_list.append(np.mean(precision_collector))
        recall_list.append(np.mean(recall_collector))
        accuracy_list.append(np.mean(accuracy_collector))
        f1_list.append(np.mean(f1_collector))
        # logging
        logging.info('epoch: {}, loss: {}, precision: {}, recall: {}, accuracy: {}, f1: {}'.format(
            epoch, loss_list[-1], precision_list[-1], recall_list[-1], accuracy_list[-1], f1_list[-1]))

        # model save
        if f1_list[-1] > best_f1:
            best_f1 = f1_list[-1]
            # save model(add epoch & time)
            local_time = time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime(time.time()))
            torch.save(model.state_dict(), 'model/model_epoch_{}_{}.pth'.format(epoch, local_time))

        # early stop
        if epoch > 10:
            # f1 raise < 0.005 in 5 epochs
            if f1_list[-1] - f1_list[-6] < 0.005:
                break

    print('best f1:', best_f1, 'at epoch:', f1_list.index(best_f1))


if __name__ == '__main__':
    # # full train
    # args = parameter_parser()
    # # args.test_mode = True
    # args.gpu_id = 0
    # args.freeze = False
    # args.peft = False
    # train(args)
    # # freeze
    # args = parameter_parser()
    # args.gpu_id = 0
    # args.freeze = True
    # args.peft = False
    # train(args)
    # LoRA r=4
    # args = parameter_parser()
    # args.gpu_id = 1
    # args.freeze = False
    # args.peft = 'lora'
    # args.LoRA_r = 4
    # train(args)
    # # LoRA r=16
    # args = parameter_parser()
    # args.gpu_id = 1
    # args.freeze = False
    # args.peft = 'lora'
    # args.LoRA_r = 16
    # train(args)
    # DoRA r=8
    args = parameter_parser()
    args.gpu_id = 1
    args.freeze = False
    args.peft = 'dora'
    args.LoRA_r = 8
    train(args)
