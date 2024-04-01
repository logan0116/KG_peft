"""
train a SequenceClassification model by using the IMDb dataset
"""

import torch.optim as optim

from utils import *
from model import *
from parser import parameter_parser

import logging
import time

logging.basicConfig(level=logging.INFO)


def train(args):
    # device
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() and args.use_gpu else 'cpu')
    logging.info('device: {}'.format(device))
    # data
    dataset_train = data_load(data_set='train')
    dataset_test = data_load(data_set='dev')
    dataloader_train = Data.DataLoader(dataset_train, args.batch_size, True)
    dataloader_test = Data.DataLoader(dataset_test, args.batch_size, True)
    logging.info('data load done!')
    # model
    model = MyNER(bert_model_path='hfl/chinese-roberta-wwm-ext-large',
                  ner_num_heads=10,
                  ner_head_size=64,
                  if_lora=args.if_lora,
                  lora_r=args.LoRA_r,
                  device=device,
                  if_rope=True)
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
    if args.if_lora:
        model.bert.print_trainable_parameters()

    # optimizer
    if args.freeze:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.init_lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    logging.info('optimizer load done!')

    # 计算平均的loss
    loss_list = []
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
            for input_ids, attention_mask, label in dataloader_train:
                # to device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                label = label.to(device)
                # loss
                optimizer.zero_grad()
                score = model(input_ids=input_ids, attention_mask=attention_mask)
                loss_ner = model.fun_loss2(score, label)
                # backend
                loss_ner.backward()
                optimizer.step()
                loss_collector.append(loss_ner.item())
                bar.update(1)
                bar.set_postfix(loss=loss_ner.item(), lr=optimizer.param_groups[0]['lr'])
        # 计算平均的loss
        loss_list.append(np.mean(loss_collector))

        # evaluate
        model.eval()
        f1_collector = []
        with tqdm(total=len(dataloader_test), desc='test') as bar:
            for input_ids, attention_mask, label in dataloader_test:
                # to device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                label = label.to(device)
                # predict
                with torch.no_grad():
                    score = model(input_ids=input_ids, attention_mask=attention_mask)
                    f1 = model.fun_evaluate(score, label)
                # evaluate
                f1_collector.append(f1.item())
                bar.update(1)
                bar.set_postfix(f1=np.mean(f1_collector))

        # 计算平均的precision, recall, accuracy, f1
        f1_list.append(np.mean(f1_collector))
        # logging
        logging.info('epoch: {}, loss: {}, f1: {}'.format(epoch, loss_list[-1], f1_list[-1]))

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

    # # LoRA r=16
    # args = parameter_parser()
    # args.gpu_id = 0
    # args.freeze = False
    # args.if_lora = True
    # args.LoRA_r = 16
    # train(args)
    # # LoRA r=8
    # args = parameter_parser()
    # args.gpu_id = 0
    # args.freeze = False
    # args.if_lora = True
    # args.LoRA_r = 8
    # train(args)
    # LoRA r=4
    args = parameter_parser()
    args.gpu_id = 0
    args.freeze = False
    args.if_lora = True
    args.LoRA_r = 4
    train(args)
