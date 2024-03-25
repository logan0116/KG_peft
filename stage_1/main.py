"""
train a SequenceClassification model by using the IMDb dataset
"""

import torch.optim as optim
import numpy as np

from model import *
from utils import *
from parser import parameter_parser

import logging

logging.basicConfig(level=logging.INFO)


def train(args):
    # device
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() and args.use_gpu else 'cpu')
    logging.info('device: {}'.format(device))
    # data
    dataset_train = data_load(data_set='train')
    dataset_test = data_load(data_set='test')
    dataloader_train = Data.DataLoader(dataset_train, args.batch_size, True)
    dataloader_test = Data.DataLoader(dataset_test, args.batch_size, True)
    logging.info('data load done!')
    # model
    model = MySequenceClassificationModel(model='bert-base-uncased', num_labels=2)
    model.to(device)
    logging.info('model load done!')
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=args.lr_reduce_factor,
                                                     patience=args.lr_schedule_patience,
                                                     verbose=True)
    logging.info('optimizer load done!')

    # 计算平均的loss
    loss_list = []
    precision_list = []
    recall_list = []
    accuracy_list = []
    f1_list = []
    best_f1 = 0
    logging.info('start training!')
    for epoch in range(args.epochs):
        # train
        model.train()
        loss_collector = []
        with tqdm(total=len(dataloader_train), desc='train---epoch:{}'.format(epoch)) as bar:
            for input_ids, attention_mask, token_type_ids, label in dataloader_train:
                # to device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                label = label.to(device)
                # loss
                optimizer.zero_grad()
                loss = model.loss(input_ids, attention_mask, token_type_ids, label)
                loss.backward()
                optimizer.step()
                loss_collector.append(loss.item())
                bar.update(1)
                bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
                scheduler.step(np.mean(loss_collector))
        # 计算平均的loss
        loss_list.append(np.mean(loss_collector))

        # evaluate
        model.eval()
        precision_collector = []
        recall_collector = []
        accuracy_collector = []
        f1_collector = []
        with tqdm(total=len(dataloader_test), desc='test') as bar:
            for input_ids, attention_mask, token_type_ids, label in dataloader_test:
                # to device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                # predict
                with torch.no_grad():
                    logits = model(input_ids, attention_mask, token_type_ids)
                    predict = torch.argmax(logits, dim=1).cpu()
                # evaluate
                precision_collector.append(precision_score(label, predict))
                recall_collector.append(recall_score(label, predict))
                accuracy_collector.append(accuracy_score(label, predict))
                f1_collector.append(f1_score(label, predict))
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
        # logging by tqdm
        tqdm.write('epoch: {}, loss: {}, precision: {}, recall: {}, accuracy: {}, f1: {}'.format(
            epoch, loss_list[-1], precision_list[-1], recall_list[-1], accuracy_list[-1], f1_list[-1]))

        # model save
        if f1_list[-1] > best_f1:
            best_f1 = f1_list[-1]
            # save model(add epoch)
            torch.save(model.state_dict(), 'model/model_epoch_{}.pth'.format(epoch))

        # early stop
        if epoch > 10:
            # f1 raise < 0.005 in 5 epochs
            if f1_list[-1] - f1_list[-6] < 0.005:
                break

    print('best f1:', best_f1, 'at epoch:', f1_list.index(best_f1))


if __name__ == '__main__':
    # 训练参数设置
    args = parameter_parser()
    args.gpu_id = 0
    args.batch_size = 16
    args.epochs = 100
    train(args)
