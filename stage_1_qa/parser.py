import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description='for GNN')
    # model
    # parser.add_argument('--model', help="Please give a value for model name",
    #                     default='LSTM', type=str)
    # parser.add_argument('--in_dim', help="Please give a value for in_dim",
    #                     default=1433, type=int)
    # parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim",
    #                     default=16, type=int)
    # parser.add_argument('--out_dim', help="Please give a value for out_dim",
    #                     default=7, type=int)
    # parser.add_argument('--n_layers', help="Please give a value for num_layers",
    #                     default=2, type=int)
    # parser.add_argument('--residual', help="Please give a value for residual",
    #                     default=False, type=bool)
    # parser.add_argument('--batch_norm', help="Please give a value for batch_norm",
    #                     default=False, type=bool)
    # parser.add_argument('--dropout', help="Please give a value for dropout",
    #                     default=0.5, type=float)
    # parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout",
    #                     default=0.5, type=float)
    # parser.add_argument('--readout', help="Please give a value for readout",
    #                     default='mean', type=str)
    # train
    parser.add_argument('--epochs', help="Please give a value for epochs",
                        default=5, type=int)
    parser.add_argument('--batch_size', help="Please give a value for batch_size",
                        default=4, type=int)
    # optimizer
    parser.add_argument('--init_lr', help="Please give a value for init_lr",
                        default=0.00002, type=float)
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor",
                        default=0.5, type=float)
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience",
                        default=10, type=int)
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay",
                        default=0.0005, type=float)
    parser.add_argument('--min_lr', help="Please give a value for min_lr",
                        default=0.00002, type=float)
    # data
    # parser.add_argument('--dataset', help="Please give a value for dataset name",
    #                     default='COLLAB', type=str)
    parser.add_argument('--data_path', help="Please give a value for data_path",
                        default='data/COLLAB', type=str)
    parser.add_argument('--test_mode', help="Please give a value for test_mode",
                        default=False, type=bool)
    # gpu
    parser.add_argument('--use_gpu', help="Please give a value for use_gpu",
                        default=True, type=bool)
    parser.add_argument('--gpu_id', help="Please give a value for gpu_id",
                        default=0, type=int)
    # freeze set
    parser.add_argument('--freeze', help="Please give a value for freeze",
                        default=False, type=bool)
    # peft set
    parser.add_argument('--peft', help="Please give a value for peft",
                        default=False, type=bool)
    parser.add_argument('--LoRA_r', help="Please give a value for LoRA_r",
                        default=8, type=int)

    return parser.parse_args()
