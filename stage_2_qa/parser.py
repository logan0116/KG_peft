import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description='for GNN')
    # train
    parser.add_argument('--part', help="Please give a value for epochs",
                        default=0, type=int)

    return parser.parse_args()
