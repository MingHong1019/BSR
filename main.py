import argparse

from args import train_argparser
from config_reader import process_configs
from fewner.fewner_trainer import FewNERTrainer
import torch.nn as nn
from tqdm import tqdm
import random
import torch
import numpy as np
import os
from transformers import BertModel, BertTokenizer
from fewner.input_reader import OntonotesReader, BioNLPReader


def _train():
    arg_parser = train_argparser()
    process_configs(target=__train, arg_parser=arg_parser)


def __train(run_args):
    device = torch.device(run_args.device)
    tokenizer = BertTokenizer.from_pretrained(run_args.tokenizer_path,
                                              do_lower_case=False)
    data_reader = OntonotesReader(run_args.train_path, tokenizer,ways=5,shots=5,qry_sen_num=5000)


    data_reader2 = BioNLPReader(run_args.test_path, tokenizer,ways=5,shots=5)

    micro = []

    for seed in range(10):
        outer_test_dataset, inner_test_dataset = data_reader2.parse_dataset(run_args.test_sample_num, seed)
        trainer = FewNERTrainer(run_args, 100)
        micro.append(trainer.train(data_reader, outer_test_dataset, inner_test_dataset))

    print('the average F1 value is :  ', sum(micro)/len(micro))

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('mode', type=str, help="Mode: 'train' or 'eval'")
    args, _ = arg_parser.parse_known_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(0)  # 为所有的GPU设置种子，以使得结果是确定的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if args.mode == 'train':
        _train()
 
    else:
        raise Exception("Mode not in ['train', 'eval', 'predict'], e.g. 'python main.py train ...'")
