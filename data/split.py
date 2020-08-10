from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import numpy as np
import random
import math
import json
from collections import Counter 
import pdb
import logging

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def proc(opt):
    tot_num_line = sum(1 for _ in open(opt.input_path, 'r'))
    labeled_count = (tot_num_line // 100) * 2
    unlabeled_count = tot_num_line = labeled_count
    with open(opt.input_path, 'r', encoding='utf-8') as f, \
         open(opt.labeled_path, 'w', encoding='utf-8') as f_labeled, \
         open(opt.unlabeled_path, 'w', encoding='utf-8') as f_unlabeled:
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            toks = line.strip().split('\t')
            if len(toks) <= 1: continue
            sent = toks[0]
            label = toks[1]
            if idx < labeled_count: # labeled
                f_labeled.write("{}\t{}\n".format(sent, label))
            if idx >= labeled_count: # unlabeled
                f_unlabeled.write("{}\t{}\n".format(sent, "UNK"))

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_path', type=str, default='train.txt')
    parser.add_argument('--labeled_path', type=str, default='labeled.txt')
    parser.add_argument('--unlabeled_path', type=str, default='unlabeled.txt')
    opt = parser.parse_args()

    proc(opt)


if __name__ == '__main__':
    main()
