import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--per', default=0.1, type=float,help='initial learning rate')
parser.add_argument('--base', default=0.1, type=float,
                    help='initial learning rate')

def main():
    args = parser.parse_args()

    print("flops: {}".format(args.base*(1-args.per)))
if __name__ == '__main__':
    main()