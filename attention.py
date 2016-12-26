# coding: utf-8

import sys
from argparse import ArgumentParser

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import utils.generators as gens
from utils.vocabulary import Vocabulary

def get_args():
    gpu = -1
    src_vocab = 2000
    trg_vocab = 1000
    embed = 100
    hidden = 200
    epoch = 10
    minibatch = 64
    generation_limit = 128

    args = ArgumentParser()
    args.add_argument('env', help="'train' or 'test'")
    args.add_argument('source', help="[in] source corpus")
    args.add_argument('target', help="[in/out] target corpus")
    args.add_argument('model', help="[in/out] model file")
    args.add_argument('--gpu', default=gpu, type=int, metavar="INT",
            help="GPU device ID (default: %(default)d [use CPU])")
    args.add_argument('--src_vocab', default=src_vocab, type=int, metavar="INT",
            help="source vocabulary size (default: %(default)d)")
    args.add_argument('--trg_vocab', default=trg_vocab, type=int, metavar="INT",
            help="target vocabulary size (default: %(default)d)")
    args.add_argument('--embed', default=embed, type=int, metavar="INT",
            help="embedding layer size (default: %(default)d)")
    args.add_argument('--hidden', default=hidden, type=int, metavar="INT",
            help="hidden layer size (default: %(default)d)")
    args.add_argument('--minibatch', default=minibatch, type=int, metavar="INT",
            help="minibatch size (default: %(default)d)")
    args.add_argument('--generation_limit', default=generation_limit, type=int, metavar="INT",
            help="maximum number of words to be generated for test input (default: %(default)d)")

    p = args.parse_args()

    try:
        if p.mode not in ["train", "test"]:
            raise ValueError("args: you must set env = 'train' or 'test'")
        if p.src_vocab < 1:
            raise ValueError("args: you must set --src_vocab >= 1")
        if p.trg_vocab < 1:
            raise ValueError("args: you must set --trg_vocab >= 1")
        if p.embed < 1:
            raise ValueError("args: you must set --embed >= 1")
        if p.hidden < 1:
            raise ValueError("args: you must set --hidden >= 1")
        if p.minibatch < 1:
            raise ValueError("args: you must set --minibatch >= 1")
        if p.generation_limit < 1:
            raise ValueError("args: you must set --generation_limit >= 1")
    except Exception as ex:
        p.print_usage(file=sys.stderr)
        print(ex, file=sys.stderr)
        sys.exit()

    return p

class Encoder(Chain):
    def __init__(self, train=False):
        super(Encoder, self).__init__(
            # TODO: write network
        )
        self.train = train

    def __call__(self):
        pass

class Attention(Chain):
    def __init__(self, train=False):
        super(Attention, self).__init__(
            # TODO: write network
        )
        self.train = train

    def __call__(self):
        pass

class Decoder(Chain):
    def __init__(self, train=False):
        super(Decoder, self).__init__(
            # TODO: write network
        )
        self.train = train

    def __call__(self):
        pass

class AttentionEncDec(Chain):
    def __init__(self, train=False):
        super(AttentionEncDec, self).__init__(
            # TODO: write network
        )
        self.train = train

    def __call__(self):
        pass

    def reset(self):
        self.cleargrads()
        self.x_list = list()

    def encode(self, x):
        pass

    def decode(self, y):
        pass

    def save_spec(self, filename):
        pass

    @staticmethod
    def load_spec(filename):
        pass

def forward():
    pass

def train(args):
    pass

def test(args):
    pass

def main():
    args = get_args()
    if args.env == "train":
        train(args)
    else args.env == "test":
        test(args)

if __name__ == "__main__":
    main()

