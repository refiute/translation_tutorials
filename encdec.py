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
    src_vocab = 2000
    trg_vocab = 1000
    embed = 100
    hidden = 200
    epoch = 10
    minibatch = 64
    generation_limit = 128

    args = ArgumentParser()
    args.add_argument('env', help="\'train\' or \'test\'")
    args.add_argument('source', help="[in] source corpus")
    args.add_argument('target', help="[in/out] target corpus")
    args.add_argument('model', help="[in/out] model file")
    args.add_argument('--gpu', default=-1, type=int, metavar="INT",
            help="GPU device ID (default: -1 [use CPU])")
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

    return args.parse_args()

class Encoder(Chain):
    def __init__(self, n_layer, vocab_size, embed_size, hidden_size, dropout=0.5, use_cudnn=True):
        super(Encoder, self).__init__(
            embed = L.EmbedID(vocab_size, embed_size),
            lstm = L.NStepLSTM(n_layer, embed_size, hidden_size, dropout, use_cudnn)
        )

    def __call__(self, x, c, h, train):
        e = self.embed(x)
        h, c, y = self.lstm(h, c, e, train=train)
        return c, h

class Decoder(Chain):
    def __init__(self, n_layer, vocab_size, embed_size, hidden_size):
        super(Decoder, self).__init__(
            embed = L.EmbedID(vocab_size, embed_size),
            lstm = L.NStepLSTM(n_layer, embed_size, hidden_size, dropout, use_cudnn),
            lf = L.Linear(hidden_size, embed_size),
            fy = L.Linear(embed_size, vocab_size)
        )

    def __call__(self, y, c, h, train):
        e = self.embed(y)
        h, c, l = self.lstm(h, c, e, train=train)
        f = F.tanh(self.lf(l))
        y = self.fy(f)
        return y, c, h

class EncoderDecoder(Chain):
    def __init__(self, src_vocab, trg_vocab, embed_size, hidden_size):
        super(EncoderDecoder, self).__init__(
            enc = Encoder(src_vocab, embed_size, hidden_size),
            dec = Decoder(trg_vocab, embed_size, hidden_size)
        )
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.embed_size = embed_size
        self.hidden_size = hidden_size

    def reset(self, batch_size):
        self.cleargrads()
        self.c = initializers.Uniform(0.08, (batch_size, self.hidden_size))
        self.h = initializers.Uniform(0.08, (batch_size, self.hidden_size))

    def encode(self, x, train=True):
        self.c, self.h = self.enc(x, self.c, self.h, train=train)

    def decode(self, y, train=True):
        y, self.c, self.h = self.dec(y, self.c, self.h, train=train)
        return y

    def save_spec(self, filename):
        with open(filename, "w", encoding="utf-8") as f:
            print(self.src_vocab, file=f)
            print(self.trg_vocab, file=f)
            print(self.embed_size, file=fp)
            print(self.hidden_size, file=fp)

    @staticmethod
    def load_spec(filename):
        with open(filename, encoding="utf-8") as f:
            src_vocab = int(next(f))
            trg_vocab = int(next(f))
            embed_size = int(next(f))
            hidden_size = int(next(f))
            return EncoderDecoder(src_vocab, trg_vocab. embed_size, hidden_size)

def forward(src_batch, trg_batch, src_vocab, trg_vocab, encdec, is_training, generation_limit):
    batch_size = len(src_batch)
    src_len = len(src_batch[0])
    trg_len = len(trg_batch[0]) if trg_batch else None
    src_stoi = src_vocab.stoi
    trg_stoi = trg_vocab.stoi
    trg_itos = trg_vocab.itos
    encdec.reset(batch_size)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        encdev.to_gpu()
        xp = cuda.cupy()
    else:
        xp = np

    x = xp.asarray([src_stoi("</s>") for _ in range(batch_size)], dtype=xp.int32)
    encdec.encode(x)
    for l in reversed(range(src_len)):
        x = xp.asarray([src_stoi(src_batch[k][l]) for k in range(batch_size)], dtype=xp.int32)
        encdec.encode(x)

    t = xp.asarray([trg_stoi("<s>") for _ in range(batch_size)], dtype=xp.int32)
    ret = [[] for _ in range(batch_size)]

    if is_training:
        loss = xp.zeros((), dtype=xp.float32)
        for l in range(trg_len):
            y = encdec.decode(t)
            t = xp.asarray([trg_stoi(trg_batch[k][l]) for k in range(batch_size)], dtype=xp.int32)
            loss += F.softmax_cross_entropy(x, t)
            output = cuda.to_cpu(y.data.argmax(1))
            for k in range(batch_size):
                ret[k].append(trg_itos(output[k]))
        return ret, loss
    else:
        while len(ret[0]) < generation_limit:
            y = encdec.decode(t)
            output = cuda.to_cpu(y.data.argmax(1))
            t = xp.asarray(output)
            for k in range(batch_size):
                ret[k].append(trg_itos(output[k]))
            if all("</s>" in ret[k] for k in range(batch_size)):
                break
        return ret

def train(args):
    pass

def test(args):
    pass

def main():
    args = get_args()
    if args.env == "train":
        train(args)
    elif args.env == "test":
        test(args)

if __name__ == "__main__":
    main()

