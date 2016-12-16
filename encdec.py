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

def get_args():

    gpu_device = 0
    src_vocab = 160000
    trg_vocab = 80000
    embed = 1000
    hidden = 1000
    epoch = 10
    minibatch = 128
    generation_limit = 128

    args = ArgumentParser(
        description="Encoder-decoder Neural Machine Translation"
    )

    # TODO: add arguments

class Encoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Encoder, self).__init__(
            xe = L.EmbedID(vocab_size, embed_size),
            eh = L.Linear(embed_size, 4 * hidden_size),
            hh = L.Linear(hidden_size, 4 * hidden_size)
        )

    def __call__(self, x, c, h):
        e = F.tanh(self.xe(x))
        return F.lstm(c, self.eh(e)+self.hh(h))

class Decoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Decoder, self).__init__(
            ye = L.EmbedID(vocab_size, embed_size),
            eh = L.Linear(embed_size, 4 * hidden_size),
            hh = L.Linear(hidden_size, 4 * hidden_size),
            hf = L.Linear(hidden_size, embed_size),
            fy = L.Linear(embed_size, vocab_size)
        )

    def __call__(self, y, c, h):
        e = F.tanh(self.ye(y))
        c, h = F.lstm(c, self.eh(e) + self.hh(h))
        f = F.tanh(self.hf(h))
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

    def reset_state(self, batch_size):
        self.zerograds()
        self.c = initializers.Uniform(0.08, (batch_size, self.hidden_size))
        self.h = initializers.Uniform(0.08, (batch_size, self.hidden_size))

    def encode(self, x):
        self.c, self.h = self.enc(x, self.c, self.h)

    def decode(self, y):
        y, self.c, self.h = self.dec(y, self.c, self.h)
        return y

def forward(src_batch, trg_batch, src_vocab, trg_vocab, encdec, is_training, generation_limit):
    pass
