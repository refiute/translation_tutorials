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
    def __init__(self, n_layer, vocab_size, embed_size, hidden_size, dropout=0.5, use_cudnn=True, train=False):
        super(Encoder, self).__init__(
            embed = L.EmbedID(vocab_size, embed_size),
            lstm = L.NStepLSTM(n_layer, embed_size, hidden_size, dropout, use_cudnn)
        )
        self.train = train

    def __call__(self, x, c, h):
        e = self.tanh(self.embed(x))
        h, c, y = self.lstm(h, c, e, train=self.train)
        return c, h

class Decoder(Chain):
    def __init__(self, n_layer, vocab_size, embed_size, hidden_size, train=False):
        super(Decoder, self).__init__(
            embed = L.EmbedID(vocab_size, embed_size),
            lstm = L.NStepLSTM(n_layer, embed_size, hidden_size, dropout, use_cudnn),
            lf = L.Linear(hidden_size, embed_size),
            fy = L.Linear(embed_size, vocab_size)
        )
        self.train = train

    def __call__(self, y, c, h):
        e = self.tanh(self.embed(y))
        h, c, l = self.lstm(h, c, e, train=self.train)
        f = F.tanh(self.lf(l))
        y = self.fy(f)
        return y, c, h

class EncoderDecoder(Chain):
    def __init__(self, src_vocab, trg_vocab, embed_size, hidden_size, train=False):
        super(EncoderDecoder, self).__init__(
            enc = Encoder(src_vocab, embed_size, hidden_size, train=train),
            dec = Decoder(trg_vocab, embed_size, hidden_size, train=train)
        )
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.train = train

    def reset(self, batch_size):
        self.cleargrads()
        self.c = initializers.Uniform(0.08, (batch_size, self.hidden_size))
        self.h = initializers.Uniform(0.08, (batch_size, self.hidden_size))

    def encode(self, x):
        self.c, self.h = self.enc(x, self.c, self.h)

    def decode(self, y):
        y, self.c, self.h = self.dec(y, self.c, self.h)
        return y

    def save_spec(self, filename):
        with open(filename, "w", encoding="utf-8") as f:
            print(self.src_vocab, file=f)
            print(self.trg_vocab, file=f)
            print(self.embed_size, file=fp)
            print(self.hidden_size, file=fp)

    @staticmethod
    def load_spec(filename, train=False):
        with open(filename, encoding="utf-8") as f:
            src_vocab = int(next(f))
            trg_vocab = int(next(f))
            embed_size = int(next(f))
            hidden_size = int(next(f))
            return EncoderDecoder(src_vocab, trg_vocab. embed_size, hidden_size, train=train)

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
    src_vocab = Vocabulary.new(gens.word_list(args.source), args.src_vocab)
    trg_vocab = Vocabulary.new(gens.word_list(args.target), args.trg_vocab)

    encdec = EncoderDecoder(args.src_vocab, args.trg_vocab, args.embed, args.hidden, train=True)
    if args.gpu >= 0:
        encdec.to_gpu()

    for epoch in range(args.epoch):

        gen1 = gens.word_list(args.source)
        gen2 = gens.word_list(args.target)
        gen3 = gens.batch(gens.concat_batches(gen1, gen2), args.minibatch)

        opt = optimizers.AdaGrad(lr=0.01)
        opt.setup(encdec)
        opt.add_hook(optimizer.GradientClipping(5))

        for src_batch, trg_batch in gen3:
            ret, loss = forward(src_batch, trg_batch, src_vocab. trg_vocab, encdec, True, 0)
            loss.backward()
            opt.update()

        prefix = args.model + ".%03.d"%(epoch+1)
        src_vocab.save(prefix + ".src.vocab")
        trg_vocab.save(prefix + ".trg.vocab")
        encdec.save_spec(prefix + ".spec")
        serializers.save_hdf5(prefix + ".model", encdec)

def test(args):
    src_vocab = Vocabulary.load(args.model + ".src.vocab")
    trg_vocab = Vocabulary.load(args.model + ".trg.vocab")
    encdec = EncoderDecoder.load_spec(args.model + ".spec")
    if args.gpu >= 0:
        encdec.to_gpu()
    serializers.load_hdf5(args.model + ".model", encdec)

    with open(args.target, "w", encoding="utf-8") as f:
        for src_batch in gens.batch(gets.word_list(args.source), args.minibatch):
            ret = forward(src_batch, None, src_vocab, trg_vocab,
                    encdec, False, args.generation_limit)

            for sent in ret:
                sent.append("</s>")
                sent = sent[:sent.index("</s>")]
                print(" ".join(sent), file=f)

def main():
    args = get_args()
    if args.env == "train":
        train(args)
    elif args.env == "test":
        test(args)

if __name__ == "__main__":
    main()

