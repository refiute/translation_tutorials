# coding: utf-8

import sys
from argparse import ArgumentParser

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers, initializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import utils.generators as gens
from utils.vocabulary import Vocabulary
from utils.functions import fill_batch, trace

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
    args.add_argument('--epoch', default=epoch, type=int, metavar="INT",
                      help="training epoch size (default: %(default)d)")
    args.add_argument('--minibatch', default=minibatch, type=int, metavar="INT",
                      help="minibatch size (default: %(default)d)")
    args.add_argument('--generation_limit', default=generation_limit, type=int, metavar="INT",
                      help="maximum number of words to be generated for test input (default: %(default)d)")

    p = args.parse_args()

    try:
        if p.env not in ["train", "test"]:
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
        args.print_usage(file=sys.stderr)
        print(ex, file=sys.stderr)
        sys.exit()

    return p

class Encoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout=0.5):
        super(Encoder, self).__init__(
            xe = L.EmbedID(vocab_size, embed_size),
            eh = L.Linear(embed_size, 4*hidden_size),
            hh = L.Linear(hidden_size, 4*hidden_size)
        )

    def __call__(self, x, c, h):
        e = F.tanh(self.xe(x))
        return F.lstm(c, self.eh(e)+self.hh(h))

class Decoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout=0.5):
        super(Decoder, self).__init__(
            ye = L.EmbedID(vocab_size, embed_size),
            eh = L.Linear(embed_size, 4*hidden_size),
            hh = L.Linear(hidden_size, 4*hidden_size),
            hf = L.Linear(hidden_size, embed_size),
            fy = L.Linear(embed_size, vocab_size)
        )

    def __call__(self, y, c, h):
        e = F.tanh(self.ye(y))
        c, h = F.lstm(c, self.eh(e)+self.hh(h))
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

    def reset(self, batch_size, xp=np):
        self.cleargrads()
        self.c = xp.zeros((batch_size, self.hidden_size), dtype=xp.float32)
        self.h = xp.zeros((batch_size, self.hidden_size), dtype=xp.float32)

    def encode(self, x):
        self.c, self.h = self.enc(x, self.c, self.h)

    def decode(self, y):
        y, self.c, self.h = self.dec(y, self.c, self.h)
        return y

    def save_spec(self, filename):
        with open(filename, "w", encoding="utf-8") as f:
            print(self.src_vocab, file=f)
            print(self.trg_vocab, file=f)
            print(self.embed_size, file=f)
            print(self.hidden_size, file=f)

    @staticmethod
    def load_spec(filename):
        with open(filename, encoding="utf-8") as f:
            src_vocab = int(next(f))
            trg_vocab = int(next(f))
            embed_size = int(next(f))
            hidden_size = int(next(f))
            return EncoderDecoder(src_vocab, trg_vocab, embed_size, hidden_size)

def forward(src_batch, trg_batch, src_vocab, trg_vocab, encdec, generation_limit, gpu=-1, is_training=False):
    batch_size = len(src_batch)
    src_len = len(src_batch[0])
    trg_len = len(trg_batch[0]) if trg_batch else None
    src_stoi = src_vocab.stoi
    trg_stoi = trg_vocab.stoi
    trg_itos = trg_vocab.itos

    xp = cuda.cupy if gpu >= 0 else np
    encdec.reset(batch_size, xp)

    x = xp.array([src_stoi("</s>") for _ in range(batch_size)], dtype=xp.int32)
    encdec.encode(x)
    for l in reversed(range(src_len)):
        x = xp.array([src_stoi(src_batch[k][l]) for k in range(batch_size)], dtype=xp.int32)
        encdec.encode(x)

    t = xp.asarray([trg_stoi("<s>") for _ in range(batch_size)], dtype=xp.int32)
    ret = [[] for _ in range(batch_size)]

    if is_training:
        loss = Variable(xp.zeros((), dtype=xp.float32))
        for l in range(trg_len):
            y = encdec.decode(t)
            t = xp.asarray([trg_stoi(trg_batch[k][l]) for k in range(batch_size)], dtype=xp.int32)
            loss += F.softmax_cross_entropy(y, t)
            output = cuda.to_cpu(y.data.argmax(1))
            for k in range(batch_size):
                ret[k].append(trg_itos(output[k]))
        return ret, loss

    else:
        while len(ret[0]) < generation_limit:
            y = encdec.decode(t)
            output = cuda.to_cpu(y.data.argmax(1))
            t = xp.asarray(output, dtype=xp.int32)
            for k in range(batch_size):
                ret[k].append(trg_itos(output[k]))
            if all("</s>" in ret[k] for k in range(batch_size)):
                break
        return ret

def train(args):
    trace("making vocabularies ...")
    src_vocab = Vocabulary.new(gens.word_list(args.source), args.src_vocab)
    trg_vocab = Vocabulary.new(gens.word_list(args.target), args.trg_vocab)

    trace("initializing model ...")
    encdec = EncoderDecoder(args.src_vocab, args.trg_vocab, args.embed, args.hidden)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        encdec.to_gpu()

    for epoch in range(args.epoch):
        trace("epoch {}/{}: ".format(epoch+1, args.epoch))
        trained = 0

        gen1 = gens.word_list(args.source)
        gen2 = gens.word_list(args.target)
        gen3 = gens.batch(gens.concat_generators(gen1, gen2), args.minibatch)

        opt = optimizers.AdaGrad(lr=0.01)
        opt.setup(encdec)
        opt.add_hook(chainer.optimizer.GradientClipping(5))

        for src_batch, trg_batch in gen3:
            src_batch = fill_batch(src_batch)
            trg_batch = fill_batch(trg_batch)

            res, loss = forward(src_batch, trg_batch, src_vocab, trg_vocab, encdec, 0, args.gpu, True)
            loss.backward()
            opt.update()

            for k in range(len(src_batch)):
                trace("epoch %d/%d, sample %8d"%(epoch+1, args.epoch, trained+k+1))
                trace("\tsrc = "+" ".join([x if x!="</s>" else "*" for x in src_batch[k]]))
                trace("\ttrg = "+" ".join([x if x!="</s>" else "*" for x in trg_batch[k]]))
                trace("\tres = "+" ".join([x if x!="</s>" else "*" for x in res[k]]))
            trained += len(src_batch)

        trace("saving model ...")
        prefix = args.model + ".%03.d"%(epoch+1)
        src_vocab.save(prefix + ".src.vocab")
        trg_vocab.save(prefix + ".trg.vocab")
        encdec.save_spec(prefix + ".spec")
        serializers.save_hdf5(prefix + ".model", encdec)

    trace("finished.")

def test(args):
    trace("loading model ...")
    src_vocab = Vocabulary.load(args.model + ".src.vocab")
    trg_vocab = Vocabulary.load(args.model + ".trg.vocab")
    encdec = EncoderDecoder.load_spec(args.model + ".spec")
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        encdec.to_gpu()
    serializers.load_hdf5(args.model + ".model", encdec)

    generated = 0
    with open(args.target, "w", encoding="utf-8") as f:
        for src_batch in gens.batch(gens.word_list(args.source), args.minibatch):
            src_batch = fill_batch(src_batch)

            trace("sample %8d - %8d ..."%(generated+1, generated+len(src_batch)))
            ret = forward(src_batch, None, src_vocab, trg_vocab,
                    encdec, args.generation_limit, args.gpu)

            for sent in ret:
                sent.append("</s>")
                sent = sent[:sent.index("</s>")]
                print(" ".join(sent), file=f)

            generated += len(src_batch)

    trace("finished.")

def main():
    args = get_args()
    if args.env == "train":
        train(args)
    elif args.env == "test":
        test(args)

if __name__ == "__main__":
    main()

