# coding: utf-8

def word_list(filename):
    with open(filename, encoding="utf-8") as f:
        for line in f:
            yield l.split()

def batch(generator, batch_size=-1):
    batch = []
    is_tuple = False
    for l in generator:
        is_tuple = isinstance(l, tuple)
        batch.append(l)
        if len(batch) == batch_size:
            yield tuple(list(x) for x in zip(*batch)) if is_tuple else batch
            batch = []
    if batch:
        yield tuple(list(x) for x in zip(*batch)) if is_tuple else batch

def concat_batches(generator1, generator2):
    gen1 = batch(generator1)
    gen2 = batch(generator2)

    for batch1, batch2 in zip(gen1, gen2):
        for x in zip(batch1, batch2):
            yield x
