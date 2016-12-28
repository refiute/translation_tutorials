# coding: utf-8

import sys
import datetime

def trace(*args):
    print(datetime.datetime.now(), "...", *args, file=sys.stderr)
    sys.stderr.flush()

def fill_batch(batch, token="</s>"):
    max_len = max(len(x) for x in batch)
    return [x + [token] * (max_len - len(x) + 1) for x in batch]
