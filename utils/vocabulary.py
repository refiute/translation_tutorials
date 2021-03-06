# coding: utf-8

from collections import defaultdict

class Vocabulary:
    def __init__(self):
        pass

    def __len__(self):
        return self.__size

    def stoi(self, s):
        return self.__stoi[s]

    def itos(self, i):
        return self.__itos[i]

    @staticmethod
    def new(generator, size):
        self = Vocabulary()
        self.__size = size

        count = defaultdict(int)
        for words in generator:
            for word in words:
                count[word] += 1

        stoi = defaultdict(lambda: 0)
        stoi["<unk>"] = 0
        stoi["<s>"] = 1
        stoi["</s>"] = 2
        for (s, _) in list(sorted(count.items(), key=lambda x: -x[1]))[:(size-3)]:
            stoi[s] = len(stoi)

        itos = [''] * size
        for (s, i) in stoi.items():
            itos[i] = s

        self.__stoi = stoi
        self.__itos = itos

        return self

    def save(self, filename):
        with open(filename, "w", encoding="utf-8") as f:
            print(self.__size, file=f)
            for i in range(self.__size):
                print(self.__itos[i], file=f)

    @staticmethod
    def load(filename):
        with open(filename, encoding="utf-8") as f:
            self = Vocabulary()
            self.__size = int(next(f))
            self.__stoi = defaultdict(lambda: 0)
            self.__itos = [""] * self.__size
            for i in range(self.__size):
                s = next(f).strip()
                if s:
                    self.__stoi[s] = i
                    self.__itos[i] = s

        return self
