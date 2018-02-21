# -*- coding: utf-8 -*-
import argparse

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

from utility import load_wordidmap


class GenerateRNN(chainer.Chain):

    def __init__(self, n_words, nodes):
        super(GenerateRNN, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(n_words, n_words)
            self.l1 = L.LSTM(n_words, nodes)
            self.l2 = L.LSTM(nodes, nodes)
            self.l3 = L.Linear(nodes, n_words)

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(h0)
        h2 = self.l2(h1)
        y = self.l3(h2)
        return y


def main():
    index2word, _ = load_wordidmap(args.index_word_text)
    index2word[0] = "<sos>"
    word2index = {word: idx for idx, word in index2word.items()}
    print(args.index_word_text)
    n_words = len(index2word)
    model = GenerateRNN(n_words, 200)

    chainer.serializers.load_hdf5(args.model, model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(0).use()
        chainer.cuda.check_cuda_available()
        model.to_gpu()

    generate_text = []
    pretext_id = word2index[args.pretext]

    generate_text.append(pretext_id)

    limit = 0
    while pretext_id != 1 and limit < 20:
        y = model(np.array([[pretext_id]], dtype=np.int32))
        z = F.softmax(y)
        z.to_cpu()
        pretext_id = np.argmax(z.data[0])
        generate_text.append(pretext_id)
        limit += 1

    print(" ".join([index2word[i] for i in generate_text[:-1]]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning RNN Language Model')

    parser.add_argument('index_word_text', action='store', nargs=None, const=None, default=None,
                        type=str, choices=None, help='txt path where index to word.', metavar=None)
    parser.add_argument('--pretext', '-p', type=str, default='')
    parser.add_argument('--model', '-m', type=str, default=0)
    parser.add_argument('--gpu', '-g', type=int, default=0)

    args = parser.parse_args()

    # if args.gpu >= 0:
    #     import cupy as cp
    # else:
    #     cp = np

    main()
