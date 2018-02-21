# -*- coding: utf-8 -*-
import argparse

import numpy as np
import chainer
from chainer import training, iterators, optimizers
from chainer.training import extensions

from rnn_laguage_model import GenerateRNN, RNNUpdater
from utility import load_wordid_text, adjust_data_length


def main():
    index_text_filepath = args.index_sentence_text
    print(index_text_filepath)

    wordid_list = load_wordid_text(index_text_filepath)
    wordid_list = adjust_data_length(wordid_list)
    n_words = max([max(x) for x in wordid_list]) + 1

    model = GenerateRNN(n_words, 200)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(0).use()
        chainer.cuda.check_cuda_available()
        model.to_gpu()

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    wordid_list = np.asarray(wordid_list, dtype=np.int32)
    train_iter = iterators.SerialIterator(wordid_list, args.batch_size, shuffle=True)
    updater = RNNUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out="result")
    trainer.extend(extensions.ProgressBar(update_interval=1))

    trainer.run()

    chainer.serializers.save_hdf5("rnnlm.hdf5", model)


def test():
    # RNNUpdater test
    dataset = [[i for i in range(10)] for _ in range(100)]
    dataset = np.asarray(dataset, dtype=np.int32)

    print(dataset)

    train_iter = iterators.SerialIterator(dataset, 10, shuffle=True)
    print(train_iter.dataset.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning RNN Language Model')

    parser.add_argument('index_sentence_text', action='store', nargs=None, const=None, default=None,
                        type=str, choices=None, help='Separated text data for learning', metavar=None)

    parser.add_argument('--batch_size', '-b', type=int, default=500,
                        help='Number of sentences in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')

    args = parser.parse_args()

    # if args.gpu >= 0:
    #     import cupy as cp
    # else:
    #     cp = np

    main()
