# -*- coding: utf-8 -*-
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
import cupy


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


class RNNUpdater(training.StandardUpdater):
    def __init__(self, train_iter, optimizer, device):
        super(RNNUpdater, self).__init__(
            train_iter,
            optimizer,
            device=device
        )
        self.xp = cupy if device == 0 else np
        
    def update_core(self):
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        model = optimizer.target

        model.reset_state()
        loss = 0
        x = self.xp.asarray(train_iter.__next__(), dtype=self.xp.int32)

        for i in range(len(x[0]) - 1):
            batch = x[:, i]
            t = x[:, i + 1]

            if self.xp.min(batch) == 1 and self.xp.max(batch) == 1:
                break

            y = model(batch)
            loss += F.softmax_cross_entropy(y, t)

        optimizer.target.cleargrads()
        loss.backward()
        optimizer.update()
