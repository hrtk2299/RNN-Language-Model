# -*- coding: utf-8 -*-


def create_id_dataset(text_iterator, end_symbol="."):
    id_dataset = []
    word2index = {"<bos>": 0, "<eos>": 1}
    for line in text_iterator:
        line = line.strip().strip(end_symbol)
        line += ' <eos>'
        for word in line.split():
            if word not in word2index:
                ind = len(word2index)
                word2index[word] = ind
            id_dataset.append(word2index[word])

    return id_dataset, word2index


def load_wordid_text(wordid_text_filepath):
    wordid_list = []
    with open(wordid_text_filepath, 'r') as f:
        for line in f:
            wordid_list.append([0] + [int(x) for x in line.strip().split(',')] + [1])

    return wordid_list


def load_wordidmap(wordidmap_filepath):
    index2word = {}
    freq_id_list = []
    with open(wordidmap_filepath, 'r') as f:
        for line in f:
            idx_str, word, count = line.strip().split(',')
            index2word[int(idx_str)] = word
            freq_id_list.append(int(count))

    return index2word, freq_id_list


def adjust_data_length(wordid_list):
    max_len = max([len(x) for x in wordid_list])
    for l in wordid_list:
        l.extend([1] * (max_len - len(l)))

    return wordid_list
