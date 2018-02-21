# -*- coding: utf-8 -*-
import argparse
from utility import create_id_dataset
from collections import Counter

def main():
    """
    Create the following two files from the separated text file
    1. File converted from text to word ID
    2. A file in which words and IDs are associated one-to-one
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('separated_text_filepath', type=str, help='Separated text filepath.')
    parser.add_argument('--end_symbol', '-e', default='.', type=str,
                        choices=['.', '。'], help='end symbol format.')

    args = parser.parse_args()
    text_filepath = args.separated_text_filepath
    end_symbol = args.end_symbol

    with open(text_filepath, "r") as f:
        id_dataset, word2index = create_id_dataset(f, end_symbol)

    word_counter = Counter(id_dataset)

    with open("all-words.txt", "w") as f:
        for word, idx in word2index.items():
            f.write(f"{idx},{word},{word_counter[idx]}\n")
    print("output: all-words.txt")

    index_list = []
    with open("all-sentences.txt", "w") as f:
        for word_id in id_dataset:
            if word_id == 1:
                s = ",".join([str(n) for n in index_list])
                f.write(f"{s}\n")
                index_list = []
            else:
                index_list.append(word_id)
    print("output: all-sentences.txt")


if __name__ == '__main__':
    main()
