import json
import os
import re
import shutil

import numpy as np


def replace_digits_zero():
    """ replace every digit in a string to zero"""

    if not os.path.exists('./data/no_digit'):
        os.mkdir('./data/no_digit')

    raw_paths = ['./data/raw/train.txt',
                 './data/raw/valid.txt', './data/raw/test.txt']
    processed_paths = ['./data/no_digit/train.txt',
                       './data/no_digit/valid.txt', './data/no_digit/test.txt']

    for raw_path, processed_path in zip(raw_paths, processed_paths):
        with open(raw_path, 'r') as f:
            l = f.readlines()
        tmp_l = [l[0], l[1]]
        for line in l[2:]:
            tmp_str = re.sub(r'\d+', '0', line)
            tmp_l.append(tmp_str)

        with open(processed_path, 'w') as f:
            for line in tmp_l:
                f.write(line)


def counter():
    """ count the number of words and tags """
    WORD, TAG = dict(), dict()
    with open('./data/no_digit/train.txt', 'r') as f:
        l = f.readlines()

    for line in l[2:]:  # ignore the redundant top 2 lines
        line = line
        if line == '\n':
            continue
        word, tag = line.split()[0].lower(), line.split()[3]
        try:
            WORD[word] += 1
        except KeyError:
            WORD[word] = 1
        try:
            TAG[tag] += 1
        except KeyError:
            TAG[tag] = 1

    # sort
    TAG = sorted(TAG.items(), key=lambda x: x[1], reverse=True)
    WORD = sorted(WORD.items(), key=lambda x: x[1], reverse=True)

    with open('./data/tag_cnt.txt', 'w') as f:
        for item in TAG:
            f.write(item[0] + '\t' + str(item[1]) + '\n')

    return list(map(lambda x: x[0], WORD))


def build_vocab(words):
    """ build vocabulary for words, chars and tags """
    word2id, char2id = dict(), dict()

    # word vocab
    word2id['pad'] = 0
    word2id['unk'] = 1
    idx = 2
    for word in words:
        if word not in word2id:
            word2id[word] = idx
            idx += 1

    # char vocab
    char2id['pad'] = 0
    char_list = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                 '0', ':', ';', '=', '?', '@', '[', ']', '`',
                 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    idx = 1
    for c in char_list:
        char2id[c] = idx
        idx += 1

    # tag vocab
    tag2id = {'O': 0, 'B-ORG': 1, 'B-MISC': 2, 'B-PER': 3, 'B-LOC': 4,
              'I-ORG': 5, 'I-MISC': 6, 'I-PER': 7, 'I-LOC': 8}

    vocab = {'tag2id': tag2id, 'char2id': char2id, 'word2id': word2id}
    with open('./data/vocab.json', 'w') as f:
        json.dump(vocab, f)

    return vocab


def process_data():
    if not os.path.exists('./data/dataset'):
        os.mkdir('./data/dataset')

    corpus_paths = ['./data/no_digit/train.txt',
                    './data/no_digit/valid.txt', './data/no_digit/test.txt']
    dest_paths = ['./data/dataset/train.json',
                  './data/dataset/valid.json', './data/dataset/test.json']

    corpus = []
    for corpus_path in corpus_paths:
        with open(corpus_path, 'r') as f:
            l = f.readlines()
        sent_list, tags_list, sent_tmp, tag_tmp, samples = [], [], [], [], []
        for line in l[2:]:  # ignore the redundant top 2 lines
            if line == '\n':  # blank lines separate each sentence
                assert len(sent_tmp) == len(tag_tmp)  # sanity check
                sent_list.append(' '.join(sent_tmp))
                tags_list.append(' '.join(tag_tmp))
                sent_tmp.clear()
                tag_tmp.clear()
                continue
            word, tag = line.split()[0], line.split()[3]
            sent_tmp.append(word)
            tag_tmp.append(tag)

        # print({len(sent_list)})  # 14986, 3465, 3683

        for i in range(len(sent_list)):
            samples.append({'sentence': sent_list[i], 'tags': tags_list[i]})

        corpus.append(samples)

    for i, path in enumerate(dest_paths):
        with open(path, 'w') as f:
            json.dump(corpus[i], f)


def load_pretrained_emb(glove_path, embed_dim, vocab):
    words, vectors = [], []

    idx = 0
    word2id = dict()
    with open(glove_path, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2id[word] = idx
            vectors.append(np.array(line[1:]).astype(np.float))
            idx += 1

    glove = {w: vectors[word2id[w]] for w in words}

    id2word = {key: value for value, key in vocab['word2id'].items()}
    weights_matrix = np.zeros((len(id2word), embed_dim))

    for idx, word in id2word.items():
        idx = int(idx)
        try:
            weights_matrix[idx] = glove[word.lower()]
        except KeyError:
            weights_matrix[idx] = np.random.normal(
                scale=0.6, size=(embed_dim,))

    np.save('./data/weights_matrix.npy', weights_matrix)


if __name__ == "__main__":
    replace_digits_zero()
    words = counter()
    vocab = build_vocab(words)
    process_data()
    load_pretrained_emb(glove_path='./data/raw/glove.6B.100d.txt',
                        embed_dim=100,
                        vocab=vocab)
    shutil.rmtree('./data/no_digit')
