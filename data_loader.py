import re

import torch
import torch.utils.data as Data
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence

from utils import load_file


def get_data_loader(dataset, batch_size, shuffle=False):
    def collate_fn(batch_data):
        batch_data.sort(key=lambda x: len(x[0]), reverse=True)
        sent, tags, chars = zip(*batch_data)
        batch_lens = torch.tensor([len(w)
                                   for w in sent], dtype=torch.long).cuda()
        sent = pad_sequence(sent, batch_first=True, padding_value=0)
        tags = pad_sequence(tags, batch_first=True, padding_value=0)

        # special padding for characters
        max_len_char = max([len(w) for s in chars for w in s])
        sequence = []
        for s in chars:
            c_padded = pad_sequence(s, batch_first=True, padding_value=0)
            c_padded = pad(c_padded, [0, max_len_char -
                                      c_padded.size(1)], mode='constant', value=0).cuda()
            sequence.append(c_padded)
        chars = pad_sequence(sequence, batch_first=True)
        return sent, tags, chars, batch_lens

    data_loader = Data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  collate_fn=collate_fn)
    return data_loader


class ConllDataset(Data.Dataset):
    def __init__(self, corpus_path, vocab):
        super(ConllDataset, self).__init__()

        samples = load_file(corpus_path)
        s_samples = [s['sentence'] for s in samples]
        t_samples = [s['tags'] for s in samples]

        word2id = vocab['word2id']
        char2id = vocab['char2id']
        tag2id = vocab['tag2id']

        self.sentences_tensors, self.tags_tensors, self.characters_tensors = [], [], []
        for s_sample, t_sample in zip(s_samples, t_samples):
            words, tags = s_sample.split(), t_sample.split()
            assert len(words) == len(tags)
            words_tensor = torch.zeros(len(words), 2, dtype=torch.long).cuda()
            tags_tensor = torch.zeros(len(tags), dtype=torch.long).cuda()
            chars_tensor = []

            for idx, (word, tag) in enumerate(zip(words, tags)):
                tags_tensor[idx] = tag2id[tag]
                if re.search(r'\d+', word):
                    if re.search(r'[a-zA-Z]+', word):
                        # mix of letters and digits
                        words_tensor[idx, 0] = 1
                    else:
                        # all digits
                        words_tensor[idx, 0] = 2
                else:
                    if word.lower() == word:
                        # all lower case letters
                        words_tensor[idx, 0] = 3
                    elif word.upper() == word:
                        # all capital letters
                        words_tensor[idx, 0] = 4
                    elif re.search(r'^[^A-Za-z0-9]*[A-Z]', word):
                        # start with capital letter
                        words_tensor[idx, 0] = 5
                    else:
                        # else conditions
                        words_tensor[idx, 0] = 0
                try:
                    words_tensor[idx, 1] = word2id[word.lower()]
                except KeyError:
                    words_tensor[idx, 1] = word2id['unk']

                char_tensor = torch.zeros(len(word), dtype=torch.long).cuda()
                for c_idx, char in enumerate(word):
                    char_tensor[c_idx] = char2id[char.lower()]
                chars_tensor.append(char_tensor)

            self.sentences_tensors.append(words_tensor)
            self.tags_tensors.append(tags_tensor)
            self.characters_tensors.append(chars_tensor)

    def __len__(self):
        return len(self.sentences_tensors)

    def __getitem__(self, index):
        sent = self.sentences_tensors[index]
        tags = self.tags_tensors[index]
        chars = self.characters_tensors[index]
        return sent, tags, chars
