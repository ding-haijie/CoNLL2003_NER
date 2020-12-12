from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CRF(nn.Module):
    def __init__(self, vocab_size):
        """ Linear CRF Model """
        super(CRF, self).__init__()

        self.tag_size = tag_size = vocab_size + 2
        self.START_TAG, self.END_TAG = tag_size - 2, tag_size - 1
        self.INF_MIN = -10000.  # log(0)

        # transition scores, entry_(i,j) is the score of transitioning *from* i *to* j.
        self.transitions = nn.Parameter(torch.randn(
            tag_size, tag_size), requires_grad=True)
        # enforce the constraints that never transmit *to* START_TAG or *from* END_TAG
        self.transitions.data[:, self.START_TAG] = self.INF_MIN
        self.transitions.data[self.END_TAG, :] = self.INF_MIN

    def forward(self, features, batch_len):
        # type: (Tensor, Tensor)  -> Tensor
        """
        :param features: FloatTensor(batch_size, seq_len, tag_size)
        :param batch_len: LongTensor(batch_size)
        :return: FloatTensor(batch_size)
        """
        batch_size = features.size(0)

        # initialize the forward variables in log space
        alpha = features.data.new_full(
            (batch_size, self.tag_size), self.INF_MIN)
        alpha[:, self.START_TAG] = 0.
        clone_batch_len = batch_len.clone()

        # FloatTensor(seq_len, batch_size, tag_size)
        features_t = features.transpose(1, 0)
        # iterate through the sentences
        for feats in features_t:  # feats: FloatTensor(batch_size, tag_size)
            emit_score = feats.unsqueeze(-1).expand(batch_size,
                                                    *self.transitions.size())
            trans_score = self.transitions.unsqueeze(0).expand_as(emit_score)
            alpha_exp = alpha.unsqueeze(1).expand_as(emit_score)
            alpha_next = log_sum_exp(
                emit_score + trans_score + alpha_exp, dim=2).squeeze(-1)

            mask = (clone_batch_len > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_next + (1 - mask) * alpha
            clone_batch_len = clone_batch_len - 1

        alpha = alpha + \
            self.transitions[self.END_TAG].unsqueeze(0).expand_as(alpha)

        return log_sum_exp(alpha, dim=1).squeeze(-1)

    def viterbi_decode(self, features, batch_len):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        """
        :param features: FloatTensor(batch_size, seq_len, tag_size)
        :param batch_len: LongTensor(batch_size)
        :return: FloatTensor(batch_size), LongTensor(batch_size, seq_len)
        """
        back_trace = []  # preserve the predicted tags
        batch_size = features.size(0)

        # initialize the viterbi variables in log space
        viterbi = features.data.new_full(
            (batch_size, self.tag_size), self.INF_MIN)
        viterbi[:, self.START_TAG] = 0.
        clone_batch_len = batch_len.clone()

        # FloatTensor(seq_len, batch_size, tag_size)
        features_t = features.transpose(1, 0)
        # iterate through the sentences
        for feats in features_t:  # feats: FloatTensor(batch_size, tag_size)
            viterbi_exp = viterbi.unsqueeze(1).expand(
                batch_size, *self.transitions.size())
            trans_exp = self.transitions.unsqueeze(0).expand_as(viterbi_exp)
            vtb_trans_sum = viterbi_exp + trans_exp

            vtb_max_value, vtb_max_idx = vtb_trans_sum.max(dim=2)
            vtb_next = vtb_max_value.squeeze(-1) + feats

            back_trace.append(vtb_max_idx.squeeze(-1).unsqueeze(0))

            mask = (clone_batch_len > 0).float(
            ).unsqueeze(-1).expand_as(vtb_next)
            viterbi = mask * vtb_next + (1 - mask) * viterbi

            mask = (clone_batch_len == 1).float(
            ).unsqueeze(-1).expand_as(vtb_next)
            viterbi += mask * \
                self.transitions[self.END_TAG].unsqueeze(0).expand_as(vtb_next)

            clone_batch_len = clone_batch_len - 1

        back_trace = torch.cat(back_trace)
        score, best_idx = viterbi.max(dim=1)
        best_path = [best_idx.unsqueeze(dim=1)]

        for bpt in reversed(back_trace):
            idx_exp = best_idx.unsqueeze(-1)
            best_idx = torch.gather(bpt, 1, idx_exp).squeeze(-1)
            best_path.insert(0, best_idx.unsqueeze(1))

        best_path = torch.cat(best_path[1:], dim=1)
        score = score.squeeze(-1)

        return score, best_path

    def transition_score(self, tags, batch_len):
        # type: (Tensor, Tensor) -> Tensor
        """
        :param tags: LongTensor(batch_size, seq_len)
        :param batch_len: LongTensor(batch_size)
        :return: Tensor(batch_size)
        """
        batch_size, seq_len = tags.size()

        # pad with <start> and <stop>
        tags_ext = tags.data.new_empty(batch_size, seq_len + 2)
        tags_ext[:, 0] = self.START_TAG
        tags_ext[:, 1:-1] = tags
        mask = sequence_mask(batch_len + 1, max_len=seq_len + 2).long()
        pad_stop = tags.data.new_full((1,), self.END_TAG)
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        tags = (1 - mask) * pad_stop + mask * tags_ext

        # obtain transition vector for each tags in batch and timestep, except the last one
        trans_exp = self.transitions.unsqueeze(0).expand(
            batch_size, *self.transitions.size())
        tags_r = tags[:, 1:]
        tags_r_exp = tags_r.unsqueeze(-1).expand(*tags_r.size(), self.tag_size)
        trans_row = torch.gather(trans_exp, 1, tags_r_exp)

        # obtain transition score from the transition vector, except the first one
        tags_l = tags[:, :-1]
        tags_l_exp = tags_l.unsqueeze(-1)
        trans_score = torch.gather(trans_row, 2, tags_l_exp)
        trans_score = trans_score.squeeze(-1)

        mask = sequence_mask(batch_len + 1).float()
        trans_score = mask * trans_score
        score = trans_score.sum(1).squeeze(-1)

        return score


class NER(nn.Module):
    def __init__(self, crf, word_size, char_size, feature_size, feature_dim,
                 word_embed_dim, char_embed_dim, hidden_dim, dropout_p):
        """ Bidirectional LSTM-CRF model """
        super(NER, self).__init__()

        self.crf = crf
        self.word_embed_dim = word_embed_dim
        self.char_embed_dim = char_embed_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.word_size = word_size
        self.char_size = char_size
        self.feature_size = feature_size
        self.tag_size = tag_size = self.crf.tag_size

        # neural networks
        self.word_embed_layer, self.char_embed_layer, self.feat_embed_layer = self._build_emb_layer()

        self.lstm_char = nn.LSTM(char_embed_dim, char_embed_dim, num_layers=1,
                                 bidirectional=True, batch_first=True)
        self.fc_lstm_char = nn.Linear(char_embed_dim * 2, char_embed_dim)

        self.fc_input = nn.Linear(
            word_embed_dim + feature_dim + char_embed_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.fc_output = nn.Linear(hidden_dim * 2, tag_size)

    def _build_emb_layer(self):
        # word embedding layer
        # word_embed_layer = nn.Embedding(self.word_size, self.word_embed_dim)
        weight = torch.from_numpy(np.load('./data/weights_matrix.npy')).float()
        word_embed_layer = nn.Embedding.from_pretrained(weight)
        word_embed_layer.weight.requires_grad = True

        # character embedding layer
        char_embed_layer = nn.Embedding(self.char_size, self.char_embed_dim)

        # feature embedding layer
        feat_embed_layer = nn.Embedding(self.feature_size, self.feature_dim)

        return word_embed_layer, char_embed_layer, feat_embed_layer

    def forward_lstm(self, sent_inp, char_inp, batch_lens):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        """
        :param sent_inp: Tensor(batch, seq_len, 2)
        :param char_inp: Tensor(batch, seq_len, word_len)
        :param batch_lens: Tensor(batch_size)
        :return Tensor(batch_size, seq_len, tag_size)
        """
        sent_feat = sent_inp[:, :, 0]
        sent_word = sent_inp[:, :, 1]
        word_embed = self.word_embed_layer(sent_word)
        feat_embed = self.feat_embed_layer(sent_feat)

        # character embedding
        char_outputs = []
        # updated char_inp of shape(seq_len, batch, char_len)
        char_inp = char_inp.permute(1, 0, 2)
        for word in char_inp:
            _char_embed = self.char_embed_layer(word)
            _, (h_n, _) = self.lstm_char(_char_embed)
            h_n = self.fc_lstm_char(torch.cat((h_n[-2], h_n[-1]), dim=1))
            char_outputs.append(h_n)
        char_embed = torch.stack(char_outputs).permute(1, 0, 2)

        embed = self.dropout(self.fc_input(
            torch.cat((word_embed, feat_embed, char_embed), dim=2)))

        # Bidirectional LSTM
        packed = pack_padded_sequence(
            embed, batch_lens.data.tolist(), batch_first=True)
        h_0 = self.init_hidden(
            bidirectional=True, num_layers=1, batch=sent_inp.size(0))
        output, _ = self.lstm(packed, h_0)
        output, _ = pad_packed_sequence(output, batch_first=True)

        features_vector = self.fc_output(output)

        return features_vector

    def gold_score(self, features, tags, batch_len):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        """
        :param features: Tensor(batch_size, seq_len, tag_size)
        :param tags: Tensor(batch_size, seq_len)
        :param batch_len: Tensor(batch_size)
        :return: Tensor
        """
        trans_score = self.crf.transition_score(tags, batch_len)

        tags_exp = tags.unsqueeze(-1)
        emit_score = torch.gather(features, 2, tags_exp).squeeze(-1)
        mask = sequence_mask(batch_len).float()
        emit_score = mask * emit_score
        emit_score = emit_score.sum(dim=1).squeeze(-1)

        score = trans_score + emit_score

        return score

    def neg_log_likelihood(self, sentence, tags, chars, batch_lens):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
        """
        :param sentence: Tensor(batch_size, seq_len)
        :param tags: Tensor(batch_len, seq_len)
        :param chars: Tensor(batch_size, seq_len, word_len)
        :param batch_lens: Tensor(batch_len)
        """
        features = self.forward_lstm(sentence, chars, batch_lens)
        gold_score = self.gold_score(features, tags, batch_lens)
        forward_score = self.crf(features, batch_lens)

        return torch.mean(forward_score - gold_score)

    def forward(self, sentence, chars, batch_len):
        # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
        features = self.forward_lstm(sentence, chars, batch_len)
        score, path = self.crf.viterbi_decode(features, batch_len)
        return score, path

    def init_hidden(self, bidirectional, num_layers, batch):
        # type: (bool, int, int) -> Tuple[Tensor, Tensor]
        if bidirectional:
            return (torch.randn(2 * num_layers, batch, self.hidden_dim).cuda(),
                    torch.randn(2 * num_layers, batch, self.hidden_dim).cuda())
        else:
            return (torch.randn(1 * num_layers, batch, self.hidden_dim).cuda(),
                    torch.randn(1 * num_layers, batch, self.hidden_dim).cuda())


def log_sum_exp(vector, dim=1):
    # type: (Tensor, int) -> Tensor
    """
    :param vector: FloatTensor(batch_size, tag_size, tag_size)
    :param dim: the dimension to reduce
    :return: Tensor(batch_size, tag_size)
    """
    max_val, _ = torch.max(vector, dim)
    max_bc = max_val.unsqueeze(-1).expand_as(vector)
    return max_val + torch.log(torch.sum(torch.exp(vector - max_bc), dim))


def sequence_mask(lens, max_len=None):
    """
    :param lens: Tensor(batch_size)
    :param max_len: int
    :return Tensor(batch_size, seq_len)
    """
    batch_size = lens.size(0)
    if max_len is None:
        max_len = lens.max().data.item()

    ranges = torch.arange(0, max_len).long().cuda()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)

    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp

    return mask
