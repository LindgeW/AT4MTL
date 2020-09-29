import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rnn import LSTM


class Embeddings(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, embed_weight=None, pad_idx=0, unk_idx=None, dropout=0.0, word_dropout=0.0):
        super(Embeddings, self).__init__()
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.dropout = dropout
        self.word_dropout = word_dropout
        self.embeddings = nn.Embedding(num_embeddings=num_embeddings,
                                       embedding_dim=embedding_dim,
                                       padding_idx=pad_idx)
        if embed_weight is None:
            self.reset_params()
        else:
            # self.embeddings = nn.Embedding.from_pretrained(torch.from_numpy(embed_weight), freeze=True)
            self.embeddings.weight.data.copy_(torch.from_numpy(embed_weight))

        if word_dropout > 0:
            assert unk_idx is not None

    def reset_params(self):
        nn.init.xavier_uniform_(self.embeddings.weight)
        with torch.no_grad():
            self.embeddings.weight[self.pad_idx].fill_(0)

    @property
    def requires_grad(self):
        return self.embeddings.weight.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        self.embeddings.weight.requires_grad = value

    @property
    def weight(self):
        return self.embeddings.weight

    def _drop_word(self, words):
        r"""
        按照一定概率随机将word设置为unk_index，这样可以使得unk这个token得到足够的训练,
        且会对网络有一定的regularize的作用。设置该值时，必须同时设置unk_index
        """
        drop_probs = torch.ones_like(words).float() * self.word_dropout
        # drop_probs = torch.full_like(words, fill_value=self.word_dropout, dtype=torch.float, device=words.device)
        drop_mask = torch.bernoulli(drop_probs).eq(1)  # dropout_word越大，越多位置为1
        pad_mask = words.ne(self.pad_idx)
        mask = drop_mask & pad_mask
        words = words.masked_fill(mask, self.unk_idx)
        return words

    def forward(self, x):
        if self.word_dropout > 0 and self.training:
            x = self._drop_word(x)

        embed = self.embeddings(x)
        embed = F.dropout(embed, p=self.dropout, training=self.training)
        return embed


def sinusoid_encoding(nb_pos, dim, pad_idx=None):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / dim)

    def get_pos_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(dim)]

    sinusoid_table = np.array([get_pos_angle_vec(pos_i) for pos_i in range(nb_pos)], dtype=np.float32)
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if pad_idx is not None:
        sinusoid_table[pad_idx] = 0.

    return sinusoid_table


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEmbedding, self).__init__()
        inv_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        '''
        :param pos_seq: (seq_len, )
        :return:
        '''
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat(tuple([sinusoid_inp.sin(), sinusoid_inp.cos()]), dim=-1)
        # return pos_emb[:, None, :]  # (seq_len, bz, dim)
        return pos_emb[None, :, :]    # (bz, seq_len, dim)

