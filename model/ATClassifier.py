import torch
import torch.nn as nn
from modules.gcnn import GCNN, MWCNN
from modules.grl import GRLayer
from modules.layers import Embeddings
import torch.nn.functional as F


# AT4MTL
class ATClassifier(nn.Module):
    def __init__(self, args, pre_embed=None):
        super(ATClassifier, self).__init__()

        self.embed_drop = args.embed_drop
        self.dropout = args.dropout
        self.adv_share = args.adv_share
        self.share = args.share
        self.nb_task = args.nb_task

        self.word_embedding = Embeddings(num_embeddings=args.num_embeddings,
                                         embedding_dim=args.embed_dim,
                                         embed_weight=pre_embed,
                                         dropout=self.embed_drop)

        # task-independent
        '''
        self.share_enc = GCNN(input_size=args.embed_dim,
                              hidden_size=args.hidden_size//2,
                              output_size=args.hidden_size,
                              nb_layers=args.nb_layers,
                              kernel_size=args.kernel_size,
                              dropout=args.dropout)
        '''
        self.share_enc = MWCNN(input_size=args.embed_dim,
                               output_size=args.hidden_size,
                               dropout=args.dropout)

        self.share_fc = nn.Linear(args.hidden_size, self.nb_task)

        # task-dependent
        '''
        self.task_encs = nn.ModuleList([GCNN(input_size=args.embed_dim,
                                             hidden_size=args.hidden_size//2,
                                             output_size=args.hidden_size,
                                             nb_layers=args.nb_layers,
                                             kernel_size=args.kernel_size,
                                             dropout=args.dropout)
                                        for _ in range(self.nb_task)])
        '''
        self.task_encs = nn.ModuleList([MWCNN(input_size=args.embed_dim,
                                              output_size=args.hidden_size,
                                              dropout=args.dropout)
                                        for _ in range(self.nb_task)])

        task_input_size = (2 if self.adv_share or self.share else 1) * args.hidden_size
        self.task_fcs = nn.ModuleList([nn.Linear(task_input_size, 2) for _ in range(self.nb_task)])

    def task_sp(self, task_id, h, adv_lmbd=0.01):
        task_h = self.task_encs[task_id](h)
        share_h = self.share_enc(h)

        if self.adv_share:
            h_out = torch.cat((share_h, task_h), dim=1).contiguous()
            share_h = GRLayer.apply(share_h, adv_lmbd)
        elif self.share:
            h_out = torch.cat((share_h, task_h), dim=1).contiguous()
        else:  # no share
            h_out = task_h

        h_out = F.dropout(h_out, p=self.dropout, training=self.training)
        task_logits = self.task_fcs[task_id](h_out)
        share_logits = self.share_fc(share_h)
        return task_logits, share_logits

    def forward(self, task_id, inp, adv_lmbd=0.01):
        embed = self.word_embedding(inp)
        task_logits, share_logits = self.task_sp(task_id, embed, adv_lmbd)
        return task_logits, share_logits












