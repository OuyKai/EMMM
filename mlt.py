import math

import numpy as np
import torch as th
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def attention(query, key, value, mask=None):
    d_k = query.shape[-1]
    scores = th.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return th.matmul(p_attn, value), p_attn


# class MultiHeadedAttention(nn.Module):
#     def __init__(self, hidden_dim, num_heads=1, dropout=0.1):
#         "Take in model size and number of heads."
#         super(MultiHeadedAttention, self).__init__()
#         assert hidden_dim % num_heads == 0
#         # We assume d_v always equals d_k
#         self.d_k = hidden_dim // num_heads
#         self.num_heads = num_heads
    
#         self.linears = self.clones(nn.Linear(hidden_dim, hidden_dim), 4)
#         self.dropout = nn.Dropout(p=dropout)

#     def clones(self, module, N):
#         "Produce N identical layers."
#         return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

#     def forward(self, query, key, value, mask=None, linear=False):
#         "Implements Figure 2"
#         if mask is not None:
#             # Same mask applied to all h heads.
#             mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
#         nbatches = query.size(0)

#         # 1) Do all the linear projections in batch from d_model => h x d_k
#         if linear is True:
#             query, key, value = [
#                 lin(x).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
#                 for lin, x in zip(self.linears, (query, key, value))
#             ]
#         else:
#             query, key, value = [
#                 x.view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
#                 for x in (query, key, value)
#             ]

#         # 2) Apply attention on all the projected vectors in batch.
#         x, _ = attention(query, key, value, mask=mask, dropout=None)
#         x = self.dropout(x)

#         # 3) "Concat" using a view and apply a final linear.
#         x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
#         return self.linears[-1](x) if linear is True else x


class ReattnLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0):
        super(ReattnLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.activate = F.relu
        self.LN = nn.LayerNorm(hidden_dim, eps=0.001)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.reattn_linear0 = nn.Linear(hidden_dim, hidden_dim)
        self.reattn_linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.reattn_linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, emb_seqs, pad_mask, tgt_mask):
        pad_mask = pad_mask.unsqueeze(1)
        query = self.activate(self.reattn_linear0(emb_seqs))
        if self.dropout is not None:
            query = self.dropout(query)
        intents, _ = attention(query, emb_seqs, emb_seqs, tgt_mask)

        residual_intents = self.activate(self.reattn_linear1(intents))
        if self.dropout is not None:
            residual_intents = self.dropout(residual_intents)
        intents = self.LN(residual_intents + intents)

        last_intent = intents[:, -1, :].unsqueeze(1)
        intent_attn, _ = attention(last_intent, intents, intents, pad_mask)

        return intent_attn.squeeze()


class DNN(nn.Module):
    def __init__(self, hidden_sizes=[300, 100], dropout=0.3):
        super(DNN, self).__init__()
        self.depth = len(hidden_sizes) - 1
        self.dnns = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(self.depth)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_size) for hidden_size in hidden_sizes[1:]])
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, x):
        for i in range(self.depth):
            x = self.dnns[i](x)
            x = self.bns[i](x)
            if self.dropout is not None:
                x = self.dropout(x)
        return x


class REATTN(nn.Module):
    def __init__(self, embedding_dim, n_items, n_pos, coefficient=20, dropout=0.3):
        super(REATTN, self).__init__()
        self.embedding = nn.Embedding(
            n_items, embedding_dim, padding_idx=0, max_norm=1.5
        )
        self.pos_embedding = nn.Embedding(
            n_pos, embedding_dim, padding_idx=0, max_norm=1.5
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.feat_drop = nn.Dropout(p=dropout)

        self.attn = ReattnLayer(embedding_dim, dropout=dropout)
        self.dnn = DNN(hidden_sizes=[3 * embedding_dim, embedding_dim], dropout=dropout)
        self.gru = nn.GRU(embedding_dim, embedding_dim, batch_first=True, dropout=dropout)

        self.scorer_linear = nn.Linear(2 * embedding_dim, embedding_dim)
        self.activate = F.relu
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=0.001)
        self.coefficient = coefficient

    def subsequent_mask(self, size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
        return th.from_numpy(subsequent_mask) == 0

    def make_std_mask(self, tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            self.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        )
        return tgt_mask

    def scorer(self, intent_attn):
        intent = intent_attn
        if self.dropout is not None:
            intent = self.dropout(intent)

        intent = intent.squeeze()
        user_intent = intent / th.norm(intent, dim=-1).unsqueeze(1)
        item_embedding = self.embedding.weight[1:].clone() / th.norm(
            self.embedding.weight[1:].clone(), dim=-1
        ).unsqueeze(1)
        score = self.coefficient * th.matmul(user_intent, item_embedding.t())
        return score

    def repeat(self, x, pos):
        pad_mask = (x != 0).float()  # B,seq
        tgt_mask = self.make_std_mask(x, 0)
        x_embeddings = self.embedding(x)  # B,seq,dim
        pos_embeddings = self.pos_embedding(pos)  # B, seq, dim
        emb_seqs = x_embeddings + pos_embeddings
        if self.feat_drop is not None:
            emb_seqs = self.feat_drop(emb_seqs)
        
        return emb_seqs, pad_mask, tgt_mask

    def forward(self, x, pos, predict=0):
        emb_seqs, pad_mask, tgt_mask = self.repeat(x, pos)
        x_embeddings = self.embedding(x)  # B,seq,dim
        if self.feat_drop is not None:
            x_embeddings = self.feat_drop(x_embeddings)
        # global
        self.intent0 = self.attn(emb_seqs, pad_mask, tgt_mask).squeeze()
        # local
        self.intent1 = self.dnn(x_embeddings[:, -3:, :].reshape(x_embeddings.shape[0], -1))
        # sequential
        self.intent2 = self.gru(x_embeddings)[0][:,-1,:]

        intent_attn = self.scorer_linear(th.cat([self.intent1, self.intent2], dim=-1))
        intent_attn = self.intent0 + intent_attn
        if self.layer_norm is not None:
            intent_attn = self.layer_norm(intent_attn)
        result = self.scorer(intent_attn)

        if predict > 0:
            rank = th.argsort(result, dim=1, descending=True)
            return rank[:, 0:predict]
        
        result0 = self.scorer(self.intent0)
        result1 = self.scorer(self.intent1)
        result2 = self.scorer(self.intent2)
        return result0, result1, result2, result

