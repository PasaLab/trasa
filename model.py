import math
from operator import pos
from typing import ItemsView
import torch
import torch.nn.functional as F
from torch import nn
from encoder import RelationEncoder
from collate import relation_vocab
from transformer import Transformer


class TRASA(nn.Module):
    def __init__(
        self, embed_dim,item_num,
        rel_dim, rnn_hidden_size, rnn_num_layers,
        graph_layers, ff_embed_dim, num_heads,
        dropout, device,
    ):
        super(TRASA, self).__init__()
        self.dropout = dropout
        self.device = device
        self.embed_dim = embed_dim

        self.item_embedding = nn.Embedding(item_num, embed_dim)
        self.graph_item_depth_embedding = nn.Embedding(200, embed_dim)
        self.graph_item_embed_layer_norm = nn.LayerNorm(embed_dim)


        self.relation_encoder = RelationEncoder(relation_vocab, rel_dim, embed_dim, rnn_hidden_size, rnn_num_layers, dropout)
        self.graph_encoder = Transformer(graph_layers, embed_dim, ff_embed_dim, num_heads, dropout)
    
        self.embed_scale = math.sqrt(embed_dim)
        self.pos_embedding = nn.Embedding(200, embed_dim)

        self.attn = nn.Parameter(torch.Tensor(embed_dim, 1))
        self.glu1 = nn.Linear(embed_dim, embed_dim)
        self.glu2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.reset_parameters()
     
    def reset_parameters(self):
        nn.init.normal_(self.attn.data, std=0.02)

    def encode_graph(self, inp):
        item_repr_g = self.embed_scale * self.item_embedding(inp['items'])  + self.graph_item_depth_embedding(inp['depths'])
        # item_repr_g = self.item_embedding(inp['items'])
        item_repr_g = self.graph_item_embed_layer_norm(item_repr_g)
        item_mask_g = torch.eq(inp['items'], inp['item_padding_idx'])

        relation = self.relation_encoder(inp['relation_bank'], inp['relation_length'])
        relation = relation.index_select(0, inp['relations'].reshape(-1)).reshape(*inp['relations'].size(), -1)
        item_repr_g = self.graph_encoder(item_repr_g, relation, self_padding_mask=item_mask_g)

        item_repr_g = item_repr_g[1:]
        item_mask = item_mask_g[1:]
        return item_repr_g
    
    def forward(self, data):
        item_repr_g = self.encode_graph(data)
        item_repr_g = item_repr_g.transpose(0, 1)
        seq_selects = data['seq_selects'].transpose(0, 1)
        item_repr_s = torch.matmul(seq_selects.to(torch.float32), item_repr_g)
        item_mask_s = torch.eq(data['seqs'], data['item_padding_idx'])
        item_repr_s = item_repr_s.transpose(0, 1)
        # without self-attention
        # item_repr_s = self.graph_item_embedding(data['seqs'])
        max_len = item_repr_s.shape[0]
        bsz = item_repr_s.shape[1]
        pos_emb = self.pos_embedding.weight[:max_len]
        pos_emb = pos_emb.unsqueeze(1).repeat(1, bsz, 1)
        item_repr_s = item_repr_s + pos_emb
        
        last_node_feat = item_repr_s[:1].repeat(max_len, 1, 1)
        # h = torch.cat([self.glu1(item_repr_s), self.glu2(last_node_feat)], dim=-1)
        h = self.glu1(item_repr_s) + self.glu2(last_node_feat)
        alpha = torch.matmul(h, self.attn)
        # alpha = self.attn(h)
        alpha.masked_fill_(item_mask_s.unsqueeze(-1), float('-inf'))
        alpha = F.softmax(alpha, dim=0)
        feats = item_repr_s * alpha
        readout = torch.sum(feats, dim=0)
        b = self.item_embedding.weight[2:]
        b = F.normalize(b, p=2, dim=1)
        scores = torch.matmul(readout, b.transpose(1, 0))
        return scores
    
    def sum_pooling(self, data):
        item_repr_g = self.encode_graph(data)
        item_repr_g = item_repr_g.transpose(0, 1)
        seq_selects = data['seq_selects'].transpose(0, 1)
        item_repr_s = torch.matmul(seq_selects.to(torch.float32), item_repr_g)
        item_mask_s = torch.eq(data['seqs'], data['item_padding_idx'])
        item_repr_s = item_repr_s.transpose(0, 1)