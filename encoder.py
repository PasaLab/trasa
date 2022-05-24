import torch
import torch.nn.functional as F
from torch import nn

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, std=0.02)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

class RelationEncoder(nn.Module):
    def __init__(self, vocab, rel_dim, embed_dim, hidden_size, num_layers, dropout, bidirectional=True):
        super(RelationEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rel_embed = Embedding(vocab.size, rel_dim, vocab.padding_idx)
        self.rnn = nn.GRU(
            input_size=rel_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout if num_layers > 1 else 0.,
            bidirectional=bidirectional
        )
        tot_dim = 2 * hidden_size if bidirectional else hidden_size
        self.out_proj = nn.Linear(tot_dim, embed_dim)
    
    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, relation_bank, relation_length):
        seq_len, bsz = relation_bank.size()

        sorted_relation_length, indices = torch.sort(relation_length, descending=True)
        sorted_relation_bank = relation_bank.index_select(1, indices)

        x = self.rel_embed(sorted_relation_bank)
        x = F.dropout(x, p=self.dropout, training=self.training)

        packed_x = nn.utils.rnn.pack_padded_sequence(x, sorted_relation_length.tolist())

        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size 
        h0 = x.data.new(*state_size).zero_()
        _, final_h = self.rnn(packed_x, h0)

        if self.bidirectional:
            def combine_bidir(outs):
                return outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous().view(self.num_layers, bsz, -1)
            final_h = combine_bidir(final_h)

        _, positions = torch.sort(indices)
        final_h = final_h.index_select(1, positions) 
        output = self.out_proj(final_h[-1]) 
        return output