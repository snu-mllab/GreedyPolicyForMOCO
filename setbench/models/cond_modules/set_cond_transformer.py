import torch
import torch.nn as nn
from setbench.models.cond_modules.base_modules import MLP, PositionalEncoding
from setbench.models.set.deepsets import CondEmbedding

class SetCondTransformer(nn.Module):
    def __init__(self, num_hid, cond_dim, max_len, vocab_size, num_actions, dropout, num_layers,
                num_head, use_cond, update_cond=False, **kwargs):
        super().__init__()
        self.use_cond = use_cond
        self.update_cond = update_cond
        assert self.use_cond or not self.update_cond, "Cannot update cond if not using cond"
        self.pos = PositionalEncoding(num_hid, dropout=dropout, max_len=max_len + 2)
        self.embedding = nn.Embedding(vocab_size, num_hid)
        encoder_layers = nn.TransformerEncoderLayer(num_hid, num_head, num_hid, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        if self.use_cond:
            self.output = MLP(num_hid + num_hid, num_actions, [4 * num_hid, 4 * num_hid], dropout)
            self.cond_embed = CondEmbedding(cond_dim, num_hid, pool='max', act='tanh', dropout_prob=dropout)
            self.Z_mod = CondEmbedding(cond_dim, num_hid, pool='max', act='tanh', dropout_prob=dropout)
        else:
            self.output = MLP(num_hid, num_actions, [2 * num_hid, 2 * num_hid], dropout)
            self.Z_mod = nn.Parameter(torch.ones(num_hid) * 30 / num_hid)
        self.logsoftmax2 = torch.nn.LogSoftmax(2)
        self.num_hid = num_hid

    def Z(self, cond_var):
        return self.Z_mod(cond_var).sum(1) if self.use_cond else self.Z_mod.sum()

    def model_params(self):
        params = list(self.pos.parameters()) + list(self.embedding.parameters()) + list(self.encoder.parameters()) + \
            list(self.output.parameters())
        if self.update_cond:
            params += list(self.cond_embed.parameters())
        return params

    def Z_param(self):
        return self.Z_mod.parameters() if self.use_cond else [self.Z_mod]

    def forward(self, x, cond, mask, cond_var=None, lens=None):
        """
        cond is separate cond for each x, same batch dim as x
        """        
        x = self.embedding(x)
        x = self.pos(x)
        # get encoded vector
        x = self.encoder(x, src_key_padding_mask=mask,
                            mask=generate_square_subsequent_mask(x.shape[0]).to(x.device))
        if lens is None:
            lens = torch.zeros(x.shape[1]).long().to(x.device)
        pooled_x = x[lens-1, torch.arange(x.shape[1])] # hidden vector of the next  token

        if self.use_cond:
            # batch x hidden_dim
            if cond_var is None:
                cond_var = self.cond_embed(cond) 
            # batch x 2 * hidden_dim 
            final_rep = torch.cat((pooled_x, cond_var), axis=-1)
        else:
            final_rep = pooled_x

        y = self.output(final_rep)
        return y

class SetCondSeqTransformer(nn.Module):
    def __init__(self, num_hid, cond_dim, max_len, vocab_size, num_actions, dropout, num_layers,
                num_head, use_cond, update_cond=False, **kwargs):
        super().__init__()
        self.pos = PositionalEncoding(num_hid, dropout=dropout, max_len=max_len + 2)
        self.use_cond = use_cond
        self.update_cond = update_cond
        assert self.use_cond or not self.update_cond, "Cannot update cond if not using cond"
        self.embedding = nn.Embedding(vocab_size, num_hid)
        encoder_layers = nn.TransformerEncoderLayer(num_hid, num_head, num_hid, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        if self.use_cond:
            self.pf_output_pos = MLP(num_hid + num_hid, 1, [4 * num_hid, 4 * num_hid], dropout)
            self.pf_output_tok = MLP(num_hid + num_hid, num_actions, [4 * num_hid, 4 * num_hid], dropout)
            self.cond_embed = CondEmbedding(cond_dim, num_hid, pool='max', act='tanh', dropout_prob=dropout)
        else:
            self.pf_output_pos = MLP(num_hid, 1, [4 * num_hid, 4 * num_hid], dropout)
            self.pf_output_tok = MLP(num_hid, num_actions, [4 * num_hid, 4 * num_hid], dropout)
        self.logsoftmax2 = torch.nn.LogSoftmax(2)
        self.num_hid = num_hid

    def model_params(self):
        params = list(self.pos.parameters()) + list(self.embedding.parameters()) + list(self.encoder.parameters()) + \
            list(self.pf_output_pos.parameters()) + list(self.pf_output_tok.parameters()) + \
            list(self.pf_output_pos.parameters()) + list(self.pf_output_tok.parameters()) 
        if self.update_cond:
            params += list(self.cond_embed.parameters())
        return params

    def forward(self, x, cond, mask, lens=None, logsoftmax=False):
        """
        cond is separate cond for each x, same batch dim as x
        """        
        x = self.embedding(x)
        x = self.pos(x)            
        x = self.encoder(x, src_key_padding_mask=mask)

        if self.use_cond:
            cond_var = self.cond_embed(cond) # batch x hidden_dim
            cond_var = torch.tile(cond_var, (x.shape[0], 1, 1))
            final_rep = torch.cat((x, cond_var), axis=-1)
        else:
            final_rep = x
        pos_logits = self.pf_output_pos(final_rep).squeeze(-1).transpose(1, 0)
        tok_logits = self.pf_output_tok(final_rep).squeeze(-1).transpose(1, 0)
        
        return pos_logits, tok_logits


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
