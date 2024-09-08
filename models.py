import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import SinusoidalPosEmb

class LSTMPolicy(nn.Module):
    def __init__(self, state_dim, cond_dim=16, hidden_dim=256, length=16):
        super(LSTMPolicy, self).__init__()

        self.state_dim = state_dim
        self.cond_dim = cond_dim
        self.length = length
        self.hidden_dim = hidden_dim
        self.cond_emb = nn.Linear(1, cond_dim)
        self.lstm = nn.LSTM(state_dim + cond_dim, hidden_dim, batch_first=True)
        self.state_linear = nn.Linear(hidden_dim, state_dim)
        self.reward_linear = nn.Linear(hidden_dim, 1)
        self.apply(init_weights)
    
    def forward(self, x, condition):
        # x shape: [batch_size, seq_len, state_dim]
        # condition shape: [batch_size, seq_len, cond_dim]
        embed_cond = self.cond_emb(condition)
        x = torch.cat([x, embed_cond], dim=-1)
        x, _ = self.lstm(x)
        state = self.state_linear(x)
        reward = self.reward_linear(x)
        return state, reward
    
    def to(self, device):
        super(LSTMPolicy, self).to(device)
        return self

class RNNPolicy(nn.Module):
    def __init__(self, state_dim, cond_dim=16, hidden_dim=128, length=16):
        super(RNNPolicy, self).__init__()

        self.state_dim = state_dim
        self.cond_dim = cond_dim
        self.length = length
        self.hidden_dim = hidden_dim
        self.cond_emb = nn.Linear(1, cond_dim)
        self.rnn = nn.RNN(state_dim + cond_dim, hidden_dim, batch_first=True)
        self.state_linear = nn.Linear(hidden_dim, state_dim)
        self.reward_linear = nn.Linear(hidden_dim, 1)
        self.apply(init_weights)
    
    def forward(self, x, condition):
        # x shape: [batch_size, seq_len, state_dim]
        # condition shape: [batch_size, seq_len, cond_dim]
        embed_cond = self.cond_emb(condition)
        x = torch.cat([x, embed_cond], dim=-1)
        x, _ = self.rnn(x)
        state = self.state_linear(x)
        reward = self.reward_linear(x)
        return state, reward
    
    def to(self, device):
        super(RNNPolicy, self).to(device)
        return self

class GRUPolicy(nn.Module):
    def __init__(self, state_dim, cond_dim=16, hidden_dim=128, length=16):
        super(GRUPolicy, self).__init__()

        self.state_dim = state_dim
        self.cond_dim = cond_dim
        self.length = length
        self.hidden_dim = hidden_dim
        self.cond_emb = nn.Linear(1, cond_dim)
        self.gru = nn.GRU(state_dim + cond_dim, hidden_dim, batch_first=True)
        self.state_linear = nn.Linear(hidden_dim, state_dim)
        self.reward_linear = nn.Linear(hidden_dim, 1)
        self.apply(init_weights)
    
    def forward(self, x, condition):
        # x shape: [batch_size, seq_len, state_dim]
        # condition shape: [batch_size, seq_len, cond_dim]
        embed_cond = self.cond_emb(condition)
        x = torch.cat([x, embed_cond], dim=-1)
        x, _ = self.gru(x)
        state = self.state_linear(x)
        reward = self.reward_linear(x)
        return state, reward
    
    def to(self, device):
        super(GRUPolicy, self).to(device)
        return self



#=================================================#
#======================Critic=====================#
#=================================================#

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, length=16):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_dim = 32
        self.length = length
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))
        self.apply(init_weights)


    def forward(self, x, a):
        x = torch.cat([x, a], dim=-1)
        return self.q1_model(x)


#=================================================#
#===================Transformer===================#
#=================================================#

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # query shape: [batch_size, query_len, d_model]
        # key shape: [batch_size, key_len, d_model]
        # value shape: [batch_size, value_len, d_model]

        batch_size = query.size(0)

        # project inputs to multi-heads
        Q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2) # [batch_size, n_heads, query_len, head_dim]
        K = self.k_linear(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2) # [batch_size, n_heads, key_len, head_dim]
        V = self.v_linear(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2) # [batch_size, n_heads, value_len, head_dim]

        # compute scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) # [batch_size, n_heads, query_len, key_len]

        # apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == True, -1e9)
        # apply softmax
        attention_weights = F.softmax(scores, dim=-1) # [batch_size, n_heads, query_len, key_len]
        out = torch.matmul(attention_weights, V) # [batch_size, n_heads, query_len, head_dim]

        # concatenate heads and apply final linear layer
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) # [batch_size, query_len, d_model]
        out = self.fc_out(out)

        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, d_model]

        # self-attention
        attn_output = self.self_attn(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        x = self.norm1(x + attn_output)

        # feed forward
        ff_output = self.feed_forward(x)
        ff_output = self.dropout2(ff_output)
        x = self.norm2(x + ff_output)

        return x
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.encoder_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # x shape: [batch_size, seq_len, d_model]

        # self-attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        attn_output = self.dropout1(attn_output)
        x = self.norm1(x + attn_output)

        # encoder-decoder attention
        attn_output = self.encoder_attn(x, encoder_output, encoder_output, src_mask)
        attn_output = self.dropout2(attn_output)
        x = self.norm2(x + attn_output)

        # feed forward
        ff_output = self.feed_forward(x)
        ff_output = self.dropout3(ff_output)
        x = self.norm3(x + ff_output)

        return x

class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout_rate) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, d_model]

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class Decoder(nn.Module):
        def __init__(self, d_model, n_heads, d_ff, n_layers, dropout_rate=0.1):
            super(Decoder, self).__init__()
            self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout_rate) for _ in range(n_layers)])
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
            # x shape: [batch_size, seq_len, d_model]

            for layer in self.layers:
                x = layer(x, encoder_output, src_mask, tgt_mask)

            return self.norm(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
    if isinstance(m, nn.Embedding):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
    if isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

class Transformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, d_ff, n_layers, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.reward_embedding = nn.Linear(1, d_model)
        
        self.pos_encoding = PositionalEncoding(d_model*2)
        self.encoder = Encoder(d_model*2, n_heads, d_ff, n_layers, dropout_rate)
        # self.decoder = Decoder(d_model, n_heads, d_ff, n_layers, dropout_rate)
        self.state_out = nn.Sequential(nn.Linear(d_model*2, d_model),
                                    nn.ReLU(),
                                    nn.Linear(d_model, input_dim))
        self.reward_out = nn.Sequential(nn.Linear(d_model*2, d_model),
                                    nn.ReLU(),
                                    nn.Linear(d_model, 1))
        self.embed_ln = nn.LayerNorm(d_model*2)


    def forward(self, x, rtg, mask=None):
        # x shape: [batch_size, seq_len, input_dim]
        # return shape: [batch_size, seq_len, input_dim]
        state_emb = self.embedding(x) # [batch_size, seq_len, d_model]
        rtg_emb = self.reward_embedding(rtg)
        x = torch.cat([state_emb, rtg_emb], dim=-1)
        x = self.pos_encoding(x)
        x = self.embed_ln(x)
        x = self.encoder(x, mask)

        reward = self.reward_out(x)
        state = self.state_out(x)

        return state, reward
    

class UncondTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, d_ff, n_layers, dropout_rate=0.1):
        super(UncondTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)

        self.pos_encoding = PositionalEncoding(d_model)
        self.encoder = Encoder(d_model, n_heads, d_ff, n_layers, dropout_rate)
        self.decoder = Decoder(d_model, n_heads, d_ff, n_layers, dropout_rate)
        self.state_out = nn.Sequential(nn.Linear(d_model, d_model),
                                    nn.ReLU(),
                                    nn.Linear(d_model, input_dim))
        self.embed_ln = nn.LayerNorm(d_model)


    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, input_dim]
        # return shape: [batch_size, seq_len, input_dim]
        state_emb = self.embedding(x) # [batch_size, seq_len, d_model]

        x = self.pos_encoding(x)
        x = self.embed_ln(x)
        x = self.encoder(x, mask)

        state = self.state_out(x)

        return state
    
    def sample(self, x, horizon):
        for _ in range(horizon-1):
            n_x = self.forward(x)
            x = torch.cat([x, n_x[:, -1:]], dim=1)
        return x

