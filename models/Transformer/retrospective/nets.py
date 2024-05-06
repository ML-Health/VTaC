import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Transformer(nn.Module):

    def __init__(self,input_dim,dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding1 = nn.Sequential(
            nn.Linear(input_dim, 128),
        )
        self.dropout = dropout
        self.encoderlayer=nn.TransformerEncoderLayer(d_model=128, nhead=8,dim_feedforward=1024,dropout=self.dropout)
        self.encoder=nn.TransformerEncoder(self.encoderlayer, num_layers=6)
        
        self.conv = nn.Conv1d(3750, 1, 1)
        self.gelu = nn.GELU()
        self.pos_encoder = PositionalEncoding(128, dropout=self.dropout, max_len=3750)
        self.classifier = nn.Sequential(
            nn.Conv1d(128, 1, 1)
        )

    def forward(self, signal):  # x (batch, time_step, input_size)

        input = self.embedding1(signal.permute(0, 2, 1))
        input = self.pos_encoder(input)

        encoder_output = self.encoder(input,mask=None)

        output = self.conv(encoder_output)
        output = torch.transpose(output, 1, 2)
        output = self.gelu(output)

        output = self.classifier(output).squeeze(2)

        return encoder_output, output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1250):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

