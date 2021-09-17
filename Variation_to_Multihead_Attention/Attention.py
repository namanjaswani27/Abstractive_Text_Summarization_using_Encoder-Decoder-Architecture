# Code adapted from https://bastings.github.io/annotated_encoder_decoder/

import torch.nn as nn
import torch
import torch.nn.functional as F

class BahadanauAttention(nn.Module):
    
    def __init__(self, hidden_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # key = Bidirectional encoder
        # query = decoder hidden state
        
        # Projecting query_size to hidden_size
        self.query_layer = nn.Linear(
            in_features=hidden_size, out_features=hidden_size, bias=False)
        
        # Projecting key_size to hidden_size
        self.key_layer = nn.Linear(
            in_features=2*hidden_size, out_features=hidden_size, bias=False)
        
        # TO get scores (scalar)
        self.energy_layer = nn.Linear(
            in_features=hidden_size, out_features=1, bias=False)
        
        
    def forward(self, encoder_output, decoder_hidden):

        # key : [ batch_size, max_len, hidden_dim ]
        key = self.key_layer(encoder_output)
        
        # query : [ batch_size, 1, hidden_dim ]
        query = self.query_layer(decoder_hidden)
        
        # energies : [ batch_size, max_len, 1 ]
        scores = self.energy_layer(torch.tanh(key + query))
        
        # scores : [ batch_size, 1, max_len ]
        scores = scores.squeeze(2).unsqueeze(1)
        
        # Attention Prob.
        # Computing attention probabilities over max_len
        # [ batch_size, 1, max_len ]
        alphas = F.softmax(
                input = scores, dim = -1)
         
        # Weighted sum of key=encoder_output and attention prob.
        # [ batch_size, 1, 2 * hidden_size ]
        context = torch.bmm(alphas, encoder_output)
        
        return context




class LuongAttention(nn.Module):
    
    def __init__(self, hidden_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # key = Bidirectional encoder
        # query = decoder hidden state
        
        # Projecting query_size to hidden_size
        self.query_layer = nn.Linear(
            in_features=hidden_size, out_features=hidden_size, bias=False)
        
        # Projecting key_size to hidden_size
        self.key_layer = nn.Linear(
            in_features=2*hidden_size, out_features=hidden_size, bias=False)

        self.W = nn.Linear(
            in_features=hidden_size, out_features=hidden_size, bias=False)
        
        
    def forward(self, encoder_output, decoder_hidden):

        # key : [ batch_size, max_len, hidden_dim ]
        key = self.key_layer(encoder_output)
        
        # query : [ batch_size, 1, hidden_dim ]
        query = self.query_layer(decoder_hidden)
        
        # scores : [ batch_size, max_len, 1 ]
        scores = self.W(key)
        scores = torch.bmm(scores, query.permute(0, 2, 1))
        
        # scores : [ batch_size, 1, max_len ]
        scores = scores.squeeze(2).unsqueeze(1)
        
        # Attention Prob.
        # Computing attention probabilities over max_len
        # [ batch_size, 1, max_len ]
        alphas = F.softmax(
                input = scores, dim = -1)
         
        # Weighted sum of key=encoder_output and attention prob.
        # [ batch_size, 1, 2 * hidden_size ]
        context = torch.bmm(alphas, encoder_output)
        
        return context
