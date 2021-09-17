# Code adapted from https://bastings.github.io/annotated_encoder_decoder/

import torch.nn as nn
import torch
import torch.nn.functional as F
from Attention import *

class Decoder(nn.Module):
    
    def __init__(self, emb_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()

        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.attention = BahadanauAttention(hidden_size)

        self.attention_1 = BahadanauAttention(hidden_size)
        self.luongattention = LuongAttention(hidden_size)
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.f = nn.GRU(input_size = emb_size + 2 * hidden_size,
                          hidden_size = hidden_size, num_layers = num_layers,
                          batch_first = True, dropout = dropout)
        
        self.g = nn.Linear(in_features = emb_size + 2*hidden_size + hidden_size,
                                          out_features = hidden_size, bias = False)
        
        self.bridge = nn.Linear(in_features = 2 * hidden_size,
                                out_features = hidden_size, bias = False)

        self.mhlinear = nn.Linear(in_features = 4 * hidden_size,
                                out_features = 2* hidden_size, bias = False)
        
        
    def forward_step(self, encoder_output, decoder_hidden,
                    prev_target_embed):
        '''
            Performs a single decoder step (1 word)
        '''
        # Prev_target_embed : [ batch_size, 1, embed_dim ]
        # decoder_hidden ( previous time step ):  [ 1, batch size, hidden_size ] -> [ batch size, 1, hidden_size ]
        # context vector : [ batch_size, 1, 2 * hidden_size ]
        context = self.attention(encoder_output,
                                decoder_hidden[-1].unsqueeze(1))

        context_1 = self.luongattention(encoder_output,
                                decoder_hidden[-1].unsqueeze(1))
        # context_multihead : [ batch_size, 1, 4 * hidden_size ]
        context_multihead = torch.cat([context, context_1], axis=2)
        # context vector : [ batch_size, 1, 2 * hidden_size ]
        context = self.mhlinear(context_multihead)
        
        # decoder_input : [ batch_size, 1, embed_dim + 2 * hidden_size ]
        decoder_input = torch.cat([prev_target_embed, context], axis=2)

        # f_output: [ batch_size, 1, num_directions * hidden_size ]
        # decoder_hidden: [ num_layers * num_directions, batch_size, hidden_size ]
        f_output, decoder_hidden = self.f(decoder_input, decoder_hidden)  
        
        # g_input : [batch_size, 1, embed + 2 * hidden_size + hidden_size]
        g_input = torch.cat([ prev_target_embed, f_output, context ], axis = 2)
        
        # g_input : [batch_size, 1, embed + 2 * hidden_size + hidden_size]
        g_input = self.dropout(g_input)
        
        # decoder_output : [batch_size, 1, hidden_size]
        decoder_output = self.g(g_input)
        return decoder_output, decoder_hidden
    
    
    def forward(self, target_embedding, encoder_output, encoder_final, decoder_hidden=None):
        '''
            Unroll deocder units one at a time
        '''
        if decoder_hidden is None:
            decoder_hidden = self.init_hidden(encoder_final)
        
        output_len = target_embedding.size(1)
        
        # Store decoder outputs
        decoder_outputs = []

        for i in range(output_len):
            
            # Get prev. target word's embeddings and feed to decoder step
            prev_target_embed = target_embedding[:, i].unsqueeze(1)
            
            decoder_output, decoder_hidden = self.forward_step(
                encoder_output = encoder_output,
                decoder_hidden = decoder_hidden,
                prev_target_embed = prev_target_embed)

            decoder_outputs.append(decoder_output)
            
        # Decoder outputs : [ batch_size, max_len, hidden_dim ]
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs, decoder_hidden
        
        
    def init_hidden(self, encoder_final):
        '''
            Use encoder_final states to initialize decoder hidden state at time = 0
        '''
        if encoder_final is None:
            return None
        return torch.tanh(self.bridge(encoder_final))
