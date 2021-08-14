# Code adapted from https://bastings.github.io/annotated_encoder_decoder/

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from self_attention import self_attention

class Encoder(nn.Module):

    def __init__(self, embedding_size, hidden_size, num_layers = 1, dropout = 0.1):
        super(Encoder, self).__init__()                           

        self.biGRU = nn.GRU(input_size=embedding_size, 
        hidden_size=hidden_size, num_layers=num_layers, 
        batch_first=True, dropout=dropout, bidirectional=True)

        self.self_attention = self_attention(dim=2*hidden_size)

    
    def forward(self, input_embedding):

        # encoder_output: [ batch_size, input_len, num_directions * hidden_size ]
        # encoder_final:  [ num_layers * num_directions, batch_size, hidden_size ]
        encoder_output, encoder_final = self.biGRU(input_embedding)

        # Manually concatenate the final state for both directions
        # final_forward:   [ 1, batch_size, hidden_size ]
        # final_backward:  [ 1, batch_size, hidden_size ]
        final_forward  = encoder_final[0 : encoder_final.size(0) : 2]
        final_backward = encoder_final[1 : encoder_final.size(0) : 2]

        # encoder_final:  [ 1, batch_size, 2 * hidden_size ]
        encoder_final = torch.cat([final_forward, final_backward], dim=2)

        return self.self_attention(encoder_output), encoder_final
