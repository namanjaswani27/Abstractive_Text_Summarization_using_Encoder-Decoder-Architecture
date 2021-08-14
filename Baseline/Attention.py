# Code adapted from https://bastings.github.io/annotated_encoder_decoder/

import torch
import torch.nn as nn
import torch.nn.functional as F

#Bahdanau et al.
class Attention(nn.Module):

    def __init__(self, hidden_size):

        super(Attention, self).__init__()

        # We assume a bi-directional encoder so key_size is 2*hidden_size
        self.ah = nn.Linear(
            in_features=2*hidden_size, out_features=hidden_size, bias=False)
        
        self.aS = nn.Linear(
            in_features=hidden_size, out_features=hidden_size, bias=False)
        
        self.a = nn.Linear(
            in_features=hidden_size, out_features=1, bias=False)

    def forward(self, encoder_output, decoder_hidden):

        # Project all the Encoder Hidden States(pre-compute this for efficiency)
        # ah: [ batch_size, input_len, hidden_size ]
        ah = self.ah(encoder_output)

        # Project the Decoder Hidden State
        # aS:  [ batch size, 1, hidden_size ]
        aS = self.aS(decoder_hidden)

        # Calculate energies
        # energies:  [ batch_size, input_len, 1 ]
        energies = self.a(torch.tanh(ah + aS))

        # energies:  [ batch_size, 1, input_len ]
        energies = energies.squeeze(2).unsqueeze(1)

        # Calculate attention
        # attention:  [ batch_size, 1, input_len ]
        attention = F.softmax(input = energies, dim = -1)

        # Context vector
        # context: [ batch_size, 1, 2 * hidden_size ]
        context = torch.bmm(attention, encoder_output)

        return context



