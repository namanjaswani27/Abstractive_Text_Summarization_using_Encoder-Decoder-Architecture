# Code adapted from https://bastings.github.io/annotated_encoder_decoder/

import torch.nn as nn
import torch
from Attention import Attention

class Decoder(nn.Module):

    def __init__(self, embedding_size, hidden_size, 
    num_layers = 1, dropout = 0.1):

        super(Decoder, self).__init__()
        self.attention = Attention(hidden_size = hidden_size)

        self.f = nn.GRU(input_size=embedding_size + 2*hidden_size, 
        hidden_size=hidden_size, num_layers=num_layers, 
        batch_first=True, dropout=dropout)

        self.g = nn.Linear(
            in_features=hidden_size + 2*hidden_size + embedding_size, 
            out_features=hidden_size, bias=False)

        self.dropout = nn.Dropout(p = dropout)

        self.init_hidden = nn.Linear(
            in_features=2*hidden_size, out_features=hidden_size, bias=True)


    def forward_step(self, encoder_output, decoder_hidden, 
    prev_target_embedding):

        """Perform a single decoder step (1 word)"""

        # Attention
        # context: [ batch_size, 1, 2 * hidden_size ]
        context = self.attention(encoder_output = encoder_output, 
        # decoder_hidden:  [ batch size, 1, hidden_size ]
        decoder_hidden = decoder_hidden[-1].unsqueeze(1))

        # Teacher Forcing
        # decoder_input: [ batch_size, 1, embedding_size + 2 * hidden_size ]
        decoder_input = torch.cat([prev_target_embedding, context], dim=2)

        # f_output: [ batch_size, 1, num_directions * hidden_size ]
        # decoder_hidden: [ num_layers * num_directions, batch_size, hidden_size ]
        f_output, decoder_hidden = self.f(decoder_input, decoder_hidden)

        # g_input: [ batch_size, 1, embedding_size + 3 * hidden_size ]
        g_input = torch.cat([prev_target_embedding, f_output, context], dim=2)

        # g_input: [ batch_size, 1, embedding_size + 3 * hidden_size ]
        g_input= self.dropout(g_input)

        # decoder_output: [ batch_size, 1, hidden_size ]
        decoder_output = self.g(g_input)

        return decoder_hidden, decoder_output


    def forward(self, target_embedding, encoder_output, encoder_final, 
    decoder_hidden = None):

        # Decoder initial hidden state results from projection of encoder final state
        # decoder_hidden:  [ 1, batch size, hidden_size ]
        if decoder_hidden is None:
            decoder_hidden = torch.tanh(self.init_hidden(encoder_final))

        #Store decoder targets for predictions
        decoder_outputs = []

        """Unroll the decoder one step at a time."""

        target_len = target_embedding.size(1)

        for i in range(target_len):

            # Feed correct previous target word embedding - Teacher Forcing
            # prev_target_embedding:  [ batch_size, 1, embedding_size ]
            prev_target_embedding = target_embedding[:, i].unsqueeze(1)

            # decoder_hidden: [ 1, batch_size, hidden_size ]
            # decoder_output: [ batch_size, 1, hidden_size ]
            decoder_hidden, decoder_output = self.forward_step(
                encoder_output = encoder_output, 
                decoder_hidden = decoder_hidden, 
                prev_target_embedding = prev_target_embedding)

            decoder_outputs.append(decoder_output)

        # output: [ batch_size, target_len, hidden_size ]
        output = torch.cat(decoder_outputs, dim=1)
        return output, decoder_hidden

