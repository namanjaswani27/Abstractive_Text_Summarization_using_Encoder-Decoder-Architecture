# Code adapted from https://bastings.github.io/annotated_encoder_decoder/

import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder

class Seq2Seq(nn.Module):

    def __init__(self, embedding, hidden_size, 
    vocab_size, num_layers = 1, dropout = 0.1):

        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(
            embedding_size = embedding.weight.size(1), 
            hidden_size = hidden_size, num_layers = num_layers, 
            dropout = dropout)

        self.decoder = Decoder(
            embedding_size = embedding.weight.size(1), 
            hidden_size = hidden_size, num_layers = num_layers, 
            dropout = dropout)

        self.embedding  = embedding

        self.generator = nn.Linear(in_features = hidden_size,
        out_features = vocab_size)
    
    def forward(self, input, target, decoder_hidden=None):

        # encoder_output: [ batch_size, input_len, 2 * hidden_size ]
        # encoder_final:  [ 1, batch size, 2 * hidden_size ]
        encoder_output, encoder_final = self.encode(input = input)

        decoder_output, _ = self.decode(target = target, 
        encoder_output = encoder_output, encoder_final = encoder_final, 
        decoder_hidden=decoder_hidden)

        return self.generator(decoder_output)

    def encode(self, input):
        
        # input_embedding: [ batch_size, input_len, embedding_size ]
        input_embedding = self.embedding(input)

        return self.encoder(input_embedding = input_embedding)

    def decode(self, target, encoder_output, encoder_final, 
    decoder_hidden = None):
        
        # target_embedding: [ batch_size, target_len, embedding_size ]
        target_embedding = self.embedding(target)

        return self.decoder(target_embedding = target_embedding,
        encoder_output = encoder_output, encoder_final = encoder_final, 
        decoder_hidden = decoder_hidden)
