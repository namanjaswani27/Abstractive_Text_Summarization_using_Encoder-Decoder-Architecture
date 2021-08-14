import torch.nn as nn
import torch
import torch.nn.functional as F

class self_attention(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        # dim = encoder_output.shape[-1]
        self.l_q = nn.Linear(dim, dim, bias=False)
        self.l_k = nn.Linear(dim, dim, bias=False)
        self.l_v = nn.Linear(dim, dim, bias=False)
    
    def forward(self, output):
         # output : [ bs, seq_len, emb_dim ]
        query, key, value = output.clone(), output.clone(), output.clone()
        query = self.l_q(query)
        key = self.l_k(key)
        value = self.l_v(value)

        # f : [ bs, seq_len, seq_len ]
        f = torch.bmm(query, key.permute(0, 2, 1))
        # f : [ bs, seq_len, seq_len ]                    # Scores for how ith word depends on jth word
        f = F.softmax(f, dim=-1)

        # output : [ bs, seq_len, emb_dim ]
        output = torch.bmm(f, value)
        
        return output
