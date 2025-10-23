import torch.nn as nn
import torch.nn.functional as F


class MLPProjector(nn.Module):
    def __init__(self, encoder_size, generator_size, num_emb_tokens, num_doc_tokens,
                 hidden_dim=None, use_ln=True):
        super().__init__()
        self.encoder_size = encoder_size
        self.generator_size = generator_size
        self.num_emb_tokens = num_emb_tokens
        self.num_doc_tokens = num_doc_tokens

        in_dim = encoder_size * num_emb_tokens
        out_dim = generator_size * num_doc_tokens
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, encoder_hidden):
        B, N, D = encoder_hidden.shape
        assert N == self.num_emb_tokens and D == self.encoder_size

        x = encoder_hidden.reshape(B, -1)
        x = self.fc(x)
        x = x.view(B, self.num_doc_tokens, self.generator_size)
        return x
