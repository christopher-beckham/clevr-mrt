import torch
from torch import nn
from .baselines import RnnEncoder

class ClevrProbeRnnOnly(nn.Module):
    def __init__(self,
                 vocab,
                 embedding_dim=300,
                 rnn_dim=256,
                 rnn_num_layers=2,
                 encoder='lstm'):
        super(ClevrProbeRnnOnly, self).__init__()

        self.n_out = len(vocab['answer_idx_to_token'])

        rnn_kwargs = {
            'token_to_idx': vocab['question_token_to_idx'],
            'wordvec_dim': embedding_dim,
            'rnn_dim': rnn_dim,
            'rnn_num_layers': rnn_num_layers,
            'rnn_dropout': 0,
            'model': encoder
        }
        print("rnn_kwargs:")
        print(rnn_kwargs)
        self.rnn = RnnEncoder(**rnn_kwargs)
        self.rnn_dim = rnn_dim

        self.fc = nn.Sequential(
            nn.Linear(rnn_dim, rnn_dim),
            nn.BatchNorm1d(rnn_dim),
            nn.ReLU(),
            nn.Linear(rnn_dim, self.n_out)
        )

    def forward(self, z, questions, camera):
        """
        """
        # Embed both question and camera.
        embedding = self.rnn(questions)
        out = self.fc(embedding)
        return out

def get_network(vocab,
                encoder='lstm',
                embedding_dim=300,
                rnn_dim=256,
                rnn_num_layers=2):
    return ClevrProbeRnnOnly(vocab,
                             encoder=encoder,
                             embedding_dim=embedding_dim,
                             rnn_dim=rnn_dim,
                             rnn_num_layers=rnn_num_layers)

if __name__ == '__main__':

    pass
