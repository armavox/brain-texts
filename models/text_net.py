import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


class ToyNN(nn.Module):
    def __init__(self, weights_matrix, hidden_dim, num_layers):
        super().__init__()
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(
            weights_matrix, True
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers,
                          batch_first=True)

    def forward(self, inp, hidden):
        return self.gru(self.embedding(inp), hidden)

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size,
                                    self.hidden_dim))


class BrainLSTM(nn.Module):
    """
    Parameters
    ----------
    """

    def __init__(self, embed_dim, hidden_dim, num_layers,
                 context_size, combine_dim, dropout=0):
        super().__init__()

        self.context_size = context_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        lstm_input_size = embed_dim * self.context_size
        self.lstm = nn.LSTM(lstm_input_size, self.hidden_dim, self.num_layers,
                            dropout=dropout, bidirectional=True)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim * self.context_size + 2 * self.hidden_dim,
                      self.hidden_dim),
            # nn.ReLU()
        )
        self.combine = nn.Sequential(
            nn.Linear(self.hidden_dim, combine_dim),
            # nn.ReLU()
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : (batch_size, seq_len, embed_dim)
        """

        x = x.unfold(1, self.context_size, 1)  # Make context vec
        x = x.transpose(2, 3)
        x = x.contiguous().view(x.size(0), x.size(1), -1)
        x = torch.transpose(x, 0, 1)
        # x.shape->(seq_len-context_size+1, batch_size, embed_dim*context_size)

        # Pass through LSTM and Linear layers
        out, (final_hidden_state, final_cell_state) = self.lstm(x)
        # out.shape -> (seq_len-context_size+1, batch_size, 2*hidden_dim)

        final_encoding = torch.cat((out, x), 2).permute(1, 0, 2)
        # final_encoding.shape -> (batch_size, seq_len-context_size+1,
        #     embed_dim * context_size + 2*hidden_dim)

        out = self.linear(final_encoding)
        # out = F.relu(out)
        # out.shape -> (batch_size, seq_len-context_size+1, hidden_dim)

        out = out.permute(0, 2, 1)
        # out.shape -> (batch_size, hidden_dim, seq_len-context_size+1)

        out = F.max_pool1d(out, out.shape[2]).squeeze(2)
        # out.shape -> (batch_size, hidden_dim)

        out = self.combine(out)

        return out




# matrix_len = len(target_vocab)
# weights_matrix = np.zeros((matrix_len, 50))
# words_found = 0

# for i, word in enumerate(target_vocab):
#     try:
#         weights_matrix[i] = glove[word]
#         words_found += 1
#     except KeyError:
#         weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))