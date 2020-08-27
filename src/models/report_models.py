import torch
import torch.nn as nn
import torch.nn.functional as F


class TextLinearClassifier(nn.Module):
    def __init__(self, num_classes=2, bert_model="base"):
        super().__init__()

        if bert_model == "base":
            in_features = 768

        elif bert_model == "large":
            in_features = 2048

        else:
            raise AssertionError("bert_model should be in ['base', 'large']")

        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features), nn.Dropout(), nn.ReLU(), nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class LSTMClassifier(nn.Module):
    def __init__(self, inp_features=768, num_classes=2, hidden_dim=1024, num_lstm_layers=2, lstm_dropout=0):
        super().__init__()

        self.lstm = nn.LSTM(inp_features, hidden_dim, num_lstm_layers, dropout=lstm_dropout, bidirectional=True)
        self.classifier = nn.Linear(2 * num_lstm_layers * hidden_dim, num_classes)

    def forward(self, x):
        """ x : (batch_size, seq_len, embed_dim) """
        x = x.permute(1, 0, 2)

        out, (final_hidden_state, final_cell_state) = self.lstm(x)
        out = final_hidden_state.transpose(0, 1).contiguous()
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class ContextLSTM(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        hidden_dim=1024,
        combine_dim=None,
        num_classes=2,
        context_size=2,
        num_lstm_layers=2,
        lstm_dropout=0,
        init_weights=False,
    ):
        super().__init__()

        self.context_size = context_size
        self.combine_dim = combine_dim
        lstm_input_size = embed_dim * context_size

        self.lstm = nn.LSTM(lstm_input_size, hidden_dim, num_lstm_layers, dropout=lstm_dropout, bidirectional=True)

        self.linear = nn.Sequential(nn.Linear(embed_dim * self.context_size + 2 * hidden_dim, hidden_dim), nn.ReLU())
        if combine_dim is not None:
            self.combine = nn.Linear(hidden_dim, combine_dim)
        else:
            self.classifier = nn.Linear(hidden_dim, num_classes)

        if init_weights:
            self.init_weights()

    def forward(self, x):
        """x : (batch_size, seq_len, embed_dim)"""

        x = x.unfold(1, self.context_size, 1)  # Make context vec
        x = x.transpose(2, 3)
        x = x.contiguous().view(x.size(0), x.size(1), -1)
        x = torch.transpose(x, 0, 1)
        # x.shape -> (seq_len-context_size+1, batch_size, embed_dim*context_size)

        # Pass through LSTM and Linear layers
        out, (final_hidden_state, final_cell_state) = self.lstm(x)
        # out.shape -> (seq_len-context_size+1, batch_size, 2*hidden_dim)

        final_encoding = torch.cat((out, x), 2).permute(1, 0, 2)
        # final_encoding.shape -> (batch_size, seq_len-context_size+1,
        #     embed_dim * context_size + 2*hidden_dim)

        out = self.linear(final_encoding)
        # out.shape -> (batch_size, seq_len-context_size+1, hidden_dim)

        out = out.permute(0, 2, 1)
        # out.shape -> (batch_size, hidden_dim, seq_len-context_size+1)

        out = F.max_pool1d(out, out.shape[2]).squeeze(2)
        # out.shape -> (batch_size, hidden_dim)

        if self.combine_dim is not None:
            out = self.combine(out)
        else:
            out = self.classifier(out)

        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                nn.init.xavier_normal_(m.weight_hh_l0)
                nn.init.xavier_normal_(m.weight_hh_l0_reverse)
                nn.init.xavier_normal_(m.weight_hh_l1)
                nn.init.xavier_normal_(m.weight_hh_l1_reverse)
                nn.init.xavier_normal_(m.weight_ih_l0)
                nn.init.xavier_normal_(m.weight_ih_l0_reverse)
                nn.init.xavier_normal_(m.weight_ih_l1)
                nn.init.xavier_normal_(m.weight_ih_l1_reverse)
                m.bias_hh_l0.data.zero_()
                m.bias_hh_l0_reverse.data.zero_()
                m.bias_hh_l1.data.zero_()
                m.bias_hh_l1_reverse.data.zero_()
                m.bias_ih_l0.data.zero_()
                m.bias_ih_l0_reverse.data.zero_()
                m.bias_ih_l1.data.zero_()
                m.bias_ih_l1_reverse.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.normal_()
