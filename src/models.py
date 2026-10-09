"""The two required classifiers: a Kim-style text CNN and a bidirectional LSTM."""
import torch
import torch.nn as nn

from text import PAD


class TextCNN(nn.Module):
    """Parallel 1-D convolutions of widths 3/4/5 over word embeddings, max-pooled over time."""

    def __init__(self, vocab_size, n_classes, emb_dim=128, n_filters=128, widths=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.convs = nn.ModuleList(nn.Conv1d(emb_dim, n_filters, w, padding=w // 2) for w in widths)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(widths), n_classes)

    def forward(self, x, lengths):
        mask = (x != PAD).unsqueeze(1)                     # (B, 1, T)
        h = self.emb(x).transpose(1, 2)                    # (B, E, T)
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(h))[:, :, : x.size(1)]     # trim the extra step even widths add
            pooled.append(c.masked_fill(~mask, -1e4).max(dim=2).values)
        return self.fc(self.drop(torch.cat(pooled, dim=1)))


class BiLSTM(nn.Module):
    """Bidirectional LSTM over word embeddings; mean- and max-pooled hidden states."""

    def __init__(self, vocab_size, n_classes, emb_dim=128, hidden=128, layers=1, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * hidden, n_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(x), lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        mask = (x != PAD).unsqueeze(2)
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        mx = out.masked_fill(~mask, -1e4).max(dim=1).values
        return self.fc(self.drop(torch.cat([mean, mx], dim=1)))


MODELS = {"cnn": TextCNN, "lstm": BiLSTM}
