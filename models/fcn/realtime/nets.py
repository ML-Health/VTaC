import torch
import torch.nn as nn


class FCN(nn.Module):
    def __init__(
        self, chan_1, chan_2, chan_3, ks1, ks2, ks3, channels, dropout_prob=0.5
    ):
        super(FCN, self).__init__()

        pd1 = (ks1 - 1) // 2
        pd2 = (ks2 - 1) // 2
        pd3 = (ks3 - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv1d(channels, chan_1, kernel_size=ks1, stride=1, padding=pd1),
            nn.BatchNorm1d(chan_1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Conv1d(chan_1, chan_2, kernel_size=ks2, stride=1, padding=pd2),
            nn.BatchNorm1d(chan_2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Conv1d(chan_2, chan_3, kernel_size=ks3, stride=1, padding=pd3),
            nn.BatchNorm1d(chan_3),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.AdaptiveMaxPool1d(1),
        )

        self.classifier = nn.Sequential(nn.Dropout(dropout_prob), nn.Linear(64, 1))

        self.signal_feature = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_prob)
        )

    def forward(self, signal, random_s=None):  # x (batch, time_step, input_size)
        signal = self.convs(signal).squeeze(-1)
        signal = signal.view(-1, signal.size(1))
        s_f = self.signal_feature(signal)

        if random_s is not None:
            random_s = self.convs(random_s).squeeze(-1)
            random_s = random_s.view(-1, random_s.size(1))
            random_s = self.signal_feature(random_s)
            return self.classifier(s_f), s_f, random_s

        return self.classifier(s_f)
