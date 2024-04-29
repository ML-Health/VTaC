import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(
        self,
        inputs,
        window_sizes=[25, 50, 100, 150],
        feature_size=64,
        window=90000,
        hidden_signal=128,
        hidden_alarm=64,
        dropout=0.0,
    ):
        super(CNNClassifier, self).__init__()

        self.window = window
        self.hidden_signal = hidden_signal
        self.hidden_alarm = hidden_alarm
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(p=0.8),
                    nn.Conv1d(inputs, feature_size, kernel_size=h, stride=5, padding=1),
                    nn.BatchNorm1d(feature_size),
                    nn.ReLU(),
                    nn.Conv1d(
                        feature_size, feature_size, kernel_size=h, stride=5, padding=1
                    ),
                    nn.BatchNorm1d(feature_size),
                    nn.AdaptiveMaxPool1d(1),
                )
                for h in window_sizes
            ]
        )

        self.signal_feature = nn.Sequential(
            nn.Linear(feature_size * len(window_sizes), self.hidden_signal),
            nn.BatchNorm1d(self.hidden_signal),
            nn.ReLU(),
        )

        self.rule_based_label = nn.Sequential(
            nn.Linear(1, self.hidden_alarm),
            nn.BatchNorm1d(self.hidden_alarm),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_signal, 1),
        )

    def forward(self, signal, random_s=None):
        signal = [conv(signal) for conv in self.convs]
        signal = torch.cat(signal, dim=1)
        signal = signal.view(-1, signal.size(1))
        s_f = self.signal_feature(signal)

        if random_s is not None:
            random_s = [conv(random_s) for conv in self.convs]

            random_s = torch.cat(random_s, dim=1)

            random_s = random_s.view(-1, random_s.size(1))

            random_s = self.signal_feature(random_s)

            # torch.cat((s_f, labels_embedding).shape = [256, 192]
            return self.classifier(s_f), s_f, random_s

        return self.classifier(s_f)
