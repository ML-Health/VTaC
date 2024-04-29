import torch
import torch.nn as nn


class ClassifierMLP(nn.Module):
    def __init__(self, input_dim, dropout_prob=0.5):
        super(ClassifierMLP, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(input_dim, 1),
            # nn.Sigmoid()
        )

    def forward(self, signal):
        prob = self.classifier(signal)
        return prob


class Encoder(nn.Module):
    def __init__(self, input_dim=2, latent_space_dim=32):
        super(Encoder, self).__init__()
        self.latent_space_dim = latent_space_dim
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(input_dim, 8, kernel_size=125, stride=5, padding=62),
            nn.BatchNorm1d(8, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=25, stride=5, padding=12),
            nn.BatchNorm1d(16, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=11, stride=5, padding=5),
            nn.BatchNorm1d(32, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=5, padding=1),
            nn.BatchNorm1d(64, track_running_stats=True),
            nn.ReLU(),
        )

        self.encoder_lin = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, latent_space_dim)
        )

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_space_dim, 128), nn.ReLU(), nn.Linear(128, 256)
        )

    def forward(self, signal):
        sig = self.encoder_cnn(signal)
        sig = sig.view(-1, 256)
        sig = self.encoder_lin(sig)
        return sig


class Decoder(nn.Module):
    def __init__(self, input_dim=2, latent_space_dim=32):
        super(Decoder, self).__init__()
        self.latent_space_dim = latent_space_dim

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_space_dim, 128), nn.ReLU(), nn.Linear(128, 256)
        )

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose1d(
                64, 32, kernel_size=3, stride=5, padding=1, output_padding=4
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(
                32, 16, kernel_size=5, stride=5, padding=2, output_padding=4
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.ConvTranspose1d(
                16, 8, kernel_size=11, stride=5, padding=5, output_padding=4
            ),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.ConvTranspose1d(
                8, input_dim, kernel_size=25, stride=5, padding=12, output_padding=4
            ),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Conv1d(input_dim, input_dim, kernel_size=125, stride=1, padding=62),
        )

    def forward(self, signal):
        sig = self.decoder_lin(signal)
        sig = sig.view(-1, 64, 4)
        sig = self.decoder_cnn(sig)
        return sig


class AutoEncoder(nn.Module):
    def __init__(self, input_dim=2, latent_space_dim=32):
        super(AutoEncoder, self).__init__()
        self.latent_space_dim = latent_space_dim
        self.encoder = Encoder(input_dim.latent_space_dim)
        self.decoder = Decoder(input_dim, latent_space_dim)

    def forward(self, signal, return_comp=False):
        sig_comp = self.Encoder(sig)
        if return_comp:
            return sig_comp
        sig = self.decoder(sig_comp)
        return sig


class SupervisedAE(nn.Module):
    def __init__(self, input_dim=3, latent_space_dim=32, dropout_prob=0.5):
        super(SupervisedAE, self).__init__()
        self.latent_space_dim = latent_space_dim
        self.encoder = Encoder(input_dim, latent_space_dim)
        self.decoder = Decoder(input_dim, latent_space_dim)
        self.mlp = ClassifierMLP(latent_space_dim, dropout_prob=dropout_prob)

        self.return_comp = False
        self.return_prob = False

    def forward(self, signal):
        sig_comp = self.encoder(signal)
        if self.return_comp:
            return sig_comp
        prob = self.mlp(sig_comp)
        if self.return_prob:
            return prob
        sig = self.decoder(sig_comp)
        return prob, sig
