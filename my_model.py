import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # [B, 32, 100, 85]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # [B, 64, 50, 43]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # [B, 128, 25, 21]
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # [B, 256, 12, 10] (примерно)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.3),

            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # [B, 512, 6, 5] (примерно)
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(512 * 6 * 5, latent_dim)
        self.fc_logvar = nn.Linear(512 * 6 * 5, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 512 * 6 * 5)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # [B, 256, 12, 10]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.3),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # [B, 128, 25, 21]
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # [B, 64, 50, 42]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=(0, 1)),  # [B, 32, 100, 85]
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, output_padding=(0, 1)),  # [B, 3, 200, 170]
            nn.Upsample(size=(200, 170), mode='bilinear', align_corners=False),

        nn.Sigmoid()
        )

        # Кроп до точного размера 200x170
        self.crop = lambda x: x[:, :, :, :170]



    def forward(self, x):
        z = self.encoder(x)
        z_flat = self.flatten(z)
        mu = self.fc_mu(z_flat)

        dec_input = self.decoder_input(mu).view(-1, 512, 6, 5)
        x_recon = self.decoder(dec_input)
        return self.crop(x_recon), mu, z

