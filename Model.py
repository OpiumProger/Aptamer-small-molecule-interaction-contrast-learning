import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')




class MicroContrastiveModel(nn.Module):
    def __init__(self, input_dim, latent_dim=128, projection_dim=64):
        super().__init__()

        self.apt_encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(latent_dim, latent_dim)
        )

        self.smi_encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(latent_dim, latent_dim)
        )

        self.projection = nn.Linear(latent_dim, projection_dim)

    def encode_aptamer(self, x):
        z = self.apt_encoder(x)
        z = self.projection(z)
        return F.normalize(z, dim=-1)

    def encode_molecule(self, x):
        z = self.smi_encoder(x)
        z = self.projection(z)
        return F.normalize(z, dim=-1)

    def forward(self, anchor_apt, positive_smi, negative_smis):

        z_anchor = self.encode_aptamer(anchor_apt)
        z_positive = self.encode_molecule(positive_smi)

        B, K, D = negative_smis.shape

        negatives_flat = negative_smis.view(B * K, D)
        z_neg = self.encode_molecule(negatives_flat)
        z_neg = z_neg.view(B, K, -1)

        return {
            'z_anchor': z_anchor,
            'z_positive': z_positive,
            'z_negatives': z_neg
        }