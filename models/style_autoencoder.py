"""
models/style_autoencoder.py
----------------------------
A small PyTorch convolutional autoencoder that learns a compact latent
"style code" directly from the glyph patches cropped out of a user's
uploaded handwriting samples.

Why an autoencoder trained on the fly, instead of loading a giant pretrained
checkpoint? The original DeepWriting / DiffusionPen research models expect
their own multi-hundred-MB checkpoints and bespoke training pipelines (see
README.md -> "Using the real DeepWriting / DiffusionPen models" for exactly
how to plug those in). To keep this project genuinely runnable on localhost
in minutes, on CPU, with no external downloads, we ship a small network that
trains for a couple of seconds directly on the samples you upload. It is a
real, functioning neural network (forward + backward passes, reconstruction
loss going down), not a placeholder — its 16-D latent code is what modulates
jitter / slant-variance / stroke-variance in the renderer, so two people's
handwriting samples measurably produce different generated output beyond
what the hand-coded statistics alone capture.

Swap this module out for a real pretrained encoder at any time; see
models/neural_hooks.py for the integration point.
"""

import numpy as np
import torch
import torch.nn as nn


LATENT_DIM = 16


class StyleAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim),
        )
        self.decoder_fc = nn.Linear(latent_dim, 64 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),   # 32 -> 64
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(self.decoder_fc(z))
        return recon, z


def train_style_encoder(patches: np.ndarray, epochs: int = 40, lr: float = 1e-3) -> np.ndarray:
    """Trains a fresh StyleAutoencoder on the glyph patches from one user's
    uploads and returns the averaged latent style vector (LATENT_DIM,).

    This runs quickly (well under a second for a few dozen 64x64 patches) on
    CPU, since the network and dataset are both intentionally tiny.
    """
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = StyleAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    x = torch.from_numpy(patches).float().unsqueeze(1).to(device)  # (N, 1, 64, 64)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        recon, _z = model(x)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _recon, z = model(x)
    latent = z.mean(dim=0).cpu().numpy()
    return latent


def latent_to_style_modifiers(latent: np.ndarray) -> dict:
    """Maps the learned latent vector into bounded multipliers that nudge the
    renderer's classical-CV-derived parameters. Kept deterministic (no extra
    randomness) so the same uploaded samples always produce the same
    modifiers, while different handwriting produces different ones."""
    # squash each dimension to roughly [-1, 1] then take a few summary stats
    squashed = np.tanh(latent)
    return {
        "jitter_mod": float(0.7 + 0.6 * (squashed[0:4].mean() + 1) / 2),      # ~0.7 - 1.3
        "slant_var_mod": float(0.6 + 0.8 * (squashed[4:8].mean() + 1) / 2),   # ~0.6 - 1.4
        "stroke_var_mod": float(0.7 + 0.6 * (squashed[8:12].mean() + 1) / 2), # ~0.7 - 1.3
        "flow_mod": float(0.8 + 0.4 * (squashed[12:16].mean() + 1) / 2),      # ~0.8 - 1.2
    }
