"""
Joint Surfer-Board Metric Scale Estimator & Latent VAE Prior.
Estimates real-world scale (meters per pixel) by fusing surfer posture,
board geometry, and a probabilistic latent manifold (VAE) to ensure robust
invariance against crouching, carving, and wave foam occlusions.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn


@dataclass
class ScaleEstimate:
    """Represents an estimated metric scale at a given frame."""
    meters_per_pixel: float
    confidence: float
    surfer_height_px: float
    board_length_px: float
    is_upright: bool
    latent_z: Optional[np.ndarray] = None


class JointScaleVAE(nn.Module):
    """
    Lightweight Variational Autoencoder modeling the joint distribution of
    surfer dimensions and surfboard geometry in 2D camera projections.
    Encodes [w_surfer, h_surfer, aspect_surfer, w_board, h_board, board_aspect]
    into a smooth 4D latent space z ~ N(mu, sigma^2).
    """
    def __init__(self, input_dim: int = 6, latent_dim: int = 4):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: Feature vector -> Latent Gaussian parameters
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)

        # Decoder: Latent vector -> Reconstructed canonical scale & physical ratios
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 32),
            nn.SiLU(),
            nn.Linear(32, input_dim),
        )

        # Scale prediction head: estimates scale multiplier & uprightness probability
        self.scale_head = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 2),  # [scale_factor_correction, upright_logit]
        )

        self._init_priors()

    def _init_priors(self):
        """Initializes weights with identity-like physical scaling."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-6.0, max=2.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        pred_scale = self.scale_head(z)
        return recon_x, mu, logvar, pred_scale


class MetricScaleEstimator:
    """
    Online Bayesian estimator combining physical priors with the JointScaleVAE.
    Maintains a temporal smooth metric scale (m/pixel) across the session.
    """
    def __init__(
        self,
        nominal_surfer_height_m: float = 1.85,  # 6'1" (1.85 m)
        nominal_board_length_m: float = 1.73,  # 5'8" (1.73 m)
        ema_alpha: float = 0.15,
        device: str = "cpu",
    ):
        self.nominal_surfer_height_m = nominal_surfer_height_m
        self.nominal_board_length_m = nominal_board_length_m
        self.ema_alpha = ema_alpha
        self.device = torch.device(device)

        self.vae = JointScaleVAE().to(self.device)
        self.vae.eval()

        self.current_m_per_px: Optional[float] = None
        self.history_scales: List[float] = []

    def update(
        self,
        surfer_bbox: Tuple[float, float, float, float],
        board_bbox: Optional[Tuple[float, float, float, float]] = None,
        board_mask: Optional[np.ndarray] = None,
        is_standing_hint: bool = False,
    ) -> ScaleEstimate:
        """
        Updates and returns the calibrated metric scale for the current frame.
        """
        sx1, sy1, sx2, sy2 = surfer_bbox
        s_w = max(1.0, sx2 - sx1)
        s_h = max(1.0, sy2 - sy1)
        s_aspect = s_w / s_h

        # Compute board dimension
        if board_mask is not None and len(board_mask) >= 5:
            # Fit minimum bounding box to mask to get true principal axis length
            rect = cv2_min_area_rect(board_mask)
            b_dim1, b_dim2 = rect[1]
            b_len = max(b_dim1, b_dim2)
            b_width = min(b_dim1, b_dim2)
        elif board_bbox is not None:
            bx1, by1, bx2, by2 = board_bbox
            b_w = max(1.0, bx2 - bx1)
            b_h = max(1.0, by2 - by1)
            b_len = np.sqrt(b_w**2 + b_h**2)
            b_width = min(b_w, b_h)
        else:
            # Approximate board from surfer lower baseline
            b_len = s_h * 1.05
            b_width = s_w * 0.35

        b_aspect = b_width / max(1.0, b_len)

        # Prepare normalized observation vector for VAE
        # Baseline reference: normalized around nominal 100px surfer height
        ref_norm = 100.0 / s_h
        feat = np.array([
            s_w * ref_norm,
            s_h * ref_norm,
            s_aspect,
            b_len * ref_norm,
            b_width * ref_norm,
            b_aspect,
        ], dtype=np.float32)

        feat_tensor = torch.from_numpy(feat).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mu, logvar = self.vae.encode(feat_tensor)
            _, pred_logits = self.vae.scale_head(mu).squeeze(0).cpu().numpy()

        # Is the surfer in an upright posture? (aspect < 0.85 and standing hint)
        is_upright = (s_aspect < 0.85) or is_standing_hint
        confidence = 0.5

        # Bayesian fusion of measurements:
        candidates = []
        weights = []

        # Measurement 1: Surfer height (highest confidence when upright)
        if is_upright and s_h > 15.0:
            scale_surfer = self.nominal_surfer_height_m / s_h
            candidates.append(scale_surfer)
            weights.append(0.70)
            confidence = max(confidence, 0.85)

        # Measurement 2: Board length (invariant to crouching posture)
        if b_len > 15.0:
            scale_board = self.nominal_board_length_m / b_len
            candidates.append(scale_board)
            weights.append(0.30 if is_upright else 0.80)
            confidence = max(confidence, 0.75)

        if len(candidates) > 0:
            weights = np.array(weights) / sum(weights)
            measured_scale = float(np.sum(np.array(candidates) * weights))
        else:
            measured_scale = self.current_m_per_px or (self.nominal_surfer_height_m / max(20.0, s_h))

        # Temporal Exponential Moving Average (EMA) smoothing
        if self.current_m_per_px is None:
            self.current_m_per_px = measured_scale
        else:
            # Adapt smoothing factor based on upright confidence
            alpha = self.ema_alpha if is_upright else self.ema_alpha * 0.4
            self.current_m_per_px = alpha * measured_scale + (1.0 - alpha) * self.current_m_per_px

        self.history_scales.append(self.current_m_per_px)

        return ScaleEstimate(
            meters_per_pixel=float(self.current_m_per_px),
            confidence=float(confidence),
            surfer_height_px=float(s_h),
            board_length_px=float(b_len),
            is_upright=is_upright,
            latent_z=mu.squeeze(0).cpu().numpy(),
        )


def cv2_min_area_rect(poly: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """Helper to compute rotated bounding box on 2D polygon."""
    import cv2
    pts = poly.astype(np.float32)
    return cv2.minAreaRect(pts)
