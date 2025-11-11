from __future__ import annotations
from typing import Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import matplotlib.pyplot as plt
from tqdm import trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gauss_logprob(x: Tensor, mean: Tensor, var: Tensor) -> Tensor:
    var = var.clamp_min(1e-12)
    return -0.5 * ((x - mean) ** 2 / var + var.log() + math.log(2.0 * math.pi)).sum(dim=-1)



def render_pendulum(theta: float, img_size: int = 32) -> np.ndarray:
    """Render pendulum matching paper implementation."""
    H, W = img_size, img_size
    cx, cy = W // 2, int(round(0.18 * H))
    L, r = 0.68 * H, 2.0
    
    bx = cx + L * np.sin(theta)
    by = cy + L * np.cos(theta)
    
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    
    dx, dy = (bx - cx), (by - cy)
    denom = dx * dx + dy * dy + 1e-8
    s = ((xx - cx) * dx + (yy - cy) * dy) / denom
    s = np.clip(s, 0.0, 1.0)
    
    projx = cx + s * dx
    projy = cy + s * dy
    dist_line = np.sqrt((xx - projx) ** 2 + (yy - projy) ** 2)
    dist_bob = np.sqrt((xx - bx) ** 2 + (yy - by) ** 2)
    
    rod = (dist_line < 0.7).astype(np.float32)
    bob = (dist_bob < r).astype(np.float32)
    
    img = np.clip(rod + bob, 0.0, 1.0)
    return img


def render_pendulum_sequence(theta_sequence: np.ndarray, img_size: int = 32) -> np.ndarray:
    """Render a sequence of pendulum images."""
    return np.stack([render_pendulum(theta, img_size) for theta in theta_sequence])


def pendulum_ode(state: Tensor, t: Tensor) -> Tensor:
    """UNDAMPED pendulum dynamics: dθ/dt = ω, dω/dt = -sin(θ)"""
    theta, omega = state[:, 0:1], state[:, 1:2]
    dtheta = omega
    domega = -torch.sin(theta)
    return torch.cat([dtheta, domega], dim=1)


def solve_ode_rk4(f, z0: Tensor, ts: float, tf: float, n_steps: int) -> tuple[Tensor, Tensor]:
    """RK4 solver for ODEs."""
    dt = (tf - ts) / n_steps
    t_eval = torch.linspace(ts, tf, n_steps + 1, device=z0.device)
    
    path = [z0]
    z = z0
    for i in range(n_steps):
        t = torch.full((z.shape[0], 1), t_eval[i], device=z.device)
        k1 = f(z, t)
        k2 = f(z + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(z + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(z + dt * k3, t + dt)
        z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        path.append(z)
    
    return t_eval, torch.stack(path)


def gen_pendulum_data(
    batch_size: int,
    n_obs: int = 40,
    t_sec: float = 30.,
    fps: int = 15,
    img_size: int = 32
) -> tuple[Tensor, Tensor]:
    """Generate pendulum image training data."""
    theta0 = (2 * torch.rand(batch_size, 1) - 1) * np.pi
    omega0 = torch.randn(batch_size, 1) * 0.5
    state0 = torch.cat([theta0, omega0], dim=1)
    
    n_steps = int(t_sec * fps)
    t_eval, state_path = solve_ode_rk4(pendulum_ode, state0, 0., t_sec, n_steps=n_steps)
    
    obs_indices = torch.linspace(0, n_steps, n_obs).long()
    state_obs = state_path[obs_indices].permute(1, 0, 2)
    
    images = []
    for batch_idx in range(batch_size):
        theta_seq = state_obs[batch_idx, :, 0].numpy()
        img_seq = render_pendulum_sequence(theta_seq, img_size)
        images.append(img_seq)
    
    images = torch.tensor(np.stack(images), dtype=torch.float32)
    timestamps = t_eval[obs_indices][None, :, None].repeat(batch_size, 1, 1)
    
    return images, timestamps


def visualise_sequences(sequences: Tensor, filename: str = "pendulum.jpg", n_show: int = 6):
    """Visualize pendulum image sequences."""
    sequences = sequences[:n_show].detach().cpu().numpy()
    n_sequences = len(sequences)
    n_frames = sequences.shape[1]
    
    frame_skip = max(1, n_frames // 10)
    frames_to_show = list(range(0, n_frames, frame_skip))[:10]
    
    fig, axes = plt.subplots(n_sequences, len(frames_to_show), figsize=(20, 2*n_sequences))
    
    if n_sequences == 1:
        axes = axes[np.newaxis, :]
    
    for seq_idx in range(n_sequences):
        for frame_idx, t in enumerate(frames_to_show):
            ax = axes[seq_idx, frame_idx]
            ax.imshow(sequences[seq_idx, t], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if seq_idx == 0:
                ax.set_title(f't={t}', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filename, format='jpg', dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Image Encoder/Decoder
# ============================================================================

class ImageEncoder(nn.Module):
    """CNN encoder: image -> latent features for posterior."""
    def __init__(self, img_size: int = 32, latent_dim: int = 64):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),     # 32 -> 16
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1),    # 16 -> 8
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1),    # 8 -> 4
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent_dim)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B, T, H, W]
        returns: [B, T, latent_dim]
        """
        B, T, H, W = x.shape
        x_flat = x.view(B * T, 1, H, W)
        z_flat = self.cnn(x_flat)
        return z_flat.view(B, T, self.latent_dim)


# ============================================================================
# Decoder Network (Latent -> Image Logits)
# ============================================================================

class TemporalDecoder(nn.Module):
    """
    Temporal decoder conditioned on latent and time.
    Maps: [B, T, latent_dim] -> [B, T, H, W] logits for Bernoulli
    """
    def __init__(self, latent_dim: int, img_size: int = 32):
        super().__init__()
        self.img_size = img_size
        self.init_size = 4
        
        # Time embedding
        self.time_embed = TimeEmbed(model_dim=64)
        
        # Combine latent + time
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + 64, 256),
            nn.SiLU(),
            nn.Linear(256, 64 * self.init_size * self.init_size)
        )
        
        # ConvTranspose decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 4 -> 8
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 8 -> 16
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),   # 16 -> 32
            nn.BatchNorm2d(8),
            nn.SiLU(),
            nn.Conv2d(8, 1, 3, 1, 1),  # Output logits (no activation)
        )
    
    def forward(self, z_latent: Tensor, t: Tensor) -> Tensor:
        """
        z_latent: [B, T, latent_dim]
        t: [B, T, 1]
        returns: logits [B, T, H, W]
        """
        B, T, D = z_latent.shape
        
        # Time embedding
        t_emb = self.time_embed(t)  # [B, T, 64]
        
        # Concatenate latent + time
        z_combined = torch.cat([z_latent, t_emb], dim=-1)  # [B, T, D+64]
        
        # Flatten batch and time for processing
        z_flat = z_combined.view(B * T, -1)
        h = self.fc(z_flat)
        h = h.view(-1, 64, self.init_size, self.init_size)
        
        # Decode to logits
        logits = self.decoder(h).squeeze(1)  # [B*T, H, W]
        
        return logits.view(B, T, self.img_size, self.img_size)


# ============================================================================
# OU path prior with exact path density & sampling (diagonal)
# ============================================================================

class DiagOUSDE(nn.Module):
    def __init__(self, D: int, init_mu: float = 0.0, init_logk: float = -0.7, init_logs: float = -0.7):
        super().__init__()
        self.mu        = nn.Parameter(torch.full((D,), init_mu))
        self.log_kappa = nn.Parameter(torch.full((D,), init_logk))
        self.log_sigma = nn.Parameter(torch.full((D,), init_logs))

    def _params(self):
        kappa = F.softplus(self.log_kappa) + 1e-6
        sigma = F.softplus(self.log_sigma) + 1e-6
        mu    = self.mu
        return mu, kappa, sigma

    @staticmethod
    def _A_Q(kappa: Tensor, sigma: Tensor, dt: Tensor) -> Tuple[Tensor, Tensor]:
        A = torch.exp(-kappa * dt)
        two_k_dt = 2.0 * kappa * dt
        small = (two_k_dt < 1e-6).to(dt.dtype)
        Q_exact  = (sigma**2) * (1.0 - torch.exp(-two_k_dt)) / (2.0 * kappa).clamp_min(1e-12)
        Q_taylor = (sigma**2) * dt * (1.0 - kappa * dt + (two_k_dt**2)/6.0)
        Q = small * Q_taylor + (1.0 - small) * Q_exact
        return A, Q

    @torch.no_grad()
    def sample_path(self, ts: Tensor, batch_size: int) -> Tensor:
        ts = ts.view(-1)
        mu, kappa, sigma = self._params()
        D = mu.numel()
        T = ts.numel()
        y = torch.empty(batch_size, T, D, device=ts.device)
        var0 = (sigma**2) / (2.0 * kappa)
        y[:, 0, :] = mu + torch.randn(batch_size, D, device=ts.device) * var0.sqrt()
        for k in range(T - 1):
            dt = (ts[k + 1] - ts[k]).clamp_min(1e-6)
            A, Q = self._A_Q(kappa, sigma, dt)
            mean = mu + A * (y[:, k, :] - mu)
            y[:, k + 1, :] = mean + torch.randn_like(mean) * Q.sqrt()
        return y

    def path_log_prob(self, y: Tensor, ts: Tensor) -> Tensor:
        B, T, D = y.shape
        if ts.dim() == 1: ts = ts[None, :, None].expand(B, -1, 1)
        if ts.dim() == 2: ts = ts[None, :, :].expand(B, -1, -1)
        mu, kappa, sigma = self._params()
        mu = mu.view(1,1,D); kappa = kappa.view(1,1,D); sigma = sigma.view(1,1,D)
        var0 = (sigma**2) / (2.0 * kappa)
        lp0 = gauss_logprob(y[:, 0, :], mu[:, 0, :], var0[:, 0, :])
        dt = (ts[:, 1:, :] - ts[:, :-1, :]).clamp_min(1e-6)
        A, Q = self._A_Q(kappa, sigma, dt)
        mean = mu + A * (y[:, :-1, :] - mu)
        lp_trans = gauss_logprob(y[:, 1:, :], mean, Q).sum(dim=1)
        return (lp0 + lp_trans).float()


# ============================================================================
# Time features & utilities
# ============================================================================

class TimeEmbed(nn.Module):
    def __init__(self, model_dim: int = 128, max_freq: float = 16.0):
        super().__init__()
        assert model_dim % 2 == 0
        freqs = torch.exp(torch.linspace(0., math.log(max_freq), model_dim // 2))
        self.register_buffer("freqs", freqs)

    def forward(self, t: Tensor) -> Tensor:
        t = t.to(self.freqs.dtype)
        ang = t * self.freqs[None, None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, zero_last: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        if zero_last:
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def bounded_exp_scale(raw_scale: Tensor, scale_factor: float) -> Tuple[Tensor, Tensor]:
    log_s = torch.tanh(raw_scale) * scale_factor
    return torch.exp(log_s), log_s


# ============================================================================
# Invertible blocks
# ============================================================================

class ReversePermute(nn.Module):
    def __init__(self, D: int):
        super().__init__()
        self.register_buffer("perm", torch.arange(D - 1, -1, -1))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return x.index_select(dim=-1, index=self.perm), torch.zeros(x.size(0), device=x.device)

    def inverse(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        inv = torch.argsort(self.perm)
        return z.index_select(dim=-1, index=inv), torch.zeros(z.size(0), device=z.device)


class AffineCoupling(nn.Module):
    """Per-time affine coupling conditioned on x_a, time features, and latent path c_t."""
    def __init__(self, D: int, cond_dim: int, model_dim: int, scale_factor: float = 3.0):
        super().__init__()
        assert D >= 2
        self.D = D
        self.d_a = D // 2
        self.d_b = D - self.d_a
        self.scale_factor = scale_factor

        self.x_proj   = nn.Linear(self.d_a, model_dim)
        self.t_embed  = TimeEmbed(model_dim)
        self.t_mlp    = MLP(model_dim, model_dim, model_dim, zero_last=False)
        self.c_proj   = nn.Linear(cond_dim, model_dim) if cond_dim > 0 else None
        in_feats      = 3 * model_dim if cond_dim > 0 else 2 * model_dim
        self.out_mlp  = MLP(in_feats, model_dim, 2 * self.d_b, zero_last=True)

        nn.init.trunc_normal_(self.x_proj.weight, std=0.02); nn.init.zeros_(self.x_proj.bias)

    def _features(self, x_a: Tensor, t: Tensor, c: Optional[Tensor]) -> Tensor:
        f_x = self.x_proj(x_a)
        f_t = self.t_mlp(self.t_embed(t))
        if self.c_proj is not None and c is not None:
            f_c = self.c_proj(c)
            return torch.cat([f_x, f_t, f_c], dim=-1)
        else:
            return torch.cat([f_x, f_t], dim=-1)

    def forward(self, x: Tensor, t: Tensor, c: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        x_a, x_b = x.split([self.d_a, self.d_b], dim=-1)
        feat = self._features(x_a, t, c)
        params = self.out_mlp(feat)
        bias, raw_scale = params.chunk(2, dim=-1)
        scale, log_s = bounded_exp_scale(raw_scale, self.scale_factor)
        z_b = (x_b + bias) * scale
        z = torch.cat([x_a, z_b], dim=-1)
        ldj = log_s.sum(dim=-1).sum(dim=1)
        return z, ldj

    def inverse(self, z: Tensor, t: Tensor, c: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        z_a, z_b = z.split([self.d_a, self.d_b], dim=-1)
        feat = self._features(z_a, t, c)
        params = self.out_mlp(feat)
        bias, raw_scale = params.chunk(2, dim=-1)
        scale, log_s = bounded_exp_scale(raw_scale, self.scale_factor)
        x_b = (z_b / scale) - bias
        x = torch.cat([z_a, x_b], dim=-1)
        ldj = -log_s.sum(dim=-1).sum(dim=1)
        return x, ldj


class FlowSequence(nn.Module):
    def __init__(self, D: int, cond_dim: int, model_dim: int, depth: int, scale_factor: float = 3.0):
        super().__init__()
        blocks = []
        for _ in range(depth):
            blocks.append(AffineCoupling(D, cond_dim, model_dim, scale_factor=scale_factor))
            blocks.append(ReversePermute(D))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor, t: Tensor, c: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        ldj = torch.zeros(x.size(0), device=x.device)
        for b in self.blocks:
            if isinstance(b, AffineCoupling):
                x, ldj_i = b(x, t, c)
            else:
                x, ldj_i = b(x)
            ldj = ldj + ldj_i
        return x, ldj

    def inverse(self, z: Tensor, t: Tensor, c: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        ldj = torch.zeros(z.size(0), device=z.device)
        for b in reversed(self.blocks):
            if isinstance(b, AffineCoupling):
                z, ldj_i = b.inverse(z, t, c)
            else:
                z, ldj_i = b.inverse(z)
            ldj = ldj + ldj_i
        return z, ldj


class SoftTimeAttention(nn.Module):
    """Soft attention over time with Gaussian kernel."""
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=False)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, h_seq: Tensor, t: Tensor) -> Tensor:
        _, T, _ = h_seq.shape
        delta = (t.unsqueeze(2) - t.unsqueeze(1)).squeeze(-1)
        logits = - (T * delta)**2 / (self.temperature.clamp_min(1e-6) ** 2)
        w = self.sm(logits)
        return w @ h_seq


# ============================================================================
# OU-VAE posterior q_phi(c|x)
# ============================================================================

class PosteriorOUEncoder(nn.Module):
    """OU posterior q_phi(c | x) with soft time-local attention."""
    def __init__(self, x_dim: int, cond_dim: int, hidden: int = 256):
        super().__init__()
        self.cond_dim = cond_dim
        self.gru = nn.GRU(input_size=x_dim, hidden_size=hidden, num_layers=1, batch_first=True)
        self.attn = SoftTimeAttention()

        self.init_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * cond_dim)
        )
        self.step_head = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 3 * cond_dim)
        )

        nn.init.zeros_(self.init_head[-1].weight); nn.init.zeros_(self.init_head[-1].bias)
        nn.init.zeros_(self.step_head[-1].weight); nn.init.zeros_(self.step_head[-1].bias)

    @staticmethod
    def _A_Q(kappa: Tensor, sigma: Tensor, dt: Tensor) -> Tuple[Tensor, Tensor]:
        A = torch.exp(-kappa * dt)
        two_k_dt = 2.0 * kappa * dt
        small = (two_k_dt < 1e-6).to(dt.dtype)
        Q_exact  = (sigma**2) * (1.0 - torch.exp(-two_k_dt)) / (2.0 * kappa).clamp_min(1e-12)
        Q_taylor = (sigma**2) * dt * (1.0 - kappa * dt + (two_k_dt**2)/6.0)
        Q = small * Q_taylor + (1.0 - small) * Q_exact
        return A, Q

    def forward(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Sample c ~ q(c|x) and compute log q(c|x)."""
        h_seq, h_last = self.gru(x)
        h_global = h_last[0]

        attn = self.attn(h_seq, t)

        h0_ctx = h_global + attn[:, 0, :]
        m0, logs0 = self.init_head(h0_ctx).chunk(2, dim=-1)
        s0 = F.softplus(logs0) + 1e-6

        ctx = torch.cat([h_global.unsqueeze(1) + attn, t], dim=-1)
        params = self.step_head(ctx[:, :-1, :])
        mu_k, logk_k, logs_k = params.chunk(3, dim=-1)
        kappa_k = F.softplus(logk_k) + 1e-6
        sigma_k = F.softplus(logs_k) + 1e-6

        dt = (t[:, 1:, :] - t[:, :-1, :]).clamp_min(1e-6)
        A_k, Q_k = self._A_Q(kappa_k, sigma_k, dt)
        c, log_q = self.ou_posterior(m0, s0, mu_k, A_k, Q_k)

        return c, log_q

    def ou_posterior(
        self,
        m0: Tensor,
        s0: Tensor,
        mu_k: Tensor,
        A_k: Tensor,
        Q_k: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Reparameterized OU posterior sampling."""
        B, Tm1, C = mu_k.shape
        T = Tm1 + 1
        device = mu_k.device
        dtype  = mu_k.dtype

        eps0 = torch.randn(B, C, device=device, dtype=dtype)
        c0   = m0 + s0 * eps0
        log_q0 = -0.5 * (((c0 - m0) / s0.clamp_min(1e-12))**2 + (s0.clamp_min(1e-12)).log()*2 + math.log(2*math.pi)).sum(-1)

        eps = torch.randn(B, Tm1, C, device=device, dtype=dtype)
        noise = eps * Q_k.clamp_min(1e-12).sqrt()
        b = (1.0 - A_k) * mu_k + noise

        one = torch.ones(B, 1, C, device=device, dtype=dtype)
        Phi_prefix = torch.cumprod(A_k, dim=1)
        Phi_full = torch.cat([one, Phi_prefix], dim=1)

        denom = Phi_full[:, 1:, :].clamp_min(1e-20)
        s = b / denom
        zero = torch.zeros(B, 1, C, device=device, dtype=dtype)
        S_prefix = torch.cumsum(s, dim=1)
        S_full = torch.cat([zero, S_prefix], dim=1)

        c = Phi_full * (c0[:, None, :] + S_full)

        mean_trans = mu_k + A_k * (c[:, :-1, :] - mu_k)
        var_trans  = Q_k.clamp_min(1e-12)
        log_q_trans = -0.5 * (
            ((c[:, 1:, :] - mean_trans)**2 / var_trans)
            + var_trans.log()
            + math.log(2.0 * math.pi)
        ).sum(-1).sum(-1)

        log_q = log_q0 + log_q_trans
        return c, log_q



class PathFlowVAE(nn.Module):
    """
    Generative model for image sequences:
      1. Sample c ~ p(c) [OU conditioning path]
      2. Sample y ~ p(y) [OU latent base path]
      3. x_latent = flow_latent^{-1}(y | c, t)
      4. logits = decoder(x_latent, t)
      5. x_img ~ Bernoulli(sigmoid(logits))
    
    Inference:
      1. Encode images for features
      2. Infer q_base(c_base|x) via base posterior encoder
      3. Transform: c = flow_posterior^{-1}(c_base | x_features, t)
      4. Compute ELBO with flow-transformed posterior
    """
    def __init__(self, img_size: int, latent_dim: int, cond_dim: int, model_dim: int, 
                 depth_latent: int, depth_posterior: int = 2):
        super().__init__()
        self.img_size = img_size

        # Encoder for posterior inference
        self.encoder = ImageEncoder(img_size=img_size, latent_dim=latent_dim)

        # Latent space flow: y -> x_latent conditioned on c
        self.latent_flow = FlowSequence(
            D=latent_dim, 
            cond_dim=cond_dim, 
            model_dim=model_dim, 
            depth=depth_latent, 
            scale_factor=3.0
        )

        # Posterior flow: c_base -> c conditioned on x_features
        # This makes the posterior more flexible
        self.posterior_flow = FlowSequence(
            D=cond_dim,
            cond_dim=latent_dim,  # Condition on encoded image features
            model_dim=model_dim,
            depth=depth_posterior,
            scale_factor=3.0
        )

        # Decoder: x_latent, t -> logits
        self.decoder = TemporalDecoder(latent_dim=latent_dim, img_size=img_size)

        # Priors
        self.prior_y = DiagOUSDE(D=latent_dim, init_logk=-1.0, init_logs=0.4)  # Base latent prior
        self.prior_c = DiagOUSDE(D=cond_dim, init_logk=-1.0, init_logs=0.4)    # Conditioning path prior

        # Base posterior (OU process)
        self.posterior_c_base = PosteriorOUEncoder(x_dim=latent_dim, cond_dim=cond_dim, hidden=model_dim)

    def log_p_latent_given_c(self, x_latent: Tensor, t: Tensor, c: Tensor) -> Tensor:
        """Compute log p(x_latent|c) via latent flow."""
        y, ldj = self.latent_flow(x_latent, t, c)
        log_p_y = self.prior_y.path_log_prob(y, t)
        return log_p_y + ldj
    
    def log_p_img_given_latent(self, x_img: Tensor, x_latent: Tensor, t: Tensor) -> Tensor:
        """
        Compute log p(x_img|x_latent) via Bernoulli likelihood.
        x_img: [B, T, H, W] binary images in {0, 1}
        """
        # Get logits from decoder
        logits = self.decoder(x_latent, t)  # [B, T, H, W]
        
        # Bernoulli log likelihood
        # log p(x|logits) = x * log(sigmoid(logits)) + (1-x) * log(1 - sigmoid(logits))
        #                 = x * log_sigmoid(logits) + (1-x) * log_sigmoid(-logits)
        log_prob = x_img * F.logsigmoid(logits) + (1 - x_img) * F.logsigmoid(-logits)
        
        # Sum over spatial dimensions, sum over time
        return log_prob.sum(dim=[2, 3]).sum(dim=1)  # [B]

    @torch.compile()
    def forward(self, x_img: Tensor, t: Tensor) -> Tuple[Tensor, dict]:
        """
        Compute negative ELBO with posterior flow.
        Returns: (total_loss, loss_dict)
        """
        B, T, H, W = x_img.shape
        
        # Encode images to latent features (for posterior)
        x_features = self.encoder(x_img)
        
        # Sample base conditioning path from base posterior: q_base(c_base|x)
        c_base, log_q_base = self.posterior_c_base(x_features, t)
        
        # Apply posterior flow: c = flow^{-1}(c_base | x_features, t)
        c, ldj_posterior = self.posterior_flow.inverse(c_base, t, x_features)
        
        # Compute log q(c|x) = log q_base(c_base|x) - log|det J|
        log_q_c = log_q_base - ldj_posterior
        
        # Prior on conditioning path: p(c)
        log_p_c = self.prior_c.path_log_prob(c, t)
        
        # KL term: log q(c|x) - log p(c)
        kl_c = log_q_c - log_p_c
        
        # Likelihood of latent given conditioning: log p(x_latent|c)
        log_p_latent_c = self.log_p_latent_given_c(x_features, t, c)
        
        # Likelihood of images given latent: log p(x_img|x_latent)
        log_p_img_latent = self.log_p_img_given_latent(x_img, x_features, t)
        
        # Total log likelihood: log p(x_img, x_latent | c)
        log_likelihood = log_p_latent_c + log_p_img_latent
        
        # ELBO = E_q(c|x)[log p(x_img, x_latent | c) + log p(c) - log q(c|x)]
        #      = E_q(c|x)[log p(x_img, x_latent | c)] - KL(q(c|x)||p(c))
        elbo = log_likelihood - kl_c
        
        # Negative ELBO (loss to minimize)
        loss = -elbo.mean()
        
        # Loss dictionary for logging
        loss_dict = {
            'total': loss.item(),
            'kl_c': kl_c.mean().item(),
            'log_q_base': log_q_base.mean().item(),
            'ldj_posterior': ldj_posterior.mean().item(),
            'log_p_latent|c': log_p_latent_c.mean().item(),
            'log_p_img|latent': log_p_img_latent.mean().item(),
            'log_likelihood': log_likelihood.mean().item(),
            'elbo': elbo.mean().item(),
        }
        
        return loss, loss_dict

    @torch.no_grad()
    def sample(self, n_samples: int, steps: int, t_start: float, t_end: float) -> Tensor:
        """Sample image sequences from the model."""
        ts = torch.linspace(t_start, t_end, steps=steps, device=device)
        tB = ts[None, :, None].repeat(n_samples, 1, 1)
        
        # Sample from priors
        c = self.prior_c.sample_path(ts, n_samples)           # [B, T, cond_dim]
        y = self.prior_y.sample_path(ts, n_samples)           # [B, T, latent_dim]
        
        # Apply inverse flow to get latent
        x_latent, _ = self.latent_flow.inverse(y, tB, c)
        
        # Decode to logits
        logits = self.decoder(x_latent, tB)
        
        probs = torch.sigmoid(logits)
        
        return probs


# ============================================================================
# Training
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    # Generate data
    batch_size = 512
    n_obs = 40
    t_sec = 30.0
    fps = 15
    img_size = 32

    print("Generating pendulum data...")
    xs_img, ts_arr = gen_pendulum_data(batch_size, n_obs, t_sec=t_sec, fps=fps, img_size=img_size)
    xs_img = xs_img.to(device)
    ts_arr = ts_arr.to(device)
    
    # Normalize timestamps to [0, 1]
    #ts_arr = (ts_arr - ts_arr.min()) / (ts_arr.max() - ts_arr.min())

    print(f"Data shape: {xs_img.shape}")
    print(f"Time shape: {ts_arr.shape}")
    
    # Visualize training data
    visualise_sequences(xs_img, 'data.jpg', n_show=6)

    # Model parameters
    latent_dim = 8          # Latent dimension for encoded features
    cond_dim = 16            # Conditioning path dimensionality
    model_dim = 128          # Hidden dimension for flows
    depth_latent = 16         # Number of flow layers in latent space
    depth_posterior = 16      # Number of flow layers in posterior

    model = PathFlowVAE(
        img_size=img_size,
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        model_dim=model_dim,
        depth_latent=depth_latent,
        depth_posterior=depth_posterior
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {n_params:,}")

    # Train
    iters = 20000
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.dim() >= 2:
                decay.append(param)
            else:
                no_decay.append(param)
    optim = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.8},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=1e-3
    )
    
    print("\nTraining...")
    print("(Saving samples every 500 iterations)\n")
    
    pbar = trange(iters)
    for step in pbar:
        loss, loss_dict = model(xs_img, ts_arr)
        
        # Update progress bar with key loss components
        desc = f"Loss: {loss_dict['total']:.4f} | "
        desc += f"KL: {loss_dict['kl_c']:.4f} | "
        desc += f"LDJ_post: {loss_dict['ldj_posterior']:.3f} | "
        desc += f"LogP(img|lat): {loss_dict['log_p_img|latent']:.3f}"
        pbar.set_description(desc)
        
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()

        if step % 500 == 0:
            with torch.no_grad():
                samples = model.sample(n_samples=6, steps=n_obs, 
                                     t_start=0., t_end=1.)
                visualise_sequences(samples, f'./samples/samples.jpg', n_show=6)
            
            # Print detailed loss breakdown
            print(f"\nStep {step}:")
            for key, val in loss_dict.items():
                print(f"  {key}: {val:.4f}")
            print()

    # Final samples
    print("\nGenerating final samples...")
    with torch.no_grad():
        samples = model.sample(n_samples=6, steps=n_obs, 
                             t_start=0., t_end=1.)
        visualise_sequences(samples, 'samples_final.jpg', n_show=6)

    print("\nDone! Check samples_*.jpg for generated sequences.")
