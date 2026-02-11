from sklearn.base import BaseEstimator
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MDNHeader(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_gaussians: int = 5,
        pi_temperature: float = 1.0,
        min_log_sigma: float = -3.0,  # σ ≈ 0.05
        max_log_sigma: float = 3.0,  # σ ≈ 20
    ):
        super().__init__()
        self.output_dim = output_dim
        self.n_gaussians = n_gaussians
        self.pi_temperature = pi_temperature
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma

        # ===== 主干网络：Linear → LayerNorm → SiLU =====
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.hidden_act = nn.SiLU()  # 数值平滑优于 GELU/softplus

        # ===== 头部 Projection 前再加一层 LayerNorm（强烈推荐）=====
        self.out_norm = nn.LayerNorm(hidden_dim)

        # mixture components
        self.pi_layer = nn.Linear(hidden_dim, n_gaussians)
        self.mu_layer = nn.Linear(hidden_dim, n_gaussians * output_dim)
        self.log_sigma_layer = nn.Linear(hidden_dim, n_gaussians * output_dim)

    def forward(self, x):
        # ---- backbone ----
        h = self.hidden_layer(x)
        h = self.hidden_norm(h)
        h = self.hidden_act(h)

        # ---- projection normalize: greatly stabilizes MDN ----
        h = self.out_norm(h)

        # ---- mixture weights use log_softmax ----
        logit_pi = self.pi_layer(h) / self.pi_temperature
        log_pi = F.log_softmax(logit_pi, dim=-1)  # [B,K]
        pi = log_pi.exp()

        # ---- mu ----
        mu = self.mu_layer(h).view(-1, self.n_gaussians, self.output_dim)

        # ---- log sigma (clamped for stability) ----
        log_sigma = self.log_sigma_layer(h).view(-1, self.n_gaussians, self.output_dim)
        log_sigma = torch.clamp(log_sigma, self.min_log_sigma, self.max_log_sigma)
        sigma = torch.exp(log_sigma)

        return pi, mu, sigma, log_pi, log_sigma


def mdn_loss(pi, mu, sigma, target, log_pi=None, log_sigma=None):
    """
    Stable MDN NLL
    """
    B, K, D = mu.shape
    y = target.unsqueeze(1).expand(B, K, D)

    if log_pi is None:
        log_pi = torch.log(pi.clamp_min(1e-12))

    if log_sigma is None:
        log_sigma = torch.log(sigma)

    # (y - μ)^2 / σ^2
    z = (y - mu) / sigma

    # log N
    log_norm = (
        -0.5 * (z * z).sum(-1) - log_sigma.sum(-1) - 0.5 * D * math.log(2 * math.pi)
    )  # [B, K]

    log_mix = torch.logsumexp(log_pi + log_norm, dim=-1)
    return -(log_mix.mean())


class MixtureDensityEstimator(BaseEstimator):
    """
    A scikit-learn compatible MDN (multivariate, diagonal covariance).
    """

    def __init__(
        self,
        hidden_dim=10,
        n_gaussians=5,
        epochs=1000,
        lr=0.01,
        weight_decay=0.0,
        pi_temperature: float = 1.0,
        min_sigma: float = 1e-3,
        device: str = None,
    ):
        self.hidden_dim = hidden_dim
        self.n_gaussians = n_gaussians
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.pi_temperature = pi_temperature
        self.min_sigma = min_sigma
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def _cast_torch(self, X, y):
        if not hasattr(self, "X_width_"):
            self.X_width_ = X.shape[1]
        if not hasattr(self, "y_dim_"):
            self.y_dim_ = y.shape[1] if y.ndim == 2 else 1
        assert X.shape[1] == self.X_width_, "Input dimension mismatch"
        y = y.reshape(len(y), self.y_dim_) if y.ndim == 1 else y
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        return X_t, y_t

    def fit(self, X, y):
        X, y = self._cast_torch(X, y)
        self.model_ = MixtureDensityNetwork(
            X.shape[1],
            self.hidden_dim,
            y.shape[1],
            self.n_gaussians,
            pi_temperature=self.pi_temperature,
            min_sigma=self.min_sigma,
        ).to(self.device)
        self.optimizer_ = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.model_.train()
        for _ in range(self.epochs):
            self.optimizer_.zero_grad()
            pi, mu, sigma = self.model_(X)
            loss = mdn_loss(pi, mu, sigma, y)
            loss.backward()
            self.optimizer_.step()
        return self

    def partial_fit(self, X, y, n_epochs=1):
        X, y = self._cast_torch(X, y)
        if not hasattr(self, "optimizer_"):
            self.optimizer_ = torch.optim.Adam(
                self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        self.model_.train()
        for _ in range(n_epochs):
            self.optimizer_.zero_grad()
            pi, mu, sigma = self.model_(X)
            loss = mdn_loss(pi, mu, sigma, y)
            loss.backward()
            self.optimizer_.step()
        return self

    @torch.no_grad()
    def forward(self, X):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        self.model_.eval()
        pi, mu, sigma = self.model_(X)
        return pi.cpu().numpy(), mu.cpu().numpy(), sigma.cpu().numpy()

    # --- 以下标量专用工具保留，但仅当 D=1 时可用 ---
    def _ensure_scalar(self):
        assert (
            getattr(self, "y_dim_", 1) == 1
        ), "pdf/cdf/predict(quantiles) 仅支持标量目标 (output_dim=1)"

    def validate_y_bound(self, y_bound, y_bound_):
        return y_bound if y_bound is not None else y_bound_

    def pdf(self, X, resolution=100, y_min=None, y_max=None):
        self._ensure_scalar()
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        pi, mu, sigma = self.forward(X)
        ys = np.linspace(
            self.validate_y_bound(y_min, mu.min()),
            self.validate_y_bound(y_max, mu.max()),
            resolution,
        )
        ys_b = np.broadcast_to(ys, (pi.shape[0], self.n_gaussians, resolution))
        pdf = np.sum(
            norm(mu[..., 0][:, :, None], sigma[..., 0][:, :, None]).pdf(ys_b)
            * pi[:, :, None],
            axis=1,
        )
        return pdf, ys

    def cdf(self, X, resolution=100):
        self._ensure_scalar()
        pdf, ys = self.pdf(X, resolution=resolution)
        cdf = pdf.cumsum(axis=1)
        cdf /= cdf[:, -1].reshape(-1, 1)
        return cdf, ys

    def predict(self, X, quantiles=None, resolution=100):
        self._ensure_scalar()
        cdf, ys = self.cdf(X, resolution=resolution)
        mean_pred = ys[np.argmax(cdf > 0.5, axis=1)]
        if not quantiles:
            return mean_pred
        q_out = np.zeros((len(X), len(quantiles)))
        for j, q in enumerate(quantiles):
            q_out[:, j] = ys[np.argmax(cdf > q, axis=1)]
        return mean_pred, q_out


class MDNInferenceMixin:
    @torch.inference_mode()
    def predict_mean(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.header(features)

    @torch.inference_mode()
    def predict_point(self, features: torch.Tensor) -> torch.Tensor:
        pi, mu, _ = self.predict_mean(features)
        return (pi.unsqueeze(-1) * mu).sum(dim=1)

    @torch.inference_mode()
    def sample(self, features: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        pi, mu, sigma = self.predict_mean(features)
        B, K, D = mu.shape
        samples = []
        for _ in range(n_samples):
            comp = torch.distributions.Categorical(pi).sample()  # [B]
            eps = torch.randn(B, D, device=mu.device)
            m = mu[torch.arange(B), comp, :]  # [B, D]
            s = sigma[torch.arange(B), comp, :]  # [B, D]
            samples.append(m + eps * s)
        return torch.stack(samples, dim=0)  # [n_samples, B, D]

    @torch.inference_mode()
    def predict_pdf(self, features: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pi, mu, sigma = self.predict_mean(features)
        B, K, D = mu.shape
        y = y.unsqueeze(1).expand(B, K, D)  # [B,K,D]
        z = (y - mu) / sigma
        log_norm = (
            -0.5 * (z * z).sum(-1)
            - torch.log(sigma).sum(-1)
            - 0.5 * D * math.log(2 * math.pi)
        )
        log_mix = torch.logsumexp(torch.log(pi + 1e-12) + log_norm, dim=-1)
        return torch.exp(log_mix)  # [B]

    @torch.inference_mode()
    def predict_cdf(self, features: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pi, mu, sigma = self.predict_mean(features)
        B, K, D = mu.shape
        y = y.unsqueeze(1).expand(B, K, D)  # [B,K,D]
        z = (y - mu) / sigma
        std_norm_cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
        cdf_k = std_norm_cdf.prod(dim=-1)  # [B,K]
        return (pi * cdf_k).sum(dim=-1)  # [B]
