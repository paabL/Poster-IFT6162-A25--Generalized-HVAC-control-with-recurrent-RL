# features_extractor_rc5.py
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class StepMLPTimeWeightedFuseExtractor(BaseFeaturesExtractor):
    r"""
    ============================================================
    GOAL (simple + physics-based)
    ============================================================

    The env returns a Dict observation:

      - now      : vecteur (B, now_dim)
                  "what I know now"
                  e.g. aggregated current weather, current internal gains,
                       comfort bounds, time (week_idx, sin/cos), etc.
                       + (Tz, setpoint) if you put them in now.

      - forecast : matrice (B, H, f_dim)
                  "future snapshots" for k=1..H
                  e.g. forecast weather, future comfort bands, future time features, etc.

    RecurrentPPO (LSTM) does NOT ingest a matrix (H, f_dim) directly:
    it expects a VECTOR per RL step.
    So we transform {now, forecast} -> z_t (B, d_out).

    ============================================================
    PIPELINE (very simple version)
    ============================================================

    Step A — Per-future-step encoding (same MLP for all horizons)
      e_{t,k} = step_mlp( forecast_{t,k} )  ∈ R^{d_step}

    Physical interpretation:
      e_{t,k} is NOT a physical variable directly.
      It is a "latent representation" that the network learns to summarize
      the energy/comfort impact of future snapshot k.
      Intuitively, an e_{t,k} can implicitly encode
      "cold risk", "solar load", "tight comfort constraints", etc.

    Step B — SIMPLE temporal summary (no complicated attention)
      We take a weighted average that depends on k (time):

        w_k ∝ gamma^(k-1)   with gamma ∈ (0,1]
        z_fc = Σ_k w_k * e_{t,k}

      - If gamma=1.0 -> uniform average (no temporal preference)
      - If gamma<1.0 -> we give more weight to the near future
        (often sensible in HVAC: myopic but stable decisions)

    Step C — Fuse with now
      z_t = fuse_mlp( [ now_t , z_fc ] )  ∈ R^{d_out}

    z_t is then fed to the LSTM at time t.

    ============================================================
    SHAPES
    ============================================================
      now      : (B, now_dim)
      forecast : (B, H, f_dim)
      e        : (B, H, d_step)
      z_fc     : (B, d_step)
      z_t      : (B, d_out)
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        d_step: int = 32,        # latent size per horizon e_{t,k}
        hidden_step: int = 64,   # hidden layer of the forecast MLP
        d_out: int = 64,         # final size fed to the LSTM
        hidden_fuse: int = 128,  # hidden layer of the fusion MLP
        gamma: float = 0.90,     # time decay (1.0 = uniform)
    ):
        assert isinstance(observation_space, spaces.Dict)

        now_space = observation_space.spaces["now"]
        fc_space = observation_space.spaces["forecast"]

        self.now_dim = int(now_space.shape[0])
        self.H = int(fc_space.shape[0])       # horizon fixed by the env
        self.f_dim = int(fc_space.shape[1])   # number of features per future step
        self.d_step = int(d_step)

        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1].")
        self.gamma = float(gamma)

        # features_dim = output dimension provided to the SB3 policy (and thus to the LSTM)
        super().__init__(observation_space, features_dim=int(d_out))

        # MLP applied to each future snapshot forecast_{t,k}
        self.step_mlp = nn.Sequential(
            nn.Linear(self.f_dim, hidden_step),
            nn.ReLU(),
            nn.Linear(hidden_step, self.d_step),
            nn.ReLU(),
        )

        # Fusion finale : [now, z_fc] -> z_t
        fuse_in = self.now_dim + self.d_step
        self.fuse_mlp = nn.Sequential(
            nn.Linear(fuse_in, hidden_fuse),
            nn.ReLU(),
            nn.Linear(hidden_fuse, d_out),
            nn.ReLU(),
        )

        # Fixed temporal weights (H,)
        # w_k ∝ gamma^(k-1), normalized to sum to 1
        if self.H > 0:
            w = (self.gamma ** torch.arange(self.H, dtype=torch.float32))  # k=0..H-1
            w = w / w.sum()
        else:
            w = torch.zeros((0,), dtype=torch.float32)

        # register_buffer => follows the device (cpu/gpu) automatically
        self.register_buffer("time_weights", w, persistent=False)

    def forward(self, obs):
        """
        obs is a dict of torch tensors (SB3 already converted from numpy).
        """
        x_now = obs["now"]        # (B, now_dim)
        x_fc = obs["forecast"]    # (B, H, f_dim)
        B = x_now.shape[0]

        # Case H=0 (if future_steps=0 in the env)
        if x_fc.shape[1] == 0:
            z_fc = torch.zeros((B, self.d_step), device=x_now.device, dtype=x_now.dtype)
            x = torch.cat([x_now, z_fc], dim=1)
            return self.fuse_mlp(x)   # <-- IMPORTANT: always return a tensor

        # Encode each future step
        # (B, H, f_dim) -> (B*H, f_dim) -> (B, H, d_step)
        B, H, F = x_fc.shape
        e = self.step_mlp(x_fc.reshape(B * H, F)).reshape(B, H, self.d_step)

        # Time-weighted average (accounts for "k")
        # time_weights: (H,) -> (1, H, 1)
        w = self.time_weights
        if w.shape[0] != H:
            # safety in case H ever changes (normally it doesn't)
            w = (self.gamma ** torch.arange(H, device=e.device, dtype=torch.float32))
            w = w / w.sum()

        z_fc = (e * w.view(1, H, 1)).sum(dim=1)   # (B, d_step)

        # Fusion
        x = torch.cat([x_now, z_fc], dim=1)       # (B, now_dim + d_step)
        return self.fuse_mlp(x)                   # (B, d_out)
