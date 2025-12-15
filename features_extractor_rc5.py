# features_extractor_rc5.py
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class StepMLPTimeWeightedFuseExtractor(BaseFeaturesExtractor):
    r"""
    ============================================================
    OBJECTIF (simple + physique)
    ============================================================

    L'env renvoie une observation Dict :

      - now      : vecteur (B, now_dim)
                  "ce que je sais maintenant"
                  ex: météo actuelle agrégée, gains internes actuels,
                      bornes de confort, temps (week_idx, sin/cos), etc.
                      + (Tz, setpoint) si tu les as mis dans now.

      - forecast : matrice (B, H, f_dim)
                  "des snapshots du futur" pour k=1..H
                  ex: météo prévue, bandes de confort futures, features temps futures, etc.

    RecurrentPPO (LSTM) n'avale PAS directement une matrice (H, f_dim) :
    il attend un VECTEUR par step RL.
    Donc on transforme {now, forecast} -> z_t (B, d_out).

    ============================================================
    PIPELINE (version très simple)
    ============================================================

    Étape A — Encodage par pas futur (même MLP pour tous les horizons)
      e_{t,k} = step_mlp( forecast_{t,k} )  ∈ R^{d_step}

    Interprétation physique :
      e_{t,k} n'est PAS une variable physique directement.
      C'est une "représentation latente" que le réseau apprend pour résumer
      l'impact énergétique/confort du snapshot futur k.
      Exemple intuitif : un e_{t,k} peut encoder implicitement
      "risque de froid", "charge solaire", "contraintes confort serrées", etc.

    Étape B — Résumé temporel SIMPLE (pas d'attention compliquée)
      On fait une moyenne pondérée qui dépend de k (le temps) :

        w_k ∝ gamma^(k-1)   avec gamma ∈ (0,1]
        z_fc = Σ_k w_k * e_{t,k}

      - Si gamma=1.0 -> moyenne uniforme (aucune préférence temporelle)
      - Si gamma<1.0 -> on donne plus d'importance au futur proche
        (souvent logique en HVAC : décisions myopes mais stables)

    Étape C — Fusion avec now
      z_t = fuse_mlp( [ now_t , z_fc ] )  ∈ R^{d_out}

    z_t est ensuite l'entrée du LSTM à l'instant t.

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
        d_step: int = 32,        # taille du latent par horizon e_{t,k}
        hidden_step: int = 64,   # couche cachée du MLP de forecast
        d_out: int = 64,         # taille finale envoyée au LSTM
        hidden_fuse: int = 128,  # couche cachée du MLP de fusion
        gamma: float = 0.90,     # décroissance temporelle (1.0 = uniforme)
    ):
        assert isinstance(observation_space, spaces.Dict)

        now_space = observation_space.spaces["now"]
        fc_space = observation_space.spaces["forecast"]

        self.now_dim = int(now_space.shape[0])
        self.H = int(fc_space.shape[0])       # horizon fixé par l'env
        self.f_dim = int(fc_space.shape[1])   # nb features par pas futur
        self.d_step = int(d_step)

        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma doit être dans (0, 1].")
        self.gamma = float(gamma)

        # features_dim = dimension de sortie fournie à la policy SB3 (et donc au LSTM)
        super().__init__(observation_space, features_dim=int(d_out))

        # MLP appliqué à chaque snapshot futur forecast_{t,k}
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

        # Poids temporels fixes (H,)
        # w_k ∝ gamma^(k-1), normalisés pour sommer à 1
        if self.H > 0:
            w = (self.gamma ** torch.arange(self.H, dtype=torch.float32))  # k=0..H-1
            w = w / w.sum()
        else:
            w = torch.zeros((0,), dtype=torch.float32)

        # register_buffer => suit le device (cpu/gpu) automatiquement
        self.register_buffer("time_weights", w, persistent=False)

    def forward(self, obs):
        """
        obs est un dict de tenseurs torch (SB3 a déjà converti le numpy).
        """
        x_now = obs["now"]        # (B, now_dim)
        x_fc = obs["forecast"]    # (B, H, f_dim)
        B = x_now.shape[0]

        # Cas H=0 (si future_steps=0 dans l'env)
        if x_fc.shape[1] == 0:
            z_fc = torch.zeros((B, self.d_step), device=x_now.device, dtype=x_now.dtype)
            x = torch.cat([x_now, z_fc], dim=1)
            return self.fuse_mlp(x)   # <-- IMPORTANT: toujours retourner un tenseur

        # Encodage de chaque pas futur
        # (B, H, f_dim) -> (B*H, f_dim) -> (B, H, d_step)
        B, H, F = x_fc.shape
        e = self.step_mlp(x_fc.reshape(B * H, F)).reshape(B, H, self.d_step)

        # Moyenne pondérée temporelle (prend en compte "k")
        # time_weights: (H,) -> (1, H, 1)
        w = self.time_weights
        if w.shape[0] != H:
            # sécurité si jamais H change (normalement non)
            w = (self.gamma ** torch.arange(H, device=e.device, dtype=torch.float32))
            w = w / w.sum()

        z_fc = (e * w.view(1, H, 1)).sum(dim=1)   # (B, d_step)

        # Fusion
        x = torch.cat([x_now, z_fc], dim=1)       # (B, now_dim + d_step)
        return self.fuse_mlp(x)                   # (B, d_out)

