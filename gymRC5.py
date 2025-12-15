import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from SIMAX.Simulation import SimulationDataset
from SIMAX.Controller import Controller_PID
from Utils.utils import RC5_steady_state_sys
import jax.numpy as jnp
import equinox as eqx
from Utils.Occup import build_occupancy, occupancy_probability
from Utils.rc5_cost import (
    DEFAULT_W_COMFORT_EUR_PER_KH,
    DEFAULT_W_ENERGY_EUR,
    DEFAULT_W_SAT_EUR_PER_UNIT_H,
    interval_reward_and_terms,
)

# Racine du projet (robuste aux exécutions depuis un sous-dossier, ex: `MPC/`)
ROOT = Path(__file__).resolve().parent

# Colonnes utilisées dans le dataset
CONTROL_COLS = ()
DISTURBANCE_COLS = (
    "InternalGainsCon[1]",
    "InternalGainsRad[1]",
    "weaSta_reaWeaHGloHor_y",
    "weaSta_reaWeaTDryBul_y",
    "reaQHeaPumCon_y",
    "LowerSetp[1]",
    "UpperSetp[1]",
)

# ------------------------------------------------------------------
#  Charger la simulation
# ------------------------------------------------------------------
sim_path = ROOT / "Models" / "sim_opti.pkl"
with sim_path.open("rb") as f:
    sim_opti_loaded = pickle.load(f)

# ------------------------------------------------------------------
#  Charger le dataset et en prendre au max N points
# ------------------------------------------------------------------
dataset = SimulationDataset.from_csv(
    str(ROOT / "datas" / "train_df.csv"),
    control_cols=CONTROL_COLS,
    disturbance_cols=DISTURBANCE_COLS,
)

# Temps (secondes) + composantes utiles
time_seconds = np.asarray(dataset.time, dtype=float)
hours = (time_seconds / 3600.0) % 24.0          # [0, 24)
days  = time_seconds / 86400.0
dow   = days % 7.0                               # day-of-week continu [0, 7)

# Prix élec (créneau)
electricity_price = np.where(
    (hours >= 17.0) & (hours < 23.0),
    0.5,
    0.2,
).astype(np.float64)

# Occupation (0/1)
occupancy = build_occupancy(time_seconds, seed=0).astype(np.float64)

# Temps : on garde week_idx (non cyclique)
week_idx = np.floor(days / 7.0).astype(np.float64)

# Temps : on remplace day_in_week/hour_in_day par sin/cos cycliques
hour_sin = np.sin(2.0 * np.pi * hours / 24.0).astype(np.float64)
hour_cos = np.cos(2.0 * np.pi * hours / 24.0).astype(np.float64)
dow_sin  = np.sin(2.0 * np.pi * dow   / 7.0 ).astype(np.float64)
dow_cos  = np.cos(2.0 * np.pi * dow   / 7.0 ).astype(np.float64)

# Reconstruction du dataset avec les features ajoutées
dataset = SimulationDataset(
    time=dataset.time,
    u=dataset.u,
    d={
        **dataset.d,
        "electricity_price": jnp.asarray(electricity_price, dtype=jnp.float64),
        "occupancy": jnp.asarray(occupancy, dtype=jnp.float64),

        # on garde week_idx
        "week_idx": jnp.asarray(week_idx, dtype=jnp.float64),

        # cycliques : jour + heure
        "dow_sin":  jnp.asarray(dow_sin,  dtype=jnp.float64),
        "dow_cos":  jnp.asarray(dow_cos,  dtype=jnp.float64),
        "hour_sin": jnp.asarray(hour_sin, dtype=jnp.float64),
        "hour_cos": jnp.asarray(hour_cos, dtype=jnp.float64),
    },
)





N = 150_000 #280_000 datas max
n_total = dataset.time.shape[0]
gamma = min(1.0, N / n_total)      # fraction dans ]0, 1]
dataset_short = dataset.take_fraction(gamma)

# ------------------------------------------------------------------
#  Exemple : état "équilibre" autour du temps 16 jours
# ------------------------------------------------------------------
target_time = 16 * 24 * 3600  # secondes
t_np = np.asarray(dataset_short.time)
idx_eq = int(np.argmin(np.abs(t_np - target_time)))  # index le + proche

ta0   = float(dataset_short.d["weaSta_reaWeaTDryBul_y"][idx_eq])
qocc0 = float(dataset_short.d["InternalGainsCon[1]"][idx_eq])
qocr0 = float(dataset_short.d["InternalGainsRad[1]"][idx_eq])
qcd0  = float(dataset_short.d["reaQHeaPumCon_y"][idx_eq])
qsol0 = float(dataset_short.d["weaSta_reaWeaHGloHor_y"][idx_eq])
theta = sim_opti_loaded.model.theta

x0 = RC5_steady_state_sys(ta0, qsol0, qocc0, qocr0, qcd0, theta)

sim = sim_opti_loaded.copy(
    x0=x0,
    time_grid=dataset_short.time[idx_eq:],
    d=dataset_short.d,
    integrator="euler",
    )


class MyMinimalEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        *,
        state_dim=5,          # dimension de l'état de ta simu SIMAX
        step_period=900,         # durée d'un pas en secondes
        past_steps=10,           # nb de pas RL passés à garder
        future_steps=0,          # nb de pas RL futurs (horizon prédictif)
        warmup_steps=50,         # nb de pas de warmup PID (approx)
        render_mode=None,
        tz_min=273.15 + 15.0,
        tz_max=273.15 + 30.0,
        base_setpoint=273.15 + 21.0,
        max_episode_length=100, # nombre max de pas par épisode (None = limité par idx_max)
        render_episodes=False,   # plot auto en fin d'épisode
        excluding_periods=None,  # liste de (start_s, end_s) en secondes
        w_energy: float = DEFAULT_W_ENERGY_EUR,
        w_comfort: float = DEFAULT_W_COMFORT_EUR_PER_KH,
        comfort_huber_k: float = 0.0,
        w_sat: float = DEFAULT_W_SAT_EUR_PER_UNIT_H,
        w_u: float = 0.0,
        w_tz: float = 0.0,
    ):
        super().__init__()

        # ---------- temps / horizons ----------
        self.step_period = float(step_period)
        # Pas de base du dataset (supposé régulier)
        dt_arr = np.diff(np.asarray(dataset_short.time, dtype=float))
        if dt_arr.size == 0:
            raise ValueError("dataset_short.time doit contenir au moins deux points.")
        base_dt = float(dt_arr[0])
        if not np.allclose(dt_arr, base_dt):
            raise ValueError("dataset_short.time doit être régulièrement échantillonné.")
        self.dataset_dt = base_dt
        self.step_n = max(1, int(round(self.step_period / self.dataset_dt)))  # nombre d'intervalles dataset par pas RL

        self.state_dim = int(state_dim)
        # Horizons exprimés directement en pas RL
        self.past_steps = max(0, int(past_steps))
        self.future_steps = max(0, int(future_steps))
        # Longueur d'historique d'état (en pas RL) = horizon passé
        self.state_hist_steps = self.past_steps
        # Warmup suffisamment long pour remplir tout l'historique d'état
        self.warmup_steps = max(int(warmup_steps), self.state_hist_steps + 1)
        self.tz_min = float(tz_min)
        self.tz_max = float(tz_max)
        self.render_episodes = bool(render_episodes)
        # On suppose que max_episode_length est toujours fourni et cohérent
        self.max_episode_length = int(max_episode_length)
        # Reward (euros) : reward = -(wE*energy + wC*comfort + wS*sat)
        self.w_energy = float(w_energy)
        self.w_comfort = float(w_comfort)
        self.comfort_huber_k = float(comfort_huber_k)
        self.w_sat = float(w_sat)
        self.w_u = float(w_u)
        self.w_tz = float(w_tz)
        # Paramètres du modèle (theta) utilisés pour la simu
        self.theta = sim.model.theta
        self.theta_idx = None

        # Longueur du dataset de perturbations (météo, gains internes, etc.)
        self.n = dataset_short.time.shape[0]

        # Cache numpy pour éviter les __getitem__ JAX coûteux dans l'observation

        self._dist_matrix = np.stack(
    [
        np.asarray(dataset_short.d["weaSta_reaWeaTDryBul_y"], dtype=np.float32),
        np.asarray(dataset_short.d["weaSta_reaWeaHGloHor_y"], dtype=np.float32),
        np.asarray(dataset_short.d["InternalGainsCon[1]"], dtype=np.float32),
        np.asarray(dataset_short.d["InternalGainsRad[1]"], dtype=np.float32),
        np.asarray(dataset_short.d["reaQHeaPumCon_y"], dtype=np.float32),
        np.asarray(dataset_short.d["LowerSetp[1]"], dtype=np.float32),
        np.asarray(dataset_short.d["UpperSetp[1]"], dtype=np.float32),
        np.asarray(dataset_short.d["occupancy"], dtype=np.float32),
        np.asarray(dataset_short.d["electricity_price"], dtype=np.float32),

        # temps
        np.asarray(dataset_short.d["week_idx"], dtype=np.float32),
        np.asarray(dataset_short.d["dow_sin"], dtype=np.float32),
        np.asarray(dataset_short.d["dow_cos"], dtype=np.float32),
        np.asarray(dataset_short.d["hour_sin"], dtype=np.float32),
        np.asarray(dataset_short.d["hour_cos"], dtype=np.float32),
    ],
    axis=1)

        self._time_np = np.asarray(dataset_short.time, dtype=np.float64)
        self._excluded_mask = self._build_excluded_mask(excluding_periods)

        # Contraintes sur l'indice de début d'épisode (en pas dataset) :
        # - warmup_steps pas RL avant idx pour le warmup PID
        # - future_steps pas RL après idx pour les observations futures + 1 pas RL simulable
        self.warmup_steps_dataset = self.warmup_steps * self.step_n
        self.idx_min = self.warmup_steps_dataset
        self.idx_max = self.n - 1 - (self.future_steps + 1) * self.step_n
        # Borne supérieure pour l'indice de début, pour qu'un épisode complet
        # de longueur max_episode_length soit possible (supposé cohérent).
        # On veut pouvoir calculer l'observation après le dernier step sans clamp sur idx_max.
        self.idx_max_start = self.idx_max - self.max_episode_length * self.step_n

        # ---------- dimension de l'observation ----------
        # Fenêtre sur les perturbations agrégées par pas RL (moyenne sur step_n pas dataset) :
        # colonnes = [Ta, qsol, qocc, qocr, qcd, lower_sp, upper_sp, occupancy, price,
        #             week_idx, dow_sin, dow_cos, hour_sin, hour_cos]
        # - passé/présent : 9 perturbations + n_time_features
        # - futur        : 6 perturbations (Ta, qsol, qocc, qocr, lower/upper, sans qcd/occup/price) + n_time_features
        # Qcd côté passé/présent est remplacé par la puissance HP simulée.
        self.n_phys_features_past = 9
        self.n_phys_features_future = 6
        self.n_time_features = int(self._dist_matrix.shape[1] - self.n_phys_features_past)
        if self.n_time_features <= 0:
            raise ValueError(
                f"Disturbance matrix invalide: {self._dist_matrix.shape[1]} colonnes (<={self.n_phys_features_past})."
            )
        self.n_features_past = self.n_phys_features_past + self.n_time_features
        if self.n_features_past != self._dist_matrix.shape[1]:
            raise ValueError(
                f"Incohérence features: n_features_past={self.n_features_past} vs dist_matrix={self._dist_matrix.shape[1]}."
            )
        self.n_features_future = self.n_phys_features_future + self.n_time_features
        # Fenêtre passée/future exprimée en pas RL
        past_len = self.past_steps + 1       # t-K, ..., t (K = past_steps)
        fut_len = self.future_steps          # t+1, ..., t+H (H = future_steps)
        dist_dim = self.n_features_past * past_len + self.n_features_future * fut_len

        # Historique des états observés : on ne garde que Tz (moyenne par pas RL)
        tz_hist_dim = (self.state_hist_steps + 1) * 1

        # Historique des setpoints (même longueur que l'historique de Tz)
        sp_hist_dim = (self.state_hist_steps + 1) * 1

        obs_dim = dist_dim + tz_hist_dim + sp_hist_dim

        # Action = setpoint Tz [K]
        self.action_space = spaces.Box(low=self.tz_min, high=self.tz_max, shape=(1,), dtype=np.float32)

        # Observation = [fenêtre perturbations][hist états][hist commandes]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.t_set = float(base_setpoint)

        # Historique de Tz (moyenne par pas RL) et des setpoints
        self.tz_hist = np.full((self.state_hist_steps + 1,), self.t_set, dtype=np.float32)
        self.sp_hist = np.full((self.state_hist_steps + 1,), self.t_set, dtype=np.float32)

        # État / commande courants de la simu
        self.x = np.zeros(self.state_dim, dtype=np.float32)
        self.current_setpoint = self.t_set

        # Index temporel courant dans le dataset
        self.idx = self.idx_min
        self.ep_steps = 0
        # Compteur global de pas (tous épisodes confondus)
        self.total_timesteps = 0

        # Logs épisode (remplis durant step, pas RL)
        self.ep_idx = []
        self.ep_time = []
        self.ep_tz = []
        self.ep_setpoint = []
        self.ep_control = []
        self.ep_rewards = []
        self.ep_indiv_rewards = []
        # Logs fins pour le plot (≈30 s)
        self.ep_idx_30s = []
        self.ep_tz_30s = []
        self.ep_u_30s = []
        self.ep_qc_30s = []
        self.ep_qe_30s = []
        self.ep_php_30s = []
        self._episode_plotted = False
        self._episode_count = 0
        self.warmup = {
            "time": np.array([], dtype=float),
            "tz": np.array([], dtype=float),
            "u": np.array([], dtype=float),
            "qc": np.array([], dtype=float),
            "qe": np.array([], dtype=float),
            "php": np.array([], dtype=float),
            "idx": (0, 0),
        }
        self._sim_qc = np.full((self.n,), np.nan, dtype=np.float32)
        self._sim_qe = np.full((self.n,), np.nan, dtype=np.float32)
        self._sim_php = np.full((self.n,), np.nan, dtype=np.float32)

        # Cache de PID pour éviter de recréer des contrôleurs (même structure = pas de recompilation)
        self._pid_cache: dict[int, Controller_PID] = {}

    def set_rollout_dir(self, rollout_dir: str | Path | None) -> None:
        self.rollout_dir = str(rollout_dir) if rollout_dir is not None else None

    # ---------------------- helpers internes ---------------------- #

    def _build_excluded_mask(self, excluding_periods) -> np.ndarray:
        """Masque des instants du dataset à exclure.

        `excluding_periods`: liste de tuples (start_s, end_s) en secondes, intervalle [start_s, end_s).
        """
        mask = np.zeros((self.n,), dtype=bool)
        if excluding_periods is None:
            return mask
        for period in excluding_periods:
            if period is None:
                continue
            if not isinstance(period, (tuple, list)) or len(period) != 2:
                raise TypeError("excluding_periods doit être une liste de tuples (start_s, end_s).")
            start_t, end_t = map(float, period)
            if end_t < start_t:
                raise ValueError(f"Période invalide (start > end): {period!r}")
            start_idx = int(np.searchsorted(self._time_np, start_t, side="left"))
            end_idx = int(np.searchsorted(self._time_np, end_t, side="left"))
            start_idx = max(0, min(self.n, start_idx))
            end_idx = max(0, min(self.n, end_idx))
            mask[start_idx:end_idx] = True
        return mask

    def _get_band_at(self, idx: int) -> tuple[float, float]:
        """Limites de confort [bas, haut] en Kelvin au pas idx."""
        row = self._dist_matrix[idx]
        return float(row[5]), float(row[6])

    def _aggregate_step_features(self, start_idx: int) -> np.ndarray:
        """Retourne la moyenne des perturbations/temps sur un pas RL (step_n points dataset)."""
        start_idx = int(start_idx)
        end_idx = min(start_idx + self.step_n, self.n)
        rows = self._dist_matrix[start_idx:end_idx]
        # Remplacer Qcd dataset par la puissance HP simulée si dispo
        php_sim = self._sim_php[start_idx:end_idx]
        if np.any(~np.isnan(php_sim)):
            rows = rows.copy()
            mask = ~np.isnan(php_sim)
            rows[mask, 4] = php_sim[mask]
        if rows.size == 0:
            # Ne devrait pas arriver si idx_min/idx_max sont corrects
            return np.zeros((self.n_features_past,), dtype=np.float32)
        return rows.mean(axis=0)

    def _build_disturbance_window(self, idx: int) -> np.ndarray:
        """Fenêtre [passé, présent, futur] sur les perturbations/temps, agrégées par pas RL."""
        # Passé + présent : (past_steps+1) pas RL (t-K, ..., t)
        past_feats = []
        first_start = idx - self.past_steps * self.step_n
        past_len = self.past_steps + 1
        for k in range(past_len):
            start_k = first_start + k * self.step_n
            past_feats.append(self._aggregate_step_features(start_k))
        past = np.stack(past_feats, axis=0)  # (n_past, n_features_past)

        # Futur : n_fut pas RL (t+1, ..., t+H), sans occupancy/price/Qcd mais avec le temps
        future_list = []
        for k in range(1, self.future_steps + 1):
            start_k = idx + k * self.step_n
            full = self._aggregate_step_features(start_k)  # (n_features_past,)
            phys = np.concatenate(
                [full[:4], full[5:7]], axis=0
            )  # Ta, qsol, qocc, qocr, lower, upper (sans Qcd futur)
            time_feats = full[self.n_phys_features_past:]  # week_idx, dow_sin, dow_cos, hour_sin, hour_cos
            future_list.append(np.concatenate([phys, time_feats], axis=0))  # (n_features_future,)

        if future_list:
            future = np.stack(future_list, axis=0)  # (n_fut, n_features_future)
            return np.concatenate(
                [past.reshape(-1), future.reshape(-1)],
                axis=0,
            )
        return past.reshape(-1)

    def _build_observation(self) -> np.ndarray:
        """Concatène : [fenêtre perturbations][hist Tz][hist setpoints]."""
        dist = self._build_disturbance_window(self.idx)
        tz_hist_flat = self.tz_hist.reshape(-1)
        sp_hist_flat = self.sp_hist.reshape(-1)
        obs = np.concatenate([dist, tz_hist_flat, sp_hist_flat], axis=0)
        obs = obs.astype(np.float32)
        expected = int(self.observation_space.shape[0])
        if obs.shape != (expected,):
            raise ValueError(
                f"Observation de taille invalide: got {obs.shape}, expected {(expected,)} "
                f"(past_steps={self.past_steps}, future_steps={self.future_steps}, n_time_features={self.n_time_features})."
            )
        return obs

    def _sample_initial_index(self, rng: np.random.Generator) -> int:
        """Choisit un idx de départ compatible avec warmup + horizons."""
        for _ in range(10_000):
            raw = int(rng.integers(self.idx_min, self.idx_max_start + 1))
            idx = raw - ((raw - self.idx_min) % self.step_n)
            if idx < self.idx_min:
                idx += self.step_n
            idx = min(idx, self.idx_max_start)

            if self._excluded_mask.any():
                warmup_start = idx - self.warmup_steps_dataset
                episode_end = idx + (self.max_episode_length + self.future_steps + 1) * self.step_n
                if warmup_start < 0 or episode_end > self.n:
                    continue
                if self._excluded_mask[warmup_start:episode_end].any():
                    continue

            return idx
        raise ValueError("Aucun index de départ valide (excluding_periods trop contraignant).")

    # ---------- Hooks à remplir avec SIMAX / PID ---------- #
    def _init_state(self, idx: int) -> np.ndarray:
        """Construit un état initial pour la simu au temps `idx`."""
        row = self._dist_matrix[int(idx)]
        ta, qsol, qocc, qocr, qcd = map(float, row[:5])
        x_init = RC5_steady_state_sys(ta, qsol, qocc, qocr, qcd, self.theta)
        return np.asarray(x_init, dtype=np.float32)

    def _make_pid(self, setpoint: float, horizon_len: int) -> Controller_PID:
        sp = jnp.full((horizon_len,), float(setpoint), dtype=jnp.float64)
        cached = self._pid_cache.get(horizon_len)
        if cached is None:
            pid = Controller_PID(k_p=0.6, k_i=0.6 / 800.0, k_d=0.0, n=1, verbose=False, SetPoints=sp)
            self._pid_cache[horizon_len] = pid
            return pid
        # On réutilise la structure existante et on ne change que la consigne
        pid = eqx.tree_at(lambda c: c.SetPoints, cached, sp)
        self._pid_cache[horizon_len] = pid
        return pid

    def _reset_episode_logs(self):
        self.ep_idx.clear()
        self.ep_time.clear()
        self.ep_tz.clear()
        self.ep_setpoint.clear()
        self.ep_control.clear()
        self.ep_rewards.clear()
        self.ep_indiv_rewards.clear()
        self.ep_idx_30s.clear()
        self.ep_tz_30s.clear()
        self.ep_u_30s.clear()
        self.ep_qc_30s.clear()
        self.ep_qe_30s.clear()
        self.ep_php_30s.clear()

    def _log_step(self, idx: int, tz: float, setpoint: float, u: float, reward: float, indiv_reward):
        self.ep_idx.append(idx)
        self.ep_time.append(float(self._time_np[idx]))
        self.ep_tz.append(float(tz))
        self.ep_setpoint.append(float(setpoint))
        self.ep_control.append(float(u))
        self.ep_rewards.append(float(reward))
        # indiv_reward = (comfort_penalty, energy_penalty, sat_penalty)
        self.ep_indiv_rewards.append(
            [
                float(indiv_reward[0]),
                float(indiv_reward[1]),
                float(indiv_reward[2]),
            ]
        )

    def _plot_episode(self):
        if not self.ep_idx:
            return

        plt.ioff()
        fig = plt.figure(figsize=(12, 9), dpi=200)
        # 8 subplots de données (sans panneau texte pour les paramètres)
        axs = fig.subplots(
            8,
            1,
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1, 1, 1, 1, 1, 1, 1]},
        )
        # Grille fine (≈30 s) pour Tz/u si dispo, sinon pas RL
        if self.ep_idx_30s:
            idx_main = np.asarray(self.ep_idx_30s, dtype=int)
            t_days_main = self._time_np[idx_main] / 86400.0
            tz_c = np.asarray(self.ep_tz_30s, dtype=float) - 273.15
            u_arr = np.asarray(self.ep_u_30s, dtype=float)
            qc_arr = np.asarray(self.ep_qc_30s, dtype=float)
            qe_arr = np.asarray(self.ep_qe_30s, dtype=float)
            php_arr = np.asarray(self.ep_php_30s, dtype=float)
        else:
            idx_main = np.asarray(self.ep_idx, dtype=int)
            t_days_main = np.asarray(self.ep_time, dtype=float) / 86400.0
            tz_c = np.asarray(self.ep_tz, dtype=float) - 273.15
            u_arr = np.asarray(self.ep_control, dtype=float)
            qc_arr = np.asarray(self._sim_qc[idx_main], dtype=float)
            qe_arr = np.asarray(self._sim_qe[idx_main], dtype=float)
            php_arr = np.asarray(self._sim_php[idx_main], dtype=float)

        # Conso (kWh) sur l'épisode (sans warmup) à partir de P_hp (W)
        if t_days_main.size >= 2:
            energy_kwh = float(np.trapezoid(np.maximum(php_arr, 0.0) / 1000.0, x=t_days_main * 24.0))
        else:
            energy_kwh = 0.0

        rows = self._dist_matrix[idx_main]
        lower_c = rows[:, 5].astype(float) - 273.15
        upper_c = rows[:, 6].astype(float) - 273.15
        ta = rows[:, 0].astype(float) - 273.15
        qsol = rows[:, 1].astype(float)
        qocc = rows[:, 2].astype(float)
        qocr = rows[:, 3].astype(float)
        occ = rows[:, 7].astype(float)
        prob = occupancy_probability(self._time_np[idx_main])

        # Setpoint / rewards restent au pas RL
        t_days_rl = np.asarray(self.ep_time, dtype=float) / 86400.0
        sp_rl = np.asarray(self.ep_setpoint, dtype=float) - 273.15
        rewards = np.asarray(self.ep_rewards, dtype=float)
        indiv = np.asarray(self.ep_indiv_rewards, dtype=float) if self.ep_indiv_rewards else None

        # Bande warmup (toutes courbes) + données warmup
        warm = self.warmup
        warm_time = warm["time"]
        warm_tz = warm["tz"]
        warm_u = warm["u"]
        warm_qc = warm.get("qc", np.array([], dtype=float))
        warm_qe = warm.get("qe", np.array([], dtype=float))
        warm_php = warm.get("php", np.array([], dtype=float))
        has_warmup = warm_time.size > 0
        warmup_span = None
        if has_warmup:
            warmup_span = (float(warm_time[0]), float(warm_time[-1]))
            i0, i1 = warm["idx"]
            warm_rows = self._dist_matrix[i0 : i1 + 1]
            w_lower_c = warm_rows[:, 5].astype(float) - 273.15
            w_upper_c = warm_rows[:, 6].astype(float) - 273.15
            w_ta = warm_rows[:, 0].astype(float) - 273.15
            w_qsol = warm_rows[:, 1].astype(float)
            w_qocc = warm_rows[:, 2].astype(float)
            w_qocr = warm_rows[:, 3].astype(float)
            w_occ = warm_rows[:, 7].astype(float)
            w_price = warm_rows[:, 8].astype(float)
            w_sp = np.full_like(warm_time, self.t_set - 273.15, dtype=float)
            w_prob = occupancy_probability(self._time_np[i0 : i1 + 1])

        # Légendes avec parts (%) + contributions (€) des sous-récompenses
        reward_label = "reward"
        comfort_label = "comfort"
        energy_label = "energy"
        sat_label = "saturation"
        if indiv is not None and indiv.shape[1] >= 3:
            total_reward = float(rewards.sum())
            total_reward_safe = total_reward if abs(total_reward) > 1e-6 else 1.0
            total_comfort = float(indiv[:, 0].sum())
            total_energy = float(indiv[:, 1].sum())
            total_sat = float(indiv[:, 2].sum())
            comfort_pct = 100.0 * total_comfort / total_reward_safe
            energy_pct = 100.0 * total_energy / total_reward_safe
            sat_pct = 100.0 * total_sat / total_reward_safe
            reward_label = f"reward (sum={total_reward:.2f}€)"
            comfort_label = f"comfort ({comfort_pct:.0f}%, {total_comfort:.2f}€)"
            energy_label = f"energy ({energy_pct:.0f}%, {total_energy:.2f}€)"
            sat_label = f"sat ({sat_pct:.0f}%, {total_sat:.2f}€)"

        if warmup_span:
            # Bande warmup sur tous les subplots temporels
            for ax in axs:
                ax.axvspan(warmup_span[0], warmup_span[1], color="khaki", alpha=0.15, zorder=0)
            axs[0].plot([warmup_span[0]], [np.nan], color="khaki", alpha=0.3, linewidth=6, label="warmup")

        # Bande de confort : uniquement les bornes, en pointillés
        axs[0].plot(t_days_main, lower_c, "--", color="seagreen", linewidth=1, label="Bande confort")
        axs[0].plot(t_days_main, upper_c, "--", color="seagreen", linewidth=1)
        if has_warmup:
            axs[0].plot(warm_time, w_sp, "-", color="gray", linewidth=1, alpha=0.8, label="warmup setpoint")
            axs[0].plot(warm_time, warm_tz, "-", color="darkorange", alpha=0.8, label="warmup Tz")
        axs[0].step(t_days_rl, sp_rl, where="post", color="gray", linewidth=1, label="Setpoint")
        axs[0].plot(t_days_main, tz_c, "-", color="darkorange", linewidth=1, label="Tz")
        axs[0].set_ylabel("Tz / setpoint\n(°C)")

        axs[1].plot(t_days_main, u_arr, "-", color="slateblue", linewidth=1)
        if has_warmup and warm_u.size:
            n = min(warm_u.size, warm_time.size)
            axs[1].plot(warm_time[:n], warm_u[:n], "-", color="slateblue", alpha=0.7)
        axs[1].set_ylabel("Commande\n(-)")

        axs[2].plot(t_days_main, php_arr, "-", color="black", linewidth=1, label="P_hp")
        if has_warmup:
            axs[2].plot(warm_time, warm_php, "-", color="black", linewidth=1, alpha=0.7)
        axs[2].set_ylabel("P_hp (W)")
        axs[2].legend(loc="upper right", fontsize=7)

        axs[3].plot(t_days_rl, rewards, "b", linewidth=1, label=reward_label)
        if indiv is not None:
            axs[3].plot(t_days_rl, indiv[:, 0], "r", linewidth=1, label=comfort_label)
            axs[3].plot(t_days_rl, indiv[:, 1], "g", linewidth=1, label=energy_label)
            if indiv.shape[1] >= 3:
                axs[3].plot(t_days_rl, indiv[:, 2], "m", linewidth=1, label=sat_label)
        axs[3].set_ylabel("Rewards")
        axs[3].legend(loc="lower left", fontsize=7)

        axs[4].plot(t_days_main, ta, color="royalblue", linewidth=1, label="Ta")
        if has_warmup:
            axs[4].plot(warm_time, w_ta, "-", color="royalblue", linewidth=1, alpha=0.7)
        axq3 = axs[4].twinx()
        axq3.plot(t_days_main, qsol, color="gold", linewidth=1, label="Qsol")
        if has_warmup:
            axq3.plot(warm_time, w_qsol, "-", color="gold", linewidth=1, alpha=0.7)
        axs[4].set_ylabel("Ta (°C)")
        axq3.set_ylabel("Qsol (W)")
        axs[4].legend(loc="upper left", fontsize=7)
        axq3.legend(loc="upper right", fontsize=7)

        # Nouveau subplot pour les gains internes
        axs[5].plot(t_days_main, qocc, color="firebrick", linewidth=1, label="Qocc")
        axs[5].plot(t_days_main, qocr, color="darkred", linewidth=1, label="Qocr")
        if has_warmup:
            axs[5].plot(warm_time, w_qocc, "-", color="firebrick", linewidth=1, alpha=0.7)
            axs[5].plot(warm_time, w_qocr, "-", color="darkred", linewidth=1, alpha=0.7)
        axs[5].set_ylabel("Internal gains (W)")
        axs[5].legend(loc="upper right", fontsize=7)

        # Subplot pour l'occupation (0/1) + profil déterministe
        axs[6].step(t_days_main, occ, where="post", color="black", linewidth=1)
        axs[6].plot(t_days_main, prob, color="red", linewidth=0.8)
        if has_warmup:
            axs[6].step(warm_time, w_occ, where="post", color="black", linewidth=1, alpha=0.7)
            axs[6].plot(warm_time, w_prob, color="red", linewidth=0.8, alpha=0.7)
        axs[6].set_ylabel("Occup\n(-)")

        # Subplot minimaliste pour le prix de l'électricité
        price = rows[:, 8].astype(float)

        # Mise en évidence (bandes pastel) des périodes où
        # - le prix est maximal
        # - l'espérance d'occupation (probabilité) est minimale
        t_main = t_days_main

        # Prix maximal
        mask_max = None
        if price.size > 0:
            p_max = price.max()
            mask_max = np.isclose(price, p_max, rtol=1e-5, atol=1e-8)
            if mask_max.any():
                idx_max = np.where(mask_max)[0]
                start = idx_max[0]
                prev = idx_max[0]
                first_span_price = True
                for k in idx_max[1:]:
                    if k == prev + 1:
                        prev = k
                    else:
                        axs[0].axvspan(
                            t_main[start],
                            t_main[prev],
                            color="lightcoral",
                            alpha=0.18,
                            zorder=0,
                            label="Prix max" if first_span_price else None,
                        )
                        first_span_price = False
                        start = prev = k
                    prev = k
                axs[0].axvspan(
                    t_main[start],
                    t_main[prev],
                    color="lightcoral",
                    alpha=0.18,
                    zorder=0,
                    label="Prix max" if first_span_price else None,
                )

        # Occupation minimale basée sur l'espérance (probabilité) : hachures bleues
        mask_min = None
        if prob.size > 0:
            o_min = prob.min()
            mask_min = np.isclose(prob, o_min, rtol=1e-5, atol=1e-8)
            if mask_min.any():
                idx_min = np.where(mask_min)[0]
                start = idx_min[0]
                prev = idx_min[0]
                first_span_occ = True
                for k in idx_min[1:]:
                    if k == prev + 1:
                        prev = k
                    else:
                        axs[0].axvspan(
                            t_main[start],
                            t_main[prev],
                            facecolor="cornflowerblue",
                            edgecolor="cornflowerblue",
                            alpha=0.25,
                            hatch="//",
                            zorder=0,
                            label="Occ. min" if first_span_occ else None,
                        )
                        first_span_occ = False
                        start = prev = k
                    prev = k
                axs[0].axvspan(
                    t_main[start],
                    t_main[prev],
                    facecolor="cornflowerblue",
                    edgecolor="cornflowerblue",
                    alpha=0.25,
                    hatch="//",
                    zorder=0,
                    label="Occ. min" if first_span_occ else None,
                )

        # Légende complète pour le subplot température, incluant bandes
        axs[0].legend(fontsize=7)

        axs[7].plot(t_days_main, price, color="black", linewidth=1)
        if has_warmup:
            axs[7].plot(warm_time, w_price, "-", color="black", linewidth=1, alpha=0.7)
        axs[7].set_ylabel("Prix\n(€/kWh)")
        axs[7].set_xlabel("Temps (jours)")

        title = f"total_timestep={self.total_timesteps}"
        if self.theta_idx is not None:
            title += f" | model={self.theta_idx}"
        title += f" | conso={energy_kwh:.1f} kWh"
        fig.suptitle(title)

        fig.tight_layout()
        rollout_dir_raw = getattr(self, "rollout_dir", None)
        rollout_dir = Path(rollout_dir_raw) if rollout_dir_raw else (ROOT / "tensorboard_logs" / "rollout")
        rollout_dir.mkdir(parents=True, exist_ok=True)
        out_path = rollout_dir / f"episode_{self._episode_count:06d}_t{self.total_timesteps:09d}.png"
        fig.savefig(out_path)
        plt.close(fig)

    def _run_warmup(self, start_idx: int, end_idx: int):
        """Warmup PID entre start_idx et end_idx (inclus),
        et remplissage des historiques d'états et de commandes.
        """
        x_init = self._init_state(start_idx)
        time_slice = dataset_short.time[start_idx : end_idx + 1]
        pid_warm = self._make_pid(self.t_set, len(time_slice))

        # Un run complet fournit états et commandes PID sur la fenêtre de warmup
        _, y_seq, states, controls = sim.run(
            self.theta,
            time_grid=time_slice,
            controller=pid_warm,
            x0=x_init,
        )

        y_arr = np.asarray(y_seq, dtype=np.float32)
        tz_traj = y_arr[:, 0]
        qc_traj = y_arr[:, 1]
        qe_traj = y_arr[:, 2]
        php_traj = qc_traj - np.abs(qe_traj)
        w_time = np.asarray(dataset_short.time[start_idx : end_idx + 1], dtype=float) / 86400.0
        w_tz = np.asarray(tz_traj, dtype=float) - 273.15
        w_qc = np.asarray(qc_traj, dtype=float)
        w_qe = np.asarray(qe_traj, dtype=float)
        w_php = np.asarray(php_traj, dtype=float)
        w_u = np.asarray(controls.get("oveHeaPumY_u", np.zeros_like(qc_traj)), dtype=np.float32)
        self.warmup = {
            "time": w_time,
            "tz": w_tz,
            "u": w_u,
            "qc": w_qc,
            "qe": w_qe,
            "php": w_php,
            "idx": (int(start_idx), int(end_idx)),
        }
        self._sim_qc[start_idx : end_idx + 1] = w_qc[: (end_idx - start_idx + 1)]
        self._sim_qe[start_idx : end_idx + 1] = w_qe[: (end_idx - start_idx + 1)]
        self._sim_php[start_idx : end_idx + 1] = w_php[: (end_idx - start_idx + 1)]
        # Moyenne de Tz sur chaque pas RL du warmup
        tz_means = [
            float(tz_traj[k * self.step_n : min((k + 1) * self.step_n, tz_traj.shape[0])].mean())
            for k in range(self.warmup_steps)
            if k * self.step_n < tz_traj.shape[0]
        ]
        self.tz_hist = np.asarray(tz_means, dtype=np.float32)[-(self.state_hist_steps + 1):]
        # État interne de la simu = dernier état du warmup
        self.x = np.asarray(states[-1], dtype=np.float32)

    # -------------------------- API Gym --------------------------- #
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        # Choix d'un instant de début d'épisode (ou forcé via options)
        opts = options or {}
        if "start_time_s" in opts or "start_idx" in opts:
            if "start_time_s" in opts:
                t0 = float(opts["start_time_s"])
                raw = int(np.searchsorted(self._time_np, t0, side="left"))
            else:
                raw = int(opts["start_idx"])
            idx = raw - ((raw - self.idx_min) % self.step_n)
            if idx < self.idx_min:
                idx += self.step_n
            self.idx = min(idx, self.idx_max_start)
        else:
            self.idx = self._sample_initial_index(rng)
        if self._excluded_mask.any():
            warmup_start = self.idx - self.warmup_steps_dataset
            episode_end = self.idx + (self.max_episode_length + self.future_steps + 1) * self.step_n
            if warmup_start < 0 or episode_end > self.n or self._excluded_mask[warmup_start:episode_end].any():
                raise ValueError("reset: idx incompatible avec excluding_periods pour un épisode complet.")

        # Si on enchaîne les épisodes, on peut afficher celui qui vient de finir
        if self.render_episodes and self.ep_idx and not self._episode_plotted and (self._episode_count % 10 == 0):
            self._plot_episode()
        self._episode_plotted = False
        self._episode_count += 1

        # Warmup PID sur `warmup_steps` pas avant idx (à faire après le plot précédent)
        warmup_start = self.idx - self.warmup_steps_dataset
        self._sim_qc.fill(np.nan)
        self._sim_qe.fill(np.nan)
        self._sim_php.fill(np.nan)
        self._run_warmup(warmup_start, self.idx)

        # reset état interne PID pour la boucle contrôle RL
        self.current_setpoint = self.t_set
        self.sp_hist.fill(self.t_set)
        self._reset_episode_logs()
        self.ep_steps = 0

        obs = self._build_observation()
        return obs, {}

    def step(self, action):
        # --- 0) Index de départ sur la grille dataset ---
        idx_start = self.idx

        # --- 1) Action RL -> setpoint ---
        tz_set = float(
            np.clip(
                np.asarray(action).reshape(-1)[0],
                self.tz_min,
                self.tz_max,
            )
        )
        self.current_setpoint = tz_set

        # Historique des setpoints (une valeur par pas RL)
        self.sp_hist[:-1] = self.sp_hist[1:]
        self.sp_hist[-1] = tz_set

        # --- 2) PID -> commande physique via SIMAX sur un pas complet ---
        # time_slice contient step_n+1 points (step_n intervalles)
        time_slice = dataset_short.time[self.idx : self.idx + self.step_n + 1]

        pid_step = self._make_pid(tz_set, len(time_slice))
        t_grid, y_seq, states, controls = sim.run(
            self.theta,
            time_grid=time_slice,
            controller=pid_step,
            x0=self.x,
        )

        # Passage en numpy (float32 pour la suite)
        y_arr      = np.asarray(y_seq,    dtype=np.float32)
        states_arr = np.asarray(states,   dtype=np.float32)

        u_hist = np.asarray(
            controls.get(
                "oveHeaPumY_u",
                np.zeros((len(time_slice),), dtype=np.float64),
            ),
            dtype=np.float32,
        )
        delta_sat_hist = np.asarray(
            controls.get(
                "delta_sat",
                np.zeros((len(time_slice),), dtype=np.float64),
            ),
            dtype=np.float32,
        )

        x_next = states_arr[-1]

        # --- 3) Logs fins (≈30 s) pour le plot uniquement ---
        # y_arr = [Tz, Qc, Qe, ...]
        tz_traj = y_arr[:, 0]
        qc_traj = y_arr[:, 1]
        qe_traj = y_arr[:, 2]
        php_traj = qc_traj - np.abs(qe_traj)

        # n_inner = nb de sous-pas effectivement loggés
        n_inner = min(len(time_slice) - 1, len(u_hist))

        # Point initial (t = idx) pour garantir la continuité avec le warmup
        self.ep_idx_30s.append(self.idx)
        self.ep_tz_30s.append(float(tz_traj[0]))
        self.ep_u_30s.append(float(u_hist[0] if len(u_hist) else 0.0))
        self.ep_qc_30s.append(float(qc_traj[0]))
        self.ep_qe_30s.append(float(qe_traj[0]))
        self.ep_php_30s.append(float(php_traj[0]))

        # Points internes (t = idx + 1 ... idx + n_inner)
        for k in range(1, n_inner + 1):
            idx_k = self.idx + k
            self.ep_idx_30s.append(idx_k)
            self.ep_tz_30s.append(float(tz_traj[k]))
            self.ep_u_30s.append(float(u_hist[k - 1]))
            self.ep_qc_30s.append(float(qc_traj[k]))
            self.ep_qe_30s.append(float(qe_traj[k]))
            self.ep_php_30s.append(float(php_traj[k]))

        # Commande / saturation sur tout le pas RL
        u_rl = u_hist           # gardé pour debug éventuel
        delta_rl = delta_sat_hist

        # Remplir les signaux simulés sur la grille dataset
        fill_len = min(len(qc_traj), self.n - idx_start)
        end_fill = idx_start + fill_len
        self._sim_qc[idx_start:end_fill]  = qc_traj[:fill_len].astype(np.float32)
        self._sim_qe[idx_start:end_fill]  = qe_traj[:fill_len].astype(np.float32)
        self._sim_php[idx_start:end_fill] = php_traj[:fill_len].astype(np.float32)

        # --- 4) Mise à jour des historiques (Tz moyen par pas RL) ---
        tz_mean = float(tz_traj.mean())
        self.tz_hist[:-1] = self.tz_hist[1:]
        self.tz_hist[-1] = tz_mean

        # État interne de la simu = dernier état du pas RL
        self.x = x_next

        # Avancer dans le dataset pour les perturbations
        self.idx += self.step_n
        self.ep_steps = getattr(self, "ep_steps", 0) + 1
        self.total_timesteps = getattr(self, "total_timesteps", 0) + 1

        # --- 5) Gestion des terminaisons / troncatures ---
        terminated = False
        truncated = False

        if self.idx > self.idx_max:
            self.idx = self.idx_max
            #terminated = True
            truncated = True

        if self.ep_steps >= self.max_episode_length:
            #terminated = True
            truncated = True

        # Observation à l’instant self.idx
        obs = self._build_observation()
        lower_sp, upper_sp = self._get_band_at(self.idx)

        # --- 6) Calcul de la récompense sur le pas RL ---
        # rows_step : step_n lignes [idx_start ... idx_start+step_n-1]
        rows_step = self._dist_matrix[np.arange(idx_start, idx_start + self.step_n)]
        n_step = rows_step.shape[0]

        # mêmes points que rows_step (step_n points / intervalles)
        t_step = np.asarray(time_slice[:-1], dtype=float)

        occ_seq   = rows_step[:, 7]
        price_seq = rows_step[:, 8]

        # Tz alignée sur les intervalles (on saute le point initial)
        tz_seq = tz_traj[1 : n_step + 1]
        lower_seq = rows_step[:, 5]
        upper_seq = rows_step[:, 6]

        u_seq = u_hist[:n_step] if u_hist.shape[0] >= n_step else None
        reward_np, (comfort_penalty_price, energy_penalty_price, sat_penalty_price) = interval_reward_and_terms(
            t_step_s=t_step,
            tz_seq_k=tz_seq,
            lower_seq_k=lower_seq,
            upper_seq_k=upper_seq,
            occ_seq=occ_seq,
            php_w_seq=php_traj[:n_step],
            price_seq=price_seq,
            u_seq=u_seq,
            delta_sat_seq=delta_rl[:n_step],
            w_energy=self.w_energy,
            w_comfort=self.w_comfort,
            comfort_huber_k=self.comfort_huber_k,
            w_sat=self.w_sat,
            w_u=self.w_u,
            w_tz=self.w_tz,
            xp=np,
        )

        # Récompense totale en euros
        reward = float(reward_np)
        comfort_penalty_price = float(comfort_penalty_price)
        energy_penalty_price = float(energy_penalty_price)
        sat_penalty_price = float(sat_penalty_price)

        tz_curr = float(tz_traj[-1])

        # Log du pas RL
        self._log_step(
            self.idx,
            tz_curr,
            tz_set,
            float(u_hist[-1]),
            reward,
            (comfort_penalty_price, energy_penalty_price, sat_penalty_price),
        )

        info = {
            "idx": self.idx,
            "Tz": tz_curr,
            "setpoint": tz_set,
            "ep_steps": self.ep_steps,
            "lower_band": lower_sp,
            "upper_band": upper_sp,
            "w_energy": self.w_energy,
            "w_comfort": self.w_comfort,
            "comfort_huber_k": self.comfort_huber_k,
            "w_sat": self.w_sat,
        }

        # --- 7) Plot éventuel de l’épisode ---
        if self.render_episodes and (terminated or truncated):
            if not self._episode_plotted and (self._episode_count % 1000 == 0):
                self._plot_episode()
                self._episode_plotted = True

        return obs, reward, terminated, truncated, info




class ResidualActionWrapper(gym.ActionWrapper):
    """
    Wrapper pour du RL résiduel : l'agent choisit un delta autour
    d'une consigne de base, en conservant l'API interne de MyMinimalEnv.
    """

    def __init__(self, env, base_action: float, max_dev: float):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Box)
        self.base_action = float(base_action)
        self.max_dev = float(max_dev)
        self._low = env.action_space.low.astype(np.float32)
        self._high = env.action_space.high.astype(np.float32)

        # L'agent voit directement un delta centré en 0
        self.action_space = spaces.Box(
            low=-self.max_dev,
            high=self.max_dev,
            shape=env.action_space.shape,
            dtype=np.float32,
        )

    def action(self, delta):
        # delta résiduel -> action réelle dans l'espace de MyMinimalEnv
        delta = np.asarray(delta, dtype=np.float32)
        delta = np.clip(delta, -self.max_dev, self.max_dev)
        action_env = self.base_action + delta
        return np.clip(action_env, self._low, self._high)


class NormalizeAction(gym.ActionWrapper):
    """
    L'agent voit des actions dans [-1, 1].
    On les rescales vers [low, high] pour l'env interne.
    """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Box)
        self._low = env.action_space.low.astype(np.float32)
        self._high = env.action_space.high.astype(np.float32)

        # Vue "agent" : actions normalisées
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32,
        )

    def action(self, action):
        # action in [-1, 1] -> [low, high]
        action = np.asarray(action, dtype=np.float32)
        return self._low + (action + 1.0) * 0.5 * (self._high - self._low)




if __name__ == "__main__":
    # avec affichage automatique à la fin de chaque épisode
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import VecNormalize
    
    import torch

    def make_env():
        def _init():
            base_sp = 273.15 + 22.0
            env = MyMinimalEnv(
                step_period=3600,
                past_steps=2*24,        # 24 h de passé
                future_steps=24,      # 24 h de futur
                warmup_steps=3*24,
                base_setpoint=base_sp,
                render_episodes=True,      # important pour que _plot_episode soit appelé
                max_episode_length=24 * 14, # Pas max par episode
            )
            # Politique constante de base (ici ≈21°C en Kelvin) + delta appris
            env = ResidualActionWrapper(env, base_action=base_sp, max_dev=5.0)
            # Optionnel : on renormalise pour SB3 (agent voit [-1, 1])
            env = NormalizeAction(env)
            env = Monitor(env)
            return env

        return _init

    n_envs = 8  # nombre d'environnements parallèles
    venv = DummyVecEnv([make_env() for _ in range(n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Logging minimaliste TensorBoard (SB3 gère les scalaires de base)
    model = PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        learning_rate=2e-4,
        device = 'cpu',
        tensorboard_log="tensorboard_logs",  # dossier où TensorBoard ira lire
    )


    # Init de la tête d'action pour partir de delta ≈ 0
    with torch.no_grad():
        actor_net = model.policy.action_net
        actor_net.weight.fill_(0.0)
        actor_net.bias.fill_(0.0)
        if hasattr(model.policy, "log_std"):
            model.policy.log_std.data.fill_(-2.0)

    model.learn(total_timesteps=10_000_000, tb_log_name="PPO_RC5")
    model.save(f"Pre_ppo_rc5_model_{model._total_timesteps}_steps")
    venv.save("vecnormalize_stats.pkl")  # pour les stats de normalisation

    venv.close()
