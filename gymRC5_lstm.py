# gymRC5_lstm.py
import numpy as np
from gymnasium import spaces

from gymRC5 import MyMinimalEnv  # <- ton env existant (inchangé)


class MyMinimalEnvLSTM(MyMinimalEnv):
    """
    ============================================================
    BUT
    ============================================================

    Adapter MyMinimalEnv pour RecurrentPPO (LSTM) de manière minimale.

    MyMinimalEnv renvoie une observation "plate" (un gros vecteur),
    qui contenait souvent du passé/futur explicitement.

    Avec un LSTM, on peut simplifier :
      - pas besoin d'empiler un historique passé dans l'observation
        (past_steps=0 suffit souvent) car le LSTM mémorise dans son état interne.
      - on peut *garder* un forecast (H pas futurs) si on veut aider
        l'agent à anticiper (météo, consignes futures, etc.)

    ============================================================
    OBSERVATION (Dict)
    ============================================================

    On renvoie un Dict avec deux parties :

      1) now      : vecteur (features au temps t) + (Tz actuel, setpoint actuel)
      2) forecast : matrice (H, feat) contenant des informations futures "connues"

    IMPORTANT (ta contrainte) :
      - On NE MET PAS occupancy ni electricity_price dans la forecast
        (tu veux que le modèle les "prédise"/inférer à partir du temps, météo, etc.)
      - En revanche on inclut les features de temps dans la forecast :
        week_idx, dow_sin/cos, hour_sin/cos, ... (déterministes)

    ============================================================
    NOTE SUR LES FEATURES "connues"
    ============================================================

    Dans ton dataset tu as :
      - météo (Ta, Qsol)
      - gains internes (Qocc/Qocr)  -> attention : ça peut implicitement contenir l'occupation
      - bandes de confort (Lower/Upper)
      - occupancy, electricity_price
      - temps cyclique

    Ici on respecte STRICTEMENT ta demande :
      - forecast : exclut occupancy + price
      - now : contient la mesure actuelle (incluant occupancy + price au temps t), car "présent connu".
    """

    def __init__(self, *args, past_steps: int = 0, future_steps: int = 24, **kwargs):
        # On laisse MyMinimalEnv gérer tout ce qui est simulation, reward, warmup, etc.
        super().__init__(*args, past_steps=past_steps, future_steps=future_steps, **kwargs)

        # ------------------------------------------------------------
        # now = "full features" au temps t + (Tz, setpoint)
        # ------------------------------------------------------------
        # self._aggregate_step_features(idx) renvoie la moyenne sur 1 pas RL :
        # full = [Ta, qsol, qocc, qocr, qcd/php, lower, upper, occ, price, time...]
        #
        # Sa taille = self.n_features_past
        self.now_dim = int(self.n_features_past + 2)  # + (tz, sp)

        # ------------------------------------------------------------
        # forecast (H, feat) : ce qu'on donne au réseau pour anticiper.
        # ------------------------------------------------------------
        # On veut du futur "connu" :
        #   - météo, consignes, (éventuellement gains internes si tu les considères connus)
        #   - temps (week_idx, hour_sin/cos, dow_sin/cos, ...)
        # On exclut explicitement :
        #   - occupancy
        #   - electricity_price
        #
        # Choix simple (phys_future) :
        #   Ta, qsol, qocc, qocr  (4)   <- si tu veux éviter de leak l'occupation, enlève qocc/qocr aussi
        #   lower, upper          (2)
        #   + time_feats          (n_time_features)
        self.forecast_phys_dim = 4 + 2
        self.forecast_feat_dim = int(self.forecast_phys_dim + self.n_time_features)

        # Définition de l'observation_space Dict pour MultiInputLstmPolicy
        self.observation_space = spaces.Dict(
            {
                "now": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.now_dim,),
                    dtype=np.float32,
                ),
                "forecast": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(int(self.future_steps), self.forecast_feat_dim),
                    dtype=np.float32,
                ),
            }
        )

    def _build_observation(self):
        """
        Construit l'observation Dict.

        now:
          - features agrégées à t
          - + tz_hist[-1] (température zone moyenne sur le dernier pas RL)
          - + sp_hist[-1] (setpoint appliqué au dernier pas RL)

        forecast:
          - H lignes (t+1 ... t+H)
          - chaque ligne : [Ta, qsol, qocc, qocr, lower, upper, time_feats...]
          - PAS d'occupancy/price dans forecast.
        """
        # ------------------------------------------------------------
        # now
        # ------------------------------------------------------------
        full = self._aggregate_step_features(self.idx).astype(np.float32)
        tz = np.float32(self.tz_hist[-1])  # température zone "résumée" au pas RL
        sp = np.float32(self.sp_hist[-1])  # consigne appliquée au pas RL
        now = np.concatenate([full, np.array([tz, sp], dtype=np.float32)], axis=0)

        # Safety check shape
        if now.shape != (self.now_dim,):
            raise ValueError(f"now shape invalide: {now.shape}, attendu {(self.now_dim,)}")

        # ------------------------------------------------------------
        # forecast
        # ------------------------------------------------------------
        H = int(self.future_steps)
        forecast = np.zeros((H, self.forecast_feat_dim), dtype=np.float32)

        for k in range(1, H + 1):
            # start_k est exprimé en index dataset (pas RL = step_n points)
            start_k = self.idx + k * self.step_n
            fk = self._aggregate_step_features(start_k)  # (n_features_past,)

            # fk indices (chez toi) :
            # 0 Ta, 1 qsol, 2 qocc, 3 qocr, 4 qcd/php, 5 lower, 6 upper, 7 occ, 8 price, 9.. time feats

            # phys futurs "connus" (selon ton choix)
            phys = np.concatenate(
                [
                    fk[0:4],   # Ta, qsol, qocc, qocr
                    fk[5:7],   # lower, upper
                ],
                axis=0,
            )

            # time feats = tout ce qui vient après les 9 features physiques "past"
            # (week_idx, dow_sin, dow_cos, hour_sin, hour_cos, ...)
            time_feats = fk[self.n_phys_features_past:]

            # On concatène => (forecast_feat_dim,)
            fvec = np.concatenate([phys, time_feats], axis=0).astype(np.float32)

            if fvec.shape != (self.forecast_feat_dim,):
                raise ValueError(
                    f"forecast row shape invalide: {fvec.shape}, attendu {(self.forecast_feat_dim,)}"
                )

            forecast[k - 1] = fvec

        return {"now": now, "forecast": forecast}
