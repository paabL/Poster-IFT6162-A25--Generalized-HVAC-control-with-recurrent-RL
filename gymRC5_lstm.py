# gymRC5_lstm.py
import numpy as np
from gymnasium import spaces

from gymRC5 import MyMinimalEnv  # <- your existing env (unchanged)


class MyMinimalEnvLSTM(MyMinimalEnv):
    """
    ============================================================
    GOAL
    ============================================================

    Adapt MyMinimalEnv to RecurrentPPO (LSTM) in a minimal way.

    MyMinimalEnv returns a "flat" observation (a large vector),
    which often explicitly contained past/future information.

    With an LSTM, we can simplify:
      - no need to stack a past history in the observation
        (past_steps=0 is often enough) because the LSTM keeps memory in its hidden state.
      - we can *keep* a forecast (H future steps) if we want to help
        the agent anticipate (weather, future setpoints, etc.)

    ============================================================
    OBSERVATION (Dict)
    ============================================================

    We return a Dict with two parts:

      1) now      : vector (features at time t) + (current Tz, current setpoint)
      2) forecast : matrix (H, feat) containing "known" future information

    IMPORTANT (your constraint):
      - We DO NOT include occupancy or electricity_price in the forecast
        (you want the model to "predict"/infer them from time, weather, etc.)
      - We do include time features in the forecast:
        week_idx, dow_sin/cos, hour_sin/cos, ... (deterministic)

    ============================================================
    NOTE ON "KNOWN" FEATURES
    ============================================================

    In your dataset you have:
      - weather (Ta, Qsol)
      - internal gains (Qocc/Qocr)  -> note: this can implicitly contain occupancy
      - comfort bands (Lower/Upper)
      - occupancy, electricity_price
      - cyclic time features

    Here we STRICTLY follow your request:
      - forecast: excludes occupancy + price
      - now: contains the current measurement (including occupancy + price at time t), because the "present is known".
    """

    def __init__(self, *args, past_steps: int = 0, future_steps: int = 24, **kwargs):
        # Let MyMinimalEnv handle everything related to simulation, reward, warmup, etc.
        super().__init__(*args, past_steps=past_steps, future_steps=future_steps, **kwargs)

        # ------------------------------------------------------------
        # now = "full features" at time t + (Tz, setpoint)
        # ------------------------------------------------------------
        # self._aggregate_step_features(idx) returns the mean over 1 RL step:
        # full = [Ta, qsol, qocc, qocr, qcd/php, lower, upper, occ, price, time...]
        #
        # Its size = self.n_features_past
        self.now_dim = int(self.n_features_past + 2)  # + (tz, sp)

        # ------------------------------------------------------------
        # forecast (H, feat): what we provide to the network to anticipate.
        # ------------------------------------------------------------
        # We want "known" future information:
        #   - weather, setpoints, (optionally internal gains if you consider them known)
        #   - time (week_idx, hour_sin/cos, dow_sin/cos, ...)
        # We explicitly exclude:
        #   - occupancy
        #   - electricity_price
        #
        # Simple choice (phys_future):
        #   Ta, qsol, qocc, qocr  (4)   <- if you want to avoid leaking occupancy, remove qocc/qocr too
        #   lower, upper          (2)
        #   + time_feats          (n_time_features)
        self.forecast_phys_dim = 4 + 2
        self.forecast_feat_dim = int(self.forecast_phys_dim + self.n_time_features)

        # Define the Dict observation_space for MultiInputLstmPolicy
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
        Build the Dict observation.

        now:
          - aggregated features at t
          - + tz_hist[-1] (zone temperature averaged over the last RL step)
          - + sp_hist[-1] (setpoint applied over the last RL step)

        forecast:
          - H rows (t+1 ... t+H)
          - each row: [Ta, qsol, qocc, qocr, lower, upper, time_feats...]
          - NO occupancy/price in forecast.
        """
        # ------------------------------------------------------------
        # now
        # ------------------------------------------------------------
        full = self._aggregate_step_features(self.idx).astype(np.float32)
        tz = np.float32(self.tz_hist[-1])  # zone temperature "summarized" at the RL step
        sp = np.float32(self.sp_hist[-1])  # setpoint applied at the RL step
        now = np.concatenate([full, np.array([tz, sp], dtype=np.float32)], axis=0)

        # Safety check shape
        if now.shape != (self.now_dim,):
            raise ValueError(f"Invalid now shape: {now.shape}, expected {(self.now_dim,)}")

        # ------------------------------------------------------------
        # forecast
        # ------------------------------------------------------------
        H = int(self.future_steps)
        forecast = np.zeros((H, self.forecast_feat_dim), dtype=np.float32)

        for k in range(1, H + 1):
            # start_k is expressed in dataset indices (RL step = step_n points)
            start_k = self.idx + k * self.step_n
            fk = self._aggregate_step_features(start_k)  # (n_features_past,)

            # fk indices (in your setup):
            # 0 Ta, 1 qsol, 2 qocc, 3 qocr, 4 qcd/php, 5 lower, 6 upper, 7 occ, 8 price, 9.. time feats

            # "known" future physical features (depending on your choice)
            phys = np.concatenate(
                [
                    fk[0:4],   # Ta, qsol, qocc, qocr
                    fk[5:7],   # lower, upper
                ],
                axis=0,
            )

            # time feats = everything after the 9 physical "past" features
            # (week_idx, dow_sin, dow_cos, hour_sin, hour_cos, ...)
            time_feats = fk[self.n_phys_features_past:]

            # Concatenate => (forecast_feat_dim,)
            fvec = np.concatenate([phys, time_feats], axis=0).astype(np.float32)

            if fvec.shape != (self.forecast_feat_dim,):
                raise ValueError(
                    f"Invalid forecast row shape: {fvec.shape}, expected {(self.forecast_feat_dim,)}"
                )

            forecast[k - 1] = fvec

        return {"now": now, "forecast": forecast}
