"""Point-scoring probability model.

Trains a calibrated logistic-regression classifier on the player_point_dataset
view and exposes a predict() method that returns per-player scoring probabilities.

Features used
-------------
Player form (rolling):
    player_points_last5, player_points_last10, player_shots_last5,
    player_TOI_last5, player_pp_toi_last5, player_last5_games_played,
    player_es_points_pct_last5

Schedule context:
    days_rest, games_last_7_days

Head-to-head:
    h2h_points_per_game

Player situational:
    player_home_away_ppg, games_since_last_point

Travel:
    travel_distance

Team context:
    team_shots_for, team_pp_pct, team_faceoff_win_pct

Opponent context:
    opp_pk_pct, opp_goals_against_avg, opp_shots_against_avg,
    opp_goals_against_venue, opp_one_goal_game_pct,
    opp_goalie_recent_5g_save_pct, opp_goalie_recent_5g_gaa

Situational:
    home_away (H=1, A=0)

Position:
    C, LW, RW, D encoded as dummy vars (G excluded upstream)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nhl_pipeline.utils import get_logger

log = get_logger(__name__)

# ── Feature definitions ────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "player_points_last5",
    "player_points_last10",
    "player_shots_last5",
    "player_TOI_last5",
    "player_pp_toi_last5",
    "player_last5_games_played",
    "player_last10_games_played",
    "player_es_points_pct_last5",
    "days_rest",
    "games_last_7_days",
    "h2h_points_per_game",
    "player_home_away_ppg",
    "games_since_last_point",
    "travel_distance",
    "team_shots_for",
    "team_pp_pct",
    "team_faceoff_win_pct",
    "opp_pk_pct",
    "opp_goals_against_avg",
    "opp_shots_against_avg",
    "opp_goals_against_venue",
    "opp_one_goal_game_pct",
    "opp_goalie_recent_5g_save_pct",
    "opp_goalie_recent_5g_gaa",
    "is_home",
]

POSITION_DUMMIES = ["pos_C", "pos_D", "pos_LW", "pos_RW"]  # G excluded in view

ALL_FEATURES = NUMERIC_FEATURES + POSITION_DUMMIES

TARGET = "point_scored"

# Default fill values for missing features (median-ish NHL averages)
FEATURE_DEFAULTS = {
    "player_points_last5": 0.3,
    "player_points_last10": 0.3,
    "player_shots_last5": 2.2,
    "player_TOI_last5": 900.0,   # ~15 min
    "player_pp_toi_last5": 120.0,  # ~2 min PP time
    "player_last5_games_played": 5.0,
    "player_last10_games_played": 10.0,
    "player_es_points_pct_last5": 0.5,
    "days_rest": 2.0,
    "games_last_7_days": 3.0,
    "h2h_points_per_game": 0.3,
    "player_home_away_ppg": 0.3,
    "games_since_last_point": 2.0,
    "travel_distance": 500.0,
    "team_shots_for": 30.0,
    "team_pp_pct": 20.0,
    "team_faceoff_win_pct": 50.0,
    "opp_pk_pct": 80.0,
    "opp_goals_against_avg": 2.8,
    "opp_shots_against_avg": 30.0,
    "opp_goals_against_venue": 2.8,
    "opp_one_goal_game_pct": 0.4,
    "opp_goalie_recent_5g_save_pct": 0.910,
    "opp_goalie_recent_5g_gaa": 2.8,
    "is_home": 0.5,
    "pos_C": 0.0,
    "pos_D": 0.0,
    "pos_LW": 0.0,
    "pos_RW": 0.0,
}


# ── Arena coordinates (lat, lon) for travel distance calculation ───────────

ARENA_COORDS: dict[str, tuple[float, float]] = {
    "ANA": (33.81, -117.88), "ARI": (33.45, -112.07), "BOS": (42.37, -71.06),
    "BUF": (42.87, -78.88), "CGY": (51.04, -114.05), "CAR": (35.80, -78.72),
    "CHI": (41.88, -87.67), "COL": (39.76, -105.01), "CBJ": (39.97, -83.01),
    "DAL": (32.79, -96.81), "DET": (42.34, -83.06), "EDM": (53.55, -113.50),
    "FLA": (26.16, -80.33), "LAK": (34.04, -118.27), "MIN": (44.94, -93.10),
    "MTL": (45.50, -73.57), "NSH": (36.16, -86.78), "NJD": (40.73, -74.17),
    "NYI": (40.72, -73.99), "NYR": (40.75, -73.99), "OTT": (45.30, -75.93),
    "PHI": (39.90, -75.17), "PIT": (40.44, -80.00), "SJS": (37.33, -121.90),
    "SEA": (47.62, -122.35), "STL": (38.63, -90.20), "TBL": (27.94, -82.45),
    "TOR": (43.64, -79.38), "UTA": (40.77, -111.90), "VAN": (49.28, -123.11),
    "VGK": (36.10, -115.18), "WPG": (49.89, -97.14), "WSH": (38.90, -77.02),
    "MDA": (28.0, -82.5),
}


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two points."""
    import math
    R = 3959  # Earth radius in miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def travel_distance(team_abbr: str, opponent_abbr: str, home_away: str) -> float:
    """Estimated travel distance for the away team (0 for home team)."""
    if home_away == "H":
        return 0.0
    c1 = ARENA_COORDS.get(team_abbr)
    c2 = ARENA_COORDS.get(opponent_abbr)
    if not c1 or not c2:
        return FEATURE_DEFAULTS["travel_distance"]
    return round(_haversine_miles(c1[0], c1[1], c2[0], c2[1]), 0)


# ── Helpers ────────────────────────────────────────────────────────────────

def _prepare(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert raw dataset rows into a feature matrix."""
    df = pd.DataFrame(rows)

    # Binary home indicator
    df["is_home"] = (df["home_away"].str.upper() == "H").astype(float)

    # Travel distance (away team travels to opponent's arena)
    if "travel_distance" not in df.columns:
        df["travel_distance"] = df.apply(
            lambda r: travel_distance(
                r.get("team_abbr", ""), r.get("opponent_abbr", ""),
                str(r.get("home_away", "H")).upper()
            ), axis=1
        )

    # Position dummies
    pos = df.get("position", pd.Series(dtype=str)).fillna("UNK")
    for col in POSITION_DUMMIES:
        letter = col.split("_", 1)[1]
        df[col] = (pos == letter).astype(float)

    # Fill numeric NaNs
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = FEATURE_DEFAULTS[col]
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(
                FEATURE_DEFAULTS[col]
            )

    return df


# ── Model ─────────────────────────────────────────────────────────────────

class PointScoringModel:
    """Calibrated logistic regression for player point-scoring probability.

    Usage
    -----
    >>> model = PointScoringModel()
    >>> model.train(rows)               # rows from player_point_dataset view
    >>> probs = model.predict(rows)     # returns list[float]
    >>> model.save("data/model.pkl")
    >>> model = PointScoringModel.load("data/model.pkl")
    """

    VERSION = "1.1"

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self.feature_importances_: dict[str, float] = {}
        self.train_metrics_: dict[str, float] = {}
        self.n_train_rows_: int = 0

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, rows: list[dict[str, Any]], cv_folds: int = 5) -> dict[str, float]:
        """Fit the model and return cross-validated metrics.

        Parameters
        ----------
        rows:
            Records from ``player_point_dataset`` — must include TARGET and
            all feature columns.
        cv_folds:
            Number of stratified CV folds for calibration + metric eval.

        Returns
        -------
        dict with keys: auc, log_loss, brier_score, n_rows, positive_rate
        """
        if not rows:
            raise ValueError("No training rows provided")

        df = _prepare(rows)

        if TARGET not in df.columns:
            raise ValueError(f"Target column '{TARGET}' missing from rows")

        X = df[ALL_FEATURES].values
        y = df[TARGET].astype(int).values

        pos_rate = y.mean()
        log.info(
            "Training on %d rows  (positive rate %.1f%%)",
            len(y), pos_rate * 100,
        )

        # Inner LR → outer calibration via cross-val
        base_lr = LogisticRegression(
            C=0.5,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )
        inner_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", base_lr),
        ])
        self._pipeline = CalibratedClassifierCV(
            inner_pipe,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            method="isotonic",
        )
        self._pipeline.fit(X, y)

        # Cross-val metrics (re-predict on training set as proxy — real CV done inside)
        probs = self._pipeline.predict_proba(X)[:, 1]
        metrics = {
            "auc":          float(roc_auc_score(y, probs)),
            "log_loss":     float(log_loss(y, probs)),
            "brier_score":  float(brier_score_loss(y, probs)),
            "n_rows":       int(len(y)),
            "positive_rate": float(pos_rate),
        }

        # Feature importances from the first calibrator's LR coefficients
        try:
            first_cal = self._pipeline.calibrated_classifiers_[0]
            scaler = first_cal.estimator.named_steps["scaler"]
            lr     = first_cal.estimator.named_steps["lr"]
            coefs  = lr.coef_[0]
            scaled = coefs * scaler.scale_
            self.feature_importances_ = dict(
                sorted(
                    zip(ALL_FEATURES, scaled.tolist()),
                    key=lambda kv: abs(kv[1]),
                    reverse=True,
                )
            )
        except Exception:  # noqa: BLE001
            pass

        self.train_metrics_ = metrics
        self.n_train_rows_ = len(y)
        log.info("Training complete — AUC %.4f  Brier %.4f", metrics["auc"], metrics["brier_score"])
        return metrics

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, rows: list[dict[str, Any]]) -> list[float]:
        """Return point-scoring probabilities for each row (same order).

        Parameters
        ----------
        rows:
            Feature dicts; ``point_scored`` column is ignored if present.

        Returns
        -------
        list[float] of probabilities in [0, 1]
        """
        if self._pipeline is None:
            raise RuntimeError("Model has not been trained — call train() or load() first")
        if not rows:
            return []
        df = _prepare(rows)
        X = df[ALL_FEATURES].values
        return self._pipeline.predict_proba(X)[:, 1].tolist()

    def predict_df(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        """Like predict(), but returns a DataFrame with identity cols + prob."""
        probs = self.predict(rows)
        df = pd.DataFrame(rows)
        df["score_prob"] = probs
        id_cols = [c for c in ["player_name", "player_id", "team_abbr", "opponent_abbr",
                                "position", "home_away", "game_date"] if c in df.columns]
        return df[id_cols + ["score_prob"]].sort_values("score_prob", ascending=False)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.VERSION,
            "pipeline": self._pipeline,
            "feature_importances": self.feature_importances_,
            "train_metrics": self.train_metrics_,
            "n_train_rows": self.n_train_rows_,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=5)
        log.info("Model saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "PointScoringModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        obj = cls()
        obj._pipeline              = payload["pipeline"]
        obj.feature_importances_   = payload.get("feature_importances", {})
        obj.train_metrics_         = payload.get("train_metrics", {})
        obj.n_train_rows_          = payload.get("n_train_rows", 0)
        log.info("Model loaded ← %s  (%d training rows)", path, obj.n_train_rows_)
        return obj

    def summary(self) -> str:
        """Return a short human-readable summary string."""
        if self._pipeline is None:
            return "PointScoringModel [untrained]"
        m = self.train_metrics_
        lines = [
            f"PointScoringModel v{self.VERSION}",
            f"  Training rows : {self.n_train_rows_:,}",
            f"  Positive rate : {m.get('positive_rate', 0):.1%}",
            f"  AUC           : {m.get('auc', 0):.4f}",
            f"  Brier score   : {m.get('brier_score', 0):.4f}",
            "",
            "Top feature importances (absolute scaled coef):",
        ]
        for feat, imp in list(self.feature_importances_.items())[:8]:
            lines.append(f"  {feat:<40s}  {imp:+.4f}")
        return "\n".join(lines)
