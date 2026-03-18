"""Prediction pipeline: given a date, produce ranked point-scoring probabilities.

Flow
----
1. Look up tonight's games in the ``games`` table (state = FUT or LIVE or FINAL).
   If the DB is empty, fall back to the provided ``game_overrides`` list.
2. For every skater in each team's roster (``rosters`` table), pull their
   latest rolling stats from the ``player_rolling_stats`` view.
3. Join with team context (``team_game_stats``) and the opponent's goalie
   rolling stats (``goalie_rolling_stats``).
4. Run the trained PointScoringModel.predict() on the assembled feature rows.
5. Return a ranked list of predictions.

When the DB is empty (e.g. fresh install or sandbox), caller can pass
``game_overrides`` — a list of (home_team, away_team) tuples — and the
predictor will use default feature values for all players in those teams'
rosters, falling back to archetype-level estimates.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nhl_pipeline.model import FEATURE_DEFAULTS, PointScoringModel
from nhl_pipeline.utils import get_logger

log = get_logger(__name__)


@dataclass
class PlayerPrediction:
    player_id: int
    player_name: str
    team_abbr: str
    opponent_abbr: str
    position: str
    home_away: str
    score_prob: float
    features: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        ha = "vs" if self.home_away == "H" else "@"
        return (
            f"{self.player_name:<28s} {self.team_abbr} {ha} {self.opponent_abbr}"
            f"  [{self.position}]  {self.score_prob:.1%}"
        )


class TonightPredictor:
    """Predict point-scoring probability for every skater in tonight's games.

    Parameters
    ----------
    db_path:
        Path to the SQLite database.
    model_path:
        Path to the saved PointScoringModel pickle.
    """

    def __init__(self, db_path: str | Path, model_path: str | Path) -> None:
        self.db_path    = Path(db_path)
        self.model_path = Path(model_path)
        self._model: PointScoringModel | None = None

    # ── Model ─────────────────────────────────────────────────────────────

    def _load_model(self) -> PointScoringModel:
        if self._model is None:
            self._model = PointScoringModel.load(self.model_path)
        return self._model

    # ── DB helpers ────────────────────────────────────────────────────────

    def _con(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _games_for_date(self, game_date: str) -> list[dict]:
        with self._con() as con:
            rows = con.execute(
                """
                SELECT game_id, home_team_abbr, away_team_abbr, game_state
                FROM   games
                WHERE  date = ?
                """,
                (game_date,),
            ).fetchall()
        return [dict(r) for r in rows]

    def _roster_for_team(self, team_abbr: str, season: str = "20252026") -> list[dict]:
        with self._con() as con:
            rows = con.execute(
                """
                SELECT player_id, full_name, position
                FROM   rosters
                WHERE  team_abbr = ? AND season = ? AND position != 'G'
                """,
                (team_abbr, season),
            ).fetchall()
        return [dict(r) for r in rows]

    def _player_rolling(self, player_id: int) -> dict:
        """Latest rolling stats for a player (last row in view)."""
        with self._con() as con:
            row = con.execute(
                """
                SELECT player_points_last5, player_points_last10,
                       player_shots_last5, player_TOI_last5,
                       player_pp_toi_last5,
                       player_last5_games_played, player_last10_games_played,
                       player_es_points_pct_last5
                FROM   player_rolling_stats
                WHERE  player_id = ?
                ORDER  BY game_date DESC
                LIMIT  1
                """,
                (player_id,),
            ).fetchone()
        return dict(row) if row else {}

    def _days_rest(self, player_id: int, game_date: str) -> int:
        """Days since the player's most recent game before game_date."""
        with self._con() as con:
            row = con.execute(
                """
                SELECT MAX(game_date) AS last_game
                FROM   player_game_logs
                WHERE  player_id = ? AND game_date < ?
                """,
                (player_id, game_date),
            ).fetchone()
        if row and row["last_game"]:
            from datetime import datetime
            last = datetime.strptime(row["last_game"], "%Y-%m-%d")
            curr = datetime.strptime(game_date, "%Y-%m-%d")
            return (curr - last).days
        return 2  # default

    def _h2h_history(self, player_id: int, opponent: str, game_date: str) -> float:
        """Average points per game vs this opponent (last 10 meetings)."""
        with self._con() as con:
            row = con.execute(
                """
                SELECT AVG(COALESCE(points, 0)) AS h2h_ppg
                FROM  (
                    SELECT points
                    FROM   player_game_logs
                    WHERE  player_id = ? AND opponent_abbr = ? AND game_date < ?
                    ORDER  BY game_date DESC
                    LIMIT  10
                )
                """,
                (player_id, opponent, game_date),
            ).fetchone()
        return row["h2h_ppg"] if row and row["h2h_ppg"] else 0.3

    def _team_recent_stats(self, team_abbr: str) -> dict:
        """Average team context over last 5 games."""
        with self._con() as con:
            row = con.execute(
                """
                SELECT AVG(shots_for)         AS team_shots_for,
                       AVG(pp_pct)            AS team_pp_pct,
                       AVG(faceoff_win_pct)   AS team_faceoff_win_pct
                FROM  (
                    SELECT shots_for, pp_pct, faceoff_win_pct
                    FROM   team_game_stats
                    WHERE  team_abbr = ?
                    ORDER  BY game_date DESC
                    LIMIT  5
                )
                """,
                (team_abbr,),
            ).fetchone()
        return dict(row) if row else {}

    def _opp_defensive_stats(self, opp_team: str) -> dict:
        """Average opponent defensive stats over last 5 games."""
        with self._con() as con:
            row = con.execute(
                """
                SELECT AVG(pk_pct)        AS opp_pk_pct,
                       AVG(goals_allowed)  AS opp_goals_against_avg,
                       AVG(shots_against)  AS opp_shots_against_avg
                FROM  (
                    SELECT pk_pct, goals_allowed, shots_against
                    FROM   team_game_stats
                    WHERE  team_abbr = ?
                    ORDER  BY game_date DESC
                    LIMIT  5
                )
                """,
                (opp_team,),
            ).fetchone()
        return dict(row) if row else {}

    def _opp_goalie_rolling(self, opp_team: str) -> dict:
        """Most-recent rolling stats for the opponent's starting goalie."""
        with self._con() as con:
            row = con.execute(
                """
                SELECT grs.recent_5g_save_pct  AS opp_goalie_recent_5g_save_pct,
                       grs.recent_5g_gaa       AS opp_goalie_recent_5g_gaa
                FROM   goalie_rolling_stats grs
                WHERE  grs.team_abbr = ?
                ORDER  BY grs.game_date DESC
                LIMIT  1
                """,
                (opp_team,),
            ).fetchone()
        return dict(row) if row else {}

    # ── Core prediction logic ─────────────────────────────────────────────

    def _build_feature_row(
        self,
        player: dict,
        team_abbr: str,
        opp_team: str,
        home_away: str,
        game_date: str,
    ) -> dict[str, Any]:
        rolling     = self._player_rolling(player["player_id"])
        team_ctx    = self._team_recent_stats(team_abbr)
        opp_def_ctx = self._opp_defensive_stats(opp_team)
        goalie_ctx  = self._opp_goalie_rolling(opp_team)

        row: dict[str, Any] = {
            "player_id":    player["player_id"],
            "player_name":  player["full_name"],
            "team_abbr":    team_abbr,
            "opponent_abbr": opp_team,
            "position":     player["position"],
            "home_away":    home_away,
            "game_date":    game_date,
        }

        # Player rolling features — fill from view or use defaults
        for feat, default in [
            ("player_points_last5",       FEATURE_DEFAULTS["player_points_last5"]),
            ("player_points_last10",      FEATURE_DEFAULTS["player_points_last10"]),
            ("player_shots_last5",        FEATURE_DEFAULTS["player_shots_last5"]),
            ("player_TOI_last5",          FEATURE_DEFAULTS["player_TOI_last5"]),
            ("player_pp_toi_last5",       FEATURE_DEFAULTS["player_pp_toi_last5"]),
            ("player_last5_games_played", FEATURE_DEFAULTS["player_last5_games_played"]),
            ("player_last10_games_played",FEATURE_DEFAULTS["player_last10_games_played"]),
            ("player_es_points_pct_last5",FEATURE_DEFAULTS["player_es_points_pct_last5"]),
        ]:
            row[feat] = rolling.get(feat) or default

        # Days rest and head-to-head
        row["days_rest"] = self._days_rest(player["player_id"], game_date)
        row["h2h_points_per_game"] = self._h2h_history(player["player_id"], opp_team, game_date)

        # Team context
        row["team_shots_for"]       = team_ctx.get("team_shots_for")       or FEATURE_DEFAULTS["team_shots_for"]
        row["team_pp_pct"]          = team_ctx.get("team_pp_pct")          or FEATURE_DEFAULTS["team_pp_pct"]
        row["team_faceoff_win_pct"] = team_ctx.get("team_faceoff_win_pct") or FEATURE_DEFAULTS["team_faceoff_win_pct"]

        # Opponent defensive context
        row["opp_pk_pct"] = opp_def_ctx.get("opp_pk_pct") or FEATURE_DEFAULTS["opp_pk_pct"]
        row["opp_goals_against_avg"] = opp_def_ctx.get("opp_goals_against_avg") or FEATURE_DEFAULTS["opp_goals_against_avg"]
        row["opp_shots_against_avg"] = opp_def_ctx.get("opp_shots_against_avg") or FEATURE_DEFAULTS["opp_shots_against_avg"]

        # Opponent goalie
        row["opp_goalie_recent_5g_save_pct"] = (
            goalie_ctx.get("opp_goalie_recent_5g_save_pct") or FEATURE_DEFAULTS["opp_goalie_recent_5g_save_pct"]
        )
        row["opp_goalie_recent_5g_gaa"] = (
            goalie_ctx.get("opp_goalie_recent_5g_gaa") or FEATURE_DEFAULTS["opp_goalie_recent_5g_gaa"]
        )

        return row

    def predict_date(
        self,
        game_date: str,
        game_overrides: list[tuple[str, str]] | None = None,
        season: str = "20252026",
        top_n: int | None = None,
    ) -> list[PlayerPrediction]:
        """Return ranked predictions for all skaters in tonight's games.

        Parameters
        ----------
        game_date:
            ISO date string, e.g. ``"2026-03-16"``.
        game_overrides:
            If the DB has no games for this date, use these (home, away) pairs.
            Useful when schedule hasn't been backfilled yet.
        season:
            Season string for roster lookup.
        top_n:
            If given, return only the top N predictions.
        """
        model = self._load_model()

        # 1. Resolve matchups
        matchups: list[tuple[str, str]] = []  # (home_team, away_team)
        db_games = self._games_for_date(game_date)
        if db_games:
            matchups = [(g["home_team_abbr"], g["away_team_abbr"]) for g in db_games]
            log.info("Found %d games in DB for %s", len(matchups), game_date)
        elif game_overrides:
            matchups = game_overrides
            log.info("Using %d game overrides for %s", len(matchups), game_date)
        else:
            log.warning("No games found for %s and no overrides provided", game_date)
            return []

        # 2. Build feature rows for every skater
        feature_rows: list[dict] = []
        for home_team, away_team in matchups:
            for team, opp, ha in [(home_team, away_team, "H"), (away_team, home_team, "A")]:
                players = self._roster_for_team(team, season=season)
                if not players:
                    log.warning("No roster data for %s — skipping", team)
                    continue
                for player in players:
                    row = self._build_feature_row(player, team, opp, ha, game_date)
                    feature_rows.append(row)

        if not feature_rows:
            log.warning("No feature rows assembled for %s", game_date)
            return []

        # 3. Predict
        probs = model.predict(feature_rows)

        # 4. Build result objects
        results: list[PlayerPrediction] = []
        for row, prob in zip(feature_rows, probs):
            results.append(PlayerPrediction(
                player_id    = row["player_id"],
                player_name  = row["player_name"],
                team_abbr    = row["team_abbr"],
                opponent_abbr= row["opponent_abbr"],
                position     = row["position"],
                home_away    = row["home_away"],
                score_prob   = prob,
                features     = {k: v for k, v in row.items()
                                if k not in ("player_id", "player_name", "team_abbr",
                                             "opponent_abbr", "position", "home_away",
                                             "game_date")},
            ))

        results.sort(key=lambda p: p.score_prob, reverse=True)
        if top_n:
            results = results[:top_n]

        return results

    def predict_from_rows(
        self,
        feature_rows: list[dict],
        top_n: int | None = None,
    ) -> list[PlayerPrediction]:
        """Run predictions directly from pre-built feature rows.

        Used when roster data comes from players_2526 rather than the DB.
        """
        model = self._load_model()
        probs = model.predict(feature_rows)

        results: list[PlayerPrediction] = []
        for row, prob in zip(feature_rows, probs):
            results.append(PlayerPrediction(
                player_id     = row["player_id"],
                player_name   = row["player_name"],
                team_abbr     = row["team_abbr"],
                opponent_abbr = row["opponent_abbr"],
                position      = row["position"],
                home_away     = row["home_away"],
                score_prob    = prob,
                features      = {k: v for k, v in row.items()
                                 if k not in ("player_id", "player_name", "team_abbr",
                                              "opponent_abbr", "position", "home_away",
                                              "game_date")},
            ))

        results.sort(key=lambda p: p.score_prob, reverse=True)
        if top_n:
            results = results[:top_n]
        return results

    def format_table(
        self,
        predictions: list[PlayerPrediction],
        top_n: int = 30,
    ) -> str:
        """Format predictions as a printable ranked table grouped by game."""
        # Build ordered game list preserving first-seen order
        seen_games: list[tuple[str, str]] = []
        game_keys: dict[frozenset, tuple[str, str]] = {}
        for p in predictions[:top_n]:
            key = frozenset([p.team_abbr, p.opponent_abbr])
            if key not in game_keys:
                home = p.team_abbr if p.home_away == "H" else p.opponent_abbr
                away = p.opponent_abbr if p.home_away == "H" else p.team_abbr
                game_keys[key] = (home, away)
                seen_games.append((home, away))

        lines: list[str] = []
        rank = 1
        for home, away in seen_games:
            lines.append(f"\n  {away} @ {home}")
            lines.append(f"  {'─' * 58}")
            lines.append(
                f"  {'#':<4} {'Player':<24} {'Team':<5} {'Pos':<4} {'H/A':<5} {'Prob':>5}"
            )
            lines.append(f"  {'─' * 58}")
            game_key = frozenset([home, away])
            game_preds = [
                p for p in predictions[:top_n]
                if frozenset([p.team_abbr, p.opponent_abbr]) == game_key
            ]
            for p in game_preds:
                ha_str = "HOME" if p.home_away == "H" else "AWAY"
                lines.append(
                    f"  {rank:<4} {p.player_name:<24} {p.team_abbr:<5}"
                    f" {p.position:<4} {ha_str:<5} {p.score_prob:>5.1%}"
                )
                rank += 1
        return "\n".join(lines)
