"""Synthetic data seeder for offline training and testing.

Generates realistic NHL player-game rows that match the schema of the
player_point_dataset view, using actual NHL scoring distributions:

  - Top-line forwards (C/LW/RW, TOI > 18 min):  ~55-60% chance of a point
  - Mid-line forwards (TOI 13-18 min):           ~35-45%
  - Bottom-line forwards (TOI < 13 min):         ~15-25%
  - Defensemen, top PP pair:                     ~35-45%
  - Defensemen, bottom pairing:                  ~15-20%

Goalie save-percentage and team shot totals follow observed 2023-25 averages.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from nhl_pipeline.utils import get_logger

log = get_logger(__name__)

# Seed for reproducibility when desired
_RNG = random.Random(2025)
_NP_RNG = np.random.default_rng(2025)


# ── Player archetypes ──────────────────────────────────────────────────────

# (name_prefix, position, toi_mean_s, toi_std_s, shots_mean, shots_std,
#  base_point_prob, pp_pct_bonus, is_pp_player)
_ARCHETYPES = [
    # Top-line C
    ("C1",  "C",  1200, 120, 3.8, 1.2, 0.56, 0.08, True),
    # Top-line wingers
    ("LW1", "LW", 1150, 110, 3.5, 1.1, 0.53, 0.07, True),
    ("RW1", "RW", 1150, 110, 3.4, 1.1, 0.52, 0.07, True),
    # 2nd-line C
    ("C2",  "C",  1020,  90, 2.8, 1.0, 0.42, 0.04, True),
    # 2nd-line wingers
    ("LW2", "LW",  980,  90, 2.5, 0.9, 0.39, 0.04, False),
    ("RW2", "RW",  980,  90, 2.4, 0.9, 0.38, 0.04, False),
    # 3rd-line C
    ("C3",  "C",   840,  80, 1.8, 0.8, 0.28, 0.01, False),
    # 3rd-line wingers
    ("LW3", "LW",  800,  75, 1.6, 0.8, 0.24, 0.01, False),
    ("RW3", "RW",  800,  75, 1.6, 0.8, 0.24, 0.01, False),
    # 4th-line
    ("C4",  "C",   660,  65, 1.0, 0.6, 0.16, 0.0,  False),
    ("LW4", "LW",  640,  65, 0.9, 0.5, 0.14, 0.0,  False),
    ("RW4", "RW",  640,  65, 0.9, 0.5, 0.14, 0.0,  False),
    # Top-pair D
    ("D1",  "D",  1380, 130, 2.6, 1.0, 0.44, 0.06, True),
    ("D2",  "D",  1320, 120, 2.2, 0.9, 0.40, 0.05, True),
    # 2nd-pair D
    ("D3",  "D",  1100, 100, 1.8, 0.8, 0.28, 0.02, False),
    ("D4",  "D",  1060,  95, 1.6, 0.7, 0.25, 0.02, False),
    # 3rd-pair D
    ("D5",  "D",   780,  75, 1.0, 0.5, 0.14, 0.0,  False),
    ("D6",  "D",   760,  70, 0.9, 0.5, 0.13, 0.0,  False),
]

_NHL_TEAMS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET",
    "EDM", "FLA", "LAK", "MDA", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR",
    "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN",
    "VGK", "WPG", "WSH",
]


def _sample_toi(mean: float, std: float) -> int:
    return max(60, int(_NP_RNG.normal(mean, std)))


def _sample_shots(mean: float, std: float) -> int:
    return max(0, int(_NP_RNG.normal(mean, std)))


def _goalie_save_pct() -> float:
    # NHL average ~.906, std ~.015
    return float(np.clip(_NP_RNG.normal(0.906, 0.015), 0.850, 0.960))


def _team_shots() -> int:
    # League average ~30 shots/game, std ~4
    return max(18, int(_NP_RNG.normal(30, 4)))


def _pp_pct() -> float:
    # ~18-22% range
    return float(np.clip(_NP_RNG.normal(20.0, 3.0), 10.0, 35.0))


def _faceoff_pct() -> float:
    return float(np.clip(_NP_RNG.normal(50.0, 3.5), 40.0, 60.0))


def _score_prob(archetype: tuple, team_pp_pct: float, opp_save_pct: float,
                is_home: bool) -> float:
    _, _, _, _, _, _, base, pp_bonus, is_pp = archetype
    prob = base
    # PP percentage effect: league avg 20%, each extra point adds a small bonus
    prob += pp_bonus * (team_pp_pct - 20.0) / 10.0
    # Goalie effect: better goalie → lower scoring rate
    prob -= (opp_save_pct - 0.906) * 2.0
    # Home advantage: ~3% higher scoring rate at home
    if is_home:
        prob += 0.03
    return float(np.clip(prob, 0.02, 0.95))


# ── Row generator ─────────────────────────────────────────────────────────

def generate_rows(
    n_games: int = 500,
    season: str = "20252026",
    game_type: int = 2,
) -> list[dict[str, Any]]:
    """Generate realistic synthetic player-game rows.

    Parameters
    ----------
    n_games:
        Number of games to simulate (each game produces ~36 skater rows).
    season:
        Season string, e.g. "20252026".
    game_type:
        2 = regular season.

    Returns
    -------
    list of dicts matching player_point_dataset schema.
    """
    rows: list[dict[str, Any]] = []
    start = date(2025, 10, 7)
    teams = list(_NHL_TEAMS)

    # Build a small 32-team world where each team has 18 archetype slots
    player_base_id = 8_000_000
    # team → list of (player_id, archetype, player_name)
    roster_map: dict[str, list[tuple[int, tuple, str]]] = {}
    for i, team in enumerate(teams):
        roster_map[team] = []
        for j, arch in enumerate(_ARCHETYPES):
            pid = player_base_id + i * 100 + j
            pname = f"{arch[0]}_{team}"
            roster_map[team].append((pid, arch, pname))

    # Build goalie map: each team has 1 starting goalie
    goalie_ids = {team: player_base_id + i * 100 + 99 for i, team in enumerate(teams)}
    # Rolling game log per player: deque of last 10 point values for rolling stats
    player_log: dict[int, list[int]] = {}
    player_shot_log: dict[int, list[int]] = {}
    player_toi_log: dict[int, list[int]] = {}
    goalie_save_log: dict[int, list[float]] = {}
    goalie_ga_log: dict[int, list[float]] = {}

    game_id = 2025_020_001

    for g in range(n_games):
        game_date = start + timedelta(days=g // 8)   # ~8 games/day schedule
        date_str = game_date.strftime("%Y-%m-%d")

        # Pick two teams
        home_team, away_team = _RNG.sample(teams, 2)

        # Team context
        home_shots  = _team_shots()
        away_shots  = _team_shots()
        home_pp_pct = _pp_pct()
        away_pp_pct = _pp_pct()
        home_fo_pct = _faceoff_pct()
        away_fo_pct = 100.0 - home_fo_pct

        # Goalie context (each team's starter)
        home_goalie_id = goalie_ids[home_team]
        away_goalie_id = goalie_ids[away_team]

        home_svpct = _goalie_save_pct()
        away_svpct = _goalie_save_pct()
        home_ga = round(away_shots * (1 - home_svpct))
        away_ga = round(home_shots * (1 - away_svpct))

        # Update goalie rolling logs
        for gid, svpct, ga in [(home_goalie_id, home_svpct, home_ga),
                                (away_goalie_id, away_svpct, away_ga)]:
            goalie_save_log.setdefault(gid, []).append(svpct)
            goalie_ga_log.setdefault(gid, []).append(ga)

        # Opposing goalie rolling stats (prior to this game)
        def goalie_rolling(gid: int) -> tuple[float, float]:
            saves = goalie_save_log.get(gid, [])
            gas   = goalie_ga_log.get(gid, [])
            prior_saves = saves[:-1][-5:]
            prior_ga    = gas[:-1][-5:]
            sv = float(np.mean(prior_saves)) if prior_saves else 0.906
            ga_val = float(np.mean(prior_ga)) if prior_ga else 2.8
            return sv, ga_val

        home_opp_sv, home_opp_ga = goalie_rolling(away_goalie_id)  # away goalie faces home skaters
        away_opp_sv, away_opp_ga = goalie_rolling(home_goalie_id)

        # Generate skater rows for both teams
        for team, opp_team, is_home, team_shots, team_pp, team_fo, opp_sv, opp_ga in [
            (home_team, away_team, True,  home_shots, home_pp_pct, home_fo_pct, home_opp_sv, home_opp_ga),
            (away_team, home_team, False, away_shots, away_pp_pct, away_fo_pct, away_opp_sv, away_opp_ga),
        ]:
            for pid, arch, pname in roster_map[team]:
                toi = _sample_toi(arch[2], arch[3])
                shots = _sample_shots(arch[4], arch[5])

                # Rolling stats BEFORE this game
                past_pts  = player_log.get(pid, [])[-10:]
                past_shots = player_shot_log.get(pid, [])[-10:]
                past_toi   = player_toi_log.get(pid, [])[-10:]

                pts5  = float(np.mean(past_pts[-5:]))   if past_pts[-5:]  else arch[6] * 0.8
                pts10 = float(np.mean(past_pts))        if past_pts       else arch[6] * 0.8
                sh5   = float(np.mean(past_shots[-5:])) if past_shots[-5:] else arch[4]
                toi5  = float(np.mean(past_toi[-5:]))   if past_toi[-5:]   else arch[2]
                lg5   = min(len(past_pts), 5)
                lg10  = min(len(past_pts), 10)

                # Scoring probability influenced by form + context
                base_prob = _score_prob(arch, team_pp, opp_sv, is_home)
                # Hot/cold streak adjustment (recent form vs. archetype baseline)
                if past_pts:
                    recent_rate = pts5 if lg5 >= 3 else arch[6]
                    base_prob = 0.6 * base_prob + 0.4 * recent_rate

                scored = int(_NP_RNG.random() < base_prob)

                # Update rolling history
                player_log.setdefault(pid, []).append(scored)
                player_shot_log.setdefault(pid, []).append(shots)
                player_toi_log.setdefault(pid, []).append(toi)

                rows.append({
                    "player_id":   pid,
                    "game_id":     game_id,
                    "season":      season,
                    "game_date":   date_str,
                    "team_abbr":   team,
                    "opponent_abbr": opp_team,
                    "home_away":   "H" if is_home else "A",
                    "player_name": pname,
                    "position":    arch[1],
                    "shoots_catches": "R",
                    # Player game stats
                    "goals":   min(scored, 1) if _RNG.random() < 0.4 else 0,
                    "assists": scored - (min(scored, 1) if _RNG.random() < 0.4 else 0),
                    "points":  scored,
                    "shots":   shots,
                    "toi_seconds": toi,
                    # Team context
                    "team_shots_for":       team_shots,
                    "team_pp_pct":          round(team_pp, 2),
                    "team_faceoff_win_pct": round(team_fo, 2),
                    # Opponent goalie context
                    "opp_goalie_recent_5g_save_pct": round(opp_sv, 4),
                    "opp_goalie_recent_5g_gaa":      round(opp_ga, 3),
                    # Player rolling features (BEFORE this game)
                    "player_points_last5":      round(pts5,  3),
                    "player_points_last10":     round(pts10, 3),
                    "player_shots_last5":       round(sh5,   3),
                    "player_TOI_last5":         round(toi5,  1),
                    "player_last5_games_played":  lg5,
                    "player_last10_games_played": lg10,
                    # Target
                    "point_scored": scored,
                })

        game_id += 1

    log.info("Generated %d synthetic skater-game rows from %d games", len(rows), n_games)
    return rows


# ── DB seeder ─────────────────────────────────────────────────────────────

def seed_database(db_path: str | Path, n_games: int = 500) -> int:
    """Insert synthetic data directly into player_game_logs (and stubs for
    team_game_stats / rosters / games) so the full pipeline can be exercised.

    Returns number of player_game_log rows inserted.
    """
    rows = generate_rows(n_games=n_games)
    path = Path(db_path)
    con = sqlite3.connect(path)
    now = "2025-10-07T00:00:00"

    inserted = 0
    for r in rows:
        try:
            con.execute(
                """
                INSERT OR IGNORE INTO player_game_logs (
                    player_id, game_id, season, game_type, game_date,
                    team_abbr, opponent_abbr, home_away,
                    goals, assists, points, plus_minus, shots, pim, shifts,
                    toi, toi_seconds, power_play_goals, power_play_points,
                    power_play_toi, power_play_toi_seconds,
                    game_winning_goals, ot_goals, shorthanded_goals,
                    shorthanded_points, fetched_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    r["player_id"], r["game_id"], r["season"], 2, r["game_date"],
                    r["team_abbr"], r["opponent_abbr"], r["home_away"],
                    r.get("goals", 0), r.get("assists", 0), r["points"],
                    0, r["shots"], 0, 0,
                    "0:00", r["toi_seconds"], 0, 0,
                    "0:00", 0, 0, 0, 0, 0,
                    now,
                ),
            )
            inserted += con.execute("SELECT changes()").fetchone()[0]
        except sqlite3.Error:
            pass

    # Also seed roster stubs so predictor can resolve player names
    players_seen: set[tuple] = set()
    for r in rows:
        key = (r["player_id"], r["team_abbr"])
        if key in players_seen:
            continue
        players_seen.add(key)
        con.execute(
            """
            INSERT OR IGNORE INTO rosters (
                player_id, team_abbr, season, first_name, last_name,
                full_name, jersey_number, position, fetched_at
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                r["player_id"], r["team_abbr"], r["season"],
                r["player_name"].split("_")[0], r["player_name"],
                r["player_name"], "0", r["position"], now,
            ),
        )

    # Seed games stubs
    games_seen: set[int] = set()
    for r in rows:
        gid = r["game_id"]
        if gid in games_seen:
            continue
        games_seen.add(gid)
        con.execute(
            """
            INSERT OR IGNORE INTO games (
                game_id, season, game_type, game_type_label, date,
                home_team_abbr, away_team_abbr, game_state, fetched_at
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                gid, r["season"], 2, "R", r["game_date"],
                r["team_abbr"] if r["home_away"] == "H" else r["opponent_abbr"],
                r["opponent_abbr"] if r["home_away"] == "H" else r["team_abbr"],
                "FINAL", now,
            ),
        )

    con.commit()
    con.close()
    log.info("Seeded DB with %d player_game_log rows", inserted)
    return inserted
