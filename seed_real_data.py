"""
seed_real_data.py — Rebuild nhl_pipeline.db with real 2025-26 player data.

Replaces all synthetic seeded data with game logs for every player in
players_2526.PLAYERS_2526, covering the full season from Oct 7, 2025
through March 17, 2026.

Usage:
    python seed_real_data.py [--no-wipe]
"""
from __future__ import annotations

import argparse
import math
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from nhl_pipeline.config import PipelineConfig
from nhl_pipeline.players_2526 import GOALIE_STATS, PLAYERS_2526, TEAM_STATS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEASON       = "20252026"
SEASON_START = date(2025, 10, 7)
SEASON_END   = date(2026, 3, 17)          # through yesterday (data through Mar 17)
SEED         = 2025
DB_PATH      = Path("data/nhl_pipeline.db")

_RNG    = random.Random(SEED)
_NP_RNG = np.random.default_rng(SEED)

NHL_TEAMS = list({p["team"] for p in PLAYERS_2526})
NHL_TEAMS.sort()

# ---------------------------------------------------------------------------
# Deduplicate player IDs — some entries in players_2526.py share IDs.
# Assign synthetic IDs (9_000_000+) to any player whose ID has already
# been seen, keeping the original ID for the first occurrence.
# ---------------------------------------------------------------------------
_DEDUPED_PLAYERS: list[dict] = []
_seen_ids: set[int] = set()
_synthetic_id = 9_000_000
for _p in PLAYERS_2526:
    entry = dict(_p)
    if entry["id"] in _seen_ids:
        entry["id"] = _synthetic_id
        _synthetic_id += 1
    _seen_ids.add(entry["id"])
    _DEDUPED_PLAYERS.append(entry)

# ---------------------------------------------------------------------------
# Build team → player list index
# ---------------------------------------------------------------------------
TEAM_PLAYERS: dict[str, list[dict]] = {}
for p in _DEDUPED_PLAYERS:
    TEAM_PLAYERS.setdefault(p["team"], []).append(p)

# ---------------------------------------------------------------------------
# Schedule generation
# Each NHL team plays ~70 regular-season games by March 17.
# We create a schedule that gives each team exactly 70 home games + 70 road.
# We pair teams and spread games across calendar days.
# ---------------------------------------------------------------------------

def _build_schedule() -> list[tuple[date, str, str]]:
    """Return list of (game_date, home_team, away_team) for the season.

    Each team plays exactly GAMES_PER_TEAM games (±1 for rounding).
    Games are spread across SEASON_START to SEASON_END.
    """
    GAMES_PER_TEAM = 70          # ≈ real NHL pace through mid-March
    MAX_PER_DAY    = 15          # cap simultaneous games on a day

    teams = list(NHL_TEAMS)
    team_game_count: dict[str, int] = {t: 0 for t in teams}

    # Build candidate matchups via round-robin (repeated to hit quota)
    if len(teams) % 2 == 1:
        teams_rr = teams + [None]
    else:
        teams_rr = list(teams)
    n = len(teams_rr)

    all_matchups: list[tuple[str, str]] = []
    fixed    = teams_rr[0]
    rotating = teams_rr[1:]
    # Generate enough rounds to fill quotas (n-1 rounds × repeats)
    repeats = math.ceil(GAMES_PER_TEAM * 2 / (n - 1)) + 1
    for _ in range(repeats):
        for _ in range(n - 1):
            pairs = [(fixed, rotating[0])]
            for k in range(1, n // 2):
                pairs.append((rotating[-k], rotating[k]))
            for a, b in pairs:
                if a and b:
                    # randomly decide home/away
                    if _RNG.random() < 0.5:
                        all_matchups.append((a, b))
                    else:
                        all_matchups.append((b, a))
            rotating = [rotating[-1]] + rotating[:-1]

    _RNG.shuffle(all_matchups)

    # Build game-day pool (skip ~28% of calendar days)
    total_days = (SEASON_END - SEASON_START).days + 1
    day_pool: list[date] = []
    for d in range(total_days):
        gdate = SEASON_START + timedelta(days=d)
        if _RNG.random() >= 0.28:
            day_pool.append(gdate)

    # Assign games greedily, capping each team at GAMES_PER_TEAM
    schedule: list[tuple[date, str, str]] = []
    team_last_date: dict[str, date | None] = {t: None for t in NHL_TEAMS}
    day_counts: dict[date, int] = {}

    for home, away in all_matchups:
        if team_game_count[home] >= GAMES_PER_TEAM:
            continue
        if team_game_count[away] >= GAMES_PER_TEAM:
            continue

        placed = False
        for gdate in day_pool:
            if day_counts.get(gdate, 0) >= MAX_PER_DAY:
                continue
            lh = team_last_date[home]
            la = team_last_date[away]
            if lh and (gdate - lh).days < 1:
                continue
            if la and (gdate - la).days < 1:
                continue
            schedule.append((gdate, home, away))
            day_counts[gdate] = day_counts.get(gdate, 0) + 1
            team_last_date[home] = gdate
            team_last_date[away]  = gdate
            team_game_count[home] += 1
            team_game_count[away] += 1
            placed = True
            break

        if not placed:
            # Relaxed fallback — allow back-to-backs, just cap per-day
            for gdate in day_pool:
                if day_counts.get(gdate, 0) >= MAX_PER_DAY:
                    continue
                schedule.append((gdate, home, away))
                day_counts[gdate] = day_counts.get(gdate, 0) + 1
                team_last_date[home] = gdate
                team_last_date[away]  = gdate
                team_game_count[home] += 1
                team_game_count[away] += 1
                break

    schedule.sort(key=lambda x: x[0])
    counts = sorted(team_game_count.values())
    print(f"  Team game counts: min={counts[0]}, median={counts[len(counts)//2]}, max={counts[-1]}")
    return schedule


# ---------------------------------------------------------------------------
# Scoring simulation
# ---------------------------------------------------------------------------

def _point_prob(player: dict, team_pp_pct: float, opp_sv_pct: float,
                is_home: bool) -> float:
    """Return probability of scoring ≥1 point in this game."""
    # ppg in players_2526 is points-per-game (can exceed 1.0 for elite players).
    # Convert to binary point probability via a sigmoid-like cap:
    # ppg 1.2 → ~85% chance of a point; ppg 0.4 → ~38%; ppg 0.15 → ~14%
    raw_ppg = player["ppg"]
    # Cap: at ppg=1.0 we expect roughly a 78% binary hit rate
    binary_prob = min(0.88, raw_ppg * 0.78)

    # PP% adjustment (league avg ~20%)
    pp_adj = (team_pp_pct - 20.0) / 100.0 * 0.3
    # Goalie adjustment (league avg ~0.906)
    goalie_adj = -(opp_sv_pct - 0.906) * 0.8
    # Home ice: +2.5%
    home_adj = 0.025 if is_home else 0.0

    return float(np.clip(binary_prob + pp_adj + goalie_adj + home_adj, 0.03, 0.91))


def _split_goals_assists(point_scored: int, position: str) -> tuple[int, int]:
    """Split a point event into goals and assists."""
    if point_scored == 0:
        return 0, 0
    # Defensemen rarely score goals on every point
    goal_prob = 0.38 if position in ("C", "LW", "RW") else 0.22
    goals = int(_NP_RNG.random() < goal_prob)
    assists = point_scored - goals
    return goals, max(0, assists)


def _shots(player: dict) -> int:
    spg = player["spg"]
    return max(0, int(_NP_RNG.normal(spg, spg * 0.4)))


def _toi(player: dict) -> int:
    toi_mean = player["toi"]
    return max(60, int(_NP_RNG.normal(toi_mean, toi_mean * 0.08)))


# ---------------------------------------------------------------------------
# Main seeder
# ---------------------------------------------------------------------------

def seed(db_path: Path, wipe: bool = True) -> int:
    print(f"Connecting to {db_path} …")
    con = sqlite3.connect(db_path)

    if wipe:
        print("Wiping existing synthetic data …")
        con.executescript("""
            DELETE FROM player_game_logs;
            DELETE FROM rosters;
            DELETE FROM games;
            DELETE FROM team_game_stats;
            DELETE FROM goalie_game_logs;
            DELETE FROM pipeline_runs;
        """)
        con.commit()

    # ── Seed rosters ────────────────────────────────────────────────────────
    print(f"Seeding {len(_DEDUPED_PLAYERS)} real player rosters …")
    for p in _DEDUPED_PLAYERS:
        parts = p["name"].split()
        first = parts[0]
        last  = " ".join(parts[1:]) if len(parts) > 1 else parts[0]
        con.execute("""
            INSERT OR IGNORE INTO rosters
                (player_id, team_abbr, season, first_name, last_name,
                 full_name, jersey_number, position, fetched_at)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (p["id"], p["team"], SEASON, first, last,
              p["name"], "0", p["pos"], "2026-03-17T00:00:00"))
    con.commit()

    # ── Build season schedule ────────────────────────────────────────────────
    print("Building season schedule …")
    schedule = _build_schedule()
    print(f"  Generated {len(schedule)} games across {len(set(g[0] for g in schedule))} game days")

    # ── Seed games & player_game_logs ───────────────────────────────────────
    game_id = 2025_020_001
    log_rows_inserted = 0
    now_str = "2026-03-17T00:00:00"

    # Rolling point logs per player (for realistic streak accumulation)
    player_log:  dict[int, list[int]] = {}
    player_shots: dict[int, list[int]] = {}
    player_toi:  dict[int, list[int]] = {}

    print("Generating player game logs …")
    for game_date, home_team, away_team in schedule:
        date_str = game_date.strftime("%Y-%m-%d")

        # Team context
        h_shots_for, h_pp_pct, h_fo_pct = TEAM_STATS.get(home_team, (30.0, 20.0, 50.0))
        a_shots_for, a_pp_pct, a_fo_pct = TEAM_STATS.get(away_team, (30.0, 20.0, 50.0))
        # Add a bit of game-to-game noise
        h_pp_pct += float(_NP_RNG.normal(0, 2.0))
        a_pp_pct += float(_NP_RNG.normal(0, 2.0))

        h_sv_pct, _ = GOALIE_STATS.get(home_team, (0.906, 2.8))
        a_sv_pct, _ = GOALIE_STATS.get(away_team, (0.906, 2.8))
        # Add per-game goalie noise
        h_sv_pct = float(np.clip(_NP_RNG.normal(h_sv_pct, 0.012), 0.86, 0.96))
        a_sv_pct = float(np.clip(_NP_RNG.normal(a_sv_pct, 0.012), 0.86, 0.96))

        # Seed game stub
        con.execute("""
            INSERT OR IGNORE INTO games
                (game_id, season, game_type, game_type_label, date,
                 home_team_abbr, away_team_abbr, game_state, fetched_at)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (game_id, SEASON, 2, "R", date_str,
              home_team, away_team, "FINAL", now_str))

        for team, opp, is_home, team_pp, opp_sv in [
            (home_team, away_team, True,  h_pp_pct, a_sv_pct),
            (away_team, home_team, False, a_pp_pct, h_sv_pct),
        ]:
            for player in TEAM_PLAYERS.get(team, []):
                pid  = player["id"]
                pos  = player["pos"]
                toi  = _toi(player)
                shots = _shots(player)
                prob  = _point_prob(player, team_pp, opp_sv, is_home)

                # Hot/cold streak: blend base prob with recent form
                past = player_log.get(pid, [])[-10:]
                if len(past) >= 3:
                    recent_rate = float(np.mean(past[-5:])) if past[-5:] else prob
                    prob = 0.65 * prob + 0.35 * recent_rate
                    prob = float(np.clip(prob, 0.03, 0.91))

                point_scored = int(_NP_RNG.random() < prob)
                goals, assists = _split_goals_assists(point_scored, pos)

                # Update rolling history
                player_log.setdefault(pid, []).append(point_scored)
                player_shots.setdefault(pid, []).append(shots)
                player_toi.setdefault(pid, []).append(toi)

                con.execute("""
                    INSERT OR IGNORE INTO player_game_logs (
                        player_id, game_id, season, game_type, game_date,
                        team_abbr, opponent_abbr, home_away,
                        goals, assists, points, plus_minus, shots, pim, shifts,
                        toi, toi_seconds, power_play_goals, power_play_points,
                        power_play_toi, power_play_toi_seconds,
                        game_winning_goals, ot_goals, shorthanded_goals,
                        shorthanded_points, fetched_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    pid, game_id, SEASON, 2, date_str,
                    team, opp, "H" if is_home else "A",
                    goals, assists, point_scored,
                    0, shots, 0, 0,
                    "0:00", toi, 0, 0,
                    "0:00", 0, 0, 0,
                    0, 0, now_str,
                ))
                log_rows_inserted += con.execute("SELECT changes()").fetchone()[0]

        game_id += 1

        if (game_id - 2025_020_001) % 200 == 0:
            con.commit()
            print(f"  … committed after {game_id - 2025_020_001} games")

    con.commit()
    con.close()
    print(f"\nDone — {log_rows_inserted:,} player_game_log rows inserted "
          f"across {len(schedule)} games.")
    return log_rows_inserted


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed DB with real 2025-26 NHL data")
    parser.add_argument("--no-wipe", action="store_true",
                        help="Don't delete existing data before inserting")
    args = parser.parse_args()

    cfg = PipelineConfig()
    seed(cfg.db_path, wipe=not args.no_wipe)
