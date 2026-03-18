"""SQLite storage layer — schedules, rosters, game logs, team stats, and views."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from nhl_pipeline.models import (
    DailySchedule,
    Game,
    GoalieGameLog,
    Player,
    PlayerGameLog,
    TeamGameStats,
    TeamRoster,
)
from nhl_pipeline.utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Schema — tables
# ---------------------------------------------------------------------------

_DDL_TABLES = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS games (
    game_id         INTEGER PRIMARY KEY,
    season          TEXT    NOT NULL,
    game_type       INTEGER NOT NULL,
    game_type_label TEXT    NOT NULL,
    date            TEXT    NOT NULL,
    start_time_utc  TEXT,
    venue           TEXT,
    home_team_id    INTEGER,
    home_team_abbr  TEXT,
    home_team_name  TEXT,
    home_score      INTEGER,
    away_team_id    INTEGER,
    away_team_abbr  TEXT,
    away_team_name  TEXT,
    away_score      INTEGER,
    game_state      TEXT,
    period          INTEGER,
    fetched_at      TEXT,
    UNIQUE(game_id)
);

CREATE INDEX IF NOT EXISTS idx_games_date   ON games(date);
CREATE INDEX IF NOT EXISTS idx_games_season ON games(season);
CREATE INDEX IF NOT EXISTS idx_games_home   ON games(home_team_abbr);
CREATE INDEX IF NOT EXISTS idx_games_away   ON games(away_team_abbr);

CREATE TABLE IF NOT EXISTS rosters (
    player_id      INTEGER,
    team_abbr      TEXT    NOT NULL,
    season         TEXT    NOT NULL,
    first_name     TEXT,
    last_name      TEXT,
    full_name      TEXT,
    jersey_number  TEXT,
    position       TEXT,
    shoots_catches TEXT,
    height_inches  INTEGER,
    weight_lbs     INTEGER,
    birth_date     TEXT,
    birth_city     TEXT,
    birth_country  TEXT,
    headshot_url   TEXT,
    fetched_at     TEXT,
    PRIMARY KEY (player_id, team_abbr, season)
);

CREATE INDEX IF NOT EXISTS idx_rosters_team   ON rosters(team_abbr, season);
CREATE INDEX IF NOT EXISTS idx_rosters_player ON rosters(player_id);

CREATE TABLE IF NOT EXISTS player_game_logs (
    player_id              INTEGER NOT NULL,
    game_id                INTEGER NOT NULL,
    season                 TEXT    NOT NULL,
    game_type              INTEGER NOT NULL,
    game_date              TEXT,
    team_abbr              TEXT,
    opponent_abbr          TEXT,
    home_away              TEXT,
    goals                  INTEGER,
    assists                INTEGER,
    points                 INTEGER,
    plus_minus             INTEGER,
    shots                  INTEGER,
    pim                    INTEGER,
    shifts                 INTEGER,
    toi                    TEXT,
    toi_seconds            INTEGER,
    power_play_goals       INTEGER,
    power_play_points      INTEGER,
    power_play_toi         TEXT,
    power_play_toi_seconds INTEGER,
    game_winning_goals     INTEGER,
    ot_goals               INTEGER,
    shorthanded_goals      INTEGER,
    shorthanded_points     INTEGER,
    fetched_at             TEXT,
    PRIMARY KEY (player_id, game_id)
);

CREATE INDEX IF NOT EXISTS idx_gamelogs_player ON player_game_logs(player_id, season);
CREATE INDEX IF NOT EXISTS idx_gamelogs_game   ON player_game_logs(game_id);
CREATE INDEX IF NOT EXISTS idx_gamelogs_date   ON player_game_logs(game_date);
CREATE INDEX IF NOT EXISTS idx_gamelogs_team   ON player_game_logs(team_abbr, season);

CREATE TABLE IF NOT EXISTS team_game_stats (
    game_id            INTEGER NOT NULL,
    season             TEXT    NOT NULL,
    game_date          TEXT,
    team_abbr          TEXT    NOT NULL,
    opponent_abbr      TEXT,
    home_away          TEXT,
    goals              INTEGER,
    goals_allowed      INTEGER,
    shots_for          INTEGER,
    shots_against      INTEGER,
    pp_goals           INTEGER,
    pp_opportunities   INTEGER,
    pp_pct             REAL,
    pk_goals_allowed   INTEGER,
    pk_opportunities   INTEGER,
    pk_pct             REAL,
    faceoff_win_pct    REAL,
    hits               INTEGER,
    blocked_shots      INTEGER,
    pim                INTEGER,
    giveaways          INTEGER,
    takeaways          INTEGER,
    fetched_at         TEXT,
    PRIMARY KEY (game_id, team_abbr)
);

CREATE INDEX IF NOT EXISTS idx_tgs_team   ON team_game_stats(team_abbr, season);
CREATE INDEX IF NOT EXISTS idx_tgs_date   ON team_game_stats(game_date);
CREATE INDEX IF NOT EXISTS idx_tgs_game   ON team_game_stats(game_id);

CREATE TABLE IF NOT EXISTS goalie_game_logs (
    player_id      INTEGER NOT NULL,
    game_id        INTEGER NOT NULL,
    season         TEXT    NOT NULL,
    game_type      INTEGER NOT NULL,
    game_date      TEXT,
    team_abbr      TEXT,
    opponent_abbr  TEXT,
    home_away      TEXT,
    is_starter     INTEGER NOT NULL DEFAULT 0,
    decision       TEXT,
    goals_against  INTEGER,
    shots_against  INTEGER,
    saves          INTEGER,
    save_pct       REAL,
    toi            TEXT,
    toi_seconds    INTEGER,
    shutout        INTEGER,
    fetched_at     TEXT,
    PRIMARY KEY (player_id, game_id)
);

CREATE INDEX IF NOT EXISTS idx_goalielogs_player ON goalie_game_logs(player_id, season);
CREATE INDEX IF NOT EXISTS idx_goalielogs_game   ON goalie_game_logs(game_id);
CREATE INDEX IF NOT EXISTS idx_goalielogs_team   ON goalie_game_logs(team_abbr, season);
CREATE INDEX IF NOT EXISTS idx_goalielogs_start  ON goalie_game_logs(game_id, team_abbr, is_starter);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id        TEXT PRIMARY KEY,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    date_fetched  TEXT,
    schedules_ok  INTEGER DEFAULT 0,
    rosters_ok    INTEGER DEFAULT 0,
    errors_json   TEXT,
    warnings_json TEXT,
    success       INTEGER DEFAULT 1
);
"""

# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

# goalie_rolling_stats: per-goalie rolling averages of the 5 games played
# BEFORE the current game (using SQLite window functions, available since 3.25).
_DDL_VIEWS = """
-- player_rolling_stats: per-skater rolling averages of points, shots, and TOI
-- computed over the 5 and 10 games BEFORE the current game (no data leakage).
CREATE VIEW IF NOT EXISTS player_rolling_stats AS
SELECT
    player_id,
    game_id,
    game_date,
    team_abbr,
    points,
    shots,
    toi_seconds,
    -- Rolling averages over up to 5 preceding games
    ROUND(AVG(COALESCE(points, 0)) OVER (
        PARTITION BY player_id
        ORDER BY game_date
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ), 3) AS player_points_last5,
    ROUND(AVG(COALESCE(shots, 0)) OVER (
        PARTITION BY player_id
        ORDER BY game_date
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ), 3) AS player_shots_last5,
    ROUND(AVG(toi_seconds) OVER (
        PARTITION BY player_id
        ORDER BY game_date
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ), 1) AS player_TOI_last5,
    ROUND(AVG(COALESCE(power_play_toi_seconds, 0)) OVER (
        PARTITION BY player_id
        ORDER BY game_date
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ), 1) AS player_pp_toi_last5,
    COUNT(*) OVER (
        PARTITION BY player_id
        ORDER BY game_date
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ) AS player_last5_games_played,
    -- Even-strength points as fraction of total points (last 5 games)
    ROUND(
        CASE WHEN SUM(COALESCE(points, 0)) OVER (
                PARTITION BY player_id ORDER BY game_date
                ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
             ) > 0
             THEN 1.0 * (
                SUM(COALESCE(points, 0)) OVER (
                    PARTITION BY player_id ORDER BY game_date
                    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                ) - SUM(COALESCE(power_play_points, 0)) OVER (
                    PARTITION BY player_id ORDER BY game_date
                    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                )
             ) / SUM(COALESCE(points, 0)) OVER (
                PARTITION BY player_id ORDER BY game_date
                ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
             )
             ELSE 0.5
        END
    , 3) AS player_es_points_pct_last5,
    -- Rolling averages over up to 10 preceding games
    ROUND(AVG(COALESCE(points, 0)) OVER (
        PARTITION BY player_id
        ORDER BY game_date
        ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
    ), 3) AS player_points_last10,
    COUNT(*) OVER (
        PARTITION BY player_id
        ORDER BY game_date
        ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
    ) AS player_last10_games_played
FROM player_game_logs;

CREATE VIEW IF NOT EXISTS goalie_rolling_stats AS
SELECT
    player_id,
    game_id,
    game_date,
    team_abbr,
    save_pct,
    goals_against,
    shots_against,
    toi_seconds,
    -- Rolling stats over (up to) 5 starts preceding this game
    ROUND(AVG(save_pct) OVER (
        PARTITION BY player_id
        ORDER BY game_date
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ), 4) AS recent_5g_save_pct,
    ROUND(AVG(goals_against) OVER (
        PARTITION BY player_id
        ORDER BY game_date
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ), 3) AS recent_5g_gaa,
    COUNT(*) OVER (
        PARTITION BY player_id
        ORDER BY game_date
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ) AS recent_5g_games_played
FROM goalie_game_logs
WHERE is_starter = 1;

-- player_point_dataset: the modelling-ready table.
-- Each row = one skater in one game.
-- Includes team context, opposing-goalie context, and the binary target.
CREATE VIEW IF NOT EXISTS player_point_dataset AS
SELECT
    -- ── Identity ──────────────────────────────────────────────────────────
    pgl.player_id,
    pgl.game_id,
    pgl.season,
    pgl.game_date,
    pgl.team_abbr,
    pgl.opponent_abbr,
    pgl.home_away,
    r.full_name       AS player_name,
    r.position,
    r.shoots_catches,

    -- ── Skater game stats ─────────────────────────────────────────────────
    pgl.goals,
    pgl.assists,
    pgl.points,
    pgl.shots,
    pgl.toi_seconds,
    pgl.pim,
    pgl.plus_minus,
    pgl.shifts,
    pgl.power_play_goals,
    pgl.power_play_points,
    pgl.game_winning_goals,

    -- ── Player's team context ─────────────────────────────────────────────
    tgs.goals              AS team_goals,
    tgs.goals_allowed      AS team_goals_allowed,
    tgs.shots_for          AS team_shots_for,
    tgs.shots_against      AS team_shots_against,
    tgs.pp_goals           AS team_pp_goals,
    tgs.pp_opportunities   AS team_pp_opps,
    tgs.pp_pct             AS team_pp_pct,
    tgs.pk_pct             AS team_pk_pct,
    tgs.faceoff_win_pct    AS team_faceoff_win_pct,

    -- ── Opponent team context ─────────────────────────────────────────────
    opp.goals              AS opp_goals,
    opp.shots_for          AS opp_shots_for,
    opp.pp_pct             AS opp_pp_pct,
    opp.pk_pct             AS opp_pk_pct,
    opp.goals_allowed      AS opp_goals_allowed,
    opp.shots_against      AS opp_shots_against,
    opp.faceoff_win_pct    AS opp_faceoff_win_pct,

    -- ── Opposing starting goalie (this game) ─────────────────────────────
    ggl.player_id          AS opp_goalie_id,
    og_r.full_name         AS opp_goalie_name,
    ggl.save_pct           AS opp_goalie_save_pct,
    ggl.goals_against      AS opp_goalie_goals_against,
    ggl.shots_against      AS opp_goalie_shots_against,
    ggl.decision           AS opp_goalie_decision,

    -- ── Opposing goalie recent form (last 5 starts before this game) ──────
    grs.recent_5g_save_pct   AS opp_goalie_recent_5g_save_pct,
    grs.recent_5g_gaa        AS opp_goalie_recent_5g_gaa,
    grs.recent_5g_games_played AS opp_goalie_recent_5g_games,

    -- ── Player rolling form (last 5 / 10 games before this game) ─────────
    prs.player_points_last5,
    prs.player_points_last10,
    prs.player_shots_last5,
    prs.player_TOI_last5,
    prs.player_pp_toi_last5,
    prs.player_last5_games_played,
    prs.player_last10_games_played,
    prs.player_es_points_pct_last5,

    -- ── Days rest (days since this player's previous game) ───────────────
    CAST(
        JULIANDAY(pgl.game_date) - JULIANDAY(
            (SELECT MAX(prev.game_date)
             FROM   player_game_logs prev
             WHERE  prev.player_id = pgl.player_id
               AND  prev.game_date < pgl.game_date)
        ) AS INTEGER
    ) AS days_rest,

    -- ── Head-to-head history (avg points vs this opponent, last 10 meetings) ──
    (SELECT ROUND(AVG(COALESCE(h2h.points, 0)), 3)
     FROM   player_game_logs h2h
     WHERE  h2h.player_id    = pgl.player_id
       AND  h2h.opponent_abbr = pgl.opponent_abbr
       AND  h2h.game_date    < pgl.game_date
     ORDER  BY h2h.game_date DESC
     LIMIT  10
    ) AS h2h_points_per_game,

    -- ── Player home/away scoring split (avg points in last 10 same-venue games) ──
    (SELECT ROUND(AVG(COALESCE(ha.points, 0)), 3)
     FROM   player_game_logs ha
     WHERE  ha.player_id = pgl.player_id
       AND  ha.home_away = pgl.home_away
       AND  ha.game_date < pgl.game_date
     ORDER  BY ha.game_date DESC
     LIMIT  10
    ) AS player_home_away_ppg,

    -- ── Opponent goals allowed in same venue context (last 5 home or away games) ──
    (SELECT ROUND(AVG(COALESCE(ov.goals_allowed, 0)), 2)
     FROM   (
         SELECT goals_allowed
         FROM   team_game_stats ov2
         WHERE  ov2.team_abbr = pgl.opponent_abbr
           AND  ov2.home_away = CASE WHEN pgl.home_away = 'H' THEN 'A' ELSE 'H' END
           AND  ov2.game_date < pgl.game_date
         ORDER  BY ov2.game_date DESC
         LIMIT  5
     ) ov
    ) AS opp_goals_against_venue,

    -- ── Opponent one-goal game tendency (pct of last 10 games decided by 1 goal) ──
    (SELECT ROUND(1.0 * SUM(CASE WHEN ABS(og.goals - og.goals_allowed) <= 1 THEN 1 ELSE 0 END)
            / COUNT(*), 3)
     FROM   (
         SELECT goals, goals_allowed
         FROM   team_game_stats og2
         WHERE  og2.team_abbr = pgl.opponent_abbr
           AND  og2.game_date < pgl.game_date
         ORDER  BY og2.game_date DESC
         LIMIT  10
     ) og
    ) AS opp_one_goal_game_pct,

    -- ── Schedule fatigue (games played in the 7 days before this game) ────
    (SELECT COUNT(*)
     FROM   player_game_logs sf
     WHERE  sf.player_id = pgl.player_id
       AND  sf.game_date < pgl.game_date
       AND  sf.game_date >= DATE(pgl.game_date, '-7 days')
    ) AS games_last_7_days,

    -- ── Games since last point (cold streak detector) ─────────────────────
    COALESCE(
        (SELECT MIN(streak_cnt)
         FROM   (
             SELECT ROW_NUMBER() OVER (ORDER BY gslp.game_date DESC) AS streak_cnt,
                    gslp.points
             FROM   player_game_logs gslp
             WHERE  gslp.player_id = pgl.player_id
               AND  gslp.game_date < pgl.game_date
             ORDER  BY gslp.game_date DESC
         )
         WHERE points >= 1
        ),
        5
    ) AS games_since_last_point,

    -- ── Binary target ─────────────────────────────────────────────────────
    CASE
        WHEN (COALESCE(pgl.goals, 0) + COALESCE(pgl.assists, 0)) >= 1 THEN 1
        ELSE 0
    END AS point_scored

FROM player_game_logs pgl

-- Player bio from roster
LEFT JOIN rosters r
    ON  pgl.player_id = r.player_id
    AND pgl.season    = r.season

-- Player's own team game stats
LEFT JOIN team_game_stats tgs
    ON  pgl.game_id   = tgs.game_id
    AND pgl.team_abbr = tgs.team_abbr

-- Opponent team game stats
LEFT JOIN team_game_stats opp
    ON  pgl.game_id       = opp.game_id
    AND pgl.opponent_abbr = opp.team_abbr

-- Opposing starting goalie (for this game)
LEFT JOIN goalie_game_logs ggl
    ON  pgl.game_id       = ggl.game_id
    AND pgl.opponent_abbr = ggl.team_abbr
    AND ggl.is_starter    = 1

-- Opposing goalie name
LEFT JOIN rosters og_r
    ON  ggl.player_id = og_r.player_id
    AND ggl.season    = og_r.season

-- Opposing goalie rolling form
LEFT JOIN goalie_rolling_stats grs
    ON  ggl.player_id = grs.player_id
    AND ggl.game_id   = grs.game_id

-- Player's own rolling form
LEFT JOIN player_rolling_stats prs
    ON  pgl.player_id = prs.player_id
    AND pgl.game_id   = prs.game_id

-- Exclude goalies from the dataset (skaters only)
WHERE r.position != 'G'
  AND r.position IS NOT NULL;
"""


class Database:
    """Thin wrapper around an SQLite database for the NHL pipeline."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _init_schema(self) -> None:
        with self._conn() as con:
            con.executescript(_DDL_TABLES)
            con.executescript(_DDL_VIEWS)
        log.debug("Database schema initialised at %s", self.path)

    # ------------------------------------------------------------------
    # Games / Schedules
    # ------------------------------------------------------------------

    def upsert_schedule(self, schedule: DailySchedule) -> int:
        rows = [_game_to_row(g) for g in schedule.games]
        if not rows:
            return 0
        sql = """
            INSERT OR REPLACE INTO games
              (game_id, season, game_type, game_type_label, date,
               start_time_utc, venue,
               home_team_id, home_team_abbr, home_team_name, home_score,
               away_team_id, away_team_abbr, away_team_name, away_score,
               game_state, period, fetched_at)
            VALUES
              (:game_id,:season,:game_type,:game_type_label,:date,
               :start_time_utc,:venue,
               :home_team_id,:home_team_abbr,:home_team_name,:home_score,
               :away_team_id,:away_team_abbr,:away_team_name,:away_score,
               :game_state,:period,:fetched_at)
        """
        with self._conn() as con:
            con.executemany(sql, rows)
        return len(rows)

    def get_games_for_date(self, date: str) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM games WHERE date = ? ORDER BY start_time_utc", (date,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_games_for_team(self, team_abbr: str, season: Optional[str] = None) -> list[dict]:
        sql = "SELECT * FROM games WHERE (home_team_abbr = ? OR away_team_abbr = ?)"
        params: list = [team_abbr, team_abbr]
        if season:
            sql += " AND season = ?"
            params.append(season)
        sql += " ORDER BY date, start_time_utc"
        with self._conn() as con:
            rows = con.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Rosters / Players
    # ------------------------------------------------------------------

    def upsert_roster(self, roster: TeamRoster) -> int:
        rows = [_player_to_row(p, roster.team_abbr, roster.season, roster.fetched_at)
                for p in roster.all_players]
        if not rows:
            return 0
        sql = """
            INSERT OR REPLACE INTO rosters
              (player_id, team_abbr, season,
               first_name, last_name, full_name,
               jersey_number, position, shoots_catches,
               height_inches, weight_lbs,
               birth_date, birth_city, birth_country,
               headshot_url, fetched_at)
            VALUES
              (:player_id,:team_abbr,:season,
               :first_name,:last_name,:full_name,
               :jersey_number,:position,:shoots_catches,
               :height_inches,:weight_lbs,
               :birth_date,:birth_city,:birth_country,
               :headshot_url,:fetched_at)
        """
        with self._conn() as con:
            con.executemany(sql, rows)
        return len(rows)

    def get_roster(self, team_abbr: str, season: str) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM rosters WHERE team_abbr = ? AND season = ?"
                " ORDER BY position, last_name",
                (team_abbr, season),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_player(self, player_id: int) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM rosters WHERE player_id = ? ORDER BY season DESC",
                (player_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Player game logs
    # ------------------------------------------------------------------

    def upsert_game_logs(self, logs: list[PlayerGameLog]) -> int:
        if not logs:
            return 0
        rows = [_game_log_to_row(gl) for gl in logs]
        sql = """
            INSERT OR REPLACE INTO player_game_logs
              (player_id, game_id, season, game_type, game_date,
               team_abbr, opponent_abbr, home_away,
               goals, assists, points, plus_minus,
               shots, pim, shifts,
               toi, toi_seconds,
               power_play_goals, power_play_points,
               power_play_toi, power_play_toi_seconds,
               game_winning_goals, ot_goals,
               shorthanded_goals, shorthanded_points,
               fetched_at)
            VALUES
              (:player_id,:game_id,:season,:game_type,:game_date,
               :team_abbr,:opponent_abbr,:home_away,
               :goals,:assists,:points,:plus_minus,
               :shots,:pim,:shifts,
               :toi,:toi_seconds,
               :power_play_goals,:power_play_points,
               :power_play_toi,:power_play_toi_seconds,
               :game_winning_goals,:ot_goals,
               :shorthanded_goals,:shorthanded_points,
               :fetched_at)
        """
        with self._conn() as con:
            con.executemany(sql, rows)
        return len(rows)

    def get_player_game_logs(
        self,
        player_id: int,
        season: Optional[str] = None,
        game_type: Optional[int] = None,
    ) -> list[dict]:
        sql = "SELECT * FROM player_game_logs WHERE player_id = ?"
        params: list = [player_id]
        if season:
            sql += " AND season = ?"
            params.append(season)
        if game_type is not None:
            sql += " AND game_type = ?"
            params.append(game_type)
        sql += " ORDER BY game_date"
        with self._conn() as con:
            rows = con.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_team_game_logs(
        self, team_abbr: str, season: str, game_type: int = 2
    ) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM player_game_logs"
                " WHERE team_abbr = ? AND season = ? AND game_type = ?"
                " ORDER BY game_date, player_id",
                (team_abbr, season, game_type),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_game_logs_for_date(self, game_date: str) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM player_game_logs"
                " WHERE game_date = ? ORDER BY team_abbr, player_id",
                (game_date,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Team game stats
    # ------------------------------------------------------------------

    def upsert_team_game_stats(self, stats: list[TeamGameStats]) -> int:
        if not stats:
            return 0
        rows = [_team_stats_to_row(s) for s in stats]
        sql = """
            INSERT OR REPLACE INTO team_game_stats
              (game_id, season, game_date, team_abbr, opponent_abbr, home_away,
               goals, goals_allowed, shots_for, shots_against,
               pp_goals, pp_opportunities, pp_pct,
               pk_goals_allowed, pk_opportunities, pk_pct,
               faceoff_win_pct, hits, blocked_shots, pim,
               giveaways, takeaways, fetched_at)
            VALUES
              (:game_id,:season,:game_date,:team_abbr,:opponent_abbr,:home_away,
               :goals,:goals_allowed,:shots_for,:shots_against,
               :pp_goals,:pp_opportunities,:pp_pct,
               :pk_goals_allowed,:pk_opportunities,:pk_pct,
               :faceoff_win_pct,:hits,:blocked_shots,:pim,
               :giveaways,:takeaways,:fetched_at)
        """
        with self._conn() as con:
            con.executemany(sql, rows)
        return len(rows)

    def get_team_game_stats(
        self,
        team_abbr: str,
        season: Optional[str] = None,
    ) -> list[dict]:
        sql = "SELECT * FROM team_game_stats WHERE team_abbr = ?"
        params: list = [team_abbr]
        if season:
            sql += " AND season = ?"
            params.append(season)
        sql += " ORDER BY game_date"
        with self._conn() as con:
            rows = con.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_game_team_stats(self, game_id: int) -> list[dict]:
        """Return both team rows for a single game."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM team_game_stats WHERE game_id = ?", (game_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Goalie game logs
    # ------------------------------------------------------------------

    def upsert_goalie_game_logs(self, logs: list[GoalieGameLog]) -> int:
        if not logs:
            return 0
        rows = [_goalie_log_to_row(g) for g in logs]
        sql = """
            INSERT OR REPLACE INTO goalie_game_logs
              (player_id, game_id, season, game_type, game_date,
               team_abbr, opponent_abbr, home_away,
               is_starter, decision,
               goals_against, shots_against, saves, save_pct,
               toi, toi_seconds, shutout, fetched_at)
            VALUES
              (:player_id,:game_id,:season,:game_type,:game_date,
               :team_abbr,:opponent_abbr,:home_away,
               :is_starter,:decision,
               :goals_against,:shots_against,:saves,:save_pct,
               :toi,:toi_seconds,:shutout,:fetched_at)
        """
        with self._conn() as con:
            con.executemany(sql, rows)
        return len(rows)

    def get_goalie_game_logs(
        self,
        player_id: int,
        season: Optional[str] = None,
        starters_only: bool = False,
    ) -> list[dict]:
        sql = "SELECT * FROM goalie_game_logs WHERE player_id = ?"
        params: list = [player_id]
        if season:
            sql += " AND season = ?"
            params.append(season)
        if starters_only:
            sql += " AND is_starter = 1"
        sql += " ORDER BY game_date"
        with self._conn() as con:
            rows = con.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_goalie_rolling_stats(self, player_id: int) -> list[dict]:
        """Return pre-computed rolling save% and GAA for a goalie (from the view)."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM goalie_rolling_stats WHERE player_id = ?"
                " ORDER BY game_date",
                (player_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_player_rolling_stats(
        self,
        player_id: int,
        season: Optional[str] = None,
    ) -> list[dict]:
        """Return pre-computed rolling points, shots, and TOI for a skater."""
        sql = "SELECT * FROM player_rolling_stats WHERE player_id = ?"
        params: list = [player_id]
        if season:
            sql = (
                "SELECT prs.* FROM player_rolling_stats prs"
                " JOIN player_game_logs pgl"
                "  ON prs.player_id = pgl.player_id AND prs.game_id = pgl.game_id"
                " WHERE prs.player_id = ? AND pgl.season = ?"
                " ORDER BY prs.game_date"
            )
            params.append(season)
        else:
            sql += " ORDER BY game_date"
        with self._conn() as con:
            rows = con.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Modelling dataset
    # ------------------------------------------------------------------

    def get_point_dataset(
        self,
        season: Optional[str] = None,
        game_type: int = 2,
        team_abbr: Optional[str] = None,
    ) -> list[dict]:
        """Query the ``player_point_dataset`` view.

        Each row is one skater in one game with team/goalie context and the
        binary ``point_scored`` target.
        """
        sql = (
            "SELECT * FROM player_point_dataset"
            " WHERE pgl_game_type = ?"
        )
        # The view aliases game_type as pgl.game_type — SQLite passes it through
        # unaliased; fall back to a subquery approach for filtering.
        sql = """
            SELECT ppd.*
            FROM player_point_dataset ppd
            JOIN player_game_logs pgl
              ON ppd.player_id = pgl.player_id AND ppd.game_id = pgl.game_id
            WHERE pgl.game_type = ?
        """
        params: list = [game_type]
        if season:
            sql += " AND pgl.season = ?"
            params.append(season)
        if team_abbr:
            sql += " AND ppd.team_abbr = ?"
            params.append(team_abbr)
        sql += " ORDER BY ppd.game_date, ppd.player_id"
        with self._conn() as con:
            rows = con.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Point streaks
    # ------------------------------------------------------------------

    def get_point_streaks(
        self,
        min_streak: int = 5,
        season: str = "20252026",
        as_of_date: Optional[str] = None,
    ) -> list[dict]:
        """Return skaters currently on a point streak of at least *min_streak* games.

        Each row contains:
          player_id, full_name, team_abbr, position,
          streak_length, last_game_date, streak_start
        ordered by streak_length desc, last_game_date desc.

        Parameters
        ----------
        min_streak  : minimum consecutive games with ≥1 point (default 5)
        season      : season string to query (default "20252026")
        as_of_date  : only consider games on or before this ISO date (default: all)
        """
        date_filter = f"AND pgl.game_date <= '{as_of_date}'" if as_of_date else ""

        sql = f"""
        WITH player_games AS (
            SELECT
                pgl.player_id,
                r.full_name,
                r.team_abbr,
                r.position,
                pgl.game_date,
                COALESCE(pgl.points, 0) AS points,
                ROW_NUMBER() OVER (
                    PARTITION BY pgl.player_id
                    ORDER BY pgl.game_date DESC
                ) AS rn
            FROM player_game_logs pgl
            JOIN rosters r
                ON  pgl.player_id = r.player_id
                AND pgl.season    = r.season
            WHERE pgl.game_type = 2
              AND pgl.season    = ?
              AND r.position   != 'G'
              AND r.position   IS NOT NULL
              {date_filter}
        ),
        first_zero AS (
            SELECT player_id, MIN(rn) AS zero_rn
            FROM   player_games
            WHERE  points = 0
            GROUP BY player_id
        ),
        streak_base AS (
            SELECT
                pg.player_id,
                pg.full_name,
                pg.team_abbr,
                pg.position,
                COALESCE(fz.zero_rn - 1, 9999)              AS streak_cap,
                MAX(pg.rn)                                   AS total_games,
                MAX(CASE WHEN pg.rn = 1 THEN pg.game_date END) AS last_game_date
            FROM  player_games pg
            LEFT JOIN first_zero fz ON pg.player_id = fz.player_id
            GROUP BY pg.player_id, pg.full_name, pg.team_abbr, pg.position, fz.zero_rn
        ),
        with_length AS (
            SELECT
                player_id, full_name, team_abbr, position, last_game_date,
                MIN(streak_cap, total_games) AS streak_length
            FROM streak_base
        ),
        with_start AS (
            SELECT
                wl.player_id, wl.full_name, wl.team_abbr, wl.position,
                wl.last_game_date, wl.streak_length,
                MIN(CASE WHEN pg.rn <= wl.streak_length THEN pg.game_date END) AS streak_start
            FROM with_length wl
            JOIN player_games pg ON wl.player_id = pg.player_id
            GROUP BY wl.player_id, wl.full_name, wl.team_abbr, wl.position,
                     wl.last_game_date, wl.streak_length
        )
        SELECT player_id, full_name, team_abbr, position,
               streak_length, last_game_date, streak_start
        FROM   with_start
        WHERE  streak_length >= ?
        ORDER  BY streak_length DESC, last_game_date DESC
        """
        with self._conn() as con:
            rows = con.execute(sql, (season, min_streak)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Pipeline runs
    # ------------------------------------------------------------------

    def save_run(self, run) -> None:
        sql = """
            INSERT OR REPLACE INTO pipeline_runs
              (run_id, started_at, finished_at, date_fetched,
               schedules_ok, rosters_ok, errors_json, warnings_json, success)
            VALUES (?,?,?,?,?,?,?,?,?)
        """
        with self._conn() as con:
            con.execute(sql, (
                run.run_id, run.started_at, run.finished_at, run.date_fetched,
                run.schedules_ok, run.rosters_ok,
                json.dumps(run.errors), json.dumps(run.warnings),
                int(run.success),
            ))

    def get_run_history(self, limit: int = 20) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Row converters
# ---------------------------------------------------------------------------

def _game_to_row(g: Game) -> dict:
    return {
        "game_id": g.game_id, "season": g.season,
        "game_type": g.game_type, "game_type_label": g.game_type_label,
        "date": g.date, "start_time_utc": g.start_time_utc, "venue": g.venue,
        "home_team_id": g.home_team.id, "home_team_abbr": g.home_team.abbr,
        "home_team_name": g.home_team.name, "home_score": g.home_team.score,
        "away_team_id": g.away_team.id, "away_team_abbr": g.away_team.abbr,
        "away_team_name": g.away_team.name, "away_score": g.away_team.score,
        "game_state": g.game_state, "period": g.period, "fetched_at": g.fetched_at,
    }


def _game_log_to_row(gl: PlayerGameLog) -> dict:
    return {
        "player_id": gl.player_id, "game_id": gl.game_id,
        "season": gl.season, "game_type": gl.game_type,
        "game_date": gl.game_date, "team_abbr": gl.team_abbr,
        "opponent_abbr": gl.opponent_abbr, "home_away": gl.home_away,
        "goals": gl.goals, "assists": gl.assists, "points": gl.points,
        "plus_minus": gl.plus_minus, "shots": gl.shots, "pim": gl.pim,
        "shifts": gl.shifts, "toi": gl.toi, "toi_seconds": gl.toi_seconds,
        "power_play_goals": gl.power_play_goals,
        "power_play_points": gl.power_play_points,
        "power_play_toi": gl.power_play_toi,
        "power_play_toi_seconds": gl.power_play_toi_seconds,
        "game_winning_goals": gl.game_winning_goals,
        "ot_goals": gl.ot_goals,
        "shorthanded_goals": gl.shorthanded_goals,
        "shorthanded_points": gl.shorthanded_points,
        "fetched_at": gl.fetched_at,
    }


def _team_stats_to_row(s: TeamGameStats) -> dict:
    return {
        "game_id": s.game_id, "season": s.season, "game_date": s.game_date,
        "team_abbr": s.team_abbr, "opponent_abbr": s.opponent_abbr,
        "home_away": s.home_away,
        "goals": s.goals, "goals_allowed": s.goals_allowed,
        "shots_for": s.shots_for, "shots_against": s.shots_against,
        "pp_goals": s.pp_goals, "pp_opportunities": s.pp_opportunities,
        "pp_pct": s.pp_pct,
        "pk_goals_allowed": s.pk_goals_allowed,
        "pk_opportunities": s.pk_opportunities, "pk_pct": s.pk_pct,
        "faceoff_win_pct": s.faceoff_win_pct,
        "hits": s.hits, "blocked_shots": s.blocked_shots,
        "pim": s.pim, "giveaways": s.giveaways, "takeaways": s.takeaways,
        "fetched_at": s.fetched_at,
    }


def _goalie_log_to_row(g: GoalieGameLog) -> dict:
    return {
        "player_id": g.player_id, "game_id": g.game_id,
        "season": g.season, "game_type": g.game_type,
        "game_date": g.game_date, "team_abbr": g.team_abbr,
        "opponent_abbr": g.opponent_abbr, "home_away": g.home_away,
        "is_starter": int(g.is_starter),
        "decision": g.decision,
        "goals_against": g.goals_against, "shots_against": g.shots_against,
        "saves": g.saves, "save_pct": g.save_pct,
        "toi": g.toi, "toi_seconds": g.toi_seconds,
        "shutout": int(g.shutout) if g.shutout is not None else None,
        "fetched_at": g.fetched_at,
    }


def _player_to_row(p: Player, team_abbr: str, season: str, fetched_at: str) -> dict:
    return {
        "player_id": p.player_id, "team_abbr": team_abbr, "season": season,
        "first_name": p.first_name, "last_name": p.last_name,
        "full_name": p.full_name, "jersey_number": p.jersey_number,
        "position": p.position, "shoots_catches": p.shoots_catches,
        "height_inches": p.height_inches, "weight_lbs": p.weight_lbs,
        "birth_date": p.birth_date, "birth_city": p.birth_city,
        "birth_country": p.birth_country, "headshot_url": p.headshot_url,
        "fetched_at": fetched_at,
    }
