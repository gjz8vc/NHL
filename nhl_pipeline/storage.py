"""SQLite storage layer — optional persistent store for schedules and rosters."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from nhl_pipeline.models import DailySchedule, Game, Player, PlayerGameLog, TeamRoster
from nhl_pipeline.utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
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

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id       TEXT PRIMARY KEY,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    date_fetched TEXT,
    schedules_ok INTEGER DEFAULT 0,
    rosters_ok   INTEGER DEFAULT 0,
    errors_json  TEXT,
    warnings_json TEXT,
    success      INTEGER DEFAULT 1
);
"""


class Database:
    """Thin wrapper around an SQLite database for the NHL pipeline."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Context manager / connection
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
            con.executescript(_DDL)
        log.debug("Database schema initialised at %s", self.path)

    # ------------------------------------------------------------------
    # Games / Schedules
    # ------------------------------------------------------------------

    def upsert_schedule(self, schedule: DailySchedule) -> int:
        """Insert or replace all games in *schedule*. Returns number of rows affected."""
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
        sql = """
            SELECT * FROM games
            WHERE (home_team_abbr = ? OR away_team_abbr = ?)
        """
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
        """Insert or replace all players in *roster*. Returns number of rows affected."""
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
                "SELECT * FROM rosters WHERE team_abbr = ? AND season = ? ORDER BY position, last_name",
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
        """Insert or replace game-log rows. Returns number of rows affected."""
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
        """Return game logs for a player, optionally filtered by season/type."""
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
        self,
        team_abbr: str,
        season: str,
        game_type: int = 2,
    ) -> list[dict]:
        """Return all player-game rows for a team in a season (great for
        building a team-level per-game dataset)."""
        sql = """
            SELECT * FROM player_game_logs
            WHERE team_abbr = ? AND season = ? AND game_type = ?
            ORDER BY game_date, player_id
        """
        with self._conn() as con:
            rows = con.execute(sql, (team_abbr, season, game_type)).fetchall()
        return [dict(r) for r in rows]

    def get_game_logs_for_date(self, game_date: str) -> list[dict]:
        """Return all player-game rows for a specific game date."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM player_game_logs WHERE game_date = ? ORDER BY team_abbr, player_id",
                (game_date,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Pipeline runs
    # ------------------------------------------------------------------

    def save_run(self, run) -> None:  # run: PipelineRun
        sql = """
            INSERT OR REPLACE INTO pipeline_runs
              (run_id, started_at, finished_at, date_fetched,
               schedules_ok, rosters_ok, errors_json, warnings_json, success)
            VALUES (?,?,?,?,?,?,?,?,?)
        """
        with self._conn() as con:
            con.execute(sql, (
                run.run_id,
                run.started_at,
                run.finished_at,
                run.date_fetched,
                run.schedules_ok,
                run.rosters_ok,
                json.dumps(run.errors),
                json.dumps(run.warnings),
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
        "game_id":         g.game_id,
        "season":          g.season,
        "game_type":       g.game_type,
        "game_type_label": g.game_type_label,
        "date":            g.date,
        "start_time_utc":  g.start_time_utc,
        "venue":           g.venue,
        "home_team_id":    g.home_team.id,
        "home_team_abbr":  g.home_team.abbr,
        "home_team_name":  g.home_team.name,
        "home_score":      g.home_team.score,
        "away_team_id":    g.away_team.id,
        "away_team_abbr":  g.away_team.abbr,
        "away_team_name":  g.away_team.name,
        "away_score":      g.away_team.score,
        "game_state":      g.game_state,
        "period":          g.period,
        "fetched_at":      g.fetched_at,
    }


def _game_log_to_row(gl: PlayerGameLog) -> dict:
    return {
        "player_id":              gl.player_id,
        "game_id":                gl.game_id,
        "season":                 gl.season,
        "game_type":              gl.game_type,
        "game_date":              gl.game_date,
        "team_abbr":              gl.team_abbr,
        "opponent_abbr":          gl.opponent_abbr,
        "home_away":              gl.home_away,
        "goals":                  gl.goals,
        "assists":                gl.assists,
        "points":                 gl.points,
        "plus_minus":             gl.plus_minus,
        "shots":                  gl.shots,
        "pim":                    gl.pim,
        "shifts":                 gl.shifts,
        "toi":                    gl.toi,
        "toi_seconds":            gl.toi_seconds,
        "power_play_goals":       gl.power_play_goals,
        "power_play_points":      gl.power_play_points,
        "power_play_toi":         gl.power_play_toi,
        "power_play_toi_seconds": gl.power_play_toi_seconds,
        "game_winning_goals":     gl.game_winning_goals,
        "ot_goals":               gl.ot_goals,
        "shorthanded_goals":      gl.shorthanded_goals,
        "shorthanded_points":     gl.shorthanded_points,
        "fetched_at":             gl.fetched_at,
    }


def _player_to_row(p: Player, team_abbr: str, season: str, fetched_at: str) -> dict:
    return {
        "player_id":      p.player_id,
        "team_abbr":      team_abbr,
        "season":         season,
        "first_name":     p.first_name,
        "last_name":      p.last_name,
        "full_name":      p.full_name,
        "jersey_number":  p.jersey_number,
        "position":       p.position,
        "shoots_catches": p.shoots_catches,
        "height_inches":  p.height_inches,
        "weight_lbs":     p.weight_lbs,
        "birth_date":     p.birth_date,
        "birth_city":     p.birth_city,
        "birth_country":  p.birth_country,
        "headshot_url":   p.headshot_url,
        "fetched_at":     fetched_at,
    }
