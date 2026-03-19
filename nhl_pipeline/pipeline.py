"""Main pipeline orchestrator for the NHL 2025-2026 season data pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Optional

from nhlpy import NHLClient

from nhl_pipeline.config import PipelineConfig
from nhl_pipeline.fetchers.game_log_fetcher import GameLogFetcher, REGULAR_SEASON, PLAYOFFS
from nhl_pipeline.fetchers.game_stats_fetcher import GameStatsFetcher
from nhl_pipeline.fetchers.roster_fetcher import RosterFetcher
from nhl_pipeline.fetchers.schedule_fetcher import ScheduleFetcher
from nhl_pipeline.models import DailySchedule, PlayerGameLog, PipelineRun, TeamRoster
from nhl_pipeline.storage import Database
from nhl_pipeline.utils import RetryError, get_logger

log = get_logger(__name__)


class NHLPipeline:
    """Orchestrates fetching, parsing, and storing NHL schedule, roster, and
    player game-log data for the 2025-2026 season.

    Typical usage::

        from nhl_pipeline.pipeline import NHLPipeline
        from nhl_pipeline.config import PipelineConfig

        config = PipelineConfig()
        pipeline = NHLPipeline(config)

        # Full daily pipeline (schedule + all rosters):
        run = pipeline.run_daily()

        # Full season game-log backfill for every player on every roster:
        pipeline.fetch_all_game_logs()

        # Individual pieces:
        schedule  = pipeline.fetch_schedule("2026-01-15")
        roster    = pipeline.fetch_roster("TOR")
        game_logs = pipeline.fetch_player_game_logs(player_id=8478402)
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        client: Optional[NHLClient] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.config.ensure_dirs()

        _client = client or NHLClient(timeout=self.config.timeout)
        self._schedule_fetcher   = ScheduleFetcher(self.config, client=_client)
        self._roster_fetcher     = RosterFetcher(self.config, client=_client)
        self._game_log_fetcher   = GameLogFetcher(self.config, client=_client)
        self._game_stats_fetcher = GameStatsFetcher(self.config, client=_client)
        self._db                 = Database(self.config.db_path)

    # ------------------------------------------------------------------
    # High-level orchestration
    # ------------------------------------------------------------------

    def run_daily(
        self,
        date: Optional[str] = None,
        *,
        fetch_rosters: bool = True,
        teams: Optional[list[str]] = None,
        force_rosters: bool = False,
        lookback_games: int = 10,
    ) -> PipelineRun:
        """Run the full daily pipeline.

        1. Walk backwards from *date* fetching schedules until we have
           *lookback_games* game-days (days with at least one NHL game).
        2. Persist schedules to JSON + SQLite.
        3. Optionally refresh rosters for teams found in those games.
        4. Fetch player game logs for those teams.
        5. Fetch team + goalie stats for the discovered games.

        Args:
            date:            Date string (YYYY-MM-DD). Defaults to today (UTC).
            fetch_rosters:   When ``True`` (default) also refresh rosters.
            teams:           Subset of team abbreviations to roster-fetch.
                             Defaults to teams found in the fetched games.
            force_rosters:   Bypass the roster cache and re-fetch from the API.
            lookback_games:  Number of game-days to look back (default 10).

        Returns:
            A :class:`~nhl_pipeline.models.PipelineRun` summary.
        """
        if date is None:
            date = datetime.utcnow().strftime("%Y-%m-%d")

        run = PipelineRun(
            run_id=str(uuid.uuid4()),
            started_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            date_fetched=date,
        )

        log.info("=" * 60)
        log.info("NHL Pipeline — daily run for %s  (run_id=%s)", date, run.run_id)
        log.info("=" * 60)

        # -- Schedule (last N game-days) -------------------------------
        target = datetime.strptime(date, "%Y-%m-%d").date()
        game_days_found = 0
        max_lookback = 30  # safety cap to avoid infinite walk-back
        game_teams: set[str] = set()
        discovered_dates: list[str] = []

        for days_back in range(max_lookback):
            day = (target - timedelta(days=days_back)).isoformat()
            try:
                schedule = self._schedule_fetcher.fetch_date(day)
                self._db.upsert_schedule(schedule)
                run.schedules_ok += 1
                if schedule.number_of_games > 0:
                    game_days_found += 1
                    discovered_dates.append(day)
                    for g in schedule.games:
                        game_teams.add(g.home_team.abbr)
                        game_teams.add(g.away_team.abbr)
                    log.info(
                        "Schedule OK — %d game(s) on %s  [%d/%d game-days]",
                        schedule.number_of_games, day,
                        game_days_found, lookback_games,
                    )
                if game_days_found >= lookback_games:
                    break
            except RetryError as exc:
                msg = f"Schedule fetch failed for {day}: {exc}"
                log.error(msg)
                run.errors.append(msg)

        log.info(
            "Schedules done — %d game-day(s), %d team(s) discovered",
            game_days_found, len(game_teams),
        )

        # -- Rosters ---------------------------------------------------
        roster_teams = list(teams or game_teams) or self.config.teams
        if fetch_rosters:
            rosters = self._roster_fetcher.fetch_all_teams(
                teams=roster_teams, force=force_rosters
            )
            for abbr, roster in rosters.items():
                try:
                    self._db.upsert_roster(roster)
                    run.rosters_ok += 1
                except Exception as exc:  # noqa: BLE001
                    msg = f"DB upsert failed for {abbr} roster: {exc}"
                    log.warning(msg)
                    run.warnings.append(msg)

        # -- Player game logs ------------------------------------------
        log.info("Fetching player game logs for %d team(s) …", len(roster_teams))
        try:
            gl_stats = self.fetch_all_game_logs(teams=roster_teams, force=False)
            total_gl = sum(gl_stats.values())
            log.info("Game logs OK — %d player-game rows", total_gl)
        except Exception as exc:  # noqa: BLE001
            msg = f"Game log fetch failed: {exc}"
            log.error(msg)
            run.errors.append(msg)

        # -- Team + goalie game stats ----------------------------------
        # Only fetch stats for games in the lookback window, not the
        # entire season (which would re-fetch hundreds of cached games
        # and overwhelm the NHL API with requests).
        lookback_game_records: list[dict] = []
        for day in discovered_dates:
            lookback_game_records.extend(self._db.get_games_for_date(day))

        log.info(
            "Fetching team/goalie stats for %d discovered game(s) …",
            len(lookback_game_records),
        )
        try:
            ts_list, gl_list = self._game_stats_fetcher.fetch_games(
                lookback_game_records, force=False,
            )
            ts_rows = self._db.upsert_team_game_stats(ts_list)
            gl_rows = self._db.upsert_goalie_game_logs(gl_list)
            log.info(
                "Game stats OK — %d team-stat rows, %d goalie-log rows",
                ts_rows, gl_rows,
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"Game stats fetch failed: {exc}"
            log.error(msg)
            run.errors.append(msg)

        run.finished_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        self._db.save_run(run)

        log.info("-" * 60)
        log.info(
            "Run complete — schedules: %d, rosters: %d, errors: %d",
            run.schedules_ok, run.rosters_ok, len(run.errors),
        )
        return run

    def run_date_range(
        self,
        start: str,
        end: str,
        *,
        fetch_rosters: bool = False,
    ) -> list[PipelineRun]:
        """Backfill schedules for a range of dates.

        Args:
            start: Start date (YYYY-MM-DD).
            end:   End date (YYYY-MM-DD), inclusive.
            fetch_rosters: Fetch/refresh rosters once (before the first date).

        Returns:
            List of :class:`~nhl_pipeline.models.PipelineRun` objects.
        """
        runs: list[PipelineRun] = []

        if fetch_rosters:
            log.info("Pre-fetching all team rosters …")
            rosters = self._roster_fetcher.fetch_all_teams()
            for roster in rosters.values():
                self._db.upsert_roster(roster)

        schedules = self._schedule_fetcher.fetch_date_range(start, end)
        for schedule in schedules:
            run = PipelineRun(
                run_id=str(uuid.uuid4()),
                started_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
                date_fetched=schedule.date,
            )
            self._db.upsert_schedule(schedule)
            run.schedules_ok = 1
            run.finished_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            self._db.save_run(run)
            runs.append(run)

        return runs

    def fetch_all_game_logs(
        self,
        teams: Optional[list[str]] = None,
        *,
        include_playoffs: bool = False,
        force: bool = False,
    ) -> dict[str, int]:
        """Fetch full-season game logs for every player on every team roster.

        Pulls rosters (from cache if available), then fetches one game-log
        per player.  Each player-game is stored as a row in the
        ``player_game_logs`` table.

        Args:
            teams:             Subset of team abbreviations. Defaults to all 32.
            include_playoffs:  Also fetch playoff game logs.
            force:             Bypass cache and re-fetch from the API.

        Returns:
            Mapping of team abbreviation → number of player-game rows stored.
        """
        teams = teams or self.config.teams
        stats: dict[str, int] = {}

        log.info("Fetching game logs for %d team(s) …", len(teams))

        # Ensure rosters are loaded
        rosters = self._roster_fetcher.fetch_all_teams(teams=teams)
        for roster in rosters.values():
            self._db.upsert_roster(roster)

        for abbr, roster in rosters.items():
            player_ids = [p.player_id for p in roster.all_players]
            log.info("Fetching game logs — %s (%d players) …", abbr, len(player_ids))

            all_logs = self._game_log_fetcher.fetch_roster_game_logs(
                player_ids, include_playoffs=include_playoffs, force=force
            )

            # Flatten all entries and bulk-upsert
            all_entries: list[PlayerGameLog] = [
                entry
                for entries in all_logs.values()
                for entry in entries
            ]
            rows_stored = self._db.upsert_game_logs(all_entries)
            stats[abbr] = rows_stored
            log.info("  %s — stored %d player-game rows", abbr, rows_stored)

        total = sum(stats.values())
        log.info("Game log fetch complete — %d total player-game rows across %d teams",
                 total, len(stats))
        return stats

    # ------------------------------------------------------------------
    # Individual fetch helpers (useful for ad-hoc use)
    # ------------------------------------------------------------------

    def fetch_schedule(self, date: str) -> DailySchedule:
        """Fetch + persist the schedule for a specific date."""
        schedule = self._schedule_fetcher.fetch_date(date)
        self._db.upsert_schedule(schedule)
        return schedule

    def fetch_roster(self, team_abbr: str, *, force: bool = False) -> TeamRoster:
        """Fetch + persist the roster for a specific team."""
        roster = self._roster_fetcher.fetch_team(team_abbr, force=force)
        self._db.upsert_roster(roster)
        return roster

    def fetch_player_game_logs(
        self,
        player_id: int,
        game_type: int = REGULAR_SEASON,
        *,
        force: bool = False,
    ) -> list[PlayerGameLog]:
        """Fetch + persist game logs for a single player.

        Returns the flat list of :class:`~nhl_pipeline.models.PlayerGameLog` entries.
        """
        season_log = self._game_log_fetcher.fetch_player(
            player_id, game_type, force=force
        )
        self._db.upsert_game_logs(season_log.entries)
        return season_log.entries

    def fetch_game_stats(
        self,
        game_id: int,
        season: str,
        game_type: int,
        game_date: str,
        home_abbr: str,
        away_abbr: str,
        home_goals: Optional[int] = None,
        away_goals: Optional[int] = None,
        *,
        force: bool = False,
    ) -> tuple[list, list]:
        """Fetch + persist team stats and goalie logs for a single completed game.

        Returns ``(team_stats_list, goalie_logs_list)``.
        """
        team_stats, goalie_logs = self._game_stats_fetcher.fetch_game(
            game_id=game_id, season=season, game_type=game_type,
            game_date=game_date, home_abbr=home_abbr, away_abbr=away_abbr,
            home_goals=home_goals, away_goals=away_goals, force=force,
        )
        self._db.upsert_team_game_stats(team_stats)
        self._db.upsert_goalie_game_logs(goalie_logs)
        return team_stats, goalie_logs

    def fetch_game_stats_for_date(
        self, date: str, *, force: bool = False
    ) -> tuple[int, int]:
        """Fetch + persist team stats and goalie logs for all FINAL games on *date*.

        Returns ``(team_stat_rows, goalie_log_rows)`` inserted.
        """
        games = self._db.get_games_for_date(date)
        ts_list, gl_list = self._game_stats_fetcher.fetch_games(games, force=force)
        ts_rows = self._db.upsert_team_game_stats(ts_list)
        gl_rows = self._db.upsert_goalie_game_logs(gl_list)
        return ts_rows, gl_rows

    def fetch_game_stats_for_season(
        self,
        season: Optional[str] = None,
        game_type: int = REGULAR_SEASON,
        *,
        force: bool = False,
    ) -> tuple[int, int]:
        """Fetch + persist team stats and goalie logs for every FINAL game in *season*.

        Returns ``(team_stat_rows, goalie_log_rows)`` inserted.
        """
        season = season or self.config.season
        with self._db._conn() as con:
            rows = con.execute(
                "SELECT * FROM games WHERE season = ? AND game_type = ?"
                " AND game_state = 'FINAL' ORDER BY date",
                (season, game_type),
            ).fetchall()
        game_records = [dict(r) for r in rows]

        ts_list, gl_list = self._game_stats_fetcher.fetch_games(
            game_records, force=force
        )
        ts_rows = self._db.upsert_team_game_stats(ts_list)
        gl_rows = self._db.upsert_goalie_game_logs(gl_list)
        log.info(
            "Season game stats complete — %d team-stat rows, %d goalie-log rows",
            ts_rows, gl_rows,
        )
        return ts_rows, gl_rows

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def games_on(self, date: str) -> list[dict]:
        """Return all persisted games for *date* from the database."""
        return self._db.get_games_for_date(date)

    def team_schedule(self, team_abbr: str, season: Optional[str] = None) -> list[dict]:
        """Return all persisted games involving *team_abbr*."""
        return self._db.get_games_for_team(team_abbr, season)

    def roster(self, team_abbr: str, season: Optional[str] = None) -> list[dict]:
        """Return all persisted roster entries for *team_abbr*."""
        return self._db.get_roster(team_abbr, season or self.config.season)

    def player_game_logs(
        self,
        player_id: int,
        season: Optional[str] = None,
        game_type: Optional[int] = None,
    ) -> list[dict]:
        """Return persisted game-log rows for *player_id*."""
        return self._db.get_player_game_logs(
            player_id,
            season=season or self.config.season,
            game_type=game_type,
        )

    def team_game_logs(
        self,
        team_abbr: str,
        season: Optional[str] = None,
        game_type: int = REGULAR_SEASON,
    ) -> list[dict]:
        """Return all persisted player-game rows for *team_abbr* (team-level dataset)."""
        return self._db.get_team_game_logs(
            team_abbr,
            season=season or self.config.season,
            game_type=game_type,
        )

    def team_game_stats(
        self, team_abbr: str, season: Optional[str] = None
    ) -> list[dict]:
        """Return persisted team-game-stats rows for *team_abbr*."""
        return self._db.get_team_game_stats(team_abbr, season or self.config.season)

    def goalie_logs(
        self,
        player_id: int,
        season: Optional[str] = None,
        starters_only: bool = True,
    ) -> list[dict]:
        """Return persisted goalie game logs."""
        return self._db.get_goalie_game_logs(
            player_id, season or self.config.season, starters_only=starters_only
        )

    def goalie_rolling_stats(self, player_id: int) -> list[dict]:
        """Return rolling save% / GAA for a goalie (last-5-start windows)."""
        return self._db.get_goalie_rolling_stats(player_id)

    def point_dataset(
        self,
        season: Optional[str] = None,
        game_type: int = REGULAR_SEASON,
        team_abbr: Optional[str] = None,
    ) -> list[dict]:
        """Return the ``player_point_dataset`` view — the modelling-ready table.

        Each row is one skater in one game with:
        - player bio and game stats
        - their team's goals / shots / PP% / PK%
        - opponent team's same stats
        - opposing starting goalie save% (current game and rolling last-5)
        - binary target ``point_scored`` (1 if goals+assists >= 1)
        """
        return self._db.get_point_dataset(
            season=season or self.config.season,
            game_type=game_type,
            team_abbr=team_abbr,
        )

    def point_streaks(
        self,
        min_streak: int = 5,
        season: Optional[str] = None,
        as_of_date: Optional[str] = None,
    ) -> list[dict]:
        """Return skaters currently on a point streak of at least *min_streak* games.

        Each row contains player_id, full_name, team_abbr, position,
        streak_length, last_game_date, and streak_start.
        """
        return self._db.get_point_streaks(
            min_streak=min_streak,
            season=season or self.config.season,
            as_of_date=as_of_date,
        )

    def run_history(self, limit: int = 20) -> list[dict]:
        """Return the most recent pipeline run records."""
        return self._db.get_run_history(limit)
