"""Main pipeline orchestrator for the NHL 2025-2026 season data pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from nhlpy import NHLClient

from nhl_pipeline.config import PipelineConfig
from nhl_pipeline.fetchers.game_log_fetcher import GameLogFetcher, REGULAR_SEASON, PLAYOFFS
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
        self._schedule_fetcher = ScheduleFetcher(self.config, client=_client)
        self._roster_fetcher   = RosterFetcher(self.config, client=_client)
        self._game_log_fetcher = GameLogFetcher(self.config, client=_client)
        self._db               = Database(self.config.db_path)

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
    ) -> PipelineRun:
        """Run the full daily pipeline.

        1. Fetch today's (or *date*'s) game schedule.
        2. Persist schedule to JSON + SQLite.
        3. Optionally refresh all (or *teams*) rosters.

        Args:
            date:          Date string (YYYY-MM-DD). Defaults to today (UTC).
            fetch_rosters: When ``True`` (default) also refresh rosters.
            teams:         Subset of team abbreviations to roster-fetch.
                           Defaults to all 32 teams.
            force_rosters: Bypass the roster cache and re-fetch from the API.

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

        # -- Schedule --------------------------------------------------
        try:
            schedule = self._schedule_fetcher.fetch_date(date)
            self._db.upsert_schedule(schedule)
            run.schedules_ok += 1
            log.info(
                "Schedule OK — %d game(s) on %s", schedule.number_of_games, date
            )
        except RetryError as exc:
            msg = f"Schedule fetch failed for {date}: {exc}"
            log.error(msg)
            run.errors.append(msg)

        # -- Rosters ---------------------------------------------------
        if fetch_rosters:
            rosters = self._roster_fetcher.fetch_all_teams(
                teams=teams, force=force_rosters
            )
            for abbr, roster in rosters.items():
                try:
                    self._db.upsert_roster(roster)
                    run.rosters_ok += 1
                except Exception as exc:  # noqa: BLE001
                    msg = f"DB upsert failed for {abbr} roster: {exc}"
                    log.warning(msg)
                    run.warnings.append(msg)

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

    def run_history(self, limit: int = 20) -> list[dict]:
        """Return the most recent pipeline run records."""
        return self._db.get_run_history(limit)
