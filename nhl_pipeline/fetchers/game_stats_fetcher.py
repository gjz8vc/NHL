"""Fetches per-game team statistics and goalie performance from the NHL API.

For each completed game this module makes **two** API calls:

1. ``gamecenter/{id}/right-rail``  → ``teamGameStats`` array
   (shots, PP%, PK%, faceoff%, hits, blocked shots, PIMs, turnovers)

2. ``gamecenter/{id}/boxscore``    → ``playerByGameStats.{homeTeam,awayTeam}.goalies``
   (per-goalie: saves, shots against, save%, TOI, decision)

The results are stored as:
- ``team_game_stats``  — one row per team per game (two rows per game)
- ``goalie_game_logs`` — one row per goalie per game

Both tables are joinable with ``player_game_logs`` on ``(game_id, team_abbr)``
for per-player enrichment with team context and opposing-goalie stats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from nhlpy import NHLClient

from nhl_pipeline.config import PipelineConfig
from nhl_pipeline.models import GoalieGameLog, TeamGameStats, to_json, from_json
from nhl_pipeline.utils import RateLimiter, RetryError, get_logger, with_retry

log = get_logger(__name__)


class GameStatsFetcher:
    """Fetches team + goalie stats for individual completed games.

    Typical usage::

        fetcher = GameStatsFetcher(config)
        team_stats, goalie_logs = fetcher.fetch_game(
            game_id=2025020700,
            season="20252026",
            game_type=2,
            game_date="2026-01-15",
            home_abbr="TOR",
            away_abbr="BOS",
            home_goals=4,
            away_goals=2,
        )
    """

    def __init__(self, config: PipelineConfig, client: Optional[NHLClient] = None) -> None:
        self.config = config
        self._client = client or NHLClient(timeout=config.timeout)
        self._limiter = RateLimiter(min_delay=config.rate_limit_delay)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_game(
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
    ) -> tuple[list[TeamGameStats], list[GoalieGameLog]]:
        """Fetch team stats + goalie logs for a single game.

        Returns:
            A tuple of ``(team_stats_list, goalie_logs_list)``.
            ``team_stats_list`` contains two items (home + away).
            ``goalie_logs_list`` may contain 2-4 items.

        Raises:
            :class:`~nhl_pipeline.utils.RetryError` if both API calls fail.
        """
        cache_path = self._cache_path(game_id)

        if not force and self.config.use_cache and cache_path.exists():
            log.debug("Cache hit — game stats for game %s", game_id)
            return self._load_from_cache(cache_path)

        log.debug("Fetching game stats for game %s (%s vs %s, %s) …",
                  game_id, home_abbr, away_abbr, game_date)

        right_rail = self._fetch_right_rail(game_id)
        boxscore   = self._fetch_boxscore(game_id)

        team_stats = self._parse_team_stats(
            right_rail, game_id, season, game_date,
            home_abbr, away_abbr, home_goals, away_goals,
        )
        goalie_logs = self._parse_goalie_logs(
            boxscore, game_id, season, game_type, game_date,
            home_abbr, away_abbr,
        )

        if self.config.save_raw:
            self._save_raw(game_id, right_rail, boxscore)

        self._save_cache(cache_path, team_stats, goalie_logs)
        return team_stats, goalie_logs

    def fetch_games(
        self,
        game_records: list[dict],
        *,
        force: bool = False,
    ) -> tuple[list[TeamGameStats], list[GoalieGameLog]]:
        """Fetch stats for a list of game records (as returned by the DB games table).

        Only fetches games that are in a ``FINAL`` state (completed).

        Args:
            game_records: Rows from the ``games`` DB table.
            force:        Bypass cache.

        Returns:
            Flat lists of all ``TeamGameStats`` and ``GoalieGameLog`` objects.
        """
        all_team: list[TeamGameStats] = []
        all_goalie: list[GoalieGameLog] = []
        errors: list[int] = []

        final_games = [g for g in game_records if g.get("game_state") == "FINAL"]
        log.info("Fetching game stats for %d completed game(s) …", len(final_games))

        for i, g in enumerate(final_games, start=1):
            try:
                self._limiter.wait()
                ts, gls = self.fetch_game(
                    game_id=g["game_id"],
                    season=g["season"],
                    game_type=g["game_type"],
                    game_date=g["date"],
                    home_abbr=g["home_team_abbr"],
                    away_abbr=g["away_team_abbr"],
                    home_goals=g.get("home_score"),
                    away_goals=g.get("away_score"),
                    force=force,
                )
                all_team.extend(ts)
                all_goalie.extend(gls)
                if i % 20 == 0 or i == len(final_games):
                    log.info("Game stats progress: %d / %d", i, len(final_games))
            except RetryError as exc:
                log.warning("Failed to fetch stats for game %s: %s", g["game_id"], exc)
                errors.append(g["game_id"])

        if errors:
            log.warning("Could not fetch stats for %d game(s): %s", len(errors), errors)
        return all_team, all_goalie

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_team_stats(
        right_rail: dict,
        game_id: int,
        season: str,
        game_date: str,
        home_abbr: str,
        away_abbr: str,
        home_goals: Optional[int],
        away_goals: Optional[int],
    ) -> list[TeamGameStats]:
        home, away = TeamGameStats.pair_from_right_rail(
            raw=right_rail,
            game_id=game_id,
            season=season,
            game_date=game_date,
            home_abbr=home_abbr,
            away_abbr=away_abbr,
            home_goals=home_goals,
            away_goals=away_goals,
        )
        return [home, away]

    @staticmethod
    def _parse_goalie_logs(
        boxscore: dict,
        game_id: int,
        season: str,
        game_type: int,
        game_date: str,
        home_abbr: str,
        away_abbr: str,
    ) -> list[GoalieGameLog]:
        by_game = boxscore.get("playerByGameStats", {})
        home_goalies_raw = by_game.get("homeTeam", {}).get("goalies", [])
        away_goalies_raw = by_game.get("awayTeam", {}).get("goalies", [])

        kwargs = dict(game_id=game_id, season=season,
                      game_type=game_type, game_date=game_date)

        home_logs = GoalieGameLog.list_from_boxscore_team(
            home_goalies_raw, home_abbr, away_abbr, "home", **kwargs
        )
        away_logs = GoalieGameLog.list_from_boxscore_team(
            away_goalies_raw, away_abbr, home_abbr, "away", **kwargs
        )
        return home_logs + away_logs

    # ------------------------------------------------------------------
    # API calls with retry
    # ------------------------------------------------------------------

    def _fetch_right_rail(self, game_id: int) -> dict:
        retry = with_retry(
            max_retries=self.config.max_retries,
            backoff_base=self.config.backoff_base,
            exceptions=(Exception,),
        )
        return retry(
            lambda: self._client.game_center.season_series_matchup(game_id=str(game_id))
        )()

    def _fetch_boxscore(self, game_id: int) -> dict:
        retry = with_retry(
            max_retries=self.config.max_retries,
            backoff_base=self.config.backoff_base,
            exceptions=(Exception,),
        )
        self._limiter.wait()
        return retry(
            lambda: self._client.game_center.boxscore(game_id=str(game_id))
        )()

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------

    def _cache_path(self, game_id: int) -> Path:
        return self.config.gamestats_dir / f"{game_id}_stats.json"

    def _save_cache(
        self,
        path: Path,
        team_stats: list[TeamGameStats],
        goalie_logs: list[GoalieGameLog],
    ) -> None:
        from dataclasses import asdict
        payload = {
            "team_stats":  [asdict(t) for t in team_stats],
            "goalie_logs": [asdict(g) for g in goalie_logs],
        }
        to_json(payload, path)

    def _save_raw(self, game_id: int, right_rail: dict, boxscore: dict) -> None:
        raw_dir = self.config.raw_dir / "gamestats"
        raw_dir.mkdir(parents=True, exist_ok=True)
        with (raw_dir / f"raw_{game_id}_right_rail.json").open("w") as fh:
            json.dump(right_rail, fh, indent=2, default=str)
        with (raw_dir / f"raw_{game_id}_boxscore.json").open("w") as fh:
            json.dump(boxscore, fh, indent=2, default=str)

    @staticmethod
    def _load_from_cache(
        path: Path,
    ) -> tuple[list[TeamGameStats], list[GoalieGameLog]]:
        data = from_json(path)

        team_stats = [
            TeamGameStats(**{k: v for k, v in t.items()})
            for t in data.get("team_stats", [])
        ]
        goalie_logs = [
            GoalieGameLog(**{k: v for k, v in g.items()})
            for g in data.get("goalie_logs", [])
        ]
        return team_stats, goalie_logs
