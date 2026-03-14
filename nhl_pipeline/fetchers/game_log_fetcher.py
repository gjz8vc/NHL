"""Fetches per-game player statistics (game logs) for the NHL 2025-2026 season.

Each player's game log is a list of per-game rows containing:
  goals, assists, points, shots, time on ice, power-play stats,
  game-winning goals, opponent, home/away flag, game date, etc.

Data is stored:
  - As JSON:   <gamelogs_dir>/<season>/<game_type>/<player_id>_gamelog.json
  - As SQLite: player_game_logs table (one row per player-game)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from nhlpy import NHLClient

from nhl_pipeline.config import PipelineConfig
from nhl_pipeline.models import PlayerGameLog, PlayerSeasonGameLog, to_json, from_json
from nhl_pipeline.utils import RateLimiter, RetryError, get_logger, with_retry

log = get_logger(__name__)

# NHL regular-season game type
REGULAR_SEASON = 2
PLAYOFFS = 3


class GameLogFetcher:
    """Fetches player game logs for the 2025-2026 NHL season.

    Usage::

        fetcher = GameLogFetcher(config)

        # Single player
        season_log = fetcher.fetch_player(player_id=8478402)

        # All players from a roster
        fetcher.fetch_players(player_ids=[8478402, 8479318, ...])
    """

    def __init__(self, config: PipelineConfig, client: Optional[NHLClient] = None) -> None:
        self.config = config
        self._client = client or NHLClient(timeout=config.timeout)
        self._limiter = RateLimiter(min_delay=config.rate_limit_delay)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_player(
        self,
        player_id: int,
        game_type: int = REGULAR_SEASON,
        *,
        force: bool = False,
    ) -> PlayerSeasonGameLog:
        """Fetch (or load from cache) the full season game log for *player_id*.

        Args:
            player_id:  NHL player ID (integer).
            game_type:  2 = regular season (default), 3 = playoffs.
            force:      Bypass cache and re-fetch from the API.

        Returns:
            A :class:`~nhl_pipeline.models.PlayerSeasonGameLog`.
        """
        out_path = self._log_path(player_id, game_type)

        if not force and self.config.use_cache and out_path.exists():
            log.debug("Cache hit — game log for player %s (type=%d)", player_id, game_type)
            return self._load_from_cache(out_path, player_id, game_type)

        log.debug("Fetching game log for player %s (season=%s, type=%d) …",
                  player_id, self.config.season, game_type)
        raw_entries = self._fetch_with_retry(player_id, game_type)

        entries = [
            PlayerGameLog.from_api(raw, player_id, self.config.season, game_type)
            for raw in raw_entries
        ]
        season_log = PlayerSeasonGameLog(
            player_id=player_id,
            season=self.config.season,
            game_type=game_type,
            entries=entries,
        )

        if self.config.save_raw:
            self._save_raw(raw_entries, player_id, game_type)

        to_json(season_log, out_path)
        log.debug(
            "Saved game log for player %s — %d game(s)", player_id, season_log.total_games
        )
        return season_log

    def fetch_players(
        self,
        player_ids: list[int],
        game_type: int = REGULAR_SEASON,
        *,
        force: bool = False,
    ) -> dict[int, PlayerSeasonGameLog]:
        """Fetch game logs for a list of player IDs.

        Returns:
            Mapping of player_id → :class:`~nhl_pipeline.models.PlayerSeasonGameLog`.
            Players that fail (after retries) are omitted and logged as errors.
        """
        results: dict[int, PlayerSeasonGameLog] = {}
        errors: list[int] = []
        total = len(player_ids)

        for i, pid in enumerate(player_ids, start=1):
            try:
                self._limiter.wait()
                results[pid] = self.fetch_player(pid, game_type, force=force)
                if i % 50 == 0 or i == total:
                    log.info("Game logs fetched: %d / %d", i, total)
            except RetryError as exc:
                log.warning("Failed to fetch game log for player %s: %s", pid, exc)
                errors.append(pid)

        if errors:
            log.warning(
                "Could not fetch game logs for %d player(s): %s",
                len(errors), errors,
            )
        return results

    def fetch_roster_game_logs(
        self,
        roster_player_ids: list[int],
        include_playoffs: bool = False,
        *,
        force: bool = False,
    ) -> dict[int, list[PlayerGameLog]]:
        """Fetch regular-season (and optionally playoff) game logs for a full roster.

        Returns:
            Mapping of player_id → flat list of all their
            :class:`~nhl_pipeline.models.PlayerGameLog` entries.
        """
        game_types = [REGULAR_SEASON]
        if include_playoffs:
            game_types.append(PLAYOFFS)

        all_logs: dict[int, list[PlayerGameLog]] = {pid: [] for pid in roster_player_ids}

        for gtype in game_types:
            season_logs = self.fetch_players(roster_player_ids, game_type=gtype, force=force)
            for pid, season_log in season_logs.items():
                all_logs[pid].extend(season_log.entries)

        return all_logs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_with_retry(self, player_id: int, game_type: int) -> list[dict]:
        retry = with_retry(
            max_retries=self.config.max_retries,
            backoff_base=self.config.backoff_base,
            exceptions=(Exception,),
        )
        return retry(lambda: self._client.stats.player_game_log(
            player_id=str(player_id),
            season_id=self.config.season,
            game_type=game_type,
        ))()

    def _log_path(self, player_id: int, game_type: int) -> Path:
        return (
            self.config.gamelogs_dir
            / self.config.season
            / str(game_type)
            / f"{player_id}_gamelog.json"
        )

    def _raw_path(self, player_id: int, game_type: int) -> Path:
        return (
            self.config.raw_dir
            / "gamelogs"
            / self.config.season
            / str(game_type)
            / f"raw_{player_id}_gamelog.json"
        )

    def _save_raw(self, raw: list[dict], player_id: int, game_type: int) -> None:
        path = self._raw_path(player_id, game_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(raw, fh, indent=2, default=str)

    @staticmethod
    def _load_from_cache(
        path: Path, player_id: int, game_type: int
    ) -> PlayerSeasonGameLog:
        data = from_json(path)
        entries = [
            PlayerGameLog(
                player_id=e["player_id"],
                game_id=e["game_id"],
                season=e["season"],
                game_type=e["game_type"],
                game_date=e["game_date"],
                team_abbr=e["team_abbr"],
                opponent_abbr=e["opponent_abbr"],
                home_away=e["home_away"],
                goals=e.get("goals"),
                assists=e.get("assists"),
                points=e.get("points"),
                plus_minus=e.get("plus_minus"),
                shots=e.get("shots"),
                pim=e.get("pim"),
                shifts=e.get("shifts"),
                toi=e.get("toi"),
                toi_seconds=e.get("toi_seconds"),
                power_play_goals=e.get("power_play_goals"),
                power_play_points=e.get("power_play_points"),
                power_play_toi=e.get("power_play_toi"),
                power_play_toi_seconds=e.get("power_play_toi_seconds"),
                game_winning_goals=e.get("game_winning_goals"),
                ot_goals=e.get("ot_goals"),
                shorthanded_goals=e.get("shorthanded_goals"),
                shorthanded_points=e.get("shorthanded_points"),
                fetched_at=e.get("fetched_at", ""),
            )
            for e in data.get("entries", [])
        ]
        return PlayerSeasonGameLog(
            player_id=data["player_id"],
            season=data["season"],
            game_type=data["game_type"],
            entries=entries,
            fetched_at=data.get("fetched_at", ""),
        )
