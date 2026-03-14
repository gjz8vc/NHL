"""Fetches and persists team rosters for the NHL 2025-2026 season."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from nhlpy import NHLClient

from nhl_pipeline.config import ALL_TEAM_ABBRS, PipelineConfig
from nhl_pipeline.models import TeamRoster, to_json, from_json
from nhl_pipeline.utils import RateLimiter, RetryError, get_logger, with_retry

log = get_logger(__name__)


class RosterFetcher:
    """Fetches and caches NHL team rosters.

    Roster files are saved to:
        <rosters_dir>/<season>/<TEAM_ABBR>_roster.json

    Raw payloads (when ``config.save_raw`` is True) go to:
        <raw_dir>/rosters/<season>/raw_<TEAM_ABBR>_roster.json
    """

    def __init__(self, config: PipelineConfig, client: Optional[NHLClient] = None) -> None:
        self.config = config
        self._client = client or NHLClient(timeout=config.timeout)
        self._limiter = RateLimiter(min_delay=config.rate_limit_delay)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_team(self, team_abbr: str, *, force: bool = False) -> TeamRoster:
        """Fetch (or load from cache) the roster for *team_abbr*.

        Args:
            team_abbr: Three-letter NHL team abbreviation (e.g. ``"TOR"``).
            force:     If ``True``, bypass the cache and re-fetch from the API.

        Returns:
            A :class:`~nhl_pipeline.models.TeamRoster` instance.
        """
        out_path = self._roster_path(team_abbr)

        if not force and self.config.use_cache and out_path.exists():
            log.info("Cache hit — loading roster for %s", team_abbr)
            return self._load_from_cache(out_path)

        log.info("Fetching roster for %s (%s) …", team_abbr, self.config.season)
        raw = self._fetch_with_retry(team_abbr)

        if self.config.save_raw:
            self._save_raw(raw, team_abbr)

        roster = TeamRoster.from_api(raw, team_abbr, self.config.season)
        to_json(roster, out_path)
        log.info(
            "Saved roster for %s — %d player(s) "
            "(%dF / %dD / %dG)",
            team_abbr,
            roster.total_players,
            len(roster.forwards),
            len(roster.defensemen),
            len(roster.goalies),
        )
        return roster

    def fetch_all_teams(
        self,
        teams: Optional[list[str]] = None,
        *,
        force: bool = False,
    ) -> dict[str, TeamRoster]:
        """Fetch rosters for every team in *teams* (defaults to all 32 NHL teams).

        Returns:
            Mapping of team abbreviation → :class:`~nhl_pipeline.models.TeamRoster`.
        """
        teams = teams or self.config.teams
        rosters: dict[str, TeamRoster] = {}
        errors: list[str] = []

        for abbr in teams:
            try:
                self._limiter.wait()
                rosters[abbr] = self.fetch_team(abbr, force=force)
            except RetryError as exc:
                log.error("Failed to fetch roster for %s: %s", abbr, exc)
                errors.append(abbr)

        if errors:
            log.warning(
                "Could not fetch rosters for %d team(s): %s",
                len(errors), ", ".join(errors),
            )
        return rosters

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_with_retry(self, team_abbr: str) -> dict:
        retry = with_retry(
            max_retries=self.config.max_retries,
            backoff_base=self.config.backoff_base,
            exceptions=(Exception,),
        )
        return retry(lambda: self._client.teams.team_roster(
            team_abbr=team_abbr,
            season=self.config.season,
        ))()

    def _roster_path(self, team_abbr: str) -> Path:
        return self.config.rosters_dir / self.config.season / f"{team_abbr}_roster.json"

    def _raw_path(self, team_abbr: str) -> Path:
        return (
            self.config.raw_dir
            / "rosters"
            / self.config.season
            / f"raw_{team_abbr}_roster.json"
        )

    def _save_raw(self, raw: dict, team_abbr: str) -> None:
        path = self._raw_path(team_abbr)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(raw, fh, indent=2, default=str)

    @staticmethod
    def _load_from_cache(path: Path) -> TeamRoster:
        data = from_json(path)
        from nhl_pipeline.models import Player

        def _players(key: str) -> list[Player]:
            return [
                Player(
                    player_id=p["player_id"],
                    first_name=p["first_name"],
                    last_name=p["last_name"],
                    full_name=p["full_name"],
                    jersey_number=p.get("jersey_number"),
                    position=p["position"],
                    shoots_catches=p.get("shoots_catches"),
                    height_inches=p.get("height_inches"),
                    weight_lbs=p.get("weight_lbs"),
                    birth_date=p.get("birth_date"),
                    birth_city=p.get("birth_city"),
                    birth_country=p.get("birth_country"),
                    headshot_url=p.get("headshot_url"),
                )
                for p in data.get(key, [])
            ]

        return TeamRoster(
            team_abbr=data["team_abbr"],
            season=data["season"],
            forwards=_players("forwards"),
            defensemen=_players("defensemen"),
            goalies=_players("goalies"),
            fetched_at=data.get("fetched_at", ""),
        )
