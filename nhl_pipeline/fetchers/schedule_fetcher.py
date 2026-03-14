"""Fetches and persists the daily NHL game schedule."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from nhlpy import NHLClient

from nhl_pipeline.config import PipelineConfig
from nhl_pipeline.models import DailySchedule, to_json, from_json
from nhl_pipeline.utils import RateLimiter, RetryError, get_logger, with_retry

log = get_logger(__name__)


class ScheduleFetcher:
    """Fetches daily NHL schedules for the 2025-2026 season.

    Each day's schedule is saved to:
        <schedules_dir>/YYYY/MM/schedule_YYYY-MM-DD.json

    Raw API payloads (when ``config.save_raw`` is True) go to:
        <raw_dir>/schedules/YYYY/MM/raw_schedule_YYYY-MM-DD.json
    """

    def __init__(self, config: PipelineConfig, client: Optional[NHLClient] = None) -> None:
        self.config = config
        self._client = client or NHLClient(timeout=config.timeout)
        self._limiter = RateLimiter(min_delay=config.rate_limit_delay)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_date(self, target_date: str) -> DailySchedule:
        """Fetch (or load from cache) the schedule for *target_date* (YYYY-MM-DD).

        Returns a :class:`~nhl_pipeline.models.DailySchedule` instance.
        Raises :class:`~nhl_pipeline.utils.RetryError` if the API is unavailable.
        """
        out_path = self._schedule_path(target_date)

        if self.config.use_cache and out_path.exists():
            log.info("Cache hit — loading schedule for %s", target_date)
            return self._load_from_cache(out_path, target_date)

        log.info("Fetching schedule for %s …", target_date)
        raw = self._fetch_with_retry(target_date)

        if self.config.save_raw:
            self._save_raw(raw, target_date)

        schedule = DailySchedule.from_api(raw, target_date)
        to_json(schedule, out_path)
        log.info(
            "Saved schedule for %s — %d game(s)",
            target_date, schedule.number_of_games,
        )
        return schedule

    def fetch_date_range(
        self,
        start: str,
        end: str,
    ) -> list[DailySchedule]:
        """Fetch schedules for every day in [*start*, *end*] (inclusive).

        Args:
            start: Start date in YYYY-MM-DD format.
            end:   End date in YYYY-MM-DD format.

        Returns:
            List of :class:`~nhl_pipeline.models.DailySchedule` objects.
        """
        current = date.fromisoformat(start)
        stop    = date.fromisoformat(end)
        results: list[DailySchedule] = []
        errors: list[str] = []

        while current <= stop:
            day_str = current.isoformat()
            try:
                self._limiter.wait()
                results.append(self.fetch_date(day_str))
            except RetryError as exc:
                log.error("Failed to fetch schedule for %s: %s", day_str, exc)
                errors.append(day_str)
            current += timedelta(days=1)

        if errors:
            log.warning(
                "Could not fetch schedules for %d date(s): %s",
                len(errors), ", ".join(errors),
            )
        return results

    def fetch_today(self) -> DailySchedule:
        """Convenience wrapper — fetch today's schedule."""
        from datetime import datetime
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return self.fetch_date(today)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_with_retry(self, target_date: str) -> dict:
        retry = with_retry(
            max_retries=self.config.max_retries,
            backoff_base=self.config.backoff_base,
            exceptions=(Exception,),
        )
        return retry(lambda: self._client.schedule.daily_schedule(date=target_date))()

    def _schedule_path(self, target_date: str) -> Path:
        year, month, _ = target_date.split("-")
        return self.config.schedules_dir / year / month / f"schedule_{target_date}.json"

    def _raw_path(self, target_date: str) -> Path:
        year, month, _ = target_date.split("-")
        return self.config.raw_dir / "schedules" / year / month / f"raw_schedule_{target_date}.json"

    def _save_raw(self, raw: dict, target_date: str) -> None:
        path = self._raw_path(target_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(raw, fh, indent=2, default=str)

    @staticmethod
    def _load_from_cache(path: Path, target_date: str) -> DailySchedule:
        data = from_json(path)
        from dataclasses import fields
        from nhl_pipeline.models import Game, TeamRef

        games = []
        for g in data.get("games", []):
            def _team(key: str) -> TeamRef:
                t = g.get(key, {})
                return TeamRef(
                    id=t.get("id", 0),
                    abbr=t.get("abbr", ""),
                    name=t.get("name", ""),
                    score=t.get("score"),
                )

            games.append(Game(
                game_id=g["game_id"],
                season=g["season"],
                game_type=g["game_type"],
                game_type_label=g["game_type_label"],
                date=g["date"],
                start_time_utc=g["start_time_utc"],
                venue=g["venue"],
                home_team=_team("home_team"),
                away_team=_team("away_team"),
                game_state=g["game_state"],
                period=g.get("period"),
                fetched_at=g.get("fetched_at", ""),
            ))

        return DailySchedule(
            date=data["date"],
            number_of_games=data["number_of_games"],
            games=games,
            next_date=data.get("next_date"),
            prev_date=data.get("prev_date"),
            fetched_at=data.get("fetched_at", ""),
        )
