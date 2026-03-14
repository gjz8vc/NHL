"""Tests for schedule and roster fetchers (mock the NHL API client)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nhl_pipeline.fetchers.roster_fetcher import RosterFetcher
from nhl_pipeline.fetchers.schedule_fetcher import ScheduleFetcher
from nhl_pipeline.models import DailySchedule, TeamRoster
from nhl_pipeline.utils import RetryError


# ---------------------------------------------------------------------------
# Helper: build a mock NHLClient
# ---------------------------------------------------------------------------

def _mock_client(schedule_payload=None, roster_payload=None):
    client = MagicMock()
    if schedule_payload is not None:
        client.schedule.daily_schedule.return_value = schedule_payload
    if roster_payload is not None:
        client.teams.team_roster.return_value = roster_payload
    return client


# ---------------------------------------------------------------------------
# ScheduleFetcher tests
# ---------------------------------------------------------------------------

class TestScheduleFetcher:
    def test_fetch_date_returns_daily_schedule(self, tmp_config, raw_schedule):
        client = _mock_client(schedule_payload=raw_schedule)
        fetcher = ScheduleFetcher(tmp_config, client=client)

        result = fetcher.fetch_date("2026-01-15")

        assert isinstance(result, DailySchedule)
        assert result.date == "2026-01-15"
        assert result.number_of_games == 2
        assert len(result.games) == 2

    def test_fetch_date_calls_api_once(self, tmp_config, raw_schedule):
        client = _mock_client(schedule_payload=raw_schedule)
        fetcher = ScheduleFetcher(tmp_config, client=client)

        fetcher.fetch_date("2026-01-15")

        client.schedule.daily_schedule.assert_called_once_with(date="2026-01-15")

    def test_fetch_date_saves_json(self, tmp_config, raw_schedule):
        client = _mock_client(schedule_payload=raw_schedule)
        fetcher = ScheduleFetcher(tmp_config, client=client)

        fetcher.fetch_date("2026-01-15")

        out = tmp_config.schedules_dir / "2026" / "01" / "schedule_2026-01-15.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["date"] == "2026-01-15"

    def test_fetch_date_saves_raw(self, tmp_config, raw_schedule):
        client = _mock_client(schedule_payload=raw_schedule)
        fetcher = ScheduleFetcher(tmp_config, client=client)

        fetcher.fetch_date("2026-01-15")

        raw = tmp_config.raw_dir / "schedules" / "2026" / "01" / "raw_schedule_2026-01-15.json"
        assert raw.exists()

    def test_fetch_date_cache_hit(self, tmp_config, raw_schedule):
        """Second call should not hit the API when caching is on."""
        client = _mock_client(schedule_payload=raw_schedule)
        fetcher = ScheduleFetcher(tmp_config, client=client)

        fetcher.fetch_date("2026-01-15")
        fetcher.fetch_date("2026-01-15")  # should use cache

        assert client.schedule.daily_schedule.call_count == 1

    def test_fetch_date_no_cache_bypass(self, tmp_config, raw_schedule):
        """With use_cache=False, the API is always called."""
        tmp_config.use_cache = False
        client = _mock_client(schedule_payload=raw_schedule)
        fetcher = ScheduleFetcher(tmp_config, client=client)

        fetcher.fetch_date("2026-01-15")
        fetcher.fetch_date("2026-01-15")

        assert client.schedule.daily_schedule.call_count == 2

    def test_fetch_date_range(self, tmp_config, raw_schedule):
        client = _mock_client(schedule_payload=raw_schedule)
        fetcher = ScheduleFetcher(tmp_config, client=client)

        results = fetcher.fetch_date_range("2026-01-13", "2026-01-15")

        assert len(results) == 3
        assert all(isinstance(r, DailySchedule) for r in results)

    def test_fetch_date_partial_failure(self, tmp_config, raw_schedule):
        """Errors on individual dates should not abort the whole range."""
        client = MagicMock()

        def side_effect(date):
            if date == "2026-01-14":
                raise RuntimeError("simulated network error")
            return raw_schedule

        client.schedule.daily_schedule.side_effect = side_effect
        tmp_config.max_retries = 0  # no retries — fail immediately
        fetcher = ScheduleFetcher(tmp_config, client=client)

        results = fetcher.fetch_date_range("2026-01-13", "2026-01-15")
        # Jan 14 always fails → only 2 days succeed
        assert len(results) == 2


# ---------------------------------------------------------------------------
# RosterFetcher tests
# ---------------------------------------------------------------------------

class TestRosterFetcher:
    def test_fetch_team_returns_roster(self, tmp_config, raw_roster):
        client = _mock_client(roster_payload=raw_roster)
        fetcher = RosterFetcher(tmp_config, client=client)

        result = fetcher.fetch_team("TOR")

        assert isinstance(result, TeamRoster)
        assert result.team_abbr == "TOR"
        assert result.total_players == 4

    def test_fetch_team_saves_json(self, tmp_config, raw_roster):
        client = _mock_client(roster_payload=raw_roster)
        fetcher = RosterFetcher(tmp_config, client=client)

        fetcher.fetch_team("TOR")

        out = tmp_config.rosters_dir / tmp_config.season / "TOR_roster.json"
        assert out.exists()

    def test_fetch_team_saves_raw(self, tmp_config, raw_roster):
        client = _mock_client(roster_payload=raw_roster)
        fetcher = RosterFetcher(tmp_config, client=client)

        fetcher.fetch_team("TOR")

        raw = tmp_config.raw_dir / "rosters" / tmp_config.season / "raw_TOR_roster.json"
        assert raw.exists()

    def test_fetch_team_cache_hit(self, tmp_config, raw_roster):
        client = _mock_client(roster_payload=raw_roster)
        fetcher = RosterFetcher(tmp_config, client=client)

        fetcher.fetch_team("TOR")
        fetcher.fetch_team("TOR")  # cache

        assert client.teams.team_roster.call_count == 1

    def test_fetch_team_force_bypasses_cache(self, tmp_config, raw_roster):
        client = _mock_client(roster_payload=raw_roster)
        fetcher = RosterFetcher(tmp_config, client=client)

        fetcher.fetch_team("TOR")
        fetcher.fetch_team("TOR", force=True)

        assert client.teams.team_roster.call_count == 2

    def test_fetch_all_teams(self, tmp_config, raw_roster):
        client = _mock_client(roster_payload=raw_roster)
        tmp_config.teams = ["TOR", "BOS", "MTL"]
        fetcher = RosterFetcher(tmp_config, client=client)

        results = fetcher.fetch_all_teams()

        assert set(results.keys()) == {"TOR", "BOS", "MTL"}
        assert client.teams.team_roster.call_count == 3

    def test_fetch_all_teams_partial_failure(self, tmp_config, raw_roster):
        def side_effect(**kwargs):
            if kwargs.get("team_abbr") == "BOS":
                raise RuntimeError("network error")
            return raw_roster

        client = MagicMock()
        client.teams.team_roster.side_effect = side_effect
        tmp_config.teams = ["TOR", "BOS", "MTL"]
        tmp_config.max_retries = 0  # no retries — BOS always fails
        fetcher = RosterFetcher(tmp_config, client=client)

        results = fetcher.fetch_all_teams()
        assert len(results) == 2  # BOS failed, TOR + MTL succeeded


# ---------------------------------------------------------------------------
# Cache round-trip tests (load from disk)
# ---------------------------------------------------------------------------

class TestCacheRoundTrip:
    def test_schedule_cache_round_trip(self, tmp_config, raw_schedule):
        client = _mock_client(schedule_payload=raw_schedule)
        fetcher = ScheduleFetcher(tmp_config, client=client)

        original = fetcher.fetch_date("2026-01-15")

        # Create a new fetcher that will load from cache
        fetcher2 = ScheduleFetcher(tmp_config, client=MagicMock())
        cached = fetcher2.fetch_date("2026-01-15")

        assert cached.date == original.date
        assert cached.number_of_games == original.number_of_games
        assert len(cached.games) == len(original.games)
        assert cached.games[0].home_team.abbr == "TOR"
        assert cached.games[0].away_team.score == 2

    def test_roster_cache_round_trip(self, tmp_config, raw_roster):
        client = _mock_client(roster_payload=raw_roster)
        fetcher = RosterFetcher(tmp_config, client=client)

        original = fetcher.fetch_team("TOR")

        fetcher2 = RosterFetcher(tmp_config, client=MagicMock())
        cached = fetcher2.fetch_team("TOR")

        assert cached.team_abbr == original.team_abbr
        assert cached.total_players == original.total_players
        assert cached.forwards[0].full_name == "Auston Matthews"
