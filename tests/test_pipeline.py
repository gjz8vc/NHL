"""Integration tests for the NHLPipeline orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nhl_pipeline.models import DailySchedule, TeamRoster
from nhl_pipeline.pipeline import NHLPipeline


def _make_pipeline(tmp_config, raw_schedule, raw_roster):
    """Build an NHLPipeline with a fully mocked NHLClient."""
    client = MagicMock()
    client.schedule.daily_schedule.return_value = raw_schedule
    client.teams.team_roster.return_value = raw_roster
    tmp_config.teams = ["TOR", "BOS"]
    pipeline = NHLPipeline(config=tmp_config, client=client)
    return pipeline, client


class TestDailyRun:
    def test_run_daily_success(self, tmp_config, raw_schedule, raw_roster):
        pipeline, _ = _make_pipeline(tmp_config, raw_schedule, raw_roster)

        run = pipeline.run_daily(date="2026-01-15")

        assert run.success
        assert run.schedules_ok == 1
        assert run.rosters_ok == 2
        assert run.date_fetched == "2026-01-15"
        assert run.finished_at is not None

    def test_run_daily_persists_run(self, tmp_config, raw_schedule, raw_roster):
        pipeline, _ = _make_pipeline(tmp_config, raw_schedule, raw_roster)

        pipeline.run_daily(date="2026-01-15")

        history = pipeline.run_history(limit=5)
        assert len(history) == 1

    def test_run_daily_no_rosters(self, tmp_config, raw_schedule, raw_roster):
        pipeline, client = _make_pipeline(tmp_config, raw_schedule, raw_roster)

        run = pipeline.run_daily(date="2026-01-15", fetch_rosters=False)

        assert run.schedules_ok == 1
        assert run.rosters_ok == 0
        client.teams.team_roster.assert_not_called()

    def test_run_daily_schedule_failure(self, tmp_config, raw_roster):
        client = MagicMock()
        client.schedule.daily_schedule.side_effect = RuntimeError("API down")
        client.teams.team_roster.return_value = raw_roster
        tmp_config.teams = ["TOR"]
        tmp_config.max_retries = 0
        pipeline = NHLPipeline(config=tmp_config, client=client)

        run = pipeline.run_daily(date="2026-01-15", fetch_rosters=False)

        assert not run.success
        assert len(run.errors) == 1


class TestQueryMethods:
    def test_games_on_returns_stored_games(self, tmp_config, raw_schedule, raw_roster):
        pipeline, _ = _make_pipeline(tmp_config, raw_schedule, raw_roster)
        pipeline.run_daily(date="2026-01-15", fetch_rosters=False)

        games = pipeline.games_on("2026-01-15")

        assert len(games) == 2

    def test_team_schedule_returns_team_games(self, tmp_config, raw_schedule, raw_roster):
        pipeline, _ = _make_pipeline(tmp_config, raw_schedule, raw_roster)
        pipeline.run_daily(date="2026-01-15", fetch_rosters=False)

        tor_games = pipeline.team_schedule("TOR")
        assert len(tor_games) == 1

    def test_roster_returns_players(self, tmp_config, raw_schedule, raw_roster):
        pipeline, _ = _make_pipeline(tmp_config, raw_schedule, raw_roster)
        pipeline.run_daily(date="2026-01-15")

        players = pipeline.roster("TOR")
        assert len(players) == 4
        names = {p["full_name"] for p in players}
        assert "Auston Matthews" in names


class TestBackfill:
    def test_backfill_date_range(self, tmp_config, raw_schedule, raw_roster):
        pipeline, client = _make_pipeline(tmp_config, raw_schedule, raw_roster)

        runs = pipeline.run_date_range("2026-01-13", "2026-01-15", fetch_rosters=False)

        assert len(runs) == 3
        assert all(r.schedules_ok == 1 for r in runs)
        # API called once per day
        assert client.schedule.daily_schedule.call_count == 3
