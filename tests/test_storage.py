"""Tests for the SQLite storage layer."""

from __future__ import annotations

import pytest

from nhl_pipeline.models import DailySchedule, PipelineRun, TeamRoster
from nhl_pipeline.storage import Database


@pytest.fixture
def db(tmp_config):
    return Database(tmp_config.db_path)


@pytest.fixture
def sample_schedule(raw_schedule):
    return DailySchedule.from_api(raw_schedule, "2026-01-15")


@pytest.fixture
def sample_roster(raw_roster):
    return TeamRoster.from_api(raw_roster, "TOR", "20252026")


class TestScheduleStorage:
    def test_upsert_returns_count(self, db, sample_schedule):
        count = db.upsert_schedule(sample_schedule)
        assert count == 2

    def test_get_games_for_date(self, db, sample_schedule):
        db.upsert_schedule(sample_schedule)
        games = db.get_games_for_date("2026-01-15")
        assert len(games) == 2

    def test_get_games_empty_date(self, db):
        games = db.get_games_for_date("1900-01-01")
        assert games == []

    def test_game_fields(self, db, sample_schedule):
        db.upsert_schedule(sample_schedule)
        games = db.get_games_for_date("2026-01-15")
        # Find the FINAL game
        final = next(g for g in games if g["game_state"] == "FINAL")
        assert final["home_team_abbr"] == "TOR"
        assert final["away_team_abbr"] == "BOS"
        assert final["home_score"] == 4
        assert final["away_score"] == 2
        assert final["venue"] == "Scotiabank Arena"

    def test_upsert_is_idempotent(self, db, sample_schedule):
        db.upsert_schedule(sample_schedule)
        db.upsert_schedule(sample_schedule)  # second upsert should not error
        games = db.get_games_for_date("2026-01-15")
        assert len(games) == 2  # still just 2, not 4

    def test_get_games_for_team(self, db, sample_schedule):
        db.upsert_schedule(sample_schedule)
        tor_games = db.get_games_for_team("TOR")
        assert len(tor_games) == 1
        assert tor_games[0]["home_team_abbr"] == "TOR"

    def test_get_games_for_team_with_season_filter(self, db, sample_schedule):
        db.upsert_schedule(sample_schedule)
        games = db.get_games_for_team("TOR", season="20252026")
        assert len(games) == 1
        games_wrong_season = db.get_games_for_team("TOR", season="20242025")
        assert len(games_wrong_season) == 0


class TestRosterStorage:
    def test_upsert_returns_count(self, db, sample_roster):
        count = db.upsert_roster(sample_roster)
        assert count == 4  # 2 forwards + 1 D + 1 G

    def test_get_roster(self, db, sample_roster):
        db.upsert_roster(sample_roster)
        players = db.get_roster("TOR", "20252026")
        assert len(players) == 4

    def test_roster_player_fields(self, db, sample_roster):
        db.upsert_roster(sample_roster)
        players = db.get_roster("TOR", "20252026")
        matthews = next(p for p in players if p["last_name"] == "Matthews")
        assert matthews["first_name"] == "Auston"
        assert matthews["position"] == "C"
        assert matthews["jersey_number"] == "34"
        assert matthews["birth_country"] == "USA"

    def test_upsert_roster_idempotent(self, db, sample_roster):
        db.upsert_roster(sample_roster)
        db.upsert_roster(sample_roster)
        players = db.get_roster("TOR", "20252026")
        assert len(players) == 4

    def test_get_player_by_id(self, db, sample_roster):
        db.upsert_roster(sample_roster)
        rows = db.get_player(8478402)
        assert len(rows) == 1
        assert rows[0]["full_name"] == "Auston Matthews"

    def test_empty_roster(self, db):
        players = db.get_roster("XYZ", "20252026")
        assert players == []


class TestPipelineRunStorage:
    def test_save_and_retrieve_run(self, db):
        run = PipelineRun(
            run_id="test-run-001",
            started_at="2026-01-15T12:00:00Z",
            finished_at="2026-01-15T12:01:00Z",
            date_fetched="2026-01-15",
            schedules_ok=1,
            rosters_ok=32,
        )
        db.save_run(run)
        history = db.get_run_history(limit=5)
        assert len(history) == 1
        assert history[0]["run_id"] == "test-run-001"
        assert history[0]["schedules_ok"] == 1
        assert history[0]["rosters_ok"] == 32
        assert history[0]["success"] == 1

    def test_run_with_errors(self, db):
        run = PipelineRun(
            run_id="test-run-002",
            started_at="2026-01-15T12:00:00Z",
            errors=["Failed to fetch TOR roster"],
        )
        db.save_run(run)
        history = db.get_run_history()
        assert history[0]["success"] == 0
