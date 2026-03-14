"""Tests for data model parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nhl_pipeline.models import (
    DailySchedule,
    Game,
    Player,
    TeamRef,
    TeamRoster,
    to_json,
    from_json,
)


class TestGameModel:
    def test_from_api_regular_game(self, raw_schedule):
        raw_game = raw_schedule["games"][0]
        game = Game.from_api(raw_game, "2026-01-15")

        assert game.game_id == 2025020700
        assert game.season == "20252026"
        assert game.game_type == 2
        assert game.game_type_label == "regular"
        assert game.date == "2026-01-15"
        assert game.venue == "Scotiabank Arena"
        assert game.game_state == "FINAL"
        assert game.period == 3

    def test_from_api_home_team(self, raw_schedule):
        game = Game.from_api(raw_schedule["games"][0], "2026-01-15")
        assert game.home_team.abbr == "TOR"
        assert game.home_team.id == 10
        assert game.home_team.score == 4

    def test_from_api_away_team(self, raw_schedule):
        game = Game.from_api(raw_schedule["games"][0], "2026-01-15")
        assert game.away_team.abbr == "BOS"
        assert game.away_team.score == 2

    def test_from_api_future_game_no_score(self, raw_schedule):
        game = Game.from_api(raw_schedule["games"][1], "2026-01-15")
        assert game.game_state == "FUT"
        assert game.home_team.score is None
        assert game.period is None

    def test_game_type_labels(self, raw_schedule):
        raw = dict(raw_schedule["games"][0])
        raw["gameType"] = 1
        assert Game.from_api(raw, "2025-09-20").game_type_label == "preseason"
        raw["gameType"] = 3
        assert Game.from_api(raw, "2026-04-20").game_type_label == "playoff"
        raw["gameType"] = 99
        assert Game.from_api(raw, "2026-04-20").game_type_label == "unknown"


class TestDailyScheduleModel:
    def test_from_api_parses_all_games(self, raw_schedule):
        schedule = DailySchedule.from_api(raw_schedule, "2026-01-15")
        assert schedule.date == "2026-01-15"
        assert schedule.number_of_games == 2
        assert len(schedule.games) == 2

    def test_from_api_pagination_fields(self, raw_schedule):
        schedule = DailySchedule.from_api(raw_schedule, "2026-01-15")
        assert schedule.next_date == "2026-01-16"
        assert schedule.prev_date == "2026-01-14"

    def test_empty_schedule(self):
        raw = {"games": [], "numberOfGames": 0}
        schedule = DailySchedule.from_api(raw, "2026-01-15")
        assert schedule.number_of_games == 0
        assert schedule.games == []


class TestPlayerModel:
    def test_from_api_forward(self, raw_roster):
        p = Player.from_api(raw_roster["forwards"][0])
        assert p.player_id == 8478402
        assert p.first_name == "Auston"
        assert p.last_name == "Matthews"
        assert p.full_name == "Auston Matthews"
        assert p.jersey_number == "34"
        assert p.position == "C"
        assert p.shoots_catches == "L"
        assert p.height_inches == 75
        assert p.weight_lbs == 220
        assert p.birth_date == "1997-09-17"
        assert p.birth_country == "USA"

    def test_from_api_goalie(self, raw_roster):
        p = Player.from_api(raw_roster["goalies"][0])
        assert p.position == "G"
        assert p.full_name == "Joseph Woll"


class TestTeamRosterModel:
    def test_from_api(self, raw_roster):
        roster = TeamRoster.from_api(raw_roster, "TOR", "20252026")
        assert roster.team_abbr == "TOR"
        assert roster.season == "20252026"
        assert len(roster.forwards) == 2
        assert len(roster.defensemen) == 1
        assert len(roster.goalies) == 1
        assert roster.total_players == 4

    def test_all_players(self, raw_roster):
        roster = TeamRoster.from_api(raw_roster, "TOR", "20252026")
        all_ids = {p.player_id for p in roster.all_players}
        assert 8478402 in all_ids  # Matthews
        assert 8476341 in all_ids  # Woll


class TestJsonSerialization:
    def test_round_trip_schedule(self, raw_schedule, tmp_path):
        schedule = DailySchedule.from_api(raw_schedule, "2026-01-15")
        path = tmp_path / "schedule.json"
        to_json(schedule, path)

        assert path.exists()
        data = from_json(path)
        assert data["date"] == "2026-01-15"
        assert len(data["games"]) == 2

    def test_round_trip_roster(self, raw_roster, tmp_path):
        roster = TeamRoster.from_api(raw_roster, "TOR", "20252026")
        path = tmp_path / "roster.json"
        to_json(roster, path)

        data = from_json(path)
        assert data["team_abbr"] == "TOR"
        assert len(data["forwards"]) == 2
