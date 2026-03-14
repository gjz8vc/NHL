"""Tests for player game log models, fetcher, and storage."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nhl_pipeline.fetchers.game_log_fetcher import GameLogFetcher, REGULAR_SEASON, PLAYOFFS
from nhl_pipeline.models import (
    PlayerGameLog,
    PlayerSeasonGameLog,
    _toi_to_seconds,
    to_json,
    from_json,
)
from nhl_pipeline.storage import Database


# ---------------------------------------------------------------------------
# Sample raw API game log entries
# ---------------------------------------------------------------------------

RAW_GAME_LOG_ENTRIES = [
    {
        "gameId": 2025020700,
        "teamAbbrev": "TOR",
        "homeRoadFlag": "H",
        "gameDate": "2026-01-15",
        "goals": 2,
        "assists": 1,
        "points": 3,
        "plusMinus": 2,
        "powerPlayGoals": 1,
        "powerPlayPoints": 2,
        "gameWinningGoals": 1,
        "otGoals": 0,
        "shots": 6,
        "shifts": 22,
        "shorthandedGoals": 0,
        "shorthandedPoints": 0,
        "opponentAbbrev": "BOS",
        "pim": 2,
        "toi": "21:34",
    },
    {
        "gameId": 2025020710,
        "teamAbbrev": "TOR",
        "homeRoadFlag": "R",
        "gameDate": "2026-01-18",
        "goals": 0,
        "assists": 2,
        "points": 2,
        "plusMinus": -1,
        "powerPlayGoals": 0,
        "powerPlayPoints": 1,
        "gameWinningGoals": 0,
        "otGoals": 0,
        "shots": 4,
        "shifts": 19,
        "shorthandedGoals": 0,
        "shorthandedPoints": 0,
        "opponentAbbrev": "MTL",
        "pim": 0,
        "toi": "18:45",
    },
]


@pytest.fixture
def raw_game_log():
    return RAW_GAME_LOG_ENTRIES


def _mock_client(game_log_payload=None):
    client = MagicMock()
    client.stats.player_game_log.return_value = game_log_payload or RAW_GAME_LOG_ENTRIES
    return client


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestTOIHelper:
    def test_converts_valid_toi(self):
        assert _toi_to_seconds("21:34") == 21 * 60 + 34

    def test_converts_zero_toi(self):
        assert _toi_to_seconds("0:00") == 0

    def test_none_input(self):
        assert _toi_to_seconds(None) is None

    def test_empty_string(self):
        assert _toi_to_seconds("") is None


class TestPlayerGameLogModel:
    def test_from_api_home_game(self, raw_game_log):
        entry = PlayerGameLog.from_api(raw_game_log[0], player_id=8478402,
                                       season="20252026", game_type=2)
        assert entry.player_id == 8478402
        assert entry.game_id == 2025020700
        assert entry.season == "20252026"
        assert entry.game_type == 2
        assert entry.game_date == "2026-01-15"
        assert entry.team_abbr == "TOR"
        assert entry.opponent_abbr == "BOS"
        assert entry.home_away == "home"

    def test_from_api_away_game(self, raw_game_log):
        entry = PlayerGameLog.from_api(raw_game_log[1], player_id=8478402,
                                       season="20252026", game_type=2)
        assert entry.home_away == "away"

    def test_from_api_stats(self, raw_game_log):
        entry = PlayerGameLog.from_api(raw_game_log[0], player_id=8478402,
                                       season="20252026", game_type=2)
        assert entry.goals == 2
        assert entry.assists == 1
        assert entry.points == 3
        assert entry.plus_minus == 2
        assert entry.shots == 6
        assert entry.pim == 2
        assert entry.shifts == 22
        assert entry.power_play_goals == 1
        assert entry.power_play_points == 2
        assert entry.game_winning_goals == 1
        assert entry.ot_goals == 0
        assert entry.shorthanded_goals == 0

    def test_from_api_toi_conversion(self, raw_game_log):
        entry = PlayerGameLog.from_api(raw_game_log[0], player_id=8478402,
                                       season="20252026", game_type=2)
        assert entry.toi == "21:34"
        assert entry.toi_seconds == 21 * 60 + 34

    def test_from_api_no_power_play_toi(self, raw_game_log):
        """power_play_toi is not in the basic game log."""
        entry = PlayerGameLog.from_api(raw_game_log[0], player_id=8478402,
                                       season="20252026", game_type=2)
        assert entry.power_play_toi is None
        assert entry.power_play_toi_seconds is None


class TestPlayerSeasonGameLog:
    def test_from_entries(self, raw_game_log):
        entries = [
            PlayerGameLog.from_api(r, 8478402, "20252026", 2)
            for r in raw_game_log
        ]
        log = PlayerSeasonGameLog(
            player_id=8478402,
            season="20252026",
            game_type=2,
            entries=entries,
        )
        assert log.total_games == 2
        assert log.entries[0].game_date == "2026-01-15"


# ---------------------------------------------------------------------------
# GameLogFetcher tests
# ---------------------------------------------------------------------------

class TestGameLogFetcher:
    def test_fetch_player_returns_season_log(self, tmp_config):
        client = _mock_client()
        fetcher = GameLogFetcher(tmp_config, client=client)

        result = fetcher.fetch_player(player_id=8478402)

        assert isinstance(result, PlayerSeasonGameLog)
        assert result.player_id == 8478402
        assert result.total_games == 2

    def test_fetch_player_calls_api(self, tmp_config):
        client = _mock_client()
        fetcher = GameLogFetcher(tmp_config, client=client)

        fetcher.fetch_player(player_id=8478402, game_type=REGULAR_SEASON)

        client.stats.player_game_log.assert_called_once_with(
            player_id="8478402",
            season_id=tmp_config.season,
            game_type=REGULAR_SEASON,
        )

    def test_fetch_player_saves_json(self, tmp_config):
        client = _mock_client()
        fetcher = GameLogFetcher(tmp_config, client=client)

        fetcher.fetch_player(player_id=8478402)

        out = (
            tmp_config.gamelogs_dir
            / tmp_config.season
            / str(REGULAR_SEASON)
            / "8478402_gamelog.json"
        )
        assert out.exists()
        data = from_json(out)
        assert data["player_id"] == 8478402
        assert len(data["entries"]) == 2

    def test_fetch_player_saves_raw(self, tmp_config):
        client = _mock_client()
        fetcher = GameLogFetcher(tmp_config, client=client)

        fetcher.fetch_player(player_id=8478402)

        raw = (
            tmp_config.raw_dir / "gamelogs" / tmp_config.season
            / str(REGULAR_SEASON) / "raw_8478402_gamelog.json"
        )
        assert raw.exists()

    def test_fetch_player_cache_hit(self, tmp_config):
        client = _mock_client()
        fetcher = GameLogFetcher(tmp_config, client=client)

        fetcher.fetch_player(8478402)
        fetcher.fetch_player(8478402)  # from cache

        assert client.stats.player_game_log.call_count == 1

    def test_fetch_player_force_bypasses_cache(self, tmp_config):
        client = _mock_client()
        fetcher = GameLogFetcher(tmp_config, client=client)

        fetcher.fetch_player(8478402)
        fetcher.fetch_player(8478402, force=True)

        assert client.stats.player_game_log.call_count == 2

    def test_fetch_players_multiple(self, tmp_config):
        client = _mock_client()
        fetcher = GameLogFetcher(tmp_config, client=client)

        results = fetcher.fetch_players([8478402, 8479318, 8480768])

        assert len(results) == 3
        assert client.stats.player_game_log.call_count == 3

    def test_fetch_players_partial_failure(self, tmp_config):
        def side_effect(player_id, **kwargs):
            if player_id == "8479318":
                raise RuntimeError("API error")
            return RAW_GAME_LOG_ENTRIES

        client = MagicMock()
        client.stats.player_game_log.side_effect = side_effect
        tmp_config.max_retries = 0
        fetcher = GameLogFetcher(tmp_config, client=client)

        results = fetcher.fetch_players([8478402, 8479318, 8480768])

        assert len(results) == 2
        assert 8479318 not in results

    def test_fetch_roster_game_logs(self, tmp_config):
        client = _mock_client()
        fetcher = GameLogFetcher(tmp_config, client=client)

        all_logs = fetcher.fetch_roster_game_logs([8478402, 8479318])

        assert set(all_logs.keys()) == {8478402, 8479318}
        # Each player has 2 game entries (from RAW_GAME_LOG_ENTRIES)
        assert len(all_logs[8478402]) == 2
        assert client.stats.player_game_log.call_count == 2  # regular season only

    def test_fetch_roster_game_logs_with_playoffs(self, tmp_config):
        client = _mock_client()
        fetcher = GameLogFetcher(tmp_config, client=client)

        fetcher.fetch_roster_game_logs([8478402], include_playoffs=True)

        # Should call API twice: regular + playoff
        assert client.stats.player_game_log.call_count == 2

    def test_cache_round_trip(self, tmp_config):
        client = _mock_client()
        fetcher = GameLogFetcher(tmp_config, client=client)
        original = fetcher.fetch_player(8478402)

        fetcher2 = GameLogFetcher(tmp_config, client=MagicMock())
        cached = fetcher2.fetch_player(8478402)

        assert cached.player_id == original.player_id
        assert cached.total_games == original.total_games
        assert cached.entries[0].goals == 2
        assert cached.entries[0].toi_seconds == 21 * 60 + 34


# ---------------------------------------------------------------------------
# Storage tests
# ---------------------------------------------------------------------------

class TestGameLogStorage:
    @pytest.fixture
    def db(self, tmp_config):
        return Database(tmp_config.db_path)

    @pytest.fixture
    def sample_logs(self, raw_game_log):
        return [
            PlayerGameLog.from_api(r, 8478402, "20252026", 2)
            for r in raw_game_log
        ]

    def test_upsert_returns_count(self, db, sample_logs):
        count = db.upsert_game_logs(sample_logs)
        assert count == 2

    def test_get_player_game_logs(self, db, sample_logs):
        db.upsert_game_logs(sample_logs)
        rows = db.get_player_game_logs(8478402)
        assert len(rows) == 2

    def test_player_log_fields(self, db, sample_logs):
        db.upsert_game_logs(sample_logs)
        rows = db.get_player_game_logs(8478402, season="20252026", game_type=2)
        first = rows[0]
        assert first["game_date"] == "2026-01-15"
        assert first["team_abbr"] == "TOR"
        assert first["opponent_abbr"] == "BOS"
        assert first["home_away"] == "home"
        assert first["goals"] == 2
        assert first["assists"] == 1
        assert first["points"] == 3
        assert first["toi"] == "21:34"
        assert first["toi_seconds"] == 21 * 60 + 34
        assert first["power_play_goals"] == 1
        assert first["game_winning_goals"] == 1

    def test_upsert_game_logs_idempotent(self, db, sample_logs):
        db.upsert_game_logs(sample_logs)
        db.upsert_game_logs(sample_logs)
        rows = db.get_player_game_logs(8478402)
        assert len(rows) == 2  # not 4

    def test_get_team_game_logs(self, db, sample_logs):
        # Add logs for a second player on TOR
        extra = PlayerGameLog.from_api(
            RAW_GAME_LOG_ENTRIES[0], player_id=8479318,
            season="20252026", game_type=2
        )
        db.upsert_game_logs(sample_logs + [extra])

        rows = db.get_team_game_logs("TOR", "20252026")
        assert len(rows) == 3  # 2 from Matthews + 1 from Marner

    def test_get_game_logs_for_date(self, db, sample_logs):
        db.upsert_game_logs(sample_logs)
        rows = db.get_game_logs_for_date("2026-01-15")
        assert len(rows) == 1
        assert rows[0]["game_date"] == "2026-01-15"

    def test_season_filter(self, db, sample_logs):
        db.upsert_game_logs(sample_logs)
        rows_correct = db.get_player_game_logs(8478402, season="20252026")
        rows_wrong   = db.get_player_game_logs(8478402, season="20242025")
        assert len(rows_correct) == 2
        assert len(rows_wrong) == 0

    def test_game_type_filter(self, db, raw_game_log):
        regular = [PlayerGameLog.from_api(r, 8478402, "20252026", 2) for r in raw_game_log]
        playoff = [PlayerGameLog.from_api(
            {**raw_game_log[0], "gameId": 2025030001}, 8478402, "20252026", 3
        )]
        db.upsert_game_logs(regular + playoff)

        reg_rows  = db.get_player_game_logs(8478402, game_type=2)
        po_rows   = db.get_player_game_logs(8478402, game_type=3)
        assert len(reg_rows) == 2
        assert len(po_rows) == 1

    def test_empty_result(self, db):
        rows = db.get_player_game_logs(9999999)
        assert rows == []
