"""Tests for TeamGameStats + GoalieGameLog models, fetcher, storage, and views."""

from __future__ import annotations

from dataclasses import asdict
from unittest.mock import MagicMock

import pytest

from nhl_pipeline.fetchers.game_stats_fetcher import GameStatsFetcher
from nhl_pipeline.models import (
    GoalieGameLog,
    PlayerGameLog,
    TeamGameStats,
    _parse_pp,
)
from nhl_pipeline.storage import Database


# ---------------------------------------------------------------------------
# Sample raw API payloads
# ---------------------------------------------------------------------------

RIGHT_RAIL = {
    "teamGameStats": [
        {"category": "sog",                "awayValue": 28,    "homeValue": 34},
        {"category": "faceoffWinningPctg", "awayValue": 0.467, "homeValue": 0.533},
        {"category": "powerPlay",          "awayValue": "1/4", "homeValue": "2/3"},
        {"category": "penaltyKillPctg",    "awayValue": 0.667, "homeValue": 0.75},
        {"category": "hits",               "awayValue": 22,    "homeValue": 18},
        {"category": "blockedShots",       "awayValue": 12,    "homeValue": 9},
        {"category": "pim",                "awayValue": 8,     "homeValue": 6},
        {"category": "giveaways",          "awayValue": 7,     "homeValue": 5},
        {"category": "takeaways",          "awayValue": 4,     "homeValue": 6},
    ]
}

BOXSCORE = {
    "playerByGameStats": {
        "homeTeam": {
            "goalies": [
                {
                    "playerId":     8476341,
                    "toi":          "59:12",
                    "decision":     "W",
                    "goalsAgainst": 2,
                    "shotsAgainst": 30,
                },
            ]
        },
        "awayTeam": {
            "goalies": [
                {
                    "playerId":     8479406,
                    "toi":          "42:00",
                    "decision":     "L",
                    "goalsAgainst": 3,
                    "shotsAgainst": 20,
                },
                {
                    "playerId":     8480786,
                    "toi":          "17:54",
                    "decision":     None,
                    "goalsAgainst": 1,
                    "shotsAgainst": 14,
                },
            ]
        },
    }
}

GAME_META = dict(
    game_id=2025020700,
    season="20252026",
    game_type=2,
    game_date="2026-01-15",
    home_abbr="TOR",
    away_abbr="BOS",
    home_goals=4,
    away_goals=2,
)

# Subset used for pair_from_right_rail (no game_type)
PAIR_META = {k: v for k, v in GAME_META.items() if k != "game_type"}


def _mock_client():
    client = MagicMock()
    client.game_center.season_series_matchup.return_value = RIGHT_RAIL
    client.game_center.boxscore.return_value = BOXSCORE
    return client


# ---------------------------------------------------------------------------
# _parse_pp helper
# ---------------------------------------------------------------------------

class TestParsePP:
    def test_normal(self):
        assert _parse_pp("2/4") == (2, 4, 0.5)

    def test_zero_goals(self):
        g, o, p = _parse_pp("0/3")
        assert g == 0 and o == 3 and p == 0.0

    def test_none(self):
        assert _parse_pp(None) == (None, None, None)

    def test_empty(self):
        assert _parse_pp("") == (None, None, None)

    def test_non_string(self):
        assert _parse_pp(0.333) == (None, None, None)

    def test_zero_opps(self):
        g, o, p = _parse_pp("0/0")
        assert p == 0.0


# ---------------------------------------------------------------------------
# TeamGameStats model
# ---------------------------------------------------------------------------

class TestTeamGameStats:
    def test_pair_from_right_rail_home(self):
        home, away = TeamGameStats.pair_from_right_rail(
            RIGHT_RAIL, **PAIR_META
        )
        assert home.team_abbr == "TOR"
        assert home.home_away == "home"
        assert home.goals == 4
        assert home.goals_allowed == 2
        assert home.shots_for == 34
        assert home.shots_against == 28

    def test_pair_from_right_rail_away(self):
        home, away = TeamGameStats.pair_from_right_rail(
            RIGHT_RAIL, **PAIR_META
        )
        assert away.team_abbr == "BOS"
        assert away.home_away == "away"
        assert away.goals == 2
        assert away.goals_allowed == 4
        assert away.shots_for == 28
        assert away.shots_against == 34

    def test_power_play_home(self):
        home, _ = TeamGameStats.pair_from_right_rail(RIGHT_RAIL, **PAIR_META)
        assert home.pp_goals == 2
        assert home.pp_opportunities == 3
        assert home.pp_pct == round(2 / 3, 4)

    def test_power_play_away(self):
        _, away = TeamGameStats.pair_from_right_rail(RIGHT_RAIL, **PAIR_META)
        assert away.pp_goals == 1
        assert away.pp_opportunities == 4
        assert away.pp_pct == 0.25

    def test_pk_home(self):
        """Home team's PK = mirrored from away PP, pk_pct from API."""
        home, _ = TeamGameStats.pair_from_right_rail(RIGHT_RAIL, **PAIR_META)
        assert home.pk_goals_allowed == 1    # away pp_goals
        assert home.pk_opportunities == 4    # away pp_opps
        assert home.pk_pct == 0.75           # from API penaltyKillPctg homeValue

    def test_pk_away(self):
        _, away = TeamGameStats.pair_from_right_rail(RIGHT_RAIL, **PAIR_META)
        assert away.pk_goals_allowed == 2    # home pp_goals
        assert away.pk_opportunities == 3
        assert away.pk_pct == 0.667

    def test_faceoff(self):
        home, away = TeamGameStats.pair_from_right_rail(RIGHT_RAIL, **PAIR_META)
        assert home.faceoff_win_pct == 0.533
        assert away.faceoff_win_pct == 0.467

    def test_misc_stats_home(self):
        home, _ = TeamGameStats.pair_from_right_rail(RIGHT_RAIL, **PAIR_META)
        assert home.hits == 18
        assert home.blocked_shots == 9
        assert home.pim == 6
        assert home.giveaways == 5
        assert home.takeaways == 6

    def test_empty_right_rail(self):
        home, away = TeamGameStats.pair_from_right_rail(
            {}, **PAIR_META
        )
        assert home.shots_for is None
        assert home.pp_goals is None
        assert away.pk_pct is None


# ---------------------------------------------------------------------------
# GoalieGameLog model
# ---------------------------------------------------------------------------

class TestGoalieGameLog:
    def _parse_home(self):
        return GoalieGameLog.list_from_boxscore_team(
            BOXSCORE["playerByGameStats"]["homeTeam"]["goalies"],
            team_abbr="TOR", opponent_abbr="BOS", home_away="home",
            game_id=2025020700, season="20252026", game_type=2,
            game_date="2026-01-15",
        )

    def _parse_away(self):
        return GoalieGameLog.list_from_boxscore_team(
            BOXSCORE["playerByGameStats"]["awayTeam"]["goalies"],
            team_abbr="BOS", opponent_abbr="TOR", home_away="away",
            game_id=2025020700, season="20252026", game_type=2,
            game_date="2026-01-15",
        )

    def test_home_single_goalie(self):
        logs = self._parse_home()
        assert len(logs) == 1
        g = logs[0]
        assert g.player_id == 8476341
        assert g.is_starter is True
        assert g.decision == "W"
        assert g.goals_against == 2
        assert g.shots_against == 30
        assert g.saves == 28
        assert g.save_pct == round(28 / 30, 4)
        assert g.shutout is False

    def test_away_two_goalies_starter_flagged(self):
        logs = self._parse_away()
        assert len(logs) == 2
        starters = [g for g in logs if g.is_starter]
        backups  = [g for g in logs if not g.is_starter]
        assert len(starters) == 1
        assert starters[0].player_id == 8479406   # more TOI: 42:00 > 17:54
        assert backups[0].player_id  == 8480786

    def test_shutout_flag(self):
        """A goalie with 0 goals against and shots > 0 should be marked shutout."""
        raw = [{"playerId": 1, "toi": "60:00", "decision": "W",
                "goalsAgainst": 0, "shotsAgainst": 25}]
        logs = GoalieGameLog.list_from_boxscore_team(
            raw, "TOR", "BOS", "home", 1, "20252026", 2, "2026-01-15"
        )
        assert logs[0].shutout is True
        assert logs[0].save_pct == 1.0

    def test_empty_goalies(self):
        logs = GoalieGameLog.list_from_boxscore_team(
            [], "TOR", "BOS", "home", 1, "20252026", 2, "2026-01-15"
        )
        assert logs == []


# ---------------------------------------------------------------------------
# GameStatsFetcher
# ---------------------------------------------------------------------------

class TestGameStatsFetcher:
    def test_fetch_game_returns_pair(self, tmp_config):
        client = _mock_client()
        fetcher = GameStatsFetcher(tmp_config, client=client)

        ts, gls = fetcher.fetch_game(**GAME_META)

        assert len(ts) == 2
        assert {t.team_abbr for t in ts} == {"TOR", "BOS"}
        assert len(gls) == 3   # 1 home goalie + 2 away goalies

    def test_fetch_game_calls_both_apis(self, tmp_config):
        client = _mock_client()
        fetcher = GameStatsFetcher(tmp_config, client=client)

        fetcher.fetch_game(**GAME_META)

        client.game_center.season_series_matchup.assert_called_once()
        client.game_center.boxscore.assert_called_once()

    def test_fetch_game_saves_cache(self, tmp_config):
        client = _mock_client()
        fetcher = GameStatsFetcher(tmp_config, client=client)

        fetcher.fetch_game(**GAME_META)

        cache = tmp_config.gamestats_dir / f"{GAME_META['game_id']}_stats.json"
        assert cache.exists()

    def test_fetch_game_saves_raw(self, tmp_config):
        client = _mock_client()
        fetcher = GameStatsFetcher(tmp_config, client=client)

        fetcher.fetch_game(**GAME_META)

        rr  = tmp_config.raw_dir / "gamestats" / f"raw_{GAME_META['game_id']}_right_rail.json"
        bs  = tmp_config.raw_dir / "gamestats" / f"raw_{GAME_META['game_id']}_boxscore.json"
        assert rr.exists()
        assert bs.exists()

    def test_fetch_game_cache_hit(self, tmp_config):
        client = _mock_client()
        fetcher = GameStatsFetcher(tmp_config, client=client)

        fetcher.fetch_game(**GAME_META)
        fetcher.fetch_game(**GAME_META)  # second call → from cache

        assert client.game_center.season_series_matchup.call_count == 1

    def test_fetch_game_force_bypasses_cache(self, tmp_config):
        client = _mock_client()
        fetcher = GameStatsFetcher(tmp_config, client=client)

        fetcher.fetch_game(**GAME_META)
        fetcher.fetch_game(**GAME_META, force=True)

        assert client.game_center.season_series_matchup.call_count == 2

    def test_cache_round_trip_team_stats(self, tmp_config):
        client = _mock_client()
        fetcher = GameStatsFetcher(tmp_config, client=client)

        original_ts, _ = fetcher.fetch_game(**GAME_META)
        cached_ts, _   = fetcher.fetch_game(**GAME_META)   # from cache

        orig_home = next(t for t in original_ts if t.home_away == "home")
        cach_home = next(t for t in cached_ts   if t.home_away == "home")

        assert cach_home.pp_goals       == orig_home.pp_goals
        assert cach_home.shots_for      == orig_home.shots_for
        assert cach_home.faceoff_win_pct == orig_home.faceoff_win_pct

    def test_fetch_games_skips_non_final(self, tmp_config):
        client = _mock_client()
        fetcher = GameStatsFetcher(tmp_config, client=client)

        game_records = [
            {"game_id": 1, "season": "20252026", "game_type": 2,
             "date": "2026-01-15", "home_team_abbr": "TOR", "away_team_abbr": "BOS",
             "home_score": 4, "away_score": 2, "game_state": "FINAL"},
            {"game_id": 2, "season": "20252026", "game_type": 2,
             "date": "2026-01-15", "home_team_abbr": "MTL", "away_team_abbr": "OTT",
             "home_score": None, "away_score": None, "game_state": "FUT"},
        ]

        ts, gls = fetcher.fetch_games(game_records)

        # Only FINAL game fetched → 2 team rows + goalies from FINAL game only
        assert all(t.team_abbr in {"TOR", "BOS"} for t in ts)
        assert client.game_center.season_series_matchup.call_count == 1


# ---------------------------------------------------------------------------
# Storage — team_game_stats table
# ---------------------------------------------------------------------------

class TestTeamGameStatsStorage:
    @pytest.fixture
    def db(self, tmp_config):
        return Database(tmp_config.db_path)

    @pytest.fixture
    def sample_stats(self):
        home, away = TeamGameStats.pair_from_right_rail(RIGHT_RAIL, **PAIR_META)
        return [home, away]

    def test_upsert_returns_count(self, db, sample_stats):
        assert db.upsert_team_game_stats(sample_stats) == 2

    def test_get_game_team_stats(self, db, sample_stats):
        db.upsert_team_game_stats(sample_stats)
        rows = db.get_game_team_stats(GAME_META["game_id"])
        assert len(rows) == 2

    def test_stored_fields(self, db, sample_stats):
        db.upsert_team_game_stats(sample_stats)
        rows = db.get_game_team_stats(GAME_META["game_id"])
        home = next(r for r in rows if r["home_away"] == "home")
        assert home["team_abbr"] == "TOR"
        assert home["goals"] == 4
        assert home["goals_allowed"] == 2
        assert home["shots_for"] == 34
        assert home["pp_goals"] == 2
        assert home["pp_opportunities"] == 3
        assert home["pk_pct"] == 0.75
        assert home["faceoff_win_pct"] == 0.533

    def test_upsert_idempotent(self, db, sample_stats):
        db.upsert_team_game_stats(sample_stats)
        db.upsert_team_game_stats(sample_stats)
        assert len(db.get_game_team_stats(GAME_META["game_id"])) == 2

    def test_get_team_game_stats_filter(self, db, sample_stats):
        db.upsert_team_game_stats(sample_stats)
        rows = db.get_team_game_stats("TOR", season="20252026")
        assert len(rows) == 1
        assert rows[0]["team_abbr"] == "TOR"


# ---------------------------------------------------------------------------
# Storage — goalie_game_logs table
# ---------------------------------------------------------------------------

class TestGoalieGameLogsStorage:
    @pytest.fixture
    def db(self, tmp_config):
        return Database(tmp_config.db_path)

    @pytest.fixture
    def sample_logs(self):
        home_logs = GoalieGameLog.list_from_boxscore_team(
            BOXSCORE["playerByGameStats"]["homeTeam"]["goalies"],
            "TOR", "BOS", "home",
            GAME_META["game_id"], GAME_META["season"], GAME_META["game_type"],
            GAME_META["game_date"],
        )
        away_logs = GoalieGameLog.list_from_boxscore_team(
            BOXSCORE["playerByGameStats"]["awayTeam"]["goalies"],
            "BOS", "TOR", "away",
            GAME_META["game_id"], GAME_META["season"], GAME_META["game_type"],
            GAME_META["game_date"],
        )
        return home_logs + away_logs

    def test_upsert_returns_count(self, db, sample_logs):
        assert db.upsert_goalie_game_logs(sample_logs) == 3

    def test_get_goalie_game_logs(self, db, sample_logs):
        db.upsert_goalie_game_logs(sample_logs)
        rows = db.get_goalie_game_logs(8476341)
        assert len(rows) == 1

    def test_starter_flag_stored(self, db, sample_logs):
        db.upsert_goalie_game_logs(sample_logs)
        rows = db.get_goalie_game_logs(8476341)
        assert rows[0]["is_starter"] == 1

    def test_backup_flag(self, db, sample_logs):
        db.upsert_goalie_game_logs(sample_logs)
        rows = db.get_goalie_game_logs(8480786)  # backup goalie
        assert rows[0]["is_starter"] == 0

    def test_save_pct_stored(self, db, sample_logs):
        db.upsert_goalie_game_logs(sample_logs)
        rows = db.get_goalie_game_logs(8476341)
        assert abs(rows[0]["save_pct"] - round(28 / 30, 4)) < 1e-6

    def test_shutout_false(self, db, sample_logs):
        db.upsert_goalie_game_logs(sample_logs)
        rows = db.get_goalie_game_logs(8476341)
        assert rows[0]["shutout"] == 0

    def test_starters_only_filter(self, db, sample_logs):
        db.upsert_goalie_game_logs(sample_logs)
        all_rows     = db.get_goalie_game_logs(8479406, starters_only=False)
        starter_rows = db.get_goalie_game_logs(8479406, starters_only=True)
        assert len(all_rows) == 1
        assert len(starter_rows) == 1  # 8479406 is a starter


# ---------------------------------------------------------------------------
# Goalie rolling stats view
# ---------------------------------------------------------------------------

def _make_goalie_log(player_id, game_id, game_date, goals_against, shots_against,
                     toi_sec=3600, is_starter=True):
    sv = shots_against - goals_against
    return GoalieGameLog(
        player_id=player_id, game_id=game_id, season="20252026", game_type=2,
        game_date=game_date, team_abbr="TOR", opponent_abbr="BOS",
        home_away="home", is_starter=is_starter,
        goals_against=goals_against, shots_against=shots_against, saves=sv,
        save_pct=round(sv / shots_against, 4) if shots_against else None,
        toi=None, toi_seconds=toi_sec, shutout=(goals_against == 0),
    )


class TestGoalieRollingStatsView:
    @pytest.fixture
    def db_with_logs(self, tmp_config):
        db = Database(tmp_config.db_path)
        logs = [
            _make_goalie_log(1001, 100, "2026-01-01", 2, 30),
            _make_goalie_log(1001, 101, "2026-01-03", 1, 25),
            _make_goalie_log(1001, 102, "2026-01-05", 3, 35),
            _make_goalie_log(1001, 103, "2026-01-07", 0, 28),
            _make_goalie_log(1001, 104, "2026-01-09", 2, 32),
            _make_goalie_log(1001, 105, "2026-01-11", 1, 27),
        ]
        db.upsert_goalie_game_logs(logs)
        return db

    def test_rolling_view_returns_rows(self, db_with_logs):
        rows = db_with_logs.get_goalie_rolling_stats(1001)
        assert len(rows) == 6

    def test_first_game_has_no_rolling(self, db_with_logs):
        rows = db_with_logs.get_goalie_rolling_stats(1001)
        first = next(r for r in rows if r["game_date"] == "2026-01-01")
        assert first["recent_5g_save_pct"] is None
        assert first["recent_5g_games_played"] == 0

    def test_rolling_after_enough_games(self, db_with_logs):
        rows = db_with_logs.get_goalie_rolling_stats(1001)
        # 6th game (Jan 11) should have rolling over 5 preceding
        sixth = next(r for r in rows if r["game_date"] == "2026-01-11")
        assert sixth["recent_5g_games_played"] == 5
        assert sixth["recent_5g_save_pct"] is not None

    def test_rolling_partial_window(self, db_with_logs):
        rows = db_with_logs.get_goalie_rolling_stats(1001)
        # 3rd game has only 2 preceding games
        third = next(r for r in rows if r["game_date"] == "2026-01-05")
        assert third["recent_5g_games_played"] == 2


# ---------------------------------------------------------------------------
# player_point_dataset view
# ---------------------------------------------------------------------------

def _insert_full_dataset(db):
    """Populate all four tables needed for the view to join."""
    from tests.conftest import RAW_SCHEDULE, RAW_ROSTER
    from nhl_pipeline.models import DailySchedule, TeamRoster, PlayerGameLog

    # Roster
    roster = TeamRoster.from_api(RAW_ROSTER, "TOR", "20252026")
    db.upsert_roster(roster)

    # Games
    schedule = DailySchedule.from_api(
        {**RAW_SCHEDULE, "games": [RAW_SCHEDULE["games"][0]]},
        "2026-01-15",
    )
    db.upsert_schedule(schedule)

    # Player game logs (for Matthews, game 2025020700)
    pgl = PlayerGameLog(
        player_id=8478402, game_id=2025020700, season="20252026", game_type=2,
        game_date="2026-01-15", team_abbr="TOR", opponent_abbr="BOS",
        home_away="home", goals=2, assists=1, points=3, plus_minus=2,
        shots=5, pim=0, shifts=22, toi="21:34", toi_seconds=1294,
        power_play_goals=1, power_play_points=2,
    )
    db.upsert_game_logs([pgl])

    # Team game stats
    home_ts, away_ts = TeamGameStats.pair_from_right_rail(
        RIGHT_RAIL, game_id=2025020700, season="20252026",
        game_date="2026-01-15", home_abbr="TOR", away_abbr="BOS",
        home_goals=4, away_goals=2,
    )
    db.upsert_team_game_stats([home_ts, away_ts])

    # Goalie game logs (BOS goalie — opposing goalie for TOR player)
    bos_goalie = GoalieGameLog(
        player_id=8479406, game_id=2025020700, season="20252026", game_type=2,
        game_date="2026-01-15", team_abbr="BOS", opponent_abbr="TOR",
        home_away="away", is_starter=True, decision="L",
        goals_against=4, shots_against=36, saves=32,
        save_pct=round(32 / 36, 4), toi="60:00", toi_seconds=3600,
        shutout=False,
    )
    db.upsert_goalie_game_logs([bos_goalie])


class TestPointDatasetView:
    @pytest.fixture
    def db(self, tmp_config):
        d = Database(tmp_config.db_path)
        _insert_full_dataset(d)
        return d

    def test_view_returns_rows(self, db):
        rows = db.get_point_dataset(season="20252026")
        # Only Matthews has a game log row inserted — view only returns
        # players who appear in player_game_logs
        assert len(rows) == 1

    def test_point_scored_target(self, db):
        rows = db.get_point_dataset(season="20252026")
        matthews = next(r for r in rows if r["player_id"] == 8478402)
        assert matthews["point_scored"] == 1     # 2G + 1A

    def test_point_scored_is_binary(self, db):
        """point_scored must be 0 or 1."""
        rows = db.get_point_dataset(season="20252026")
        for row in rows:
            assert row["point_scored"] in (0, 1)

    def test_team_context_joined(self, db):
        rows = db.get_point_dataset(season="20252026")
        matthews = next(r for r in rows if r["player_id"] == 8478402)
        assert matthews["team_goals"] == 4
        assert matthews["team_shots_for"] == 34
        assert abs(matthews["team_pp_pct"] - round(2 / 3, 4)) < 1e-4
        assert matthews["team_pk_pct"] == 0.75

    def test_opponent_context_joined(self, db):
        rows = db.get_point_dataset(season="20252026")
        matthews = next(r for r in rows if r["player_id"] == 8478402)
        assert matthews["opp_goals"] == 2
        assert matthews["opp_shots_for"] == 28

    def test_goalie_context_joined(self, db):
        rows = db.get_point_dataset(season="20252026")
        matthews = next(r for r in rows if r["player_id"] == 8478402)
        assert matthews["opp_goalie_id"] == 8479406
        assert matthews["opp_goalie_goals_against"] == 4
        assert matthews["opp_goalie_save_pct"] == round(32 / 36, 4)

    def test_goalies_excluded(self, db):
        """Goalies from the roster should not appear in the dataset."""
        rows = db.get_point_dataset(season="20252026")
        positions = {r["position"] for r in rows if r["position"]}
        assert "G" not in positions

    def test_rolling_columns_present(self, db):
        """player_point_dataset should include rolling player form columns."""
        rows = db.get_point_dataset(season="20252026")
        assert len(rows) >= 1
        row = rows[0]
        for col in ("player_points_last5", "player_points_last10",
                    "player_shots_last5", "player_TOI_last5",
                    "player_last5_games_played", "player_last10_games_played"):
            assert col in row, f"Missing column: {col}"

    def test_rolling_null_for_first_game(self, db):
        """Rolling stats are NULL for a player's very first game."""
        rows = db.get_point_dataset(season="20252026")
        matthews = next(r for r in rows if r["player_id"] == 8478402)
        # Only one game log inserted for Matthews → no preceding games
        assert matthews["player_points_last5"] is None
        assert matthews["player_last5_games_played"] == 0


# ---------------------------------------------------------------------------
# Player rolling stats view
# ---------------------------------------------------------------------------

def _make_player_log(player_id, game_id, game_date, goals, assists, shots,
                     toi_sec=1200, team="TOR", opp="BOS"):
    return PlayerGameLog(
        player_id=player_id, game_id=game_id, season="20252026", game_type=2,
        game_date=game_date, team_abbr=team, opponent_abbr=opp, home_away="home",
        goals=goals, assists=assists, points=goals + assists,
        plus_minus=0, shots=shots, pim=0, shifts=20,
        toi=None, toi_seconds=toi_sec,
    )


class TestPlayerRollingStatsView:
    @pytest.fixture
    def db_with_logs(self, tmp_config):
        db = Database(tmp_config.db_path)
        logs = [
            _make_player_log(999, 200, "2026-01-01", 1, 1, 4, toi_sec=1200),
            _make_player_log(999, 201, "2026-01-03", 0, 1, 3, toi_sec=1100),
            _make_player_log(999, 202, "2026-01-05", 2, 0, 6, toi_sec=1300),
            _make_player_log(999, 203, "2026-01-07", 0, 0, 2, toi_sec=900),
            _make_player_log(999, 204, "2026-01-09", 1, 2, 5, toi_sec=1400),
            _make_player_log(999, 205, "2026-01-11", 0, 1, 4, toi_sec=1250),
            _make_player_log(999, 206, "2026-01-13", 1, 0, 3, toi_sec=1150),
        ]
        db.upsert_game_logs(logs)
        return db

    def test_view_returns_all_rows(self, db_with_logs):
        rows = db_with_logs.get_player_rolling_stats(999)
        assert len(rows) == 7

    def test_first_game_no_rolling(self, db_with_logs):
        rows = db_with_logs.get_player_rolling_stats(999)
        first = next(r for r in rows if r["game_date"] == "2026-01-01")
        assert first["player_points_last5"] is None
        assert first["player_shots_last5"] is None
        assert first["player_TOI_last5"] is None
        assert first["player_last5_games_played"] == 0

    def test_partial_window_second_game(self, db_with_logs):
        rows = db_with_logs.get_player_rolling_stats(999)
        second = next(r for r in rows if r["game_date"] == "2026-01-03")
        # Only 1 preceding game
        assert second["player_last5_games_played"] == 1
        assert second["player_points_last5"] == pytest.approx(2.0, abs=0.01)
        assert second["player_shots_last5"] == pytest.approx(4.0, abs=0.01)

    def test_full_window_sixth_game(self, db_with_logs):
        rows = db_with_logs.get_player_rolling_stats(999)
        # 6th game (Jan 11) has 5 preceding games → full 5-game window
        sixth = next(r for r in rows if r["game_date"] == "2026-01-11")
        assert sixth["player_last5_games_played"] == 5
        # points over games 1-5: 2,1,2,0,3 = 8 → avg 1.6
        expected_pts = round((2 + 1 + 2 + 0 + 3) / 5, 3)
        assert sixth["player_points_last5"] == pytest.approx(expected_pts, abs=0.001)

    def test_last10_window(self, db_with_logs):
        rows = db_with_logs.get_player_rolling_stats(999)
        # 7th game (Jan 13) has 6 preceding games → last10 games_played == 6
        seventh = next(r for r in rows if r["game_date"] == "2026-01-13")
        assert seventh["player_last10_games_played"] == 6
        # last5 also capped at 5
        assert seventh["player_last5_games_played"] == 5

    def test_shots_rolling_value(self, db_with_logs):
        rows = db_with_logs.get_player_rolling_stats(999)
        # 3rd game (Jan 05) has 2 preceding: shots 4 and 3 → avg 3.5
        third = next(r for r in rows if r["game_date"] == "2026-01-05")
        assert third["player_shots_last5"] == pytest.approx(3.5, abs=0.01)

    def test_toi_rolling_value(self, db_with_logs):
        rows = db_with_logs.get_player_rolling_stats(999)
        # 3rd game has 2 preceding: toi_seconds 1200 and 1100 → avg 1150
        third = next(r for r in rows if r["game_date"] == "2026-01-05")
        assert third["player_TOI_last5"] == pytest.approx(1150.0, abs=0.5)

    def test_season_filter(self, db_with_logs):
        rows_filtered = db_with_logs.get_player_rolling_stats(999, season="20252026")
        rows_all = db_with_logs.get_player_rolling_stats(999)
        assert len(rows_filtered) == len(rows_all)
