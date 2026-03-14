"""Typed data models for the NHL pipeline (dataclasses + JSON serialisation)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def _toi_to_seconds(toi: Optional[str]) -> Optional[int]:
    """Convert 'MM:SS' time-on-ice string to total seconds."""
    if not toi:
        return None
    try:
        minutes, seconds = toi.split(":")
        return int(minutes) * 60 + int(seconds)
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def to_json(obj: Any, path: Path, *, indent: int = 2) -> None:
    """Serialise a dataclass (or plain dict/list) to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        if hasattr(obj, "__dataclass_fields__"):
            json.dump(asdict(obj), fh, indent=indent, default=str)
        else:
            json.dump(obj, fh, indent=indent, default=str)


def from_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Game / Schedule models
# ---------------------------------------------------------------------------

@dataclass
class TeamRef:
    """Lightweight team reference as it appears inside a game record."""
    id: int
    abbr: str
    name: str
    score: Optional[int] = None


@dataclass
class Game:
    game_id: int
    season: str
    game_type: int          # 1=preseason, 2=regular, 3=playoff
    game_type_label: str
    date: str               # YYYY-MM-DD
    start_time_utc: str
    venue: str
    home_team: TeamRef
    away_team: TeamRef
    game_state: str         # FINAL, LIVE, FUT, etc.
    period: Optional[int] = None
    fetched_at: str = field(default_factory=_now_iso)

    @classmethod
    def from_api(cls, raw: dict, date: str) -> "Game":
        def _team(key: str) -> TeamRef:
            t = raw.get(key, {})
            return TeamRef(
                id=t.get("id", 0),
                abbr=t.get("abbrev", ""),
                name=(t.get("commonName") or t.get("placeName") or {}).get("default", ""),
                score=t.get("score"),
            )

        game_type = raw.get("gameType", 2)
        type_labels = {1: "preseason", 2: "regular", 3: "playoff"}

        return cls(
            game_id=raw.get("id", 0),
            season=str(raw.get("season", "")),
            game_type=game_type,
            game_type_label=type_labels.get(game_type, "unknown"),
            date=date,
            start_time_utc=raw.get("startTimeUTC", ""),
            venue=(raw.get("venue") or {}).get("default", ""),
            home_team=_team("homeTeam"),
            away_team=_team("awayTeam"),
            game_state=raw.get("gameState", ""),
            period=raw.get("period"),
        )


@dataclass
class DailySchedule:
    date: str
    number_of_games: int
    games: list[Game]
    next_date: Optional[str] = None
    prev_date: Optional[str] = None
    fetched_at: str = field(default_factory=_now_iso)

    @classmethod
    def from_api(cls, raw: dict, date: str) -> "DailySchedule":
        games = [Game.from_api(g, date) for g in raw.get("games", [])]
        return cls(
            date=date,
            number_of_games=raw.get("numberOfGames", len(games)),
            games=games,
            next_date=raw.get("nextStartDate"),
            prev_date=raw.get("previousStartDate"),
        )


# ---------------------------------------------------------------------------
# Roster / Player models
# ---------------------------------------------------------------------------

@dataclass
class Player:
    player_id: int
    first_name: str
    last_name: str
    full_name: str
    jersey_number: Optional[str]
    position: str           # C, LW, RW, D, G
    shoots_catches: Optional[str]
    height_inches: Optional[int]
    weight_lbs: Optional[int]
    birth_date: Optional[str]
    birth_city: Optional[str]
    birth_country: Optional[str]
    headshot_url: Optional[str] = None

    @classmethod
    def from_api(cls, raw: dict) -> "Player":
        first = (raw.get("firstName") or {}).get("default", "")
        last  = (raw.get("lastName")  or {}).get("default", "")
        return cls(
            player_id=raw.get("id", 0),
            first_name=first,
            last_name=last,
            full_name=f"{first} {last}".strip(),
            jersey_number=str(raw.get("sweaterNumber", "")) or None,
            position=raw.get("positionCode", ""),
            shoots_catches=raw.get("shootsCatches"),
            height_inches=raw.get("heightInInches"),
            weight_lbs=raw.get("weightInPounds"),
            birth_date=raw.get("birthDate"),
            birth_city=(raw.get("birthCity") or {}).get("default"),
            birth_country=raw.get("birthCountry"),
            headshot_url=raw.get("headshot"),
        )


@dataclass
class TeamRoster:
    team_abbr: str
    season: str
    forwards: list[Player]
    defensemen: list[Player]
    goalies: list[Player]
    fetched_at: str = field(default_factory=_now_iso)

    @property
    def all_players(self) -> list[Player]:
        return self.forwards + self.defensemen + self.goalies

    @property
    def total_players(self) -> int:
        return len(self.all_players)

    @classmethod
    def from_api(cls, raw: dict, team_abbr: str, season: str) -> "TeamRoster":
        def _parse(group: str) -> list[Player]:
            return [Player.from_api(p) for p in raw.get(group, [])]

        return cls(
            team_abbr=team_abbr,
            season=season,
            forwards=_parse("forwards"),
            defensemen=_parse("defensemen"),
            goalies=_parse("goalies"),
        )


# ---------------------------------------------------------------------------
# Player game log models
# ---------------------------------------------------------------------------

@dataclass
class PlayerGameLog:
    """One row = one player in one game.

    Fields present in the NHL API ``player/{id}/game-log/{season}/{game_type}``
    endpoint.  ``power_play_toi`` is *not* provided by the basic game-log
    endpoint; it remains ``None`` unless enriched from another source.
    """

    # Identity
    player_id: int
    game_id: int
    season: str
    game_type: int          # 2 = regular, 3 = playoff

    # Game context
    game_date: str          # YYYY-MM-DD
    team_abbr: str
    opponent_abbr: str
    home_away: str          # "home" or "away"  (API gives "H" / "R")

    # Skater stats (None for goalies — goalies have a separate log shape)
    goals: Optional[int] = None
    assists: Optional[int] = None
    points: Optional[int] = None
    plus_minus: Optional[int] = None
    shots: Optional[int] = None
    pim: Optional[int] = None
    shifts: Optional[int] = None

    # Time on ice
    toi: Optional[str] = None           # raw "MM:SS" string
    toi_seconds: Optional[int] = None   # derived integer

    # Power play
    power_play_goals: Optional[int] = None
    power_play_points: Optional[int] = None
    power_play_toi: Optional[str] = None         # not in basic log; None by default
    power_play_toi_seconds: Optional[int] = None

    # Special teams / misc
    game_winning_goals: Optional[int] = None
    ot_goals: Optional[int] = None
    shorthanded_goals: Optional[int] = None
    shorthanded_points: Optional[int] = None

    fetched_at: str = field(default_factory=_now_iso)

    @classmethod
    def from_api(
        cls,
        raw: dict,
        player_id: int,
        season: str,
        game_type: int,
    ) -> "PlayerGameLog":
        toi_str = raw.get("toi")
        pp_toi_str = raw.get("powerPlayToi")  # present in some enriched endpoints
        home_away_flag = raw.get("homeRoadFlag", "")
        home_away = "home" if home_away_flag == "H" else "away"

        return cls(
            player_id=player_id,
            game_id=raw.get("gameId", 0),
            season=season,
            game_type=game_type,
            game_date=raw.get("gameDate", ""),
            team_abbr=raw.get("teamAbbrev", ""),
            opponent_abbr=raw.get("opponentAbbrev", ""),
            home_away=home_away,
            goals=raw.get("goals"),
            assists=raw.get("assists"),
            points=raw.get("points"),
            plus_minus=raw.get("plusMinus"),
            shots=raw.get("shots"),
            pim=raw.get("pim"),
            shifts=raw.get("shifts"),
            toi=toi_str,
            toi_seconds=_toi_to_seconds(toi_str),
            power_play_goals=raw.get("powerPlayGoals"),
            power_play_points=raw.get("powerPlayPoints"),
            power_play_toi=pp_toi_str,
            power_play_toi_seconds=_toi_to_seconds(pp_toi_str),
            game_winning_goals=raw.get("gameWinningGoals"),
            ot_goals=raw.get("otGoals"),
            shorthanded_goals=raw.get("shorthandedGoals"),
            shorthanded_points=raw.get("shorthandedPoints"),
        )


@dataclass
class PlayerSeasonGameLog:
    """All game logs for one player in one season."""
    player_id: int
    season: str
    game_type: int
    entries: list[PlayerGameLog]
    fetched_at: str = field(default_factory=_now_iso)

    @property
    def total_games(self) -> int:
        return len(self.entries)


# ---------------------------------------------------------------------------
# Pipeline run summary
# ---------------------------------------------------------------------------

@dataclass
class PipelineRun:
    run_id: str
    started_at: str
    finished_at: Optional[str] = None
    date_fetched: Optional[str] = None
    schedules_ok: int = 0
    rosters_ok: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0
