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
# Team game stats (one row per team per game)
# ---------------------------------------------------------------------------

def _parse_pp(raw_value: Optional[str]) -> tuple[Optional[int], Optional[int], Optional[float]]:
    """Parse a power-play string like ``'2/4'`` into (goals, opportunities, pct).

    Returns ``(None, None, None)`` if the value is missing or unparseable.
    """
    if not raw_value or not isinstance(raw_value, str) or "/" not in raw_value:
        return None, None, None
    try:
        goals_str, opps_str = raw_value.split("/")
        goals = int(goals_str)
        opps  = int(opps_str)
        pct   = round(goals / opps, 4) if opps > 0 else 0.0
        return goals, opps, pct
    except (ValueError, ZeroDivisionError):
        return None, None, None


@dataclass
class TeamGameStats:
    """Per-team statistics for a single game.

    Populated from:
    - **right-rail** (``gamecenter/{id}/right-rail``): SOG, faceoff%,
      power-play string (``"goals/opportunities"``), PK%, hits, blocked shots,
      PIM, giveaways, takeaways.
    - **games table** (already stored in DB): goals / goals_allowed.

    Two rows are produced per game — one for each team.
    """

    game_id: int
    season: str
    game_date: str
    team_abbr: str
    opponent_abbr: str
    home_away: str          # "home" | "away"

    # Scoring
    goals: Optional[int] = None
    goals_allowed: Optional[int] = None

    # Shot volume
    shots_for: Optional[int] = None
    shots_against: Optional[int] = None

    # Power play
    pp_goals: Optional[int] = None
    pp_opportunities: Optional[int] = None
    pp_pct: Optional[float] = None   # derived: pp_goals / pp_opportunities

    # Penalty kill (mirrored from opponent PP)
    pk_goals_allowed: Optional[int] = None   # = opponent pp_goals
    pk_opportunities: Optional[int] = None   # = opponent pp_opportunities
    pk_pct: Optional[float] = None           # from API

    # Advanced context
    faceoff_win_pct: Optional[float] = None
    hits: Optional[int] = None
    blocked_shots: Optional[int] = None
    pim: Optional[int] = None
    giveaways: Optional[int] = None
    takeaways: Optional[int] = None

    fetched_at: str = field(default_factory=_now_iso)

    @classmethod
    def pair_from_right_rail(
        cls,
        raw: dict,
        game_id: int,
        season: str,
        game_date: str,
        home_abbr: str,
        away_abbr: str,
        home_goals: Optional[int] = None,
        away_goals: Optional[int] = None,
    ) -> tuple["TeamGameStats", "TeamGameStats"]:
        """Build two ``TeamGameStats`` rows (home + away) from a right-rail payload.

        Args:
            raw:        Parsed JSON of the right-rail response.
            game_id:    NHL game ID.
            season:     Season string (e.g. ``"20252026"``).
            game_date:  Date string (``YYYY-MM-DD``).
            home_abbr:  Home team abbreviation.
            away_abbr:  Away team abbreviation.
            home_goals: Final score for home team (from games table).
            away_goals: Final score for away team (from games table).
        """
        # Flatten teamGameStats array → {category: (away_val, home_val)}
        stat: dict[str, tuple] = {}
        for item in raw.get("teamGameStats", []):
            cat = item.get("category", "")
            stat[cat] = (item.get("awayValue"), item.get("homeValue"))

        def _float(val) -> Optional[float]:
            try:
                return float(val) if val is not None else None
            except (TypeError, ValueError):
                return None

        def _int(val) -> Optional[int]:
            try:
                return int(val) if val is not None else None
            except (TypeError, ValueError):
                return None

        sog_away,   sog_home   = stat.get("sog",                 (None, None))
        fow_away,   fow_home   = stat.get("faceoffWinningPctg",  (None, None))
        pp_away_s,  pp_home_s  = stat.get("powerPlay",           (None, None))
        pk_away,    pk_home    = stat.get("penaltyKillPctg",      (None, None))
        hits_away,  hits_home  = stat.get("hits",                (None, None))
        blk_away,   blk_home   = stat.get("blockedShots",        (None, None))
        pim_away,   pim_home   = stat.get("pim",                 (None, None))
        give_away,  give_home  = stat.get("giveaways",           (None, None))
        take_away,  take_home  = stat.get("takeaways",           (None, None))

        pp_away_g, pp_away_o, pp_away_p = _parse_pp(pp_away_s)
        pp_home_g, pp_home_o, pp_home_p = _parse_pp(pp_home_s)

        home = cls(
            game_id=game_id, season=season, game_date=game_date,
            team_abbr=home_abbr, opponent_abbr=away_abbr, home_away="home",
            goals=home_goals, goals_allowed=away_goals,
            shots_for=_int(sog_home), shots_against=_int(sog_away),
            pp_goals=pp_home_g, pp_opportunities=pp_home_o, pp_pct=pp_home_p,
            pk_goals_allowed=pp_away_g, pk_opportunities=pp_away_o,
            pk_pct=_float(pk_home),
            faceoff_win_pct=_float(fow_home),
            hits=_int(hits_home), blocked_shots=_int(blk_home),
            pim=_int(pim_home), giveaways=_int(give_home), takeaways=_int(take_home),
        )
        away = cls(
            game_id=game_id, season=season, game_date=game_date,
            team_abbr=away_abbr, opponent_abbr=home_abbr, home_away="away",
            goals=away_goals, goals_allowed=home_goals,
            shots_for=_int(sog_away), shots_against=_int(sog_home),
            pp_goals=pp_away_g, pp_opportunities=pp_away_o, pp_pct=pp_away_p,
            pk_goals_allowed=pp_home_g, pk_opportunities=pp_home_o,
            pk_pct=_float(pk_away),
            faceoff_win_pct=_float(fow_away),
            hits=_int(hits_away), blocked_shots=_int(blk_away),
            pim=_int(pim_away), giveaways=_int(give_away), takeaways=_int(take_away),
        )
        return home, away


# ---------------------------------------------------------------------------
# Goalie game log (one row per goalie per game)
# ---------------------------------------------------------------------------

@dataclass
class GoalieGameLog:
    """Per-goalie statistics for a single game, parsed from the boxscore.

    ``is_starter`` is ``True`` for the goalie with the most TOI in the game
    (typically the starter), which is the relevant record for modelling
    opposing-goalie difficulty.
    """

    player_id: int
    game_id: int
    season: str
    game_type: int
    game_date: str
    team_abbr: str
    opponent_abbr: str
    home_away: str          # "home" | "away"

    is_starter: bool        # True for the goalie with most TOI

    decision: Optional[str] = None      # "W", "L", "O", or None
    goals_against: Optional[int] = None
    shots_against: Optional[int] = None
    saves: Optional[int] = None
    save_pct: Optional[float] = None
    toi: Optional[str] = None
    toi_seconds: Optional[int] = None
    shutout: Optional[bool] = None

    fetched_at: str = field(default_factory=_now_iso)

    @classmethod
    def list_from_boxscore_team(
        cls,
        goalies_raw: list[dict],
        team_abbr: str,
        opponent_abbr: str,
        home_away: str,
        game_id: int,
        season: str,
        game_type: int,
        game_date: str,
    ) -> list["GoalieGameLog"]:
        """Parse all goalies for one team side from a boxscore response.

        The goalie with the highest TOI is flagged as ``is_starter=True``.
        """
        if not goalies_raw:
            return []

        # Determine starter by max TOI
        max_toi = max(
            (_toi_to_seconds(g.get("toi")) or 0) for g in goalies_raw
        )

        logs = []
        for raw in goalies_raw:
            toi_str = raw.get("toi")
            toi_sec = _toi_to_seconds(toi_str)
            shots   = raw.get("shotsAgainst", 0) or 0
            ga      = raw.get("goalsAgainst", 0) or 0
            sv      = shots - ga
            sv_pct  = round(sv / shots, 4) if shots > 0 else None

            logs.append(cls(
                player_id=raw.get("playerId", 0),
                game_id=game_id,
                season=season,
                game_type=game_type,
                game_date=game_date,
                team_abbr=team_abbr,
                opponent_abbr=opponent_abbr,
                home_away=home_away,
                is_starter=(toi_sec == max_toi and max_toi > 0),
                decision=raw.get("decision"),
                goals_against=ga,
                shots_against=shots,
                saves=sv,
                save_pct=sv_pct,
                toi=toi_str,
                toi_seconds=toi_sec,
                shutout=(ga == 0 and shots > 0),
            ))
        return logs


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
