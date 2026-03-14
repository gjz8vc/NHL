"""Shared test fixtures."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from nhl_pipeline.config import PipelineConfig


# ---------------------------------------------------------------------------
# Fixture helpers: raw API response shapes (based on real NHL API payloads)
# ---------------------------------------------------------------------------

RAW_SCHEDULE = {
    "nextStartDate": "2026-01-16",
    "previousStartDate": "2026-01-14",
    "numberOfGames": 2,
    "games": [
        {
            "id": 2025020700,
            "season": 20252026,
            "gameType": 2,
            "startTimeUTC": "2026-01-15T18:00:00Z",
            "venue": {"default": "Scotiabank Arena"},
            "homeTeam": {
                "id": 10,
                "abbrev": "TOR",
                "commonName": {"default": "Maple Leafs"},
                "score": 4,
            },
            "awayTeam": {
                "id": 6,
                "abbrev": "BOS",
                "commonName": {"default": "Bruins"},
                "score": 2,
            },
            "gameState": "FINAL",
            "period": 3,
        },
        {
            "id": 2025020701,
            "season": 20252026,
            "gameType": 2,
            "startTimeUTC": "2026-01-15T21:00:00Z",
            "venue": {"default": "Rogers Arena"},
            "homeTeam": {
                "id": 23,
                "abbrev": "VAN",
                "commonName": {"default": "Canucks"},
                "score": None,
            },
            "awayTeam": {
                "id": 22,
                "abbrev": "EDM",
                "commonName": {"default": "Oilers"},
                "score": None,
            },
            "gameState": "FUT",
            "period": None,
        },
    ],
}

RAW_ROSTER = {
    "forwards": [
        {
            "id": 8478402,
            "firstName": {"default": "Auston"},
            "lastName": {"default": "Matthews"},
            "sweaterNumber": 34,
            "positionCode": "C",
            "shootsCatches": "L",
            "heightInInches": 75,
            "weightInPounds": 220,
            "birthDate": "1997-09-17",
            "birthCity": {"default": "San Ramon"},
            "birthCountry": "USA",
            "headshot": "https://assets.nhle.com/mugs/nhl/20252026/TOR/8478402.png",
        },
        {
            "id": 8479318,
            "firstName": {"default": "Mitch"},
            "lastName": {"default": "Marner"},
            "sweaterNumber": 16,
            "positionCode": "RW",
            "shootsCatches": "R",
            "heightInInches": 72,
            "weightInPounds": 175,
            "birthDate": "1997-05-05",
            "birthCity": {"default": "Markham"},
            "birthCountry": "CAN",
            "headshot": "https://assets.nhle.com/mugs/nhl/20252026/TOR/8479318.png",
        },
    ],
    "defensemen": [
        {
            "id": 8480768,
            "firstName": {"default": "Morgan"},
            "lastName": {"default": "Rielly"},
            "sweaterNumber": 44,
            "positionCode": "D",
            "shootsCatches": "L",
            "heightInInches": 73,
            "weightInPounds": 195,
            "birthDate": "1994-03-09",
            "birthCity": {"default": "Vancouver"},
            "birthCountry": "CAN",
            "headshot": "https://assets.nhle.com/mugs/nhl/20252026/TOR/8480768.png",
        },
    ],
    "goalies": [
        {
            "id": 8476341,
            "firstName": {"default": "Joseph"},
            "lastName": {"default": "Woll"},
            "sweaterNumber": 60,
            "positionCode": "G",
            "shootsCatches": "L",
            "heightInInches": 74,
            "weightInPounds": 197,
            "birthDate": "1998-07-12",
            "birthCity": {"default": "Dardenne Prairie"},
            "birthCountry": "USA",
            "headshot": "https://assets.nhle.com/mugs/nhl/20252026/TOR/8476341.png",
        },
    ],
}


@pytest.fixture
def tmp_config(tmp_path: Path) -> PipelineConfig:
    """A PipelineConfig wired to a temporary directory."""
    cfg = PipelineConfig(
        schedules_dir=tmp_path / "schedules",
        rosters_dir=tmp_path / "rosters",
        gamelogs_dir=tmp_path / "gamelogs",
        gamestats_dir=tmp_path / "gamestats",
        raw_dir=tmp_path / "raw",
        db_path=tmp_path / "test.db",
        use_cache=True,
        save_raw=True,
    )
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def raw_schedule() -> dict:
    return RAW_SCHEDULE


@pytest.fixture
def raw_roster() -> dict:
    return RAW_ROSTER
