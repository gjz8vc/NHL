"""Pipeline configuration for the NHL 2025-2026 season."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Season constants
# ---------------------------------------------------------------------------
CURRENT_SEASON = "20252026"
SEASON_START_DATE = "2025-10-07"
SEASON_END_DATE = "2026-04-18"

# All 32 NHL team abbreviations (2025-2026 roster)
ALL_TEAM_ABBRS: list[str] = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
    "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD",
    "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
    "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
]

# ---------------------------------------------------------------------------
# Retry / HTTP settings
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT_SECONDS: int = 30
MAX_RETRIES: int = 4
RETRY_BACKOFF_BASE: float = 2.0   # seconds — doubles each attempt
RATE_LIMIT_DELAY: float = 0.5     # seconds between requests

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(os.getenv("NHL_DATA_DIR", Path(__file__).parent.parent / "data"))

SCHEDULES_DIR  = BASE_DIR / "schedules"
ROSTERS_DIR    = BASE_DIR / "rosters"
GAMELOGS_DIR   = BASE_DIR / "gamelogs"
GAMESTATS_DIR  = BASE_DIR / "gamestats"
RAW_DIR        = BASE_DIR / "raw"
DB_PATH        = BASE_DIR / "nhl_pipeline.db"


@dataclass
class PipelineConfig:
    """Runtime configuration for the pipeline."""

    season: str = CURRENT_SEASON
    season_start: str = SEASON_START_DATE
    season_end: str = SEASON_END_DATE

    teams: list[str] = field(default_factory=lambda: list(ALL_TEAM_ABBRS))

    schedules_dir: Path  = SCHEDULES_DIR
    rosters_dir: Path    = ROSTERS_DIR
    gamelogs_dir: Path   = GAMELOGS_DIR
    gamestats_dir: Path  = GAMESTATS_DIR
    raw_dir: Path        = RAW_DIR
    db_path: Path        = DB_PATH

    timeout: int        = DEFAULT_TIMEOUT_SECONDS
    max_retries: int    = MAX_RETRIES
    backoff_base: float = RETRY_BACKOFF_BASE
    rate_limit_delay: float = RATE_LIMIT_DELAY

    # When True the pipeline persists raw API responses alongside parsed data
    save_raw: bool = True
    # When True skip a fetch if a file already exists for that date/team
    use_cache: bool = True

    def ensure_dirs(self) -> None:
        for d in (self.schedules_dir, self.rosters_dir, self.gamelogs_dir,
                  self.gamestats_dir, self.raw_dir):
            d.mkdir(parents=True, exist_ok=True)
