#!/usr/bin/env python3
"""Command-line interface for the NHL 2025-2026 data pipeline.

Examples
--------
# Run today's full daily pipeline (schedule + all rosters):
    python run_pipeline.py daily

# Run for a specific date, no roster refresh:
    python run_pipeline.py daily --date 2026-01-15 --no-rosters

# Backfill schedules for a date range:
    python run_pipeline.py backfill --start 2025-10-07 --end 2025-10-31

# Fetch a single team's roster:
    python run_pipeline.py roster --team TOR

# Fetch game logs for all players on all teams (full season):
    python run_pipeline.py gamelogs

# Fetch game logs for specific teams only:
    python run_pipeline.py gamelogs --teams TOR BOS MTL

# Fetch game logs for a single player:
    python run_pipeline.py gamelogs --player 8478402

# Query game logs from the local DB:
    python run_pipeline.py query-logs --player 8478402
    python run_pipeline.py query-logs --team TOR

# Query games stored in the local DB:
    python run_pipeline.py games --date 2026-01-15
    python run_pipeline.py games --team TOR

# Show recent pipeline run history:
    python run_pipeline.py history
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime

from nhl_pipeline.config import PipelineConfig
from nhl_pipeline.pipeline import NHLPipeline


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NHL 2025-2026 season data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Bypass the file cache and always re-fetch from the API"
    )
    parser.add_argument(
        "--no-raw", action="store_true",
        help="Do not save raw API responses alongside parsed data"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- daily --------------------------------------------------------
    p_daily = sub.add_parser("daily", help="Run the full daily pipeline")
    p_daily.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Target date (default: today UTC)",
    )
    p_daily.add_argument(
        "--no-rosters", dest="rosters", action="store_false",
        help="Skip roster refresh",
    )
    p_daily.add_argument(
        "--teams", nargs="+", metavar="ABBR",
        help="Restrict roster refresh to these team abbreviations",
    )
    p_daily.add_argument(
        "--force-rosters", action="store_true",
        help="Re-fetch rosters even if cached",
    )

    # ---- backfill -----------------------------------------------------
    p_fill = sub.add_parser("backfill", help="Backfill schedules for a date range")
    p_fill.add_argument("--start", required=True, metavar="YYYY-MM-DD")
    p_fill.add_argument("--end",   required=True, metavar="YYYY-MM-DD")
    p_fill.add_argument(
        "--rosters", action="store_true",
        help="Also fetch rosters once before backfilling",
    )

    # ---- roster -------------------------------------------------------
    p_roster = sub.add_parser("roster", help="Fetch a single team roster")
    p_roster.add_argument("--team", required=True, metavar="ABBR", help="e.g. TOR")
    p_roster.add_argument("--force", action="store_true", help="Bypass cache")

    # ---- games (query) ------------------------------------------------
    p_games = sub.add_parser("games", help="Query persisted games from the local DB")
    grp = p_games.add_mutually_exclusive_group(required=True)
    grp.add_argument("--date", metavar="YYYY-MM-DD")
    grp.add_argument("--team", metavar="ABBR")

    # ---- gamelogs -----------------------------------------------------
    p_gl = sub.add_parser(
        "gamelogs",
        help="Fetch player game logs (full season, per-player per-game stats)",
    )
    p_gl.add_argument(
        "--teams", nargs="+", metavar="ABBR",
        help="Restrict to these team abbreviations (default: all 32 teams)",
    )
    p_gl.add_argument(
        "--player", type=int, metavar="PLAYER_ID",
        help="Fetch logs for a single player ID instead of teams",
    )
    p_gl.add_argument(
        "--playoffs", action="store_true",
        help="Also fetch playoff game logs",
    )
    p_gl.add_argument("--force", action="store_true", help="Bypass cache")

    # ---- query-logs ---------------------------------------------------
    p_ql = sub.add_parser("query-logs", help="Query persisted game logs from the local DB")
    grp2 = p_ql.add_mutually_exclusive_group(required=True)
    grp2.add_argument("--player", type=int, metavar="PLAYER_ID")
    grp2.add_argument("--team", metavar="ABBR")
    p_ql.add_argument(
        "--game-type", type=int, default=2, metavar="INT",
        help="1=pre, 2=regular (default), 3=playoff",
    )

    # ---- history ------------------------------------------------------
    p_hist = sub.add_parser("history", help="Show recent pipeline run history")
    p_hist.add_argument("--limit", type=int, default=10)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)

    config = PipelineConfig(
        use_cache=not args.no_cache,
        save_raw=not args.no_raw,
    )
    pipeline = NHLPipeline(config)

    # ------------------------------------------------------------------
    if args.command == "daily":
        run = pipeline.run_daily(
            date=args.date,
            fetch_rosters=args.rosters,
            teams=args.teams,
            force_rosters=args.force_rosters,
        )
        print(json.dumps({
            "run_id":      run.run_id,
            "date":        run.date_fetched,
            "schedules_ok": run.schedules_ok,
            "rosters_ok":  run.rosters_ok,
            "errors":      run.errors,
            "success":     run.success,
        }, indent=2))
        return 0 if run.success else 1

    # ------------------------------------------------------------------
    elif args.command == "backfill":
        runs = pipeline.run_date_range(
            args.start, args.end, fetch_rosters=args.rosters
        )
        ok    = sum(r.schedules_ok for r in runs)
        fails = sum(len(r.errors) for r in runs)
        print(json.dumps({
            "dates_processed": len(runs),
            "schedules_ok": ok,
            "total_errors": fails,
        }, indent=2))
        return 0 if fails == 0 else 1

    # ------------------------------------------------------------------
    elif args.command == "roster":
        try:
            roster = pipeline.fetch_roster(args.team.upper(), force=args.force)
            print(json.dumps({
                "team":    roster.team_abbr,
                "season":  roster.season,
                "total":   roster.total_players,
                "forwards":   len(roster.forwards),
                "defensemen": len(roster.defensemen),
                "goalies":    len(roster.goalies),
                "players": [
                    {"name": p.full_name, "number": p.jersey_number, "pos": p.position}
                    for p in sorted(roster.all_players, key=lambda p: p.last_name)
                ],
            }, indent=2))
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    # ------------------------------------------------------------------
    elif args.command == "games":
        if args.date:
            rows = pipeline.games_on(args.date)
            label = f"date={args.date}"
        else:
            rows = pipeline.team_schedule(args.team.upper())
            label = f"team={args.team.upper()}"
        print(json.dumps({"query": label, "count": len(rows), "games": rows}, indent=2))
        return 0

    # ------------------------------------------------------------------
    elif args.command == "gamelogs":
        if args.player:
            try:
                entries = pipeline.fetch_player_game_logs(args.player, force=args.force)
                print(json.dumps({
                    "player_id": args.player,
                    "games": len(entries),
                    "logs": [
                        {
                            "date": e.game_date,
                            "opponent": e.opponent_abbr,
                            "home_away": e.home_away,
                            "goals": e.goals,
                            "assists": e.assists,
                            "points": e.points,
                            "shots": e.shots,
                            "toi": e.toi,
                            "pp_goals": e.power_play_goals,
                            "pp_points": e.power_play_points,
                            "gwg": e.game_winning_goals,
                        }
                        for e in entries
                    ],
                }, indent=2))
                return 0
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1
        else:
            stats = pipeline.fetch_all_game_logs(
                teams=args.teams,
                include_playoffs=args.playoffs,
                force=args.force,
            )
            total = sum(stats.values())
            print(json.dumps({
                "teams_processed": len(stats),
                "total_player_game_rows": total,
                "by_team": stats,
            }, indent=2))
            return 0

    # ------------------------------------------------------------------
    elif args.command == "query-logs":
        if args.player:
            rows = pipeline.player_game_logs(args.player, game_type=args.game_type)
            print(json.dumps({
                "player_id": args.player,
                "count": len(rows),
                "logs": rows,
            }, indent=2))
        else:
            rows = pipeline.team_game_logs(args.team.upper(), game_type=args.game_type)
            print(json.dumps({
                "team": args.team.upper(),
                "count": len(rows),
                "logs": rows,
            }, indent=2))
        return 0

    # ------------------------------------------------------------------
    elif args.command == "history":
        rows = pipeline.run_history(limit=args.limit)
        print(json.dumps(rows, indent=2))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
