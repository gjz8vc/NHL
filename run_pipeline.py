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

# Fetch team game stats + goalie logs for a specific date:
    python run_pipeline.py gamestats --date 2026-01-15

# Fetch team game stats + goalie logs for the whole season:
    python run_pipeline.py gamestats --season

# Export the modelling dataset (skater-game rows + binary target):
    python run_pipeline.py dataset
    python run_pipeline.py dataset --team TOR --out tor_dataset.json

# Query games stored in the local DB:
    python run_pipeline.py games --date 2026-01-15
    python run_pipeline.py games --team TOR

# Show recent pipeline run history:
    python run_pipeline.py history

# ── ML commands ──────────────────────────────────────────────────────────

# Seed DB with synthetic data (useful for offline testing / first-time setup):
    python run_pipeline.py seed
    python run_pipeline.py seed --games 800

# Train the point-scoring probability model on data in the DB:
    python run_pipeline.py train
    python run_pipeline.py train --model data/my_model.pkl

# Predict tonight's point scorers (requires a trained model):
    python run_pipeline.py predict
    python run_pipeline.py predict --date 2026-03-16 --top 25
    python run_pipeline.py predict --games "BOS NJD" "LAK NYR" --top 20

# Show players currently on point streaks:
    python run_pipeline.py streaks
    python run_pipeline.py streaks --min 3
    python run_pipeline.py streaks --tonight
    python run_pipeline.py streaks --date 2025-12-20 --min 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from collections import defaultdict

from nhl_pipeline.config import PipelineConfig, BASE_DIR
from nhl_pipeline.pipeline import NHLPipeline


def _build_real_name_map() -> dict[str, str]:
    """Return a mapping of seeded placeholders (e.g. 'C1_EDM') to real player names.

    Uses the players_2526.py database: for each (team, position) group, players
    are ranked by points-per-game descending so C1_EDM = highest-PPG centre on EDM.
    """
    from nhl_pipeline.players_2526 import PLAYERS_2526

    by_team_pos: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in PLAYERS_2526:
        by_team_pos[(p["team"], p["pos"])].append(p)

    name_map: dict[str, str] = {}
    for (team, pos), players in by_team_pos.items():
        for rank, player in enumerate(
            sorted(players, key=lambda x: x["ppg"], reverse=True), start=1
        ):
            name_map[f"{pos}{rank}_{team}"] = player["name"]
    return name_map

_DEFAULT_MODEL_PATH = BASE_DIR / "model.pkl"


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

    # ---- gamestats ----------------------------------------------------
    p_gs = sub.add_parser(
        "gamestats",
        help="Fetch per-game team stats and goalie logs (right-rail + boxscore)",
    )
    gs_grp = p_gs.add_mutually_exclusive_group(required=True)
    gs_grp.add_argument("--date",   metavar="YYYY-MM-DD",
                        help="Fetch stats for all FINAL games on this date")
    gs_grp.add_argument("--season", action="store_true",
                        help="Fetch stats for every FINAL game this season")
    p_gs.add_argument("--force", action="store_true", help="Bypass cache")

    # ---- dataset ------------------------------------------------------
    p_ds = sub.add_parser(
        "dataset",
        help="Export the player_point_dataset view (modelling-ready, with binary target)",
    )
    p_ds.add_argument("--team", metavar="ABBR", help="Filter to one team")
    p_ds.add_argument(
        "--out", metavar="FILE",
        help="Write JSON output to FILE instead of stdout",
    )

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

    # ---- seed ---------------------------------------------------------
    p_seed = sub.add_parser(
        "seed",
        help="Seed the DB with synthetic data for offline training / testing",
    )
    p_seed.add_argument(
        "--games", type=int, default=500,
        help="Number of games to simulate (default: 500, ~18k rows)",
    )

    # ---- train --------------------------------------------------------
    p_train = sub.add_parser(
        "train",
        help="Train the point-scoring probability model on data in the DB",
    )
    p_train.add_argument(
        "--model", metavar="FILE", default=None,
        help=f"Where to save the model pickle (default: {_DEFAULT_MODEL_PATH})",
    )
    p_train.add_argument(
        "--team", metavar="ABBR",
        help="Restrict training data to one team (for quick testing)",
    )

    # ---- predict ------------------------------------------------------
    p_pred = sub.add_parser(
        "predict",
        help="Predict point-scoring probabilities for tonight's skaters",
    )
    p_pred.add_argument(
        "--date", metavar="YYYY-MM-DD", default=None,
        help="Date to predict for (default: today UTC)",
    )
    p_pred.add_argument(
        "--model", metavar="FILE", default=None,
        help=f"Path to trained model pickle (default: {_DEFAULT_MODEL_PATH})",
    )
    p_pred.add_argument(
        "--top", type=int, default=30, metavar="N",
        help="Show top N players (default: 30)",
    )
    p_pred.add_argument(
        "--games", nargs="+", metavar="HOME AWAY",
        help=(
            "Override tonight's matchups when schedule isn't in DB. "
            "Provide pairs: --games 'BOS NJD' 'LAK NYR'"
        ),
    )
    p_pred.add_argument(
        "--json", dest="output_json", action="store_true",
        help="Output raw JSON instead of formatted table",
    )
    p_pred.add_argument(
        "--real-players", dest="real_players", action="store_true",
        help=(
            "Use the built-in 2025-26 real player database instead of DB rosters. "
            "Useful when the NHL API hasn't been backfilled yet."
        ),
    )

    # ---- streaks ------------------------------------------------------
    p_streak = sub.add_parser(
        "streaks",
        help="Show players currently on point streaks",
    )
    p_streak.add_argument(
        "--min", type=int, default=5, dest="min_streak", metavar="N",
        help="Minimum streak length (default: 5)",
    )
    p_streak.add_argument(
        "--date", metavar="YYYY-MM-DD", default=None,
        help="Check streaks as of this date (default: most recent data in DB)",
    )
    p_streak.add_argument(
        "--tonight", action="store_true",
        help="Only show players who are playing tonight (uses --date for game lookup)",
    )
    p_streak.add_argument(
        "--real-players", dest="real_players", action="store_true",
        help="Map DB placeholder names to real 2025-26 player names",
    )

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
    elif args.command == "gamestats":
        if args.date:
            ts_rows, gl_rows = pipeline.fetch_game_stats_for_date(
                args.date, force=args.force
            )
        else:
            ts_rows, gl_rows = pipeline.fetch_game_stats_for_season(force=args.force)
        print(json.dumps({
            "team_stat_rows":  ts_rows,
            "goalie_log_rows": gl_rows,
        }, indent=2))
        return 0

    # ------------------------------------------------------------------
    elif args.command == "dataset":
        rows = pipeline.point_dataset(
            team_abbr=args.team.upper() if args.team else None
        )
        payload = json.dumps({
            "count": len(rows),
            "columns": list(rows[0].keys()) if rows else [],
            "rows": rows,
        }, indent=2)
        if args.out:
            import pathlib
            pathlib.Path(args.out).write_text(payload)
            print(f"Wrote {len(rows)} rows to {args.out}")
        else:
            print(payload)
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

    # ------------------------------------------------------------------
    elif args.command == "seed":
        from nhl_pipeline.seeder import seed_database
        inserted = seed_database(config.db_path, n_games=args.games)
        print(json.dumps({
            "status":  "ok",
            "games_simulated": args.games,
            "rows_inserted":   inserted,
            "db_path": str(config.db_path),
        }, indent=2))
        return 0

    # ------------------------------------------------------------------
    elif args.command == "train":
        from nhl_pipeline.model import PointScoringModel
        model_path = Path(args.model) if args.model else _DEFAULT_MODEL_PATH

        rows = pipeline.point_dataset(
            team_abbr=args.team.upper() if args.team else None
        )
        if not rows:
            print(
                "ERROR: No training data found in the DB.\n"
                "Run  python run_pipeline.py seed  to generate synthetic data, or\n"
                "backfill real data first with the gamelogs / gamestats commands.",
                file=sys.stderr,
            )
            return 1

        model = PointScoringModel()
        metrics = model.train(rows)
        model.save(model_path)

        print(model.summary())
        print()
        print(json.dumps({
            "model_path": str(model_path),
            "metrics":    metrics,
        }, indent=2))
        return 0

    # ------------------------------------------------------------------
    elif args.command == "predict":
        from nhl_pipeline.predictor import TonightPredictor
        model_path = Path(args.model) if args.model else _DEFAULT_MODEL_PATH

        if not model_path.exists():
            print(
                f"ERROR: Model not found at {model_path}.\n"
                "Train a model first:  python run_pipeline.py train",
                file=sys.stderr,
            )
            return 1

        game_date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Parse --games overrides: each value is "HOME AWAY"
        game_overrides: list[tuple[str, str]] | None = None
        if args.games:
            game_overrides = []
            for pair in args.games:
                parts = pair.upper().split()
                if len(parts) != 2:
                    print(
                        f"ERROR: --games expects 'HOME AWAY' pairs, got: {pair!r}",
                        file=sys.stderr,
                    )
                    return 1
                game_overrides.append((parts[0], parts[1]))

        predictor = TonightPredictor(config.db_path, model_path)

        if args.real_players:
            # Use the curated 2025-26 real player database
            from nhl_pipeline.players_2526 import build_feature_rows

            # Resolve matchups from DB or --games overrides
            if game_overrides:
                matchups = game_overrides
            else:
                db_games = predictor._games_for_date(game_date)
                if not db_games:
                    print(
                        f"No games found in DB for {game_date}.\n"
                        "Pass --games 'HOME AWAY' pairs, e.g.: --games 'TOR NYI' 'BOS MTL'",
                        file=sys.stderr,
                    )
                    return 1
                matchups = [(g["home_team_abbr"], g["away_team_abbr"]) for g in db_games]

            feature_rows = build_feature_rows(matchups, game_date=game_date)
            if not feature_rows:
                print(
                    "No feature rows built — check that team abbreviations match players_2526.py",
                    file=sys.stderr,
                )
                return 1
            preds = predictor.predict_from_rows(feature_rows, top_n=args.top)
        else:
            preds = predictor.predict_date(
                game_date=game_date,
                game_overrides=game_overrides,
                top_n=args.top,
            )

        if not preds:
            print(
                f"No predictions generated for {game_date}.\n"
                "Use --real-players for the built-in 2025-26 player database, or\n"
                "pass --games HOME AWAY pairs when the schedule isn't in the DB.",
                file=sys.stderr,
            )
            return 1

        if args.output_json:
            print(json.dumps([{
                "rank":         i + 1,
                "player_name":  p.player_name,
                "player_id":    p.player_id,
                "team":         p.team_abbr,
                "opponent":     p.opponent_abbr,
                "position":     p.position,
                "home_away":    p.home_away,
                "score_prob":   round(p.score_prob, 4),
            } for i, p in enumerate(preds)], indent=2))
        else:
            print(f"\nPoint-scoring predictions for {game_date}")
            print(predictor.format_table(preds, top_n=args.top))

        return 0

    # ------------------------------------------------------------------
    elif args.command == "streaks":
        from datetime import datetime, timezone

        as_of = args.date  # may be None → uses all data in DB

        rows = pipeline.point_streaks(
            min_streak=args.min_streak,
            as_of_date=as_of,
        )

        if not rows:
            print(
                f"No players found on a {args.min_streak}+ game point streak"
                + (f" as of {as_of}" if as_of else "") + "."
            )
            return 0

        # Optional: filter to players whose team is in tonight's games
        if args.tonight:
            game_date = as_of or datetime.now(timezone.utc).strftime("%Y-%m-%d")
            tonight_games = pipeline.games_on(game_date)
            tonight_teams = {
                t
                for g in tonight_games
                for t in (g["home_team_abbr"], g["away_team_abbr"])
            }
            rows = [r for r in rows if r["team_abbr"] in tonight_teams]
            if not rows:
                print(
                    f"No streak players found playing on {game_date}. "
                    "Check that the schedule has been loaded for that date."
                )
                return 0

        name_map = _build_real_name_map() if args.real_players else {}

        # Header
        date_label = f" as of {as_of}" if as_of else ""
        tonight_label = " (tonight only)" if args.tonight else ""
        print(
            f"\n{args.min_streak}+ game point streaks{date_label}{tonight_label}\n"
        )
        print(
            f"  {'#':>3}  {'Player':<28} {'Team':>5} {'Pos':>4}  "
            f"{'Streak':>6}  {'Streak Start':>12}  {'Last Game':>12}"
        )
        print("  " + "-" * 74)
        for i, r in enumerate(rows, 1):
            if not r.get("last_game_date"):
                continue
            display_name = name_map.get(r["full_name"], r["full_name"])
            print(
                f"  {i:>3}  {display_name:<28} {r['team_abbr']:>5} {r['position']:>4}  "
                f"{r['streak_length']:>6}  {r['streak_start']:>12}  {r['last_game_date']:>12}"
            )
        print(f"\n  {len(rows)} player(s) on {args.min_streak}+ game point streaks")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
