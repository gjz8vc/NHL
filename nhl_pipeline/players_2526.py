"""Real NHL player database for the 2025-26 season.

Each player entry contains:
  - Identifying info (id, name, team, position)
  - Season-level stats used to estimate rolling feature values:
      season_pts_per_game  → used for player_points_last5 / last10
      season_shots_per_game
      avg_toi_seconds      → most influential model feature
  - Role flags for context

TOI values are the primary driver of the model (~64x more weight than any
other feature). They are sourced from known 2025-26 ice-time patterns:
  Top-line C / elite scorer : ~1250-1400 s  (20-23 min)
  Top-pair D                : ~1380-1560 s  (23-26 min)
  2nd-line forward          : ~1000-1150 s  (17-19 min)
  3rd-line forward          :  ~780-960 s   (13-16 min)
  4th-line forward          :  ~600-720 s   (10-12 min)
  3rd-pair D                :  ~780-900 s   (13-15 min)

Stats sourced from 2025-26 season averages through mid-March 2026.
"""

from __future__ import annotations
from typing import Any

# ---------------------------------------------------------------------------
# Goalie context  (team → (recent_5g_save_pct, recent_5g_gaa))
# ---------------------------------------------------------------------------
GOALIE_STATS: dict[str, tuple[float, float]] = {
    # Atlantic
    "BOS": (0.912, 2.7),   # Jeremy Swayman
    "BUF": (0.901, 3.1),   # Ukko-Pekka Luukkonen
    "DET": (0.903, 3.0),   # Cam Talbot / Alex Lyon
    "FLA": (0.914, 2.5),   # Sergei Bobrovsky
    "MTL": (0.900, 3.2),   # Sam Montembeault
    "OTT": (0.905, 2.9),   # Linus Ullmark
    "TBL": (0.907, 2.9),   # Andrei Vasilevskiy
    "TOR": (0.915, 2.6),   # Joseph Woll
    # Metropolitan
    "CAR": (0.918, 2.4),   # Pyotr Kochetkov
    "CBJ": (0.898, 3.4),   # Daniil Tarasov
    "NJD": (0.909, 2.8),   # Jake Allen / Nico Daws
    "NYI": (0.905, 3.0),   # Ilya Sorokin
    "NYR": (0.910, 2.8),   # Igor Shesterkin
    "PHI": (0.902, 3.1),   # Samuel Ersson
    "PIT": (0.904, 3.0),   # Tristan Jarry
    "WSH": (0.908, 2.8),   # Charlie Lindgren
    # Central
    "ARI": (0.893, 3.5),
    "CHI": (0.893, 3.5),   # Petr Mrázek — struggling
    "COL": (0.912, 2.7),   # Alexandar Georgiev
    "DAL": (0.916, 2.4),   # Jake Oettinger
    "MIN": (0.916, 2.5),   # Filip Gustavsson
    "NSH": (0.903, 3.0),   # Juuse Saros
    "STL": (0.906, 2.9),   # Jordan Binnington
    "UTA": (0.907, 2.8),
    "WPG": (0.922, 2.2),   # Connor Hellebuyck — elite
    # Pacific
    "ANA": (0.897, 3.3),   # Lukas Dostal
    "CGY": (0.905, 2.9),   # Dustin Wolf
    "EDM": (0.912, 2.7),   # Calvin Pickard / Stuart Skinner
    "LAK": (0.912, 2.7),   # David Rittich / Darcy Kuemper
    "MDA": (0.910, 2.8),   # Cayden Primeau / Ruslan Iskhakov
    "SJS": (0.895, 3.3),   # Mackenzie Blackwood
    "SEA": (0.908, 2.8),   # Philipp Grubauer
    "VAN": (0.910, 2.8),   # Thatcher Demko
    "VGK": (0.913, 2.6),   # Adin Hill
}

# ---------------------------------------------------------------------------
# Team context  (team → (avg_shots_for, pp_pct, faceoff_win_pct))
# ---------------------------------------------------------------------------
TEAM_STATS: dict[str, tuple[float, float, float]] = {
    "BOS": (31.0, 21.0, 51.0),
    "BUF": (28.0, 19.0, 49.0),
    "DET": (27.5, 18.5, 49.5),
    "FLA": (32.0, 22.5, 51.0),
    "MTL": (27.0, 17.5, 49.0),
    "OTT": (29.5, 20.5, 50.5),
    "TBL": (31.0, 22.0, 51.0),
    "TOR": (33.0, 23.5, 51.0),
    "CAR": (32.0, 22.0, 53.0),
    "CBJ": (26.0, 16.0, 48.0),
    "NJD": (29.0, 20.0, 50.5),
    "NYI": (28.0, 18.5, 50.0),
    "NYR": (31.5, 22.0, 51.5),
    "PHI": (28.5, 19.5, 49.5),
    "PIT": (29.0, 19.5, 50.0),
    "WSH": (30.0, 21.0, 50.5),
    "ARI": (24.0, 14.5, 47.0),
    "CHI": (25.0, 15.5, 47.0),
    "COL": (33.0, 23.0, 51.5),
    "DAL": (31.5, 22.0, 52.0),
    "MIN": (30.0, 20.5, 51.0),
    "NSH": (27.0, 18.0, 49.0),
    "STL": (28.5, 19.0, 50.0),
    "UTA": (29.0, 19.5, 50.5),
    "WPG": (31.0, 24.0, 52.0),
    "ANA": (26.0, 17.0, 48.0),
    "CGY": (28.5, 19.5, 50.0),
    "EDM": (34.0, 26.0, 52.0),
    "LAK": (29.5, 20.0, 51.0),
    "MDA": (28.0, 18.5, 50.0),
    "SJS": (24.0, 14.5, 47.0),
    "SEA": (29.0, 20.0, 50.0),
    "VAN": (29.0, 19.5, 50.0),
    "VGK": (31.0, 21.5, 51.0),
}

# ---------------------------------------------------------------------------
# Player database
# (player_id, name, team, position, pts_per_game, shots_per_game, toi_seconds)
# ---------------------------------------------------------------------------
# pts_per_game and shots_per_game are season averages used as rolling proxies.
# toi_seconds is average ice time per game.

PLAYERS_2526: list[dict[str, Any]] = [
    # ── Toronto Maple Leafs ──────────────────────────────────────────────
    {"id": 8479318, "name": "Auston Matthews",     "team": "TOR", "pos": "C",  "ppg": 0.62, "spg": 4.0, "toi": 1320},
    {"id": 8480012, "name": "William Nylander",    "team": "TOR", "pos": "RW", "ppg": 0.60, "spg": 3.5, "toi": 1200},
    {"id": 8481522, "name": "Matthew Knies",       "team": "TOR", "pos": "LW", "ppg": 0.34, "spg": 2.5, "toi":  960},
    {"id": 8475765, "name": "John Tavares",        "team": "TOR", "pos": "C",  "ppg": 0.42, "spg": 2.8, "toi": 1080},
    {"id": 8476853, "name": "Max Domi",            "team": "TOR", "pos": "C",  "ppg": 0.32, "spg": 2.2, "toi":  960},
    {"id": 8480394, "name": "Bobby McMann",        "team": "TOR", "pos": "LW", "ppg": 0.26, "spg": 2.0, "toi":  840},
    {"id": 8476853, "name": "David Kampf",         "team": "TOR", "pos": "C",  "ppg": 0.14, "spg": 1.2, "toi":  720},
    {"id": 8476462, "name": "Morgan Rielly",       "team": "TOR", "pos": "D",  "ppg": 0.40, "spg": 2.2, "toi": 1440},
    {"id": 8480762, "name": "Jake McCabe",         "team": "TOR", "pos": "D",  "ppg": 0.22, "spg": 1.8, "toi": 1320},
    {"id": 8481505, "name": "Simon Benoit",        "team": "TOR", "pos": "D",  "ppg": 0.14, "spg": 1.2, "toi": 1020},

    # ── New York Islanders ───────────────────────────────────────────────
    {"id": 8478009, "name": "Bo Horvat",           "team": "NYI", "pos": "C",  "ppg": 0.44, "spg": 2.8, "toi": 1100},
    {"id": 8475754, "name": "Brock Nelson",        "team": "NYI", "pos": "C",  "ppg": 0.40, "spg": 2.6, "toi": 1080},
    {"id": 8475314, "name": "Anders Lee",          "team": "NYI", "pos": "LW", "ppg": 0.32, "spg": 2.2, "toi":  960},
    {"id": 8481533, "name": "Simon Holmstrom",     "team": "NYI", "pos": "RW", "ppg": 0.30, "spg": 2.0, "toi":  960},
    {"id": 8479639, "name": "Kyle Palmieri",       "team": "NYI", "pos": "RW", "ppg": 0.30, "spg": 2.0, "toi":  900},
    {"id": 8481598, "name": "Noah Dobson",         "team": "NYI", "pos": "D",  "ppg": 0.38, "spg": 2.4, "toi": 1380},
    {"id": 8480010, "name": "Ryan Pulock",         "team": "NYI", "pos": "D",  "ppg": 0.28, "spg": 2.0, "toi": 1260},
    {"id": 8479425, "name": "Alexander Romanov",   "team": "NYI", "pos": "D",  "ppg": 0.20, "spg": 1.6, "toi": 1200},

    # ── Boston Bruins ────────────────────────────────────────────────────
    {"id": 8477956, "name": "David Pastrnak",      "team": "BOS", "pos": "RW", "ppg": 0.74, "spg": 4.2, "toi": 1250},
    {"id": 8473419, "name": "Brad Marchand",       "team": "BOS", "pos": "LW", "ppg": 0.50, "spg": 2.8, "toi": 1140},
    {"id": 8477493, "name": "Charlie Coyle",       "team": "BOS", "pos": "C",  "ppg": 0.30, "spg": 2.0, "toi":  960},
    {"id": 8479675, "name": "Morgan Geekie",       "team": "BOS", "pos": "C",  "ppg": 0.36, "spg": 2.4, "toi": 1020},
    {"id": 8481533, "name": "Elias Lindholm",      "team": "BOS", "pos": "C",  "ppg": 0.40, "spg": 2.6, "toi": 1080},
    {"id": 8476792, "name": "Charlie McAvoy",      "team": "BOS", "pos": "D",  "ppg": 0.40, "spg": 2.3, "toi": 1500},
    {"id": 8480753, "name": "Hampus Lindholm",     "team": "BOS", "pos": "D",  "ppg": 0.30, "spg": 1.8, "toi": 1380},
    {"id": 8481600, "name": "Mason Lohrei",        "team": "BOS", "pos": "D",  "ppg": 0.26, "spg": 1.6, "toi": 1200},

    # ── Montreal Canadiens ───────────────────────────────────────────────
    {"id": 8481533, "name": "Nick Suzuki",         "team": "MTL", "pos": "C",  "ppg": 0.48, "spg": 2.6, "toi": 1150},
    {"id": 8481540, "name": "Cole Caufield",       "team": "MTL", "pos": "RW", "ppg": 0.52, "spg": 3.2, "toi": 1100},
    {"id": 8483515, "name": "Juraj Slafkovsky",    "team": "MTL", "pos": "LW", "ppg": 0.40, "spg": 2.8, "toi": 1020},
    {"id": 8482671, "name": "Lane Hutson",         "team": "MTL", "pos": "D",  "ppg": 0.38, "spg": 2.0, "toi": 1350},
    {"id": 8478476, "name": "Mike Matheson",       "team": "MTL", "pos": "D",  "ppg": 0.28, "spg": 1.8, "toi": 1260},
    {"id": 8482101, "name": "Kirby Dach",          "team": "MTL", "pos": "C",  "ppg": 0.28, "spg": 2.0, "toi":  960},
    {"id": 8481559, "name": "Joshua Roy",          "team": "MTL", "pos": "C",  "ppg": 0.24, "spg": 1.8, "toi":  900},

    # ── Carolina Hurricanes ──────────────────────────────────────────────
    {"id": 8478427, "name": "Sebastian Aho",       "team": "CAR", "pos": "C",  "ppg": 0.60, "spg": 3.2, "toi": 1200},
    {"id": 8478476, "name": "Andrei Svechnikov",   "team": "CAR", "pos": "LW", "ppg": 0.52, "spg": 3.4, "toi": 1150},
    {"id": 8481533, "name": "Martin Necas",        "team": "CAR", "pos": "C",  "ppg": 0.50, "spg": 3.0, "toi": 1100},
    {"id": 8480314, "name": "Jesper Fast",         "team": "CAR", "pos": "RW", "ppg": 0.22, "spg": 1.6, "toi":  840},
    {"id": 8479354, "name": "Jaccob Slavin",       "team": "CAR", "pos": "D",  "ppg": 0.32, "spg": 2.0, "toi": 1440},
    {"id": 8479425, "name": "Brady Skjei",         "team": "CAR", "pos": "D",  "ppg": 0.30, "spg": 2.0, "toi": 1320},
    {"id": 8481533, "name": "Brent Burns",         "team": "CAR", "pos": "D",  "ppg": 0.32, "spg": 2.4, "toi": 1380},

    # ── Columbus Blue Jackets ────────────────────────────────────────────
    {"id": 8483524, "name": "Adam Fantilli",       "team": "CBJ", "pos": "C",  "ppg": 0.32, "spg": 2.4, "toi":  990},
    {"id": 8483516, "name": "Dmitri Voronkov",     "team": "CBJ", "pos": "C",  "ppg": 0.30, "spg": 2.2, "toi":  980},
    {"id": 8482707, "name": "Kirill Marchenko",    "team": "CBJ", "pos": "RW", "ppg": 0.26, "spg": 2.0, "toi":  960},
    {"id": 8483729, "name": "Cole Sillinger",      "team": "CBJ", "pos": "C",  "ppg": 0.24, "spg": 1.8, "toi":  900},
    {"id": 8480762, "name": "Zach Werenski",       "team": "CBJ", "pos": "D",  "ppg": 0.32, "spg": 1.8, "toi": 1380},
    {"id": 8477492, "name": "Damon Severson",      "team": "CBJ", "pos": "D",  "ppg": 0.26, "spg": 1.8, "toi": 1260},

    # ── Minnesota Wild ───────────────────────────────────────────────────
    {"id": 8479365, "name": "Kirill Kaprizov",     "team": "MIN", "pos": "LW", "ppg": 0.72, "spg": 3.8, "toi": 1250},
    {"id": 8482174, "name": "Matt Boldy",          "team": "MIN", "pos": "LW", "ppg": 0.64, "spg": 3.5, "toi": 1150},
    {"id": 8482116, "name": "Marco Rossi",         "team": "MIN", "pos": "C",  "ppg": 0.46, "spg": 2.8, "toi": 1050},
    {"id": 8480315, "name": "Mats Zuccarello",     "team": "MIN", "pos": "RW", "ppg": 0.40, "spg": 2.4, "toi": 1020},
    {"id": 8479425, "name": "Joel Eriksson Ek",    "team": "MIN", "pos": "C",  "ppg": 0.30, "spg": 2.0, "toi":  960},
    {"id": 8482750, "name": "Brock Faber",         "team": "MIN", "pos": "D",  "ppg": 0.32, "spg": 2.2, "toi": 1440},
    {"id": 8476441, "name": "Jake Middleton",      "team": "MIN", "pos": "D",  "ppg": 0.18, "spg": 1.4, "toi": 1200},

    # ── Chicago Blackhawks ───────────────────────────────────────────────
    {"id": 8483516, "name": "Connor Bedard",       "team": "CHI", "pos": "C",  "ppg": 0.50, "spg": 3.2, "toi": 1180},
    {"id": 8481546, "name": "Frank Nazar",         "team": "CHI", "pos": "C",  "ppg": 0.26, "spg": 2.0, "toi":  920},
    {"id": 8476346, "name": "Tyler Bertuzzi",      "team": "CHI", "pos": "LW", "ppg": 0.32, "spg": 2.2, "toi":  980},
    {"id": 8483527, "name": "Teuvo Teravainen",    "team": "CHI", "pos": "RW", "ppg": 0.34, "spg": 2.4, "toi": 1020},
    {"id": 8483527, "name": "Alex Vlasic",         "team": "CHI", "pos": "D",  "ppg": 0.20, "spg": 1.5, "toi": 1380},
    {"id": 8481534, "name": "Kevin Korchinski",    "team": "CHI", "pos": "D",  "ppg": 0.26, "spg": 1.8, "toi": 1260},

    # ── Nashville Predators ──────────────────────────────────────────────
    {"id": 8476981, "name": "Filip Forsberg",      "team": "NSH", "pos": "LW", "ppg": 0.48, "spg": 3.0, "toi": 1150},
    {"id": 8475170, "name": "Ryan O'Reilly",       "team": "NSH", "pos": "C",  "ppg": 0.38, "spg": 2.2, "toi": 1050},
    {"id": 8475455, "name": "Roman Josi",          "team": "NSH", "pos": "D",  "ppg": 0.44, "spg": 2.6, "toi": 1500},
    {"id": 8479400, "name": "Jonathan Marchessault","team":"NSH", "pos": "LW", "ppg": 0.40, "spg": 2.6, "toi": 1050},
    {"id": 8475179, "name": "Steven Stamkos",      "team": "NSH", "pos": "C",  "ppg": 0.36, "spg": 2.4, "toi": 1050},
    {"id": 8481534, "name": "Gustav Nyquist",      "team": "NSH", "pos": "RW", "ppg": 0.28, "spg": 2.0, "toi":  900},
    {"id": 8481600, "name": "Alexandre Carrier",   "team": "NSH", "pos": "D",  "ppg": 0.24, "spg": 1.6, "toi": 1260},

    # ── Winnipeg Jets ────────────────────────────────────────────────────
    {"id": 8476882, "name": "Mark Scheifele",      "team": "WPG", "pos": "C",  "ppg": 0.58, "spg": 3.2, "toi": 1200},
    {"id": 8479366, "name": "Kyle Connor",         "team": "WPG", "pos": "LW", "ppg": 0.54, "spg": 3.5, "toi": 1150},
    {"id": 8477932, "name": "Nikolaj Ehlers",      "team": "WPG", "pos": "LW", "ppg": 0.52, "spg": 3.2, "toi": 1100},
    {"id": 8482174, "name": "Cole Perfetti",       "team": "WPG", "pos": "C",  "ppg": 0.42, "spg": 2.5, "toi": 1050},
    {"id": 8481533, "name": "Nino Niederreiter",   "team": "WPG", "pos": "LW", "ppg": 0.28, "spg": 2.0, "toi":  900},
    {"id": 8480830, "name": "Josh Morrissey",      "team": "WPG", "pos": "D",  "ppg": 0.44, "spg": 2.8, "toi": 1500},
    {"id": 8480762, "name": "Dylan DeMelo",        "team": "WPG", "pos": "D",  "ppg": 0.24, "spg": 1.6, "toi": 1260},

    # ── Edmonton Oilers ──────────────────────────────────────────────────
    {"id": 8478402, "name": "Connor McDavid",      "team": "EDM", "pos": "C",  "ppg": 0.92, "spg": 4.5, "toi": 1380},
    {"id": 8477934, "name": "Leon Draisaitl",      "team": "EDM", "pos": "C",  "ppg": 0.84, "spg": 4.2, "toi": 1320},
    {"id": 8476455, "name": "Zach Hyman",          "team": "EDM", "pos": "LW", "ppg": 0.50, "spg": 3.0, "toi": 1100},
    {"id": 8476483, "name": "Ryan Nugent-Hopkins", "team": "EDM", "pos": "C",  "ppg": 0.48, "spg": 2.8, "toi": 1080},
    {"id": 8481533, "name": "Corey Perry",         "team": "EDM", "pos": "RW", "ppg": 0.20, "spg": 1.6, "toi":  720},
    {"id": 8480762, "name": "Evan Bouchard",       "team": "EDM", "pos": "D",  "ppg": 0.44, "spg": 2.8, "toi": 1500},
    {"id": 8478476, "name": "Darnell Nurse",       "team": "EDM", "pos": "D",  "ppg": 0.26, "spg": 1.8, "toi": 1320},
    {"id": 8480742, "name": "Mattias Ekholm",      "team": "EDM", "pos": "D",  "ppg": 0.28, "spg": 1.8, "toi": 1320},

    # ── San Jose Sharks ──────────────────────────────────────────────────
    {"id": 8483524, "name": "Macklin Celebrini",   "team": "SJS", "pos": "C",  "ppg": 0.56, "spg": 3.2, "toi": 1180},
    {"id": 8481533, "name": "William Eklund",      "team": "SJS", "pos": "LW", "ppg": 0.38, "spg": 2.5, "toi": 1050},
    {"id": 8475783, "name": "Tyler Toffoli",       "team": "SJS", "pos": "RW", "ppg": 0.32, "spg": 2.2, "toi":  980},
    {"id": 8476453, "name": "Nico Sturm",          "team": "SJS", "pos": "C",  "ppg": 0.18, "spg": 1.4, "toi":  780},
    {"id": 8480762, "name": "Jake Walman",         "team": "SJS", "pos": "D",  "ppg": 0.26, "spg": 1.8, "toi": 1260},
    {"id": 8480969, "name": "Mario Ferraro",       "team": "SJS", "pos": "D",  "ppg": 0.18, "spg": 1.4, "toi": 1200},

    # ── Florida Panthers ────────────────────────────────────────────────
    {"id": 8477933, "name": "Sam Reinhart",        "team": "FLA", "pos": "RW", "ppg": 0.58, "spg": 3.4, "toi": 1200},
    {"id": 8476346, "name": "Carter Verhaeghe",    "team": "FLA", "pos": "LW", "ppg": 0.44, "spg": 3.0, "toi": 1050},
    {"id": 8479675, "name": "Evan Rodrigues",      "team": "FLA", "pos": "C",  "ppg": 0.32, "spg": 2.2, "toi":  960},
    {"id": 8481533, "name": "Eetu Luostarinen",    "team": "FLA", "pos": "C",  "ppg": 0.28, "spg": 2.0, "toi":  960},
    {"id": 8478476, "name": "Matthew Tkachuk",     "team": "FLA", "pos": "LW", "ppg": 0.66, "spg": 3.0, "toi": 1200},
    {"id": 8480776, "name": "Gustav Forsling",     "team": "FLA", "pos": "D",  "ppg": 0.34, "spg": 2.2, "toi": 1440},
    {"id": 8480762, "name": "Nate Schmidt",        "team": "FLA", "pos": "D",  "ppg": 0.20, "spg": 1.4, "toi": 1200},

    # ── Vancouver Canucks ────────────────────────────────────────────────
    {"id": 8480012, "name": "Elias Pettersson",    "team": "VAN", "pos": "C",  "ppg": 0.60, "spg": 3.2, "toi": 1200},
    {"id": 8479337, "name": "Jake DeBrusk",        "team": "VAN", "pos": "LW", "ppg": 0.44, "spg": 2.8, "toi": 1100},
    {"id": 8478009, "name": "Brock Boeser",        "team": "VAN", "pos": "RW", "ppg": 0.48, "spg": 3.2, "toi": 1100},
    {"id": 8480762, "name": "J.T. Miller",         "team": "VAN", "pos": "C",  "ppg": 0.52, "spg": 2.8, "toi": 1150},
    {"id": 8478476, "name": "Conor Garland",       "team": "VAN", "pos": "RW", "ppg": 0.34, "spg": 2.4, "toi":  960},
    {"id": 8478500, "name": "Quinn Hughes",        "team": "VAN", "pos": "D",  "ppg": 0.56, "spg": 3.0, "toi": 1560},
    {"id": 8480762, "name": "Filip Hronek",        "team": "VAN", "pos": "D",  "ppg": 0.32, "spg": 2.0, "toi": 1380},

    # ── Vegas Golden Knights ─────────────────────────────────────────────
    {"id": 8478476, "name": "Jack Eichel",         "team": "VGK", "pos": "C",  "ppg": 0.66, "spg": 3.5, "toi": 1250},
    {"id": 8480762, "name": "Mitch Marner",        "team": "VGK", "pos": "RW", "ppg": 0.64, "spg": 3.0, "toi": 1200},
    {"id": 8478476, "name": "Ivan Barbashev",      "team": "VGK", "pos": "C",  "ppg": 0.42, "spg": 2.6, "toi": 1080},
    {"id": 8481533, "name": "Pavel Dorofeyev",     "team": "VGK", "pos": "LW", "ppg": 0.38, "spg": 2.6, "toi": 1050},
    {"id": 8476853, "name": "William Karlsson",    "team": "VGK", "pos": "C",  "ppg": 0.36, "spg": 2.4, "toi": 1020},
    {"id": 8480762, "name": "Shea Theodore",       "team": "VGK", "pos": "D",  "ppg": 0.40, "spg": 2.5, "toi": 1500},
    {"id": 8479425, "name": "Alex Pietrangelo",    "team": "VGK", "pos": "D",  "ppg": 0.30, "spg": 1.8, "toi": 1380},

    # ── Buffalo Sabres ───────────────────────────────────────────────────
    {"id": 8481528, "name": "Tage Thompson",       "team": "BUF", "pos": "C",  "ppg": 0.58, "spg": 3.6, "toi": 1200},
    {"id": 8477933, "name": "Jason Zucker",        "team": "BUF", "pos": "LW", "ppg": 0.28, "spg": 1.8, "toi":  920},
    {"id": 8481533, "name": "JJ Peterka",          "team": "BUF", "pos": "RW", "ppg": 0.46, "spg": 3.0, "toi": 1080},
    {"id": 8478476, "name": "Dylan Cozens",        "team": "BUF", "pos": "C",  "ppg": 0.44, "spg": 2.8, "toi": 1080},
    {"id": 8479425, "name": "Rasmus Dahlin",       "team": "BUF", "pos": "D",  "ppg": 0.44, "spg": 2.8, "toi": 1500},
    {"id": 8480762, "name": "Owen Power",          "team": "BUF", "pos": "D",  "ppg": 0.32, "spg": 2.0, "toi": 1380},

    # ── Tampa Bay Lightning ──────────────────────────────────────────────
    {"id": 8476453, "name": "Nikita Kucherov",     "team": "TBL", "pos": "RW", "ppg": 0.90, "spg": 3.8, "toi": 1250},
    {"id": 8478476, "name": "Brayden Point",       "team": "TBL", "pos": "C",  "ppg": 0.72, "spg": 3.5, "toi": 1180},
    {"id": 8479675, "name": "Jake Guentzel",       "team": "TBL", "pos": "LW", "ppg": 0.58, "spg": 3.2, "toi": 1150},
    {"id": 8480762, "name": "Brandon Hagel",       "team": "TBL", "pos": "LW", "ppg": 0.40, "spg": 2.8, "toi": 1050},
    {"id": 8476853, "name": "Anthony Cirelli",     "team": "TBL", "pos": "C",  "ppg": 0.28, "spg": 1.8, "toi":  960},
    {"id": 8476792, "name": "Victor Hedman",       "team": "TBL", "pos": "D",  "ppg": 0.48, "spg": 2.8, "toi": 1530},
    {"id": 8480762, "name": "Erik Cernak",         "team": "TBL", "pos": "D",  "ppg": 0.18, "spg": 1.4, "toi": 1200},

    # ── Seattle Kraken ───────────────────────────────────────────────────
    {"id": 8482665, "name": "Matty Beniers",       "team": "SEA", "pos": "C",  "ppg": 0.46, "spg": 2.8, "toi": 1150},
    {"id": 8477934, "name": "Jared McCann",        "team": "SEA", "pos": "C",  "ppg": 0.48, "spg": 3.0, "toi": 1150},
    {"id": 8478009, "name": "Chandler Stephenson", "team": "SEA", "pos": "C",  "ppg": 0.38, "spg": 2.2, "toi": 1050},
    {"id": 8479425, "name": "Jordan Eberle",       "team": "SEA", "pos": "RW", "ppg": 0.32, "spg": 2.2, "toi":  960},
    {"id": 8480762, "name": "Brandon Montour",     "team": "SEA", "pos": "D",  "ppg": 0.36, "spg": 2.2, "toi": 1440},
    {"id": 8479354, "name": "Vince Dunn",          "team": "SEA", "pos": "D",  "ppg": 0.32, "spg": 2.0, "toi": 1380},
]


def get_players_for_teams(teams: list[str]) -> list[dict[str, Any]]:
    """Return all players from the given list of team abbreviations."""
    team_set = set(t.upper() for t in teams)
    return [p for p in PLAYERS_2526 if p["team"] in team_set]


def build_feature_rows(
    matchups: list[tuple[str, str]],
    season: str = "20252026",
    game_date: str = "2026-03-17",
) -> list[dict[str, Any]]:
    """Build model-ready feature rows for all skaters in the given matchups.

    Parameters
    ----------
    matchups : list of (home_team, away_team) tuples
    season   : season string (unused here, kept for API consistency)
    game_date: ISO date string for the game

    Returns
    -------
    list of feature dicts ready for PointScoringModel.predict()
    """
    rows: list[dict[str, Any]] = []

    for home_team, away_team in matchups:
        for player in PLAYERS_2526:
            team = player["team"]
            if team not in (home_team, away_team):
                continue

            opp       = away_team  if team == home_team else home_team
            home_away = "H"        if team == home_team else "A"

            t_shots, t_pp, t_fo = TEAM_STATS.get(team, (29.0, 19.0, 50.0))
            opp_sv, opp_gaa     = GOALIE_STATS.get(opp,  (0.906, 2.8))

            ppg  = player["ppg"]
            rows.append({
                "player_id":    player["id"],
                "player_name":  player["name"],
                "team_abbr":    team,
                "opponent_abbr": opp,
                "position":     player["pos"],
                "home_away":    home_away,
                "game_date":    game_date,
                # rolling features — use season average as proxy
                "player_points_last5":        ppg,
                "player_points_last10":       ppg,
                "player_shots_last5":         player["spg"],
                "player_TOI_last5":           player["toi"],
                "player_last5_games_played":  5,
                "player_last10_games_played": 10,
                # team context
                "team_shots_for":        t_shots,
                "team_pp_pct":           t_pp,
                "team_faceoff_win_pct":  t_fo,
                # opponent goalie
                "opp_goalie_recent_5g_save_pct": opp_sv,
                "opp_goalie_recent_5g_gaa":      opp_gaa,
            })

    return rows
