"""Where the experiments write: what the deck loads under site/data/talk/ (published by site/deploy.sh, never
committed), and their sweeps and exports under site/.experiments/ (local, ignored by git)."""

from pathlib import Path

SITE = Path(__file__).resolve().parents[3]
DECK_DATA = SITE / "data/talk"
WORK = SITE / ".experiments"
