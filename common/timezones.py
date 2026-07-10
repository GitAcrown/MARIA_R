"""Fuseaux horaires partagés par les cogs."""

from datetime import datetime
import zoneinfo

PARIS_TZ = zoneinfo.ZoneInfo("Europe/Paris")

_WEEKDAYS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
_MONTHS_FR = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]


def format_french_date(dt: datetime) -> str:
    """Formate une date en français : « vendredi 10 juillet 2026 »."""
    return f"{_WEEKDAYS_FR[dt.weekday()]} {dt.day} {_MONTHS_FR[dt.month - 1]} {dt.year}"
