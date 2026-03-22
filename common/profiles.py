"""Profils utilisateur — champ fixe (éditable) + notes dynamiques (via outil)."""

import logging
from pathlib import Path

from common.dataio import CogData, DictTableBuilder, Database

logger = logging.getLogger("profiles")

# Schéma : user_id -> (profile TEXT, dynamic_notes TEXT)
# profile = édité par l'utilisateur
# dynamic_notes = maintenu par le modèle principal via outil update_user_notes


class ProfileStore:
    """Stockage profils par utilisateur. Utilise CogData chat."""

    def __init__(self):
        self._data = CogData("chat")
        self._data.set_builders(
            "global",
            DictTableBuilder("user_profiles"),
        )
        self._db = self._data.get("global")

    def get_profile(self, user_id: int) -> str:
        """Profil fixe édité par l'utilisateur."""
        s = self._db.settings("user_profiles")
        return s.get(f"profile_{user_id}", default="") or ""

    def set_profile(self, user_id: int, text: str) -> None:
        s = self._db.settings("user_profiles")
        s.set(f"profile_{user_id}", text)

    def get_notes(self, user_id: int) -> str:
        """Notes dynamiques maintenues par le modèle."""
        s = self._db.settings("user_profiles")
        return s.get(f"notes_{user_id}", default="") or ""

    def set_notes(self, user_id: int, text: str) -> None:
        s = self._db.settings("user_profiles")
        s.set(f"notes_{user_id}", text)

    def append_notes(self, user_id: int, addition: str) -> None:
        """Ajoute au paragraphe dynamique (utilisé par l'outil)."""
        current = self.get_notes(user_id)
        new = (current + "\n" + addition).strip() if current else addition
        if len(new) > 1500:
            new = new[-1500:]  # garder les 1500 derniers caractères
        self.set_notes(user_id, new)

    def get_full(self, user_id: int) -> str:
        """Profil complet = fixe + notes."""
        p = self.get_profile(user_id)
        n = self.get_notes(user_id)
        if not p and not n:
            return ""
        if not n:
            return p
        if not p:
            return f"[Notes récentes]\n{n}"
        return f"{p}\n\n[Notes récentes]\n{n}"

    def delete(self, user_id: int) -> None:
        s = self._db.settings("user_profiles")
        s.delete(f"profile_{user_id}")
        s.delete(f"notes_{user_id}")
