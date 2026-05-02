"""Profils utilisateur — notes dynamiques maintenues par le modèle."""

import logging

from common.dataio import CogData, DictTableBuilder

logger = logging.getLogger("profiles")

# Format attendu des notes : lignes "[catégorie] info"
# Ex: "[identité] Théo, 24 ans, dev à Lyon"
#     "[préférences] Déteste les films de zombies, aime la techno"
#     "[projets] Travaille sur un jeu indie en Godot"
# Les notes sont maintenues exclusivement par le modèle via update_user_notes.

NOTES_MAX = 2000
NOTES_KEEP_HEAD = 800    # infos stables (début) toujours conservées
NOTES_KEEP_TAIL = 1193   # infos récentes (fin) — 2000 - 800 - 7 (séparateur \n[…]\n)


class ProfileStore:
    """Stockage des notes utilisateur. Utilise CogData chat / table globale."""

    def __init__(self):
        self._data = CogData("chat")
        self._data.set_builders(
            "global",
            DictTableBuilder("user_profiles"),
        )
        self._db = self._data.get("global")

    def get_notes(self, user_id: int) -> str:
        s = self._db.settings("user_profiles")
        return s.get(f"notes_{user_id}", default="") or ""

    def set_notes(self, user_id: int, text: str) -> None:
        s = self._db.settings("user_profiles")
        s.set(f"notes_{user_id}", text)

    def append_notes(self, user_id: int, addition: str) -> None:
        """Ajoute des infos. Si la limite est dépassée, conserve le début
        (infos fondamentales stables) et la fin (infos récentes)."""
        addition = addition.strip()
        if not addition:
            return
        current = self.get_notes(user_id)
        new = (current + "\n" + addition).strip() if current else addition
        if len(new) > NOTES_MAX:
            head = new[:NOTES_KEEP_HEAD]
            tail = new[-(NOTES_KEEP_TAIL):]
            # Couper sur une frontière de ligne pour éviter les fragments
            head_cut = head.rfind("\n")
            if head_cut > NOTES_KEEP_HEAD // 2:
                head = head[:head_cut]
            tail_cut = tail.find("\n")
            if 0 < tail_cut < 80:
                tail = tail[tail_cut + 1:]
            new = head + "\n[…]\n" + tail
        self.set_notes(user_id, new)

    def get_full(self, user_id: int) -> str:
        """Retourne les notes complètes, ou chaîne vide si inexistantes."""
        return self.get_notes(user_id)

    def get_all_with_notes(self) -> dict[int, str]:
        """Retourne {user_id: notes} pour tous les utilisateurs ayant des notes."""
        s = self._db.settings("user_profiles")
        result: dict[int, str] = {}
        for key, value in s.all().items():
            if key.startswith("notes_") and value:
                try:
                    uid = int(key[len("notes_"):])
                    result[uid] = value
                except ValueError:
                    pass
        return result

    def delete(self, user_id: int) -> None:
        s = self._db.settings("user_profiles")
        s.delete(f"notes_{user_id}")
