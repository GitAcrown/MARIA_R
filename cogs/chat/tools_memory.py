"""Outil LLM de consultation de la mémoire long terme (lecture seule)."""

from datetime import datetime, timezone
from typing import Optional

from common.llm import Tool, ToolCallRecord, ToolResponseRecord
from common.memory.store import VALID_CATEGORIES, MemoryStore

_MAX_RESULTS = 20


def build_memory_tools(store: MemoryStore) -> list[Tool]:
    """Construit l'outil search_memory (énumération / filtre, pas d'écriture)."""

    async def _tool_search_memory(tc: ToolCallRecord, ctx) -> ToolResponseRecord:
        if not ctx or not ctx.trigger_message or not ctx.trigger_message.guild:
            return ToolResponseRecord(
                tc.id, {"error": "Disponible uniquement sur un serveur"}, datetime.now(timezone.utc),
            )
        guild_id = ctx.trigger_message.guild.id
        args = tc.arguments or {}
        query = (args.get("query") or "").strip()
        category = (args.get("category") or "").strip().lower() or None
        if category and category not in VALID_CATEGORIES:
            category = None
        user_id: Optional[int] = None
        raw_uid = (args.get("user_id") or "").strip()
        if raw_uid:
            try:
                user_id = int(raw_uid)
            except ValueError:
                return ToolResponseRecord(
                    tc.id, {"error": "user_id invalide"}, datetime.now(timezone.utc),
                )
            # Cible un membre → mémoires perso uniquement
            category = "user"

        memories = store.search_active(
            guild_id,
            query=query,
            category=category,
            user_id=user_id,
            limit=_MAX_RESULTS,
        )
        items = [
            {
                "category": m.category,
                "user_id": str(m.user_id) if m.user_id else None,
                "content": m.content,
                "confidence": round(m.confidence, 2),
            }
            for m in memories
        ]
        return ToolResponseRecord(tc.id, {
            "count": len(items),
            "memories": items,
            "_llm_summary": (
                f"{len(items)} souvenir(s) trouvé(s)."
                if items
                else "Aucun souvenir correspondant."
            ),
        }, datetime.now(timezone.utc))

    return [
        Tool(
            name="search_memory",
            description=(
                "Mémoire long terme (lecture seule). Les PROFILS du prompt couvrent déjà "
                "auteur + mentions — ne pas rappeler pour ça. "
                "Pour : membre/sujet ABSENT des profils, énumérer, filtrer par mot-clé. "
                "Pas d'écriture (mémoire auto ou /moi). "
                "query optionnel ; category user|server|event ou omit ; user_id pour un membre."
            ),
            properties={
                "query": {
                    "type": "string",
                    "description": "Mot-clé dans le contenu (ex: anniversaire, café). Vide = tout.",
                },
                "category": {
                    "type": "string",
                    "enum": list(VALID_CATEGORIES),
                    "description": "Filtrer par catégorie. Omettre pour user+server+event.",
                },
                "user_id": {
                    "type": "string",
                    "description": "Id Discord du membre (mémoires user uniquement).",
                },
            },
            optional_props=["query", "category", "user_id"],
            function=_tool_search_memory,
        ),
    ]
