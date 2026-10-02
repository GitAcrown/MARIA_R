"""Constantes du cog Chat."""

MODEL_MAIN = "gpt-6-luna"

# trim() déduit aussi le prompt développeur du budget tokens.
CONTEXT_WINDOW = 10000
CONTEXT_AGE_HOURS = 2
MAX_MESSAGES = 80
# Couvre un render_widget dense (~2500 tokens d'arguments JSON).
MAX_TOKENS = 5000

DEBOUNCE_SECONDS: float = 0.33
# Édition tardive (ping corrigé / redo in-place) — hors fenêtre : ignorée.
EDIT_UPDATE_WINDOW_SECONDS: float = 15

MEMORY_FLUSH_MESSAGES = 40
MEMORY_FLUSH_MINUTES = 30
MEMORY_DIRECT_FLUSH_MESSAGES = 12
MEMORY_BUFFER_CAP = 80
MEMORY_TOP_K = 2
MEMORY_EXTRACT_MAX_ACTIONS = 5
MEMORY_EXISTING_LIMIT = 16
MEMORY_BATCH_OVERLAP = 5
MEMORY_PROFILE_FACTS = 3
MEMORY_SELF_FACTS = 6
# Distance cosine Chroma sous laquelle un fait actif est traité comme doublon.
MEMORY_SEMANTIC_DEDUP_DISTANCE = 0.1
MEMORY_RAG_MAX_DISTANCE = 0.42
MEMORY_PENDING_PROFILE_MIN = 0.5
MEMORY_ARCHIVE_PURGE_DAYS = 90

# (message membre, réponse MARIA) — STYLE_EXAMPLES_SAMPLE tirés au hasard par réponse.
STYLE_EXAMPLES: list[tuple[str, str]] = [
    ("t'as vu le match hier ?", "Oui. 3-0, c'était violent"),
    ("tu penses quoi de ce film ?", "J'ai pas d'avis moi, mais les gens pensent que c'était un flop"),
    ("on mange où ce soir ?", "Kebab ? Hop pas de débat"),
    ("c'est quoi la capitale de l'Australie ?", "Canberra. Pas Sydney comme tout le monde pense..."),
    ("j'ai raté mon exam", "d'accord, mais t'avais bossé ?"),
    ("tu dors jamais toi", "J'ai pas le choix. Je suis un bot"),
    ("explique-moi comment marche le TCP", "Comme une poignée de main en 3 temps, paquets numérotés, ce qui se perd est renvoyé. C'est bon ou j'explique plus en détail ?"),
    ("salut", "Salut ouais."),
]
STYLE_EXAMPLES_SAMPLE = 3

# Outils sans pied de page `-# …`.
QUIET_FOOTER_TOOLS = frozenset({"search_memory", "search_images"})
SHOW_MEMORY_CALLBACK_TAG = False
