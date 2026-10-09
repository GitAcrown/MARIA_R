"""Constantes du cog Chat."""

MODEL_MAIN = "gpt-6-luna"

# trim() déduit aussi le prompt développeur du budget tokens.
CONTEXT_WINDOW = 10000
CONTEXT_AGE_HOURS = 1
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
MEMORY_TOP_K = 4
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

# Outils sans pied de page `-# …`.
QUIET_FOOTER_TOOLS = frozenset({"search_memory", "search_images"})
SHOW_MEMORY_CALLBACK_TAG = False
