"""Constantes de configuration du cog Chat (modèles, fenêtre de contexte, debounce).

Centralise les valeurs qui étaient auparavant éparpillées et incohérentes entre
`cogs/chat/chat.py`, `common/llm/api.py` et `common/llm/context.py`.
"""

# Modèles OpenAI
MODEL_MAIN = "gpt-5.6-luna"   # réponses conversationnelles
MODEL_NANO = "gpt-5.4-nano"   # tâches structurées simples (rappels, calculs)

# Fenêtre de contexte / budget de la session de chat
CONTEXT_WINDOW = 8000
CONTEXT_AGE_HOURS = 1
MAX_MESSAGES = 40
MAX_TOKENS = 2800

# Debounce des réponses (regroupe les messages rapprochés en un seul appel)
DEBOUNCE_SECONDS: float = 0.5

# Mémoire long terme — flush hybride + RAG
# Moins d'appels nano, lots plus gros = meilleur contexte (gags, attribution).
MEMORY_FLUSH_MESSAGES = 40
MEMORY_FLUSH_MINUTES = 30
MEMORY_BUFFER_CAP = 80
# RAG complémentaire (le perso vient surtout des profils injectés).
MEMORY_TOP_K = 3
MEMORY_EXTRACT_MAX_ACTIONS = 6
MEMORY_EXISTING_LIMIT = 25
# Chevauchement entre lots : contexte du lot précédent, sans re-create.
MEMORY_BATCH_OVERLAP = 8
# Mini-profils injectés à chaque réponse (auteur + mentions/reply).
MEMORY_PROFILE_FACTS = 5
MEMORY_PROFILE_MAX_OTHERS = 3
