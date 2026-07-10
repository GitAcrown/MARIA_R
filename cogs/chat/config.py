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
