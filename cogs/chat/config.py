"""Constantes de configuration du cog Chat (modèles, fenêtre de contexte, debounce).

Centralise les valeurs qui étaient auparavant éparpillées et incohérentes entre
`cogs/chat/chat.py`, `common/llm/api.py` et `common/llm/context.py`.
"""

# Modèles OpenAI
MODEL_MAIN = "gpt-5.6-luna"

# Fenêtre de contexte / budget de la session de chat
# CONTEXT_WINDOW est la vraie limite (en tokens) ; MAX_MESSAGES=0 = pas de plafond
# de comptage — seul le budget tokens décide, en retirant des messages ENTIERS
# (jamais de troncature au milieu d'un message, cf. ConversationContext.trim()).
# ATTENTION : trim() déduit le prompt développeur ENTIER (instructions + contexte
# salon + goûts + profils + mémoire, souvent 1800-2600 tokens une fois tout injecté)
# de ce budget avant de calculer la place pour l'historique. 6000 s'est révélé encore
# trop juste en pratique (amnésie dès qu'un render_widget ou un salon actif traîne
# dans l'historique récent) → 10000 pour laisser une vraie marge à la conversation,
# tout en restant loin de la fenêtre réelle du modèle.
CONTEXT_WINDOW = 10000
CONTEXT_AGE_HOURS = 3
MAX_MESSAGES = 0
# Plafond, pas un coût fixe (une réponse courte ne consomme que ce qu'elle écrit).
# Doit couvrir le pire cas d'un tool call render_widget rempli à fond : 12 blocs
# à ~800 caractères chacun ≈ 2500 tokens rien que pour l'argument JSON — d'où la marge.
MAX_TOKENS = 4000

# Debounce des réponses (regroupe les messages rapprochés d'UNE MÊME personne en un seul appel)
DEBOUNCE_SECONDS: float = 0.33

# Streaming Discord (édition progressive du message)
# Discord limite ~5 PATCH / 5 s par message : 1,2 s laisse une marge, plus l'edit final.
STREAM_EDIT_INTERVAL: float = 1.2
# Ne pas poster le premier message à la première lettre (évite un flash "M").
STREAM_MIN_FIRST_CHARS: int = 8

# Mémoire long terme — flush hybride + RAG (extraction via MODEL_MAIN)
# Lots plus gros = meilleur contexte (gags, attribution).
# Flush hybride : lecture passive plus lente ; dialogue avec MARIA flush plus tôt.
MEMORY_FLUSH_MESSAGES = 25
MEMORY_FLUSH_MINUTES = 20
MEMORY_DIRECT_FLUSH_MESSAGES = 8  # si le lot contient des msgs → MARIA
MEMORY_BUFFER_CAP = 80
# RAG complémentaire (le perso vient surtout des profils injectés).
MEMORY_TOP_K = 6
MEMORY_EXTRACT_MAX_ACTIONS = 8
MEMORY_EXISTING_LIMIT = 30
# Chevauchement entre lots : contexte du lot précédent, sans re-create.
MEMORY_BATCH_OVERLAP = 8
# Mini-profils injectés à chaque réponse (auteur + mentions/reply).
MEMORY_PROFILE_FACTS = 5
MEMORY_PROFILE_MAX_OTHERS = 3
# Goûts / faits sur MARIA injectés à chaque réponse (constance des avis).
MEMORY_SELF_FACTS = 8
# Dédup sémantique à la création (distance cosine Chroma) : en dessous de ce seuil,
# un souvenir actif existant est considéré comme "le même fait" et bloque la création.
MEMORY_SEMANTIC_DEDUP_DISTANCE = 0.1
