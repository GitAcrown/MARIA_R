"""Constantes de configuration du cog Chat (modèles, fenêtre de contexte, debounce).

Centralise les valeurs qui étaient auparavant éparpillées et incohérentes entre
`cogs/chat/chat.py`, `common/llm/api.py` et `common/llm/context.py`.
"""

# Modèles OpenAI
MODEL_MAIN = "gpt-6-luna"

# Fenêtre de contexte / budget de la session de chat
# CONTEXT_WINDOW est la vraie limite (en tokens) ; MAX_MESSAGES est un filet
# de comptage — le budget tokens décide aussi, en retirant des messages ENTIERS
# (jamais de troncature au milieu d'un message, cf. ConversationContext.trim()).
# ATTENTION : trim() déduit le prompt développeur ENTIER (instructions + contexte
# salon + goûts + profils + mémoire, souvent 1800-2600 tokens une fois tout injecté)
# de ce budget avant de calculer la place pour l'historique. 6000 s'est révélé encore
# trop juste en pratique (amnésie dès qu'un render_widget ou un salon actif traîne
# dans l'historique récent) → 10000 pour laisser une vraie marge à la conversation,
# tout en restant loin de la fenêtre réelle du modèle.
CONTEXT_WINDOW = 10000
CONTEXT_AGE_HOURS = 2
# Filet conservateur : évite qu'un salon très actif garde 3 h de pavés.
MAX_MESSAGES = 80
# Plafond, pas un coût fixe (une réponse courte ne consomme que ce qu'elle écrit).
# Doit couvrir le pire cas d'un tool call render_widget rempli à fond : 12 blocs
# à ~800 caractères chacun ≈ 2500 tokens rien que pour l'argument JSON — d'où la marge.
MAX_TOKENS = 5000

# Debounce des réponses (regroupe les messages rapprochés d'UNE MÊME personne en un seul appel)
DEBOUNCE_SECONDS: float = 0.33
# Fenêtre unique pour les éditions (ping corrigé « marie » → « maria », ou
# mise à jour in-place d'une réponse déjà postée). Au-delà : on ignore —
# trop long = risque de relancer un message déjà modéré.
EDIT_TRIGGER_SECONDS: float = 15
EDIT_UPDATE_WINDOW_SECONDS: float = 15

# Mémoire long terme — carte d'identité, pas le journal du salon.
# Le résumé de session couvre déjà le fil récent : on flush peu, on extrait peu.
MEMORY_FLUSH_MESSAGES = 40
MEMORY_FLUSH_MINUTES = 30
MEMORY_DIRECT_FLUSH_MESSAGES = 12  # dialogue → MARIA : un peu plus tôt, pas gourmand
MEMORY_BUFFER_CAP = 80
# RAG complémentaire (le perso vient surtout des 3 faits de profil).
MEMORY_TOP_K = 2
MEMORY_EXTRACT_MAX_ACTIONS = 5
MEMORY_EXISTING_LIMIT = 16
# Chevauchement entre lots : contexte du lot précédent, sans re-create.
MEMORY_BATCH_OVERLAP = 5
# Mini-profils injectés à chaque réponse (auteur + mentions/reply).
MEMORY_PROFILE_FACTS = 3
# Goûts / faits sur MARIA injectés à chaque réponse (constance des avis).
MEMORY_SELF_FACTS = 6
# Dédup sémantique à la création (distance cosine Chroma) : en dessous de ce seuil,
# un souvenir actif existant est considéré comme "le même fait" et bloque la création.
MEMORY_SEMANTIC_DEDUP_DISTANCE = 0.1
# Retrieval : au-delà, le voisin Chroma n'est pas assez proche (sauf match FTS).
MEMORY_RAG_MAX_DISTANCE = 0.42
# Pending auteur injecté dans le profil au-dessus de ce seuil.
MEMORY_PENDING_PROFILE_MIN = 0.5
# Archives SQLite purgées après ce délai.
MEMORY_ARCHIVE_PURGE_DAYS = 90
