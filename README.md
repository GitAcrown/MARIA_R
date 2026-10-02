# Maria

Bot Discord GPT conçu pour s'intégrer naturellement dans une communauté — pas un assistant, quelqu'un qui est là.

Personnalité directe et Gen Z, Maria s'adapte au ton de chaque salon, et cherche sur le web quand elle ne sait pas.

---

## Ce qu'elle fait

- **Conversation** — contexte restreint par salon
- **Mémoire long terme** — souvenirs durables (préférences, gags, événements) via RAG
- **Recherche web** — Brave Search (+ fallback DuckDuckGo) avec crawling de pages
- **Rappels** — planification et envoi de rappels personnalisés
- **Médias** — analyse d'images, transcription audio
- **Personnalité par salon** — configurable par la modération via `/chatbot personality`

## Stack

- [discord.py](https://discordpy.readthedocs.io/) — interface Discord
- [OpenAI](https://platform.openai.com/) — `gpt-6-luna` · `gpt-4o-transcribe`
- [TypeSafe JEV](https://docs.typesafe.ai/api) — décisions structurées (optionnel)
- [Brave Search API](https://brave.com/search/api/) — recherche web (optionnel)
- SQLite + [Chroma](https://www.trychroma.com/) — persistance locale et recherche sémantique

## TypeSafe / JEV (optionnel)

Avec `TYPESAFE_API_KEY` dans `.env`, MARIA utilise [JEV](https://docs.typesafe.ai/api) pour :

| Usage | Question | Seuil | Sans clé / erreur API |
|-------|----------|-------|------------------------|
| Mode **greedy** (nom cité) | Noul « s’adresse au bot ? » | `≥ 0.65` | Répond comme avant (regex) |
| Routage d’outils | Choice `force_level` + catégorie | confidence `≥ 0.5` | Regex `capabilities.py` |
| RAG mémoire | Score pertinence souvenir↔query | score `≥ 1.0` et conf `≥ 0.4` | Ranking embedding actuel |
| Extraction mémoire | Noul « fait durable ? » | `≥ 0.6` | Toutes les actions LLM |

Sans clé, le comportement legacy est inchangé. Mentions Discord et replies au bot restent déterministes (pas de JEV).

## Licence

[MIT](LICENSE) — Acrone, 2026
