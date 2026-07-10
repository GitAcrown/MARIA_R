"""Helpers Brave News — partagés par le hub /me."""

import logging

import requests

logger = logging.getLogger("MARIA.News")


def brave_news(api_key: str, query: str, n: int = 6) -> list[dict]:
    """Appel Brave News Search API."""
    try:
        r = requests.get(
            "https://api.search.brave.com/res/v1/news/search",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            },
            params={"q": query, "count": min(n * 2, 20), "search_lang": "fr", "country": "FR"},
            timeout=8,
        )
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:
        logger.warning(f"Brave news ({query!r}): {e}")
        return []


def build_news_summary(results: list[dict], date_str: str) -> str:
    """Construit un résumé compact (avec liens Markdown) à partir des résultats Brave."""
    lines: list[str] = []
    seen: set[str] = set()
    for r in results:
        title = (r.get("title") or "").strip()
        if not title or title in seen:
            continue
        seen.add(title)
        url = (r.get("url") or "").strip()
        desc = (r.get("description") or "").strip()
        line = f"- {title}"
        if desc:
            line += f" : {desc[:120]}"
        if url:
            line += " ([lien](" + url + "))"
        lines.append(line)
        if len(lines) >= 7:
            break
    if not lines:
        return ""
    return f"ACTU DU {date_str} :\n" + "\n".join(lines)
