from __future__ import annotations

"""reddit_enhanced.py
Advanced Reddit helper that combines the following in **one call**:

• Fetch up to 100 search results.
• Rank them with BM25 and select top-K (default 20).
• For each selected post, hydrate the top comments.
• Compute a volume-trend metric via :func:`reddit_analysis.compute_volume_trend`.

Returns a single JSON-serialisable ``dict`` so that the OpenAI tool layer can
hand the LLM both granular examples and an aggregate trend snapshot in one
response.
"""

from typing import Any, Dict, List
from urllib.parse import quote

import os
import re
import requests
from rank_bm25 import BM25Okapi

from reddit_search import _get_bearer_token  # type: ignore
from reddit_analysis import compute_volume_trend  # type: ignore

__all__ = ["search_reddit_full"]

# ---------------------------------------------------------------------------
# Basic tokenisation --------------------------------------------------------
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    """Very simple whitespace & punctuation splitter suitable for BM25."""
    # Lower-case and split on non-word chars
    return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------------------------
# Core helper ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def _search_raw(
    query: str,
    *,
    limit: int = 100,
    time_filter: str | None = None,
    client_id: str,
    client_secret: str,
    user_agent: str,
) -> List[Dict[str, Any]]:
    """Return Reddit search listing without comments."""

    token = _get_bearer_token(client_id, client_secret, user_agent)
    headers = {"Authorization": f"bearer {token}", "User-Agent": user_agent}

    params = {
        "q": query,
        "limit": min(100, limit),
        "sort": "new",
        "type": "link",
    }
    if time_filter is not None:
        params["t"] = time_filter

    resp = requests.get("https://oauth.reddit.com/search", headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    return [child["data"] for child in resp.json()["data"]["children"]]


def _fetch_comments_bulk(
    post_ids: List[str],
    *,
    comment_limit: int,
    headers: Dict[str, str],
) -> Dict[str, List[Dict[str, str]]]:
    """Return mapping post_id -> list of comments."""
    out: Dict[str, List[Dict[str, str]]] = {}
    for pid in post_ids:
        url = f"https://oauth.reddit.com/comments/{quote(pid, safe='')}"
        try:
            resp = requests.get(
                url,
                headers=headers,
                params={"limit": comment_limit, "depth": 1, "sort": "top"},
                timeout=15,
            )
            resp.raise_for_status()
            listing = resp.json()
            if len(listing) < 2:
                continue
            comments_children = listing[1]["data"].get("children", [])
            comments: List[Dict[str, str]] = []
            for child in comments_children:
                if child.get("kind") != "t1":
                    continue
                cdata = child["data"]
                comments.append({"author": cdata.get("author"), "body": cdata.get("body")})
                if len(comments) >= comment_limit:
                    break
            out[pid] = comments
        except Exception:
            # Silently continue; individual failures shouldn't abort entire run
            out[pid] = []
    return out


# ---------------------------------------------------------------------------
# Public entry-point ---------------------------------------------------------
# ---------------------------------------------------------------------------

def search_reddit_full(
    query: str,
    *,
    top_k: int = 20,
    comment_limit: int = 5,
    time_filter: str | None = None,
    user_agent: str = "reddit_enhanced/0.1 (by github.com/example)",
    client_id: str | None = None,
    client_secret: str | None = None,
) -> Dict[str, Any]:
    """Return BM25-ranked posts (+comments) *and* volume-trend snapshot."""

    # Resolve credentials ---------------------------------------------------
    client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
    client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing Reddit credentials (REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET)."
        )

    # Raw search ------------------------------------------------------------
    raw_posts = _search_raw(
        query,
        limit=100,
        time_filter=time_filter,
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )
    if not raw_posts:
        return {"trend": compute_volume_trend(query), "posts": []}

    # Build BM25 corpus ------------------------------------------------------
    corpus = [_tokenise((p.get("title", "") + " " + p.get("selftext", ""))) for p in raw_posts]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenise(query))

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    selected_posts: List[Dict[str, Any]] = [raw_posts[i] for i in ranked_indices]

    # Auth headers for further calls ---------------------------------------
    token = _get_bearer_token(client_id, client_secret, user_agent)
    headers = {"Authorization": f"bearer {token}", "User-Agent": user_agent}

    # Hydrate comments ------------------------------------------------------
    ids = [p["id"] for p in selected_posts]
    comments_map = _fetch_comments_bulk(ids, comment_limit=comment_limit, headers=headers)

    # Assemble final list ---------------------------------------------------
    posts_out: List[Dict[str, Any]] = []
    for p in selected_posts:
        pid = p["id"]
        permalink = p.get("permalink", "")
        url = f"https://reddit.com{permalink}" if permalink else ""
        posts_out.append(
            {
                "id": pid,
                "title": p.get("title"),
                "subreddit": p.get("subreddit"),
                "url": url,
                "selftext": p.get("selftext", ""),
                "score": p.get("score"),
                "created_utc": p.get("created_utc"),
                "comments": comments_map.get(pid, []),
            }
        )

    # Trend statistics ------------------------------------------------------
    trend = compute_volume_trend(query, client_id=client_id, client_secret=client_secret, user_agent=user_agent)

    return {"trend": trend, "posts": posts_out} 