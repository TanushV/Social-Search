import os
import requests
from typing import Any, Dict, List

__all__ = [
    "query_reddit",
    "query_bsky",
    "query_meta",
    "search_web",
    "search_reddit_full",
]


# ---------------------------------------------------------------------------
# Reddit API ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def _reddit_auth(client_id: str, client_secret: str, user_agent: str) -> str:
    """Return a bearer token for the Reddit OAuth2 *script* flow.

    Parameters
    ----------
    client_id, client_secret : str
        Credentials from https://www.reddit.com/prefs/apps.
    user_agent : str
        Something like "myapp/0.1 by <reddit_username>". Reddit requires this.
    """
    auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
    data = {"grant_type": "client_credentials"}
    headers = {"User-Agent": user_agent}

    r = requests.post("https://www.reddit.com/api/v1/access_token", auth=auth, data=data, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()["access_token"]


def query_reddit(
    subreddit: str,
    *,
    client_id: str | None = None,
    client_secret: str | None = None,
    user_agent: str = "api_clients/0.1 (by github.com/example)",
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Return the newest posts in *subreddit* (default 10).

    Either pass `client_id`/`client_secret` explicitly or rely on
    environment variables `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET`.
    """
    client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
    client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError("Reddit credentials missing. Provide client_id/client_secret or set env vars.")

    token = _reddit_auth(client_id, client_secret, user_agent)
    headers = {"Authorization": f"bearer {token}", "User-Agent": user_agent}

    url = f"https://oauth.reddit.com/r/{subreddit}/new?limit={limit}"
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()

    return resp.json()["data"]["children"]  # Each child has a "data" dict with post info


# ---------------------------------------------------------------------------
# Bluesky / AT-Protocol ------------------------------------------------------
# ---------------------------------------------------------------------------

def _bsky_login(handle: str, app_password: str) -> str:
    """Return a JWT session token using basic username/password auth."""
    payload = {"identifier": handle, "password": app_password}
    r = requests.post("https://bsky.social/xrpc/com.atproto.server.createSession", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()["accessJwt"]


def query_bsky(
    keyword: str,
    *,
    handle: str | None = None,
    app_password: str | None = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Search public posts containing *keyword*.

    If no credentials are provided, the public AppView endpoint is used and
    results are anonymous. Providing credentials can improve rate limits.
    """
    headers: Dict[str, str] = {}
    if handle and app_password:
        token = _bsky_login(handle, app_password)
        headers["Authorization"] = f"Bearer {token}"
        endpoint = "https://bsky.social/xrpc/app.bsky.feed.searchPosts"
    else:
        endpoint = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"

    params = {"q": keyword, "limit": limit}
    resp = requests.get(endpoint, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json().get("posts", [])


# ---------------------------------------------------------------------------
# Meta (Facebook/Instagram) Graph API ---------------------------------------
# ---------------------------------------------------------------------------

def query_meta(
    path: str,
    *,
    access_token: str | None = None,
    api_version: str = "v19.0",
    **params: Any,
) -> Dict[str, Any]:
    """GET `/{path}` from the Facebook Graph API.

    Example: `query_meta('me', access_token='EAA...', fields='id,name')`
    """
    access_token = access_token or os.getenv("META_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("Meta Graph API access_token missing. Pass it explicitly or set META_ACCESS_TOKEN env var.")

    base = f"https://graph.facebook.com/{api_version}/{path}"
    params = {"access_token": access_token, **params}
    resp = requests.get(base, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Simple web search via DuckDuckGo -----------------------------------------
# ---------------------------------------------------------------------------

def search_web(query: str, *, max_results: int = 10) -> List[str]:
    """Return a list of result titles from DuckDuckGo HTML search.

    NOTE: This scrapes the public HTML endpoint; heavy usage may be blocked.
    Use a real API (SerpAPI, ContextualWeb, etc.) for production systems.
    """
    params = {"q": query, "kl": "us-en"}
    html = requests.get("https://duckduckgo.com/html/", params=params, timeout=10).text

    # Very small & dirty extraction; avoids extra dependencies.
    import re

    titles = re.findall(r"<a[^>]+class=\"result__a\"[^>]*>(.*?)</a>", html)
    # Remove HTML tags/entities crudely.
    clean = re.compile(r"<.*?>")
    out = [re.sub(clean, "", t) for t in titles]
    return out[: max_results]


# ---------------------------------------------------------------------------
# Higher-level Reddit search: fetch posts + comments -------------------------
# ---------------------------------------------------------------------------

def _reddit_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"bearer {token}", "User-Agent": "search_assistant/1.0"}


def search_reddit_full(
    query: str,
    *,
    limit: int = 10,
    comments_limit: int = 5,
    client_id: str | None = None,
    client_secret: str | None = None,
) -> List[Dict[str, Any]]:
    """Search Reddit and return full post text plus top comments.

    Each result dict contains: title, subreddit, url, selftext, comments[].
    """

    client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
    client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError("Reddit credentials missing. Provide client_id/client_secret or set env vars.")

    token = _reddit_auth(client_id, client_secret, "search_assistant/1.0")
    headers = _reddit_headers(token)

    search_params = {
        "q": query,
        "limit": limit,
        "sort": "new",
        "type": "link",
        "restrict_sr": False,
    }

    import requests, json as _json  # local import to keep top clean

    search_resp = requests.get("https://oauth.reddit.com/search", headers=headers, params=search_params, timeout=15)
    search_resp.raise_for_status()
    posts = search_resp.json()["data"]["children"]

    results: List[Dict[str, Any]] = []

    for p in posts:
        d = p["data"]
        post_id = d["id"]  # base36 without t3_
        permalink = d.get("permalink", "")
        post_url = "https://reddit.com" + permalink

        # fetch comments listing
        comm_url = f"https://oauth.reddit.com/comments/{post_id}"
        comm_params = {"limit": comments_limit, "depth": 1, "sort": "top"}
        comm_resp = requests.get(comm_url, headers=headers, params=comm_params, timeout=15)
        comm_resp.raise_for_status()
        listing = comm_resp.json()
        comments_listing = listing[1]["data"]["children"] if len(listing) > 1 else []

        comments: List[Dict[str, str]] = []
        for c in comments_listing:
            if c.get("kind") != "t1":
                continue
            cdata = c["data"]
            comments.append({"author": cdata.get("author"), "body": cdata.get("body")})

        results.append(
            {
                "title": d.get("title"),
                "subreddit": d.get("subreddit"),
                "url": post_url,
                "selftext": d.get("selftext", ""),
                "comments": comments,
            }
        )

    return results 