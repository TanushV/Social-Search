from __future__ import annotations

import os
import json
import requests
import requests.auth
from typing import Any, Dict, List


__all__ = [
    "fetch_posts_and_comments",
]


# ---------------------------------------------------------------------------
# Internal helpers ----------------------------------------------------------
# ---------------------------------------------------------------------------


def _get_bearer_token(client_id: str, client_secret: str, user_agent: str) -> str:  # noqa: D401
    """Return OAuth2 *application* bearer token.

    This uses the simple *client credentials* flow which is sufficient for
    **read-only** endpoints such as /search and /comments when the application
    has the *public data* scope.
    """
    auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
    data = {"grant_type": "client_credentials"}
    headers = {"User-Agent": user_agent}

    resp = requests.post(
        "https://www.reddit.com/api/v1/access_token",
        auth=auth,
        data=data,
        headers=headers,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


# ---------------------------------------------------------------------------
# Public API ----------------------------------------------------------------
# ---------------------------------------------------------------------------


def fetch_posts_and_comments(
    search_query: str,
    post_limit: int = 10,
    comment_limit: int = 5,
    *,
    client_id: str | None = None,
    client_secret: str | None = None,
    user_agent: str = "reddit_search/0.1 (by github.com/example)",
    time_filter: str | None = None,
) -> List[Dict[str, Any]]:
    """Search Reddit and return posts along with their top comments.

    Parameters
    ----------
    search_query:
        The query string to search for.
    post_limit:
        Maximum number of posts to retrieve.
    comment_limit:
        Maximum number of top-level comments to fetch for each post.
    client_id, client_secret:
        Reddit application credentials.  If *None*, the function falls back to
        the ``REDDIT_CLIENT_ID`` and ``REDDIT_CLIENT_SECRET`` environment
        variables.  You can obtain these at https://www.reddit.com/prefs/apps.
    user_agent:
        A descriptive user-agent string.  Reddit requires this header.
    time_filter:
        Optional time filter for Reddit search.  Possible values: "hour", "day", "week", "month", "year", "all".

    Returns
    -------
    list[dict[str, Any]]
        Each dict contains ``title``, ``subreddit``, ``url``, ``selftext``, and
        a ``comments`` list (author/body pairs).
    """

    # Resolve credentials ---------------------------------------------------
    client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
    client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing Reddit credentials.  Set the REDDIT_CLIENT_ID and "
            "REDDIT_CLIENT_SECRET environment variables or pass them "
            "explicitly."
        )

    # Authenticate and prepare headers -------------------------------------
    token = _get_bearer_token(client_id, client_secret, user_agent)
    headers = {"Authorization": f"bearer {token}", "User-Agent": user_agent}

    # Perform the search ----------------------------------------------------
    search_params = {
        "q": search_query,
        "limit": post_limit,
        "sort": "new",  # you can change to 'relevance', 'top', etc.
        "type": "link",  # only return link (i.e. post) results
    }

    # Optional time filter (hour, day, week, month, year, all)
    if time_filter is not None:
        search_params["t"] = time_filter

    search_resp = requests.get(
        "https://oauth.reddit.com/search",
        headers=headers,
        params=search_params,
        timeout=15,
    )
    search_resp.raise_for_status()
    posts_listing = search_resp.json()["data"]["children"]

    results: List[Dict[str, Any]] = []

    for post in posts_listing:
        pdata = post["data"]
        post_id = pdata["id"]  # base36 without t3_ prefix
        permalink = pdata.get("permalink", "")
        post_url = f"https://reddit.com{permalink}" if permalink else ""

        # Fetch comments ----------------------------------------------------
        comments_url = f"https://oauth.reddit.com/comments/{post_id}"
        comments_resp = requests.get(
            comments_url,
            headers=headers,
            params={"limit": comment_limit, "depth": 1, "sort": "top"},
            timeout=15,
        )
        comments_resp.raise_for_status()

        # The comments endpoint returns a list: [post_listing, comments_listing]
        listing = comments_resp.json()
        if len(listing) < 2:
            comments_children: list[dict[str, Any]] = []
        else:
            comments_children = listing[1]["data"].get("children", [])

        comments: List[Dict[str, str]] = []
        for child in comments_children:
            if child.get("kind") != "t1":  # only actual comments
                continue
            cdata = child["data"]
            comments.append(
                {
                    "author": cdata.get("author"),
                    "body": cdata.get("body"),
                }
            )
            if len(comments) >= comment_limit:
                break

        results.append(
            {
                "title": pdata.get("title"),
                "subreddit": pdata.get("subreddit"),
                "url": post_url,
                "selftext": pdata.get("selftext", ""),
                "comments": comments,
            }
        )

    return results


# ---------------------------------------------------------------------------
# CLI helper ----------------------------------------------------------------
# ---------------------------------------------------------------------------


def _cli() -> None:  # pragma: no cover – simple util
    import argparse

    parser = argparse.ArgumentParser(description="Search Reddit and dump JSON with posts & comments.")
    parser.add_argument("query", help="Search string")
    parser.add_argument("--posts", type=int, default=10, help="Number of posts to retrieve (default: 10)")
    parser.add_argument(
        "--comments", type=int, default=5, help="Number of comments to retrieve per post (default: 5)"
    )
    args = parser.parse_args()

    data = fetch_posts_and_comments(args.query, post_limit=args.posts, comment_limit=args.comments)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    _cli() 