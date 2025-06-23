"""bluesky_search.py
A minimal Bluesky (AT Protocol) search helper.

Usage examples:
    # Simple anonymous search
    python bluesky_search.py search "openai" --limit 5

    # Authenticated search to improve rate-limits (set env vars first)
    export BLUESKY_HANDLE="you.bsky.social"
    export BLUESKY_APP_PASSWORD="xxxx-xxxx-xxxx-xxxx"
    python bluesky_search.py search "openai" --limit 5 --sort top

    # Hydrate specific posts
    python bluesky_search.py posts at://did:plc:foo/app.bsky.feed.post/3kjxhs at://did:plc:bar/app.bsky.feed.post/6f2abc

    # Fetch a full thread
    python bluesky_search.py thread at://did:plc:foo/app.bsky.feed.post/3kjxhs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Sequence

import requests

# ---------------------------------------------------------------------------
# Helper functions -----------------------------------------------------------
# ---------------------------------------------------------------------------


def _login(handle: str, app_password: str) -> str:
    """Return an access JWT for *handle* using an *app_password*."""
    payload = {"identifier": handle, "password": app_password}
    r = requests.post(
        "https://bsky.social/xrpc/com.atproto.server.createSession",
        json=payload,
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["accessJwt"]


class BlueskyClient:
    """Lightweight wrapper around a few public Bluesky endpoints."""

    # As of mid-2025, the canonical hostname for unauthenticated access is
    # https://api.bsky.app (the older public.api.bsky.app often returns 403).
    BASE_PUBLIC = "https://api.bsky.app"
    BASE_AUThed = "https://bsky.social"  # for PDS login; proxying enabled

    def __init__(self, handle: str | None = None, app_password: str | None = None):
        self._sess = requests.Session()
        if handle and app_password:
            token = _login(handle, app_password)
            self._sess.headers.update({"Authorization": f"Bearer {token}"})
            self._base = self.BASE_AUThed
        else:
            self._base = self.BASE_PUBLIC

    # -------------------------------------
    # Core endpoints
    # -------------------------------------

    def search_posts(
        self,
        q: str,
        *,
        limit: int = 25,
        sort: str | None = None,
        since: str | None = None,
        until: str | None = None,
        mentions: str | None = None,
        author: str | None = None,
        lang: str | None = None,
        domain: str | None = None,
        url: str | None = None,
        tag: Sequence[str] | None = None,
        cursor: str | None = None,
    ) -> Dict[str, Any]:
        """Call `app.bsky.feed.searchPosts` and return the JSON response."""
        params: Dict[str, Any] = {"q": q, "limit": limit}
        if sort:
            params["sort"] = sort
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        if mentions:
            params["mentions"] = mentions
        if author:
            params["author"] = author
        if lang:
            params["lang"] = lang
        if domain:
            params["domain"] = domain
        if url:
            params["url"] = url
        if tag:
            # Multiple tags can be provided; requests will repeat the key.
            params["tag"] = list(tag)
        if cursor:
            params["cursor"] = cursor

        endpoint = f"{self._base}/xrpc/app.bsky.feed.searchPosts"
        r = self._sess.get(endpoint, params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def get_posts(self, uris: Sequence[str]) -> Dict[str, Any]:
        """Hydrate a list of post URIs via `app.bsky.feed.getPosts`."

        Bluesky allows up to 25 URIs per call.
        """
        if not uris:
            raise ValueError("At least one URI must be supplied")
        if len(uris) > 25:
            raise ValueError("Maximum 25 URIs per call")

        params = [("uris", u) for u in uris]  # repeat key for each value
        endpoint = f"{self._base}/xrpc/app.bsky.feed.getPosts"
        r = self._sess.get(endpoint, params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def get_post_thread(
        self,
        uri: str,
        *,
        depth: int | None = None,
        parent_height: int | None = None,
    ) -> Dict[str, Any]:
        """Return thread context for a given post via `app.bsky.feed.getPostThread`."""
        params: Dict[str, Any] = {"uri": uri}
        if depth is not None:
            params["depth"] = depth
        if parent_height is not None:
            params["parentHeight"] = parent_height

        endpoint = f"{self._base}/xrpc/app.bsky.feed.getPostThread"
        r = self._sess.get(endpoint, params=params, timeout=15)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# CLI helper ---------------------------------------------------------------
# ---------------------------------------------------------------------------


def _pretty_print(obj: Any):
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="Bluesky search utility")
    sub = parser.add_subparsers(dest="command", required=True)

    # ------------------- search -------------------
    p_search = sub.add_parser("search", help="Search posts")
    p_search.add_argument("query", help="Search string")
    p_search.add_argument("--limit", type=int, default=10, help="Max results (1-100)")
    p_search.add_argument("--sort", choices=["top", "latest"], help="Result ordering")
    p_search.add_argument("--min-likes", type=int, default=0, help="Minimum likeCount to keep a post")
    p_search.add_argument("--min-reposts", type=int, default=0, help="Minimum repostCount to keep a post")
    p_search.add_argument("--min-replies", type=int, default=0, help="Minimum replyCount to keep a post")

    # ------------------- posts -------------------
    p_posts = sub.add_parser("posts", help="Hydrate one or more post URIs")
    p_posts.add_argument("uris", nargs="+", help="One or more at:// post URIs")

    # ------------------- thread ------------------
    p_thread = sub.add_parser("thread", help="Fetch a thread for a post URI")
    p_thread.add_argument("uri", help="Post at:// URI")
    p_thread.add_argument("--depth", type=int, help="Reply depth to include")
    p_thread.add_argument("--parent-height", type=int, help="Ancestor depth to include")

    args = parser.parse_args(argv)

    # Acquire credentials from env if present
    handle = os.getenv("BLUESKY_HANDLE")
    app_password = os.getenv("BLUESKY_APP_PASSWORD")

    client = BlueskyClient(handle=handle, app_password=app_password)

    try:
        if args.command == "search":
            def _gather_replies(root_uri: str) -> List[str]:
                """Return list of reply texts (depth ≤3) for the given post URI."""
                try:
                    thread = client.get_post_thread(root_uri, depth=3)
                except requests.HTTPError:
                    return []

                out: List[str] = []

                def walk(node):
                    if not isinstance(node, dict):
                        return
                    rec = node.get("post") or node
                    if rec.get("record") and isinstance(rec["record"], dict):
                        t = rec["record"].get("text")
                        if t and (rec.get("uri") or node.get("uri")) != root_uri:
                            out.append(t)
                    for child in node.get("replies", []) or []:
                        walk(child)

                walk(thread.get("thread", {}))
                return out

            collected: List[Dict[str, Any]] = []  # final simplified posts
            cursor: str | None = None

            while len(collected) < args.limit:
                batch_limit = min(100, args.limit * 2)  # fetch generously
                try:
                    raw = client.search_posts(
                        args.query,
                        limit=batch_limit,
                        sort=args.sort,
                        cursor=cursor,
                    )
                except requests.HTTPError:
                    # Some public endpoints return 403 when cursor is used. Stop pagination.
                    break

                posts = raw.get("posts", [])
                if not posts:
                    break

                for p in posts:
                    rec = p.get("record", {})
                    text = rec.get("text", "")
                    uri = p.get("uri", "")
                    replies = _gather_replies(uri)

                    if (
                        p.get("likeCount", 0) >= args.min_likes
                        and p.get("repostCount", 0) >= args.min_reposts
                        and len(replies) >= args.min_replies
                    ):
                        collected.append({"text": text, "replies": replies})
                        if len(collected) >= args.limit:
                            break

                if len(collected) >= args.limit:
                    break

                cursor = raw.get("cursor")
                if not cursor:
                    break  # no more pages

            _pretty_print({"posts": collected})
        elif args.command == "posts":
            resp = client.get_posts(args.uris)
            _pretty_print(resp)
        elif args.command == "thread":
            resp = client.get_post_thread(args.uri, depth=args.depth, parent_height=args.parent_height)
            _pretty_print(resp)
    except requests.HTTPError as e:
        print("HTTP Error", e, file=sys.stderr)
        try:
            print(e.response.json(), file=sys.stderr)
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main() 