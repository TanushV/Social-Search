from __future__ import annotations

"""reddit_analysis.py
Analytical helpers that sit *on top* of ``reddit_search.fetch_posts_and_comments``
so that we can provide **insight** (trend detection & delta summaries) rather than
just raw posts.  This implements step #4 of the recommendations, but for Reddit
only.

Current capabilities
--------------------
* ``compute_volume_trend`` – compare the number of posts in the last 24 h with
  the trailing-7-day window and report a percentage change.
* ``quick_report`` – fetch the two buckets, compute the delta and return a
  Markdown string ready for a human-readable dashboard.

Both helpers avoid persistent state; they make two on-demand API calls and keep
results in memory only.  They depend solely on ``reddit_search`` for data
collection, so they inherit its credential handling & env-var requirements.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple

from reddit_search import fetch_posts_and_comments  # type: ignore

__all__ = [
    "compute_volume_trend",
    "quick_report",
]

# ---------------------------------------------------------------------------
# Internal helpers ----------------------------------------------------------
# ---------------------------------------------------------------------------

def _window_bounds(hours_back: int) -> Tuple[int, int]:
    """Return (after, before) epoch seconds for *hours_back*-long window."""
    before = datetime.now(tz=timezone.utc)
    after = before - timedelta(hours=hours_back)
    return int(after.timestamp()), int(before.timestamp())


# ---------------------------------------------------------------------------
# Public API ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def compute_volume_trend(
    query: str,
    *,
    client_id: str | None = None,
    client_secret: str | None = None,
    user_agent: str = "reddit_analysis/0.1 (by github.com/example)",
) -> Dict[str, Any]:
    """Return a dict with counts and percentage change for *query*.

    The function issues two Reddit searches:
    1. Posts from the **last 24 h** (t="day").
    2. Posts from the **previous 7 d** (t="week").

    It then computes ``pct_change = (day/week_avg) - 1``.  Negative values
    imply a downturn, positive values an uptick.
    """

    # Recent window (1 day) --------------------------------------------------
    posts_day = fetch_posts_and_comments(
        query,
        post_limit=100,
        comment_limit=0,
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        time_filter="day",
    )
    n_day = len(posts_day)

    # Baseline window (1 week) ---------------------------------------------
    posts_week = fetch_posts_and_comments(
        query,
        post_limit=350,  # 50/posts per day × 7 ≈ 350 (approximation)
        comment_limit=0,
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        time_filter="week",
    )
    n_week = len(posts_week)

    # Avoid division by zero -------------------------------------------------
    avg_per_day = n_week / 7 if n_week else 0.0
    pct_change = ((n_day / avg_per_day) - 1.0) * 100 if avg_per_day else None

    return {
        "query": query,
        "last_24h": n_day,
        "weekly_avg_per_day": avg_per_day,
        "pct_change": pct_change,
    }


def quick_report(**kwargs) -> str:  # type: ignore[override]
    """Human-readable Markdown summary wrapper around ``compute_volume_trend``."""
    stats = compute_volume_trend(**kwargs)

    if stats["pct_change"] is None:
        return (
            "### Trend Report (Reddit)\n"
            "Not enough baseline data to compute a trend for the query: "
            f"`{stats['query']}`."
        )

    arrow = "⬆️" if stats["pct_change"] > 0 else "⬇️"
    return (
        "### Trend Report (Reddit)\n"
        f"**Query:** `{stats['query']}`\n\n"
        f"• Last 24 h: **{stats['last_24h']}** posts\n"
        f"• Baseline (avg/day over prev-7-d): **{stats['weekly_avg_per_day']:.1f}** posts\n"
        f"• Change: **{stats['pct_change']:+.1f}%** {arrow}"
    ) 