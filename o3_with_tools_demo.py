"""
===============================================================================
 o3_with_tools_demo.py
===============================================================================
A **fully-independent** Python 3 script that demonstrates how to use the *OpenAI*
`o3` model with **function-calling (a.k.a. tool use)**.

The goal is to provide a clear, heavily-commented example that you can copy into
any project and run immediately (after installing `openai` and `requests`).

Key features
------------
1. Shows how to define JSON-schema tool specifications.
2. Implements a couple of tiny but useful tools – a DuckDuckGo web search and a
   simple math helper – *inside* this file so there are **no external
   dependencies** beyond the two pip packages.
3. Demonstrates the full chat loop:
       user  →  assistant (tool call)  →  tool  →  assistant (final answer)
4. Tracks token usage and prints a tiny cost report at the end.

Prerequisites
-------------
1. `pip install openai requests`
2. Set the environment variable `OPENAI_API_KEY` with your key.

Usage
-----
```bash
python o3_with_tools_demo.py "Why is the sky blue? Also, what is 123 * 456?"
```
Feel free to tweak `QUESTION` at the bottom of this file.
"""

from __future__ import annotations

###############################################################################
# Standard library imports
###############################################################################
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

###############################################################################
# Third-party imports – keep the list minimal so the script stays independent
###############################################################################
import requests  # type: ignore – only used for a tiny web search helper
from openai import OpenAI  # type: ignore – pip install openai

###############################################################################
# Configuration – tweak these to taste
###############################################################################

# The o3 snapshot model name (April 2025).  Adjust if OpenAI releases a newer
# snapshot.
O3_MODEL = "o3-2025-04-16"

# DuckDuckGo imposes rate limits.  Keep *max_results* small for demo purposes.
DEFAULT_WEB_LIMIT = 5

###############################################################################
# Helper: ultra-lightweight DuckDuckGo HTML scraper
###############################################################################

def search_web(query: str, *, limit: int = DEFAULT_WEB_LIMIT) -> List[str]:
    """Very small helper that scrapes DuckDuckGo's HTML results page.

    It returns a *list of result titles* so the response remains short and cheap.

    Note: For anything production-grade you would use a proper search API with
    an official terms-of-service, *not* HTML scraping.  This is only for demo
    purposes.
    """

    # DuckDuckGo's HTML endpoint.  The JS-heavy mainstream page is harder to
    # scrape without a headless browser.
    DDG_HTML = "https://duckduckgo.com/html/"
    params = {"q": query, "kl": "us-en"}

    try:
        html = requests.get(DDG_HTML, params=params, timeout=10).text
    except Exception as exc:  # noqa: BLE001 – catch all for demo simplicity
        return [f"<error: {exc}>"]

    # Results are in <a class="result__a">TITLE</a>
    titles = re.findall(r"<a[^>]+class=\"result__a\"[^>]*>(.*?)</a>", html)

    # Strip HTML tags inside the title (rare, but keep it clean)
    clean = re.compile(r"<.*?>")
    out = [re.sub(clean, "", t) for t in titles]

    return out[:limit]

###############################################################################
# Another toy tool: basic multiplication (just to show multiple tools)
###############################################################################

def multiply(a: float, b: float) -> float:  # noqa: D401 – tiny helper
    """Return *a* multiplied by *b* (float)."""

    # A deliberately trivial example.  The goal is to showcase how the model can
    # call *any* deterministic function you expose – from file I/O to DB queries
    # or micro-services.
    return a * b

###############################################################################
# Build the JSON schema that we will pass to the ChatCompletion call
###############################################################################

def build_tool_specs() -> List[Dict[str, Any]]:
    """Return the list of *tool* definitions in OpenAI's JSON schema format."""

    return [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search DuckDuckGo and return a list of result titles.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search string"},
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of result titles (<=25)",
                            "default": DEFAULT_WEB_LIMIT,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "Return the product of two numbers (a * b).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First factor"},
                        "b": {"type": "number", "description": "Second factor"},
                    },
                    "required": ["a", "b"],
                },
            },
        },
    ]

###############################################################################
# Token usage tracking – completely optional, but nice for cost awareness
###############################################################################

token_usage: dict[str, dict[str, int]] = defaultdict(lambda: {"in": 0, "out": 0})


def _update_usage(model: str, resp) -> None:  # noqa: ANN001 – OpenAI resp obj
    """Accumulate token usage stats from an OpenAI response object."""

    try:
        usage = resp.usage
        token_usage[model]["in"] += getattr(usage, "prompt_tokens", 0) or 0
        token_usage[model]["out"] += getattr(usage, "completion_tokens", 0) or 0
    except AttributeError:
        # Safety: if openai changes naming we don't want the demo to break
        pass

###############################################################################
# Main chat loop with tool execution
###############################################################################

def chat_with_o3(question: str) -> str:
    """Run *question* through o3 with our two demo tools and return the answer."""

    tools = build_tool_specs()
    client = OpenAI()

    # ➤ Keep a running list of messages.  Start with system instructions & the
    #   user's question.
    messages: list[dict[str, str | list[dict[str, Any]]]] = [
        {
            "role": "system",
            "content": (
                "You are a concise, knowledgeable assistant. When relevant, "
                "use the provided tools to look up information or compute "
                "results. If you use a tool, explain the answer afterwards."
            ),
        },
        {"role": "user", "content": question},
    ]

    while True:
        # ------------------------------------------------------------------
        # 1. Ask the model for the *next* step (answer or tool call)
        # ------------------------------------------------------------------
        resp = client.chat.completions.create(
            model=O3_MODEL,
            messages=messages,  # our running context
            tools=tools,       # JSON schema definitions
            # The model is *encouraged* but not *forced* to use tools.  It decides.
            tool_choice="auto",
        )

        # Track token usage so we can print a mini cost report later.
        _update_usage(O3_MODEL, resp)

        msg = resp.choices[0].message  # shorthand

        # ------------------------------------------------------------------
        # 2. If the model wants to call a tool… do it!
        # ------------------------------------------------------------------
        if msg.tool_calls:
            # (The vast majority of cases will be a single call; loop for safety.)
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                print(f"[MODEL] Requested tool: {fn_name}({args})")

                # Dispatch to our local Python functions -------------------
                if fn_name == "search_web":
                    result = search_web(**args)
                elif fn_name == "multiply":
                    result = multiply(**args)
                else:  # ← should never happen unless spec & impl drift
                    result = f"<Unhandled tool name: {fn_name}>"

                # Append the tool *response* so the model can continue the chat
                messages.append(
                    {
                        "role": "tool",
                        "name": fn_name,
                        "content": json.dumps(result),  # must be a string
                    }
                )

            # Once we've executed the call(s), continue the `while` loop so the
            # model can *read* the tool outputs and produce the final answer.
            continue

        # ------------------------------------------------------------------
        # 3. Model returned a *direct* answer.  We're done!
        # ------------------------------------------------------------------
        answer = msg.content or ""
        return answer

###############################################################################
# CLI helper
###############################################################################

def _pretty_usage() -> str:
    """Return a one-liner string with token usage & estimated cost (USD)."""

    # Very rough cost approximation (April 2025 pricing – adjust as needed).
    COST_PER_1K_INPUT = 0.05  # $/1K tokens – invented number for demo only
    COST_PER_1K_OUTPUT = 0.15

    total_cost = 0.0
    parts: list[str] = []
    for model, usage in token_usage.items():
        in_tok = usage["in"]
        out_tok = usage["out"]
        cost = (in_tok / 1000) * COST_PER_1K_INPUT + (out_tok / 1000) * COST_PER_1K_OUTPUT
        total_cost += cost
        parts.append(f"{model}: in {in_tok}, out {out_tok}, cost ~${cost:.4f}")

    parts.append(f"TOTAL ≈ ${total_cost:.4f}")
    return " | ".join(parts)

###############################################################################
# Main guard – allows `python o3_with_tools_demo.py` to run directly
###############################################################################

if __name__ == "__main__":
    # You can change this question (or accept it from argv, env, etc.)
    QUESTION = sys.argv[1] if len(sys.argv) > 1 else "What's the capital of France, and what is 12 * 9?"

    start = time.time()
    final_answer = chat_with_o3(QUESTION)
    duration = time.time() - start

    print("\n============================== ANSWER ==============================")
    print(final_answer)
    print("==================================================================\n")

    print(f"Token usage: {_pretty_usage()}")
    print(f"Elapsed: {duration:.1f}s") 