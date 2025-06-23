# Multisource Research Assistant

A command-line helper that spins up one or more autonomous **search agents** powered by OpenAI (o3 or o4-mini).  Each agent can gather information from:

* DuckDuckGo Web Search (HTML scrape)
* Reddit posts (plus top comments)
* Bluesky posts

The collected evidence is summarised and passed to a **report compiler** model that produces the final analysis.  If the compiler feels the results are insufficient, it can request another search round via an internal `redo_search` tool.

## Why?

Quickly answer exploratory questions that benefit from perspectives across traditional web pages, social-media discussion (Reddit) and emerging networks (Bluesky) without juggling multiple APIs or manual browsing.

## Architecture

```
┌──────────────┐   1. strategy_model (o3/o4-mini)
│ generate     │      • Produces 3-8 search queries
│ search       │
│ objectives   │
└──────┬───────┘
       │queries[]
┌──────▼───────┐   2. N search agents (agent_model)
│ each agent   │      • Picks one query
│ uses tools   │      • Calls search_web / search_reddit / search_bsky
└──────┬───────┘
       │summaries[]
┌──────▼───────┐   3. report_model
│ compile      │      • Writes final report
│ final report │      • May call redo_search to loop back
└──────────────┘
```

## Installation

```bash
# Clone & enter workspace
$ git clone <repo> && cd Social-Search

# Recommended: Python 3.10+
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt  # see below
```

### Required Python packages

The project purposely avoids heavy dependencies.  Install the few we need:

```text
openai>=1.13
requests>=2.31
rich>=13.7  # optional but nice colours
python-dotenv>=1.0  # optional env-file support
```

Create a `requirements.txt` with the above if you prefer `pip -r`.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Your OpenAI key (mandatory) |
| `REDDIT_CLIENT_ID` & `REDDIT_CLIENT_SECRET` | Reddit app credentials |
| `BLUESKY_HANDLE` & `BLUESKY_APP_PASSWORD` | Bluesky login (optional – improves rate-limits) |

Place them in `.env` for convenience.

## Usage

```bash
$ python search_assistant.py  # or socialsearch init
```

1. Clarification phase — the assistant asks follow-up questions until it proposes a final objective:

   ```
   READY: <concise objective>
   ```
   You must then confirm with `y` (accept) or `n` (explain why, loop continues). Press `z` at any prompt to skip the phase entirely.

2. Source selection  
   • Reddit **and** Bluesky are always enabled (sentiment sources).  
   • DuckDuckGo Web search is optional (default **n**).

3. Model selection / number of agents — same as before.

During execution you will see, *for each generated search query*:

```
Search — <query>
Tool calls:
  1. search_reddit(query='…')
  2. search_bsky(query='…')
  …
Total cost so far: $0.0123
```

The agent is encouraged to perform deep dives (many tool calls) on Reddit and Bluesky to surface real user sentiment. The report compiler will call `redo_search` automatically if it feels more evidence is required.

### Skip everything quickly
If you want a totally non-interactive run you can feed inputs through stdin, e.g.

```bash
echo -e "My question\nz\n\n\no4-mini\no3\n" | socialsearch init | cat
```

(The first `z` skips clarifications.)

## Extending functionality

* Add new search back-ends by:
  1. Implementing a simple Python function that returns JSON-serialisable results.
  2. Registering a matching schema in `build_tool_specs()`.
  3. Handling the tool call in `execute_tool()`.
* All OpenAI calls are centralised, so adding caching or async batching is straightforward.

## Caveats

* DuckDuckGo HTML scraping is brittle and for light personal use only.
* The project is **not** production-grade: no rate-limit back-off, retries or robust error handling.
* Token counts rely on the `usage` field from the OpenAI response and may vary slightly from billing.

## License

MIT – do what you like, just don't blame me. 