import os
import sys
import json
from typing import Any, Dict, List, DefaultDict, Optional

# NEW IMPORTS --------------------------------------------------------------
# Remove dependency on api_clients and instead use our local helper modules.
from bluesky_search import BlueskyClient  # type: ignore
from reddit_enhanced import search_reddit_full  # type: ignore
import requests
import re
from collections import defaultdict

# ---------------------------------------------------------------------------
# Optional environment & rich setup -----------------------------------------
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass  # dotenv is optional

try:
    from rich import print
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
except ModuleNotFoundError:

    def print(*args, **kwargs):  # type: ignore
        __builtins__.print(*args, **kwargs)

    class Console:  # type: ignore
        def rule(self, *args, **kwargs):
            print("-" * 80)

    class Prompt:  # type: ignore
        @staticmethod
        def ask(msg: str, **kwargs):
            return input(msg + " ")

    class Confirm:  # type: ignore
        @staticmethod
        def ask(msg: str, **kwargs):
            return input(msg + " [y/N] ").lower().startswith("y")

    class Table:  # type: ignore
        def __init__(self, *args, **kwargs):
            self.rows: List[List[str]] = []

        def add_column(self, *args, **kwargs):
            pass

        def add_row(self, *cols):
            self.rows.append(list(cols))

        def __rich_console__(self, *args, **kwargs):
            for row in self.rows:
                print(" | ".join(row))

console = Console()

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config --------------------------------------------------------------------
# ---------------------------------------------------------------------------

O3_MODEL = "o3-2025-04-16"
O4_MODEL = "o4-mini-2025-04-16"

# ---------------------------------------------------------------------------
# Token usage tracking -------------------------------------------------------
# ---------------------------------------------------------------------------

token_usage: DefaultDict[str, Dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0})

def _add_usage(model: str, resp) -> None:
    """Accumulate token usage from an OpenAI response object."""
    try:
        usage = resp.usage
        token_usage[model]["input"] += getattr(usage, "prompt_tokens", 0) or 0
        token_usage[model]["output"] += getattr(usage, "completion_tokens", 0) or 0
    except AttributeError:
        pass

# ---------------------------------------------------------------------------
# Source-specific helper functions ------------------------------------------
# ---------------------------------------------------------------------------

def search_web(query: str, *, max_results: int = 10) -> List[str]:
    """Very lightweight DuckDuckGo HTML scraper returning result titles."""
    params = {"q": query, "kl": "us-en"}
    try:
        html = requests.get("https://duckduckgo.com/html/", params=params, timeout=10).text
    except Exception as e:
        return [f"[error] {e}"]

    titles = re.findall(r"<a[^>]+class=\"result__a\"[^>]*>(.*?)</a>", html)
    clean = re.compile(r"<.*?>")
    out = [re.sub(clean, "", t) for t in titles]
    return out[: max_results]


def _search_bsky(
    query: str,
    *,
    limit: int = 10,
    sort: str | None = None,
    min_likes: int = 0,
    min_reposts: int = 0,
    min_replies: int = 0,
) -> List[Dict[str, str]]:
    """Return simplified Bluesky posts filtered by engagement."""
    client = BlueskyClient(handle=os.getenv("BLUESKY_HANDLE"), app_password=os.getenv("BLUESKY_APP_PASSWORD"))

    # Fetch generously to allow filtering
    raw = client.search_posts(query, limit=min(100, limit * 2), sort=sort)
    posts = raw.get("posts", [])

    results: List[Dict[str, str]] = []
    for p in posts:
        if (
            p.get("likeCount", 0) >= min_likes
            and p.get("repostCount", 0) >= min_reposts
            and p.get("replyCount", 0) >= min_replies
        ):
            rec = p.get("record", {})
            text = rec.get("text", "")
            uri = p.get("uri", "")
            results.append({"text": text, "uri": uri})
            if len(results) >= limit:
                break
    return results


# ---------------------------------------------------------------------------
# Tool Definitions ----------------------------------------------------------
# ---------------------------------------------------------------------------

def build_tool_specs(sources: List[str]) -> List[Dict[str, Any]]:
    """Return OpenAI tool schema for selected sources plus the redo_search tool."""

    specs: List[Dict[str, Any]] = []

    if "web" in sources:
        specs.append(
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
                                "default": 10,
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        )

    if "reddit" in sources:
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": "search_reddit",
                    "description": "Return BM25-ranked Reddit posts (full text + top comments) plus a volume-trend snapshot. Results are limited to the top *top_k* posts (default 5) most relevant to the query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search string"},
                            "top_k": {
                                "type": "integer",
                                "description": "Number of posts to return after BM25 ranking (<=25)",
                                "default": 5,
                            },
                            "comment_limit": {
                                "type": "integer",
                                "description": "Top comments per post (<=10)",
                                "default": 5,
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        )

    if "bsky" in sources:
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": "search_bsky",
                    "description": "Return Bluesky posts relevant to the query using app.bsky.feed.searchPosts.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search string"},
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of posts (<=25)",
                                "default": 10,
                            },
                            "sort": {
                                "type": "string",
                                "enum": ["top", "latest"],
                                "description": "Ordering of results (optional)",
                            },
                            "min_likes": {
                                "type": "integer",
                                "default": 0,
                                "description": "Filter out posts with fewer likes (optional)",
                            },
                            "min_reposts": {
                                "type": "integer",
                                "default": 0,
                                "description": "Filter out posts with fewer reposts (optional)",
                            },
                            "min_replies": {
                                "type": "integer",
                                "default": 0,
                                "description": "Filter out posts with fewer replies (optional)",
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        )

    # Allow the report agent to trigger another search cycle if needed
    specs.append(
        {
            "type": "function",
            "function": {
                "name": "redo_search",
                "description": "Signal that additional or revised searches are needed to answer the question.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "Why further searches are necessary"}
                    },
                    "required": ["reason"],
                },
            },
        }
    )

    return specs


# Map OpenAI tool-call names to local python functions

def execute_tool(name: str, args: Dict[str, Any]) -> str:
    query = args.get("query")
    limit = int(args.get("limit", 10))

    if name == "search_web":
        results = search_web(query, max_results=limit)
        return json.dumps(results)

    if name == "search_reddit":
        top_k = int(args.get("top_k", 5))
        top_k = max(1, min(top_k, 25))  # safety
        comment_limit = int(args.get("comment_limit", 5))
        comment_limit = max(0, min(comment_limit, 10))
        data = search_reddit_full(query, top_k=top_k, comment_limit=comment_limit)
        return json.dumps(data)

    if name == "search_bsky":
        sort = args.get("sort")
        min_likes = int(args.get("min_likes", 0))
        min_reposts = int(args.get("min_reposts", 0))
        min_replies = int(args.get("min_replies", 0))
        data = _search_bsky(
            query,
            limit=limit,
            sort=sort,
            min_likes=min_likes,
            min_reposts=min_reposts,
            min_replies=min_replies,
        )
        return json.dumps(data)

    if name == "redo_search":
        return "OK"

    raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Conversation Helpers ------------------------------------------------------
# ---------------------------------------------------------------------------

client = OpenAI()

def generate_search_queries(question: str, *, strategy_model: str, context: str = "", max_queries: int | None = None) -> List[str]:
    """Return a list of search phrases, optionally capped at *max_queries*."""
    directive = (
        "You are a strategist whose goal is to design search phrases for social-media sentiment research. "
        "Given the user's research objective and background context, output a JSON object with a key 'queries' containing 4–8 concise search phrases. "
        "Guidelines:\n"
        "• Prioritise wording that uncovers real user opinions: pain points, complaints, wish-lists, success stories.\n"
        "• Favour plain language a Reddit user would type (≤10 words, no unnecessary dates).\n"
        "• Cover multiple angles of the topic so that subsequent social searches reveal sentiment diversity.\n"
        "Return ONLY valid JSON."
    )
    if max_queries:
        directive += f" Limit the list to at most {max_queries} items."

    messages = [
        {"role": "system", "content": directive},
        {"role": "user", "content": f"Objective: {question}\n\nBackground:\n{context}"},
    ]

    resp = client.chat.completions.create(
        model=strategy_model,
        response_format={"type": "json_object"},
        messages=messages,
    )
    _add_usage(strategy_model, resp)
    data = json.loads(resp.choices[0].message.content)
    queries: List[str] = data.get("queries", [])
    # Enforce hard upper-bound of 100 searches to avoid runaway cost.
    if max_queries:
        max_queries = min(max_queries, 100)
        queries = queries[:max_queries]
    return queries


def gather_results_for_query(query: str, tools: List[Dict[str, Any]], *, agent_model: str):
    """Run a single search agent; return (summary, call_logs)."""
    call_logs: List[Dict[str, Any]] = []
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a Vociro agent focused on measuring USER SENTIMENT. Deep-dive: make as many tool calls as necessary (do NOT worry about limits) to capture diverse viewpoints. Prioritise Reddit for opinions & pain points, Bluesky for fresh takes, and use Web only if social tools are empty. After gathering, summarise sentiment strictly from tool results; do NOT hallucinate."
            ),
        },
        {"role": "user", "content": query},
    ]

    max_rounds = 20  # safeguard to avoid infinite tool loops
    rounds = 0
    while True:
        if rounds >= max_rounds:
            # Stop the loop and return whatever we have
            summary = "[aborted] tool loop exceeded safety limit"
            return summary, call_logs

        resp = client.chat.completions.create(
            model=agent_model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        _add_usage(agent_model, resp)
        msg = resp.choices[0].message

        if msg.tool_calls:
            # Append assistant message ONCE, then handle each tool call
            messages.append(msg)
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                try:
                    result = execute_tool(name, args)
                except Exception as e:
                    result = json.dumps({"error": str(e)})

                # record call log
                call_logs.append({"name": name, "args": args})

                # Echo tool result back to model
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": result,
                    }
                )
            rounds += 1
            continue  # continue agent loop

        # No tool calls → agent produced its summary
        messages.append(msg)
        summary = (msg.content or "").strip()
        return summary, call_logs


def produce_final_report(question: str, summaries: List[str], tools: List[Dict[str, Any]], *, report_model: str):
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are an expert analyst. Using the data provided, write a clear, well-structured report that answers the user's question. If you feel more evidence is needed, call the redo_search tool with an explanation."
            ),
        },
        {"role": "user", "content": question},
        {"role": "assistant", "content": "\n\n".join(summaries)},
    ]

    max_rounds = 5  # avoid endless redo loops
    rounds = 0
    while True:
        if rounds >= max_rounds:
            # Give up and return concatenated summaries
            fallback = "\n\n".join(summaries)
            return True, fallback

        resp = client.chat.completions.create(
            model=report_model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        _add_usage(report_model, resp)
        msg = resp.choices[0].message

        if msg.tool_calls and msg.tool_calls[0].function.name == "redo_search":
            reason = json.loads(msg.tool_calls[0].function.arguments)["reason"]
            print(
                f"[yellow]{report_model} requested another search: {reason}[/yellow]",
            )
            return False, reason

        return True, (msg.content or "").strip()


# ---------------------------------------------------------------------------
# Orchestration -------------------------------------------------------------
# ---------------------------------------------------------------------------

def run():
    console.rule("Vociro — Multisource Research Assistant")
    initial_question = Prompt.ask("Enter your research question")

    # Clarification loop returns refined objective and dialog context
    question, dialog_ctx = clarification_phase(initial_question)

    # ----------------------------------------
    # Source selection
    # ----------------------------------------
    sources: List[str] = ["reddit", "bsky"]  # mandatory social sources
    if Confirm.ask("Include Web search (DuckDuckGo)?", default=False):
        sources.append("web")

    if not sources:
        print("[red]You must select at least one data source.[/red]")
        return

    # ----------------------------------------
    # Model choices
    # ----------------------------------------
    agent_model_inp = Prompt.ask("Model for search agents (o3 / o4-mini)", default="o3")
    agent_model = O3_MODEL if agent_model_inp.startswith("o3") else O4_MODEL

    report_model_inp = Prompt.ask("Model for report compilation (o3 / o4-mini)", default="o3")
    report_model = O3_MODEL if report_model_inp.startswith("o3") else O4_MODEL

    # ----------------------------------------
    # Number of agents (i.e. search queries)
    # ----------------------------------------
    agents_raw = Prompt.ask("Maximum number of search agents (press Enter for auto)", default="")
    max_agents = int(agents_raw) if agents_raw.isdigit() and int(agents_raw) > 0 else None
    if max_agents and max_agents > 100:
        print("[yellow]Capping maximum agents to 100.[/yellow]")
        max_agents = 100

    tools = build_tool_specs(sources)

    # ----------------------------------------
    # Main loop with redo capability
    # ----------------------------------------
    while True:
        console.rule(f"Generating search queries with {agent_model} ...")
        searches = generate_search_queries(question, strategy_model=agent_model, context=dialog_ctx, max_queries=max_agents)
        print(f"[bold]Queries:[/bold] {searches}")

        summaries: List[str] = []
        agent_logs: List[Dict[str, Any]] = []
        for q in searches:
            console.rule(f"Search — {q}")
            summary, call_logs = gather_results_for_query(q, tools, agent_model=agent_model)

            # Display tool calls
            if call_logs:
                print("[cyan]Tool calls:[/cyan]")
                for idx, cl in enumerate(call_logs, 1):
                    arg_preview = ", ".join(f"{k}={repr(v)}" for k, v in cl["args"].items())
                    print(f"  {idx}. {cl['name']}({arg_preview})")
            print(f"[green]Total cost so far: ${_current_cost():.4f}[/green]\n")

            summaries.append(f"### {q}\n{summary}")
            agent_logs.append({"query": q, "tool_calls": call_logs})

        console.rule(f"Compiling final report with {report_model} ...")
        ok, report_or_reason = produce_final_report(question, summaries, tools, report_model=report_model)
        if ok:
            console.rule("Report")
            print(report_or_reason)
            break
        else:
            print("Redoing search as requested...")
            # Loop restarts to satisfy redo

    # ----------------------------------------
    # Cost summary
    # ----------------------------------------
    console.rule("Token & Cost Summary")
    total_cost = 0.0
    for model, stats in token_usage.items():
        in_tok = stats["input"]
        out_tok = stats["output"]
        if model.startswith("o3"):
            cost = in_tok * 2e-6 + out_tok * 8e-6
        else:
            cost = in_tok * 1.1e-6 + out_tok * 4.4e-6
        total_cost += cost
        print(f"{model}: {in_tok} input, {out_tok} output tokens -> ${cost:.4f}")
    print(f"[bold]Estimated total cost: ${total_cost:.4f}[/bold]")


def _current_cost() -> float:
    total = 0.0
    for model, stats in token_usage.items():
        in_tok = stats["input"]
        out_tok = stats["output"]
        if model.startswith("o3"):
            total += in_tok * 2e-6 + out_tok * 8e-6
        else:
            total += in_tok * 1.1e-6 + out_tok * 4.4e-6
    return total


def clarification_phase(initial_q: str, model: str = O3_MODEL):
    """Interactive loop returning (objective, dialog_str)."""
    console.rule("Clarification Phase")
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Ask concise clarifying questions until you fully understand the user's research objective. "
                "When you have enough information, respond with ONLY the single line 'READY: <concise objective>'."
            ),
        },
        {"role": "user", "content": initial_q},
    ]
    dialog_log = [f"USER: {initial_q}"]
    max_turns = 10  # prevent infinite questioning
    turns = 0
    while True:
        if turns >= max_turns:
            objective = initial_q
            dialog_str = "\n".join(dialog_log)
            return objective, dialog_str

        resp = client.chat.completions.create(model=model, messages=messages)
        _add_usage(model, resp)
        reply = resp.choices[0].message.content.strip()
        if reply.upper().startswith("READY:"):
            objective = reply[len("READY:"):].strip() or initial_q
            print(f"[green]{reply}[/green]")
            confirm = Prompt.ask("Is this an accurate objective? (y/n)", default="y")
            if confirm.lower().startswith("y"):
                dialog_str = "\n".join(dialog_log)
                return objective, dialog_str
            # user disagrees -> solicit reason and continue loop
            reason = Prompt.ask("Please provide clarification (or 'z' to skip)")
            if reason.lower() == 'z':
                dialog_str = "\n".join(dialog_log)
                return objective, dialog_str
            dialog_log.append(f"USER: {reason}")
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": reason})
            turns += 1
            continue
        print(f"[yellow]{reply}[/yellow]")
        user_ans = Prompt.ask("Your answer (or 'z' to skip)")
        if user_ans.lower() == 'z':
            objective = initial_q
            dialog_str = "\n".join(dialog_log)
            return objective, dialog_str
        dialog_log.append(f"ASSISTANT: {reply}")
        dialog_log.append(f"USER: {user_ans}")
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": user_ans})
        turns += 1


# ----------------------------------------------------------------------------
# Ensure UTF-8 console encoding (esp. Windows) -------------------------------
# ----------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        # If reconfigure isn't supported (older Python or redirected I/O), ignore.
        pass

# ---------------------------------------------------------------------------
# Public wrapper -------------------------------------------------------------
# ---------------------------------------------------------------------------

def search(
    question: str,
    *,
    include_web: bool = False,
    agent_model: str = "o3",
    report_model: str = "o3",
    max_agents: int | None = None,
    return_json: bool = False,
):
    """Run a complete Vociro search programmatically and return the final report.

    Parameters
    ----------
    question : str
        The research question / objective.
    include_web : bool, default False
        Whether to search the public web (DuckDuckGo) in addition to Reddit & Bluesky.
    agent_model : {"o3", "o4-mini"}, default "o3"
        OpenAI model identifier for the search agents.
    report_model : {"o3", "o4-mini"}, default "o3"
        Model used to compile the final report.
    max_agents : int | None, optional
        Maximum number of generated search queries (agents). None = auto (3-8).
    return_json : bool, default False
        Whether to return the result as a JSON object including tool calls.

    Returns
    -------
    str | dict[str, Any]
        By default a Markdown report string.  If *return_json* is True, a dict
        with keys:

        * "response" – compiled report (Markdown)
        * "agents" – list of objects `{"query", "tool_calls"}`
    """

    sources: list[str] = ["reddit", "bsky"]
    if include_web:
        sources.append("web")

    agent_model_id = O3_MODEL if str(agent_model).startswith("o3") else O4_MODEL
    report_model_id = O3_MODEL if str(report_model).startswith("o3") else O4_MODEL

    tools = build_tool_specs(sources)

    if max_agents and max_agents > 100:
        max_agents = 100  # enforce global safety cap

    searches = generate_search_queries(
        question,
        strategy_model=agent_model_id,
        max_queries=max_agents,
    )
    summaries: list[str] = []
    agent_logs: list[dict[str, Any]] = []
    for q in searches:
        summary, call_logs = gather_results_for_query(q, tools, agent_model=agent_model_id)
        summaries.append(f"### {q}\n{summary}")
        agent_logs.append({"query": q, "tool_calls": call_logs})

    ok, report = produce_final_report(question, summaries, tools, report_model=report_model_id)
    if ok and (not report or not report.strip()):
        # Rare edge-case: model returned empty content. Fallback to concatenated summaries.
        report = "\n\n".join(summaries)

    if return_json:
        return {"response": report, "agents": agent_logs, "queries": searches}
    return report


if __name__ == "__main__":
    run() 