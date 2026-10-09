"""Host-side research source tools.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json  # noqa: F401
import os  # noqa: F401
import time  # noqa: F401
import random
import re
import threading
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote
from html.parser import HTMLParser
import xml.etree.ElementTree  # noqa: F401  (arXiv answers with Atom XML)

import httpx  # noqa: F401
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn with capped exponential backoff and jitter for RetryableError only; never sleep after the final attempt."""
    if attempts < 1 or base < 0 or cap < 0:
        raise ValueError("attempts must be positive; base and cap must be non-negative")
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            delay = (float(exc.retry_after) if exc.retry_after is not None
                     else base * 2 ** attempt + random.uniform(0, base))
            time.sleep(min(cap, max(0, delay)))


def _retry_after(value):
    if not value:
        return None
    try:
        return max(0, float(value))
    except ValueError:
        try:
            return max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None


def _request(method, url, **kwargs):
    try:
        response = httpx.request(method, url, timeout=45, follow_redirects=True, **kwargs)
    except httpx.TransportError as exc:
        raise RetryableError(str(exc)) from exc
    if response.status_code in {429, 500, 502, 503, 504}:
        raise RetryableError(f"HTTP {response.status_code}", _retry_after(response.headers.get("Retry-After")))
    response.raise_for_status()
    return response


def _redact(text):
    for name, key in os.environ.items():
        if key and ("KEY" in name or "TOKEN" in name or "SECRET" in name):
            text = text.replace(key, "[REDACTED]").replace(quote(key, safe=""), "[REDACTED]")
    return re.sub(r"(?i)(exaApiKey=)[^&\s\"']+", r"\1[REDACTED]", text)


def _error(exc):
    return _redact(f"ERROR: {type(exc).__name__}: {exc}")


def _compact(text):
    return " ".join(str(text or "").split())


def _records(items):
    return json.dumps(items, ensure_ascii=False) if items else "NO RESULTS"


_arxiv_lock = threading.Lock()
_last_arxiv_call = None


def _arxiv_request(params):
    global _last_arxiv_call
    # The lock covers retries and parallel researchers sharing this host.
    with _arxiv_lock:
        if _last_arxiv_call is not None:
            time.sleep(max(0, 3 - (time.monotonic() - _last_arxiv_call)))
        _last_arxiv_call = time.monotonic()
        return _request("GET", ARXIV_URL, params=params)


# ---- arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    try:
        terms = [term for term in re.findall(r"[^\W_]+(?:-[^\W_]+)*", query)
                 if term.upper() not in {"AND", "OR", "NOT", "ALL", "TI", "AU", "ABS", "CAT"}]
        if not terms:
            return "NO RESULTS"
        params = {"search_query": " AND ".join(f"all:{term}" for term in terms),
                  "sortBy": "submittedDate", "sortOrder": "descending", "start": 0,
                  "max_results": max(1, min(30, max_results))}
        response = with_retry(lambda: _arxiv_request(params),
                              attempts=int(os.getenv("ARXIV_RETRY_ATTEMPTS", "7")),
                              cap=float(os.getenv("ARXIV_RETRY_CAP", "60")))
        ns = {"a": "http://www.w3.org/2005/Atom"}
        items = []
        for entry in xml.etree.ElementTree.fromstring(response.text).findall("a:entry", ns):
            raw_id = entry.findtext("a:id", "", ns).strip()
            if "/abs/" not in raw_id:
                continue
            paper_id = re.sub(r"v\d+$", "", raw_id.split("/abs/", 1)[1])
            items.append({"id": paper_id, "url": f"https://arxiv.org/abs/{paper_id}", "source": "arxiv",
                          "published": entry.findtext("a:published", "", ns)[:10],
                          "title": _compact(entry.findtext("a:title", "", ns)),
                          "summary": _compact(entry.findtext("a:summary", "", ns))[:600]})
        return _records(items)
    except Exception as exc:
        return _error(exc)


def _hf_records(payload, prefer_ai=False):
    items = []
    for item in payload:
        paper = item.get("paper") or {}
        if not paper.get("id"):
            continue
        summary = ((paper.get("ai_summary") or item.get("ai_summary")) if prefer_ai else None)
        items.append({"id": paper["id"], "url": f"https://huggingface.co/papers/{paper['id']}",
                      "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
                      "title": _compact(paper.get("title") or item.get("title")),
                      "summary": _compact(summary or paper.get("summary") or item.get("summary"))[:600],
                      "upvotes": paper.get("upvotes") or item.get("upvotes") or 0,
                      "github": paper.get("githubRepo") or item.get("githubRepo") or "",
                      "stars": paper.get("githubStars") or item.get("githubStars") or 0})
    return items


# ---- Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        params = {"limit": max(1, min(100, limit))}
        if date:
            datetime.strptime(date, "%Y-%m-%d")
            params["date"] = date
        response = with_retry(lambda: _request("GET", HF_DAILY_URL, params=params))
        items = _hf_records(response.json())
        for item in items:
            item['source'] = 'hf-daily'
        if keyword:
            items = [item for item in items if keyword.casefold() in (item["title"] + " " + item["summary"]).casefold()]
        return _records(sorted(items, key=lambda item: item["upvotes"], reverse=True))
    except Exception as exc:
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        if not query.strip():
            return "NO RESULTS"
        response = with_retry(lambda: _request("GET", HF_SEARCH_URL,
                              params={"q": query, "limit": max(1, min(50, limit))}))
        items = _hf_records(response.json(), prefer_ai=True)
        for item in items:
            item['source'] = 'hf-search'
        return _records(items)
    except Exception as exc:
        return _error(exc)


def _mcp_call(name, arguments):
    key = os.getenv("EXA_API_KEY", "").strip()
    response = _request("POST", EXA_URL, params={"exaApiKey": key} if key else {},
                        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
                        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                              "params": {"name": name, "arguments": arguments}})
    if response.headers.get("content-type", "").startswith("application/json"):
        envelopes = [response.json()]
    else:
        envelopes = []
        # SSE data may span multiple lines in one event.
        for event in re.split(r"\r?\n\r?\n", response.text):
            data = "\n".join(line[5:].lstrip() for line in event.splitlines() if line.startswith("data:"))
            if data and data != "[DONE]":
                envelopes.append(json.loads(data))
    for envelope in envelopes:
        result = envelope.get("result") or {}
        content = "\n".join(block.get("text", "") for block in result.get("content", []) if block.get("type") == "text")
        meta = result.get("_meta") or {}
        # Vendors have used several flag names; recognize semantic rate-limit flags
        # as well as the HTTP-200 error message, never returning it as page evidence.
        meta_text = json.dumps(meta).lower()
        rate_flag = any(value and re.search(r"rate.?limit|throttl", str(field), re.I) for field, value in meta.items())
        error_text = json.dumps(envelope.get("error") or {})
        rate_message = re.search(r"rate[ _-]?limit\s+(?:exceeded|reached)|(?:hit|exceed)[^\n]{0,100}rate[ _-]?limit|too many requests|quota exceeded", content, re.I)
        rpc_rate_error = re.search(r"rate[ _-]?limit|too many requests|quota exceeded", error_text, re.I)
        if rate_flag or rate_message or rpc_rate_error or (
                "rate" in meta_text and "limit" in meta_text and "true" in meta_text):
            raise RetryableError("Exa rate limit", _retry_after(str(meta.get("retryAfter") or "")))
        if envelope.get("error"):
            raise RuntimeError(_redact(error_text))
        if result.get("isError"):
            raise RuntimeError(_redact(content or "Exa tool failed"))
        if "result" in envelope:
            return _redact(content.strip()) or "NO RESULTS"
    raise ValueError("Exa returned no JSON-RPC result")


# ---- web search / fetch through the Exa MCP endpoint ----
@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        if not query.strip():
            return "NO RESULTS"
        return with_retry(lambda: _mcp_call("web_search_exa", {"query": query,
                          "objective": objective or f"Find reliable sources about {query}",
                          "numResults": max(1, min(10, num_results))}), attempts=int(os.getenv("EXA_RETRY_ATTEMPTS", "7")),
                          cap=float(os.getenv("EXA_RETRY_CAP", "60")))
    except Exception as exc:
        return _error(exc)


@tool
def web_fetch(url: str) -> str:
    """Read one HTTP(S) page with Exa; optional direct-HTTP fallback when Exa is unavailable. Returns up to 12000 characters with URL and retrieval method."""
    try:
        if not url.startswith(("http://", "https://")):
            return "ERROR: expected an HTTP(S) URL"
        text = with_retry(lambda: _mcp_call("web_fetch_exa", {"urls": [url], "maxCharacters": 12000}),
                          attempts=int(os.getenv("EXA_RETRY_ATTEMPTS", "7")), cap=float(os.getenv("EXA_RETRY_CAP", "60")))
        return f"SOURCE_FAMILY: web\nURL: {url}\n" + text[:12000]
    except Exception as exc:
        if isinstance(exc, RetryableError) and os.getenv("WEB_FETCH_DIRECT_FALLBACK") == "1":
            try:
                response = with_retry(lambda: _request("GET", url), attempts=3)
                content_type = response.headers.get("content-type", "")
                if "text/html" in content_type:
                    parser = _PageText()
                    parser.feed(response.text)
                    text = "\n".join(parser.parts)
                elif "text/" in content_type or "json" in content_type:
                    text = response.text
                else:
                    return "ERROR: direct fetch returned an unsupported document type"
                return _redact(f"SOURCE_FAMILY: web\nURL: {url}\nFETCH_BACKEND: direct-http (Exa unavailable)\n" + text)[:12000]
            except Exception as fallback_exc:
                return _error(fallback_exc)
        return _error(exc)


class _PageText(HTMLParser):
    """Extract page text, excluding executable/style content."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(_compact(data))


# ---- registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        print(f"== {name}\n{fn.invoke(args)[:400]}\n", flush=True)
