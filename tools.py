"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

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
import random
import re
import time  # noqa: F401
import xml.etree.ElementTree as ET  # noqa: F401  (arXiv answers with Atom XML)

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

_LAST_ARXIV_CALL = 0.0

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


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError or temporary network errors, wait and call it again."""
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as e:
            if attempt == attempts - 1:
                raise
            delay = e.retry_after if e.retry_after is not None else (base * (2 ** attempt) + random.uniform(0.0, 1.0))
            delay = min(delay, cap)
            time.sleep(delay)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status not in (429, 500, 502, 503, 504):
                raise
            if attempt == attempts - 1:
                raise
            retry_after = None
            ra_hdr = e.response.headers.get("Retry-After")
            if ra_hdr:
                try:
                    retry_after = float(ra_hdr)
                except ValueError:
                    pass
            delay = retry_after if retry_after is not None else (base * (2 ** attempt) + random.uniform(0.0, 1.0))
            delay = min(delay, cap)
            time.sleep(delay)
        except httpx.TransportError:
            if attempt == attempts - 1:
                raise
            delay = min(base * (2 ** attempt) + random.uniform(0.0, 1.0), cap)
            time.sleep(delay)


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    global _LAST_ARXIV_CALL
    try:
        terms = re.findall(r"[\w-]+", query)
        if not terms:
            return "NO RESULTS"
        search_query = " AND ".join(f"all:{t}" for t in terms)
        clamped_max = max(1, min(max_results, 30))
        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": clamped_max,
        }

        def _fetch():
            global _LAST_ARXIV_CALL
            now = time.time()
            elapsed = now - _LAST_ARXIV_CALL
            if elapsed < 3.0:
                time.sleep(3.0 - elapsed)
            _LAST_ARXIV_CALL = time.time()

            with httpx.Client(timeout=30.0) as client:
                resp = client.get(ARXIV_URL, params=params)
                resp.raise_for_status()
                return resp.text

        xml_text = with_retry(_fetch, attempts=2, base=1.0, cap=3.0)
        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if not entries:
            return "NO RESULTS"

        records = []
        for entry in entries:
            id_elem = entry.find("atom:id", ns)
            title_elem = entry.find("atom:title", ns)
            summary_elem = entry.find("atom:summary", ns)
            pub_elem = entry.find("atom:published", ns)

            raw_id = (id_elem.text or "").strip()
            id_part = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id
            clean_id = re.sub(r"v\d+$", "", id_part)
            url = f"https://arxiv.org/abs/{clean_id}"

            published = (pub_elem.text or "").strip()[:10] if pub_elem is not None else ""
            title = " ".join((title_elem.text or "").split()) if title_elem is not None else "Untitled"
            summary = " ".join((summary_elem.text or "").split())[:600] if summary_elem is not None else ""

            records.append({
                "id": clean_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
            })

        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        clamped_limit = max(1, min(limit, 100))
        params = {"limit": clamped_limit}
        if date.strip():
            params["date"] = date.strip()

        def _fetch():
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(HF_DAILY_URL, params=params)
                resp.raise_for_status()
                return resp.json()

        data = with_retry(_fetch, attempts=5, base=1.0, cap=30.0)
        if not isinstance(data, list) or not data:
            return "NO RESULTS"

        records = []
        for item in data:
            paper = item.get("paper") if isinstance(item, dict) and "paper" in item else item
            if not isinstance(paper, dict):
                continue
            paper_id = paper.get("id")
            if not paper_id:
                continue

            raw_title = paper.get("title") or item.get("title") or ""
            raw_summary = paper.get("summary") or item.get("summary") or ""
            clean_title = " ".join(str(raw_title).split())
            clean_summary = " ".join(str(raw_summary).split())[:600]

            if keyword.strip():
                kw = keyword.strip().lower()
                if kw not in clean_title.lower() and kw not in clean_summary.lower():
                    continue

            records.append({
                "id": paper_id,
                "url": f"https://huggingface.co/papers/{paper_id}",
                "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
                "title": clean_title,
                "summary": clean_summary,
                "upvotes": paper.get("upvotes", 0) or 0,
                "github": paper.get("githubRepo") or None,
                "stars": paper.get("githubStars") or None,
            })

        records.sort(key=lambda r: r.get("upvotes", 0) or 0, reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        clamped_limit = max(1, min(limit, 50))
        params = {"q": query.strip(), "limit": clamped_limit}

        def _fetch():
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(HF_SEARCH_URL, params=params)
                resp.raise_for_status()
                return resp.json()

        data = with_retry(_fetch, attempts=5, base=1.0, cap=30.0)
        if not isinstance(data, list) or not data:
            return "NO RESULTS"

        records = []
        for item in data:
            paper = item.get("paper") if isinstance(item, dict) and "paper" in item else item
            if not isinstance(paper, dict):
                continue
            paper_id = paper.get("id")
            if not paper_id:
                continue

            raw_title = paper.get("title") or item.get("title") or ""
            summary_cand = paper.get("ai_summary") or paper.get("summary") or item.get("summary") or ""
            clean_title = " ".join(str(raw_title).split())
            clean_summary = " ".join(str(summary_cand).split())[:600]

            records.append({
                "id": paper_id,
                "url": f"https://huggingface.co/papers/{paper_id}",
                "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
                "title": clean_title,
                "summary": clean_summary,
                "upvotes": paper.get("upvotes", 0) or 0,
                "github": paper.get("githubRepo") or None,
                "stars": paper.get("githubStars") or None,
            })

        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _call_exa_mcp(tool_name: str, arguments: dict) -> str:
    exa_key = (os.getenv("EXA_API_KEY") or "").strip()
    url = f"{EXA_URL}?exaApiKey={exa_key}" if exa_key else EXA_URL
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }

    def _fetch():
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            text = resp.text

            data_json = None
            for line in text.splitlines():
                if line.startswith("data:"):
                    raw = line[len("data:"):].strip()
                    if raw:
                        try:
                            data_json = json.loads(raw)
                            break
                        except json.JSONDecodeError:
                            continue
            if data_json is None:
                try:
                    data_json = json.loads(text)
                except json.JSONDecodeError:
                    raise RuntimeError("Failed to parse Exa MCP response")

            if "error" in data_json:
                err_msg = str(data_json["error"])
                if "rate limit" in err_msg.lower():
                    raise RetryableError(f"Exa rate limited: {err_msg}", retry_after=20)
                raise RuntimeError(f"Exa error: {err_msg}")

            result = data_json.get("result", {})
            meta = result.get("_meta", {})
            if meta.get("rate_limited") or meta.get("rateLimit") or "rate limit" in str(meta).lower():
                raise RetryableError("Exa rate limit flag in _meta", retry_after=20)

            contents = result.get("content", [])
            extracted = []
            for c in contents:
                c_text = c.get("text", "")
                if "rate limit" in c_text.lower() and "exceeded" in c_text.lower():
                    raise RetryableError(f"Exa rate limit text: {c_text[:100]}", retry_after=20)
                if c.get("type") == "text" and c_text.strip():
                    extracted.append(c_text.strip())

            return "\n\n".join(extracted)

    try:
        content = with_retry(_fetch, attempts=6, base=2.0, cap=60.0)
        return content.strip() if content.strip() else "NO RESULTS"
    except Exception as e:
        msg = f"ERROR: {type(e).__name__}: {e}"
        if exa_key:
            msg = msg.replace(exa_key, "<REDACTED>")
        return msg


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    obj = objective.strip() or f"search for research on {query.strip()}"
    num = max(1, min(num_results, 10))
    return _call_exa_mcp("web_search_exa", {"query": query.strip(), "objective": obj, "numResults": num})


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    content = _call_exa_mcp("web_fetch_exa", {"urls": [url.strip()]})
    if content.startswith("ERROR:") or content == "NO RESULTS":
        return content
    return content[:12000]


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
