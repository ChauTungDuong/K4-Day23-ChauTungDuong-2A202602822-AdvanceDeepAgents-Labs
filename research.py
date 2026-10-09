"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json  # noqa: F401
import os  # noqa: F401
import re  # noqa: F401
import sys
import time  # noqa: F401
from collections import Counter  # noqa: F401
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent  # noqa: F401
from model import make_model  # noqa: F401
from sandbox import download, open_sandbox, upload  # noqa: F401

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    s = str(topic).strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "-", s).strip("-")
    if not s:
        s = "topic"
    return s[:60].rstrip("-")


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Conduct a comprehensive literature survey on the topic: '{topic}'.\n\n"
        "Requirements:\n"
        "1. Start by planning with `write_todos` and split the topic into at least 3-4 distinct sub-questions.\n"
        "2. Delegate each sub-question to the `researcher` subagent via the `task` tool in parallel, providing full context (topic, sub-question, notes path in /tmp/work/research/notes/, and require sources from arxiv, huggingface (hf-search or hf-daily), and web in each note).\n"
        "3. Review the generated notes and ensure AT LEAST 3 distinct source families (arxiv, hf-search or hf-daily, and web) are represented in sources.json. If missing any of the 3 families, delegate a targeted search to fill the gap.\n"
        "4. Consolidate sources into /tmp/work/research/sources.json.\n"
        "5. Draft the complete survey report in /tmp/work/report/report.md following REPORT_TEMPLATE.md (do NOT include a ## References section).\n"
        "6. Execute /tmp/work/research/finalize_citations.py via the `execute` tool to automatically generate references and update sources.json.\n"
        "7. Execute /tmp/work/research/check_citations.py to validate citations. Iterate if necessary until it passes with OK.\n"
        "8. Delegate spot-checking of claims to `citation-checker`.\n"
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}."""
    tool_counts = Counter()
    subagent_calls = 0
    input_tokens = 0
    output_tokens = 0

    for msg in messages:
        calls = getattr(msg, "tool_calls", None) or (msg.get("tool_calls") if isinstance(msg, dict) else None)
        if calls:
            for c in calls:
                name = c.get("name") if isinstance(c, dict) else getattr(c, "name", None)
                if name:
                    tool_counts[name] += 1
                    if name == "task":
                        subagent_calls += 1

        meta = getattr(msg, "usage_metadata", None) or (msg.get("usage_metadata") if isinstance(msg, dict) else None)
        if meta and isinstance(meta, dict):
            input_tokens += meta.get("input_tokens", 0) or 0
            output_tokens += meta.get("output_tokens", 0) or 0

    return {
        "model": model_name,
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(tool_counts),
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
        },
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)

    if not report_bytes or not report_bytes.strip():
        raise RuntimeError(f"Report file {REPORT_PATH} is missing or empty in sandbox.")

    if not sources_bytes or not sources_bytes.strip():
        raise RuntimeError(f"Sources file {SOURCES_PATH} is missing or empty in sandbox.")

    try:
        sources_data = json.loads(sources_bytes.decode("utf-8"))
        if not isinstance(sources_data, list):
            raise ValueError("sources.json is not a JSON list")
    except Exception as e:
        raise RuntimeError(f"Failed to parse {SOURCES_PATH} as JSON: {e}")

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    slug = slugify(topic)
    summary_data = summarize(messages, elapsed, model_name)

    n_sources = len(sources_data)
    families = sorted(list({s.get("source") for s in sources_data if s.get("source")}))

    meta_data = {
        "topic": topic,
        "model": summary_data["model"],
        "elapsed_s": summary_data["elapsed_s"],
        "subagent_calls": summary_data["subagent_calls"],
        "tool_calls": summary_data["tool_calls"],
        "tokens": summary_data["tokens"],
        "n_sources": n_sources,
        "source_families": families,
    }

    report_file = reports_dir / f"{slug}.md"
    sources_file = reports_dir / f"{slug}.sources.json"
    meta_file = reports_dir / f"{slug}.meta.json"

    sources_file.write_text(json.dumps(sources_data, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_file.write_text(json.dumps(meta_data, ensure_ascii=False, indent=2), encoding="utf-8")
    report_file.write_bytes(report_bytes)

    return report_file


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic_str = str(topic).strip()
    if not topic_str:
        print("Usage: python research.py '<topic>'", file=sys.stderr)
        return 2

    model = make_model()
    if hasattr(model, "max_retries"):
        model.max_retries = 10
    model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or os.getenv("LAB_MODEL", "model")
    start = time.monotonic()

    print(f"[*] Starting research pipeline for topic: '{topic_str}'")
    print(f"[*] Model: {model_name}")

    try:
        with open_sandbox() as backend:
            print("[*] Sandbox opened successfully.")
            backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            upload(backend, {
                VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
            })

            agent = build_lead_agent(backend, model)
            print("[*] Invoking Lead Agent (this may take 2-4 minutes)...")
            result = agent.invoke(
                {"messages": [{"role": "user", "content": build_prompt(topic_str)}]},
                config={"recursion_limit": 1000},
            )

            messages = result.get("messages", [])
            elapsed = time.monotonic() - start
            print(f"[*] Agent finished in {elapsed:.1f}s. Saving outputs...")

            report_file = save_outputs(backend, topic_str, messages, elapsed, model_name)
            print(f"[+] SUCCESS: Report saved to {report_file}")
            return 0
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
