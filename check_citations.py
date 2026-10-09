"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


import re
from urllib.parse import urlparse

_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")   # [3]  [1, 2]  [1-3]  [2-3]; not [3](link)
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_URL = re.compile(r"https?://[^\s)\]]+")


def _group_numbers(group):
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            numbers.append(int(part))
    return numbers


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    problems = []
    if not isinstance(sources, list) or len(sources) == 0:
        return ["no sources in sources.json"]

    source_by_n = {}
    seen_urls = {}
    for idx, s in enumerate(sources):
        if not isinstance(s, dict):
            problems.append(f"source at index {idx} is not a dictionary")
            continue
        n = s.get("n")
        if not isinstance(n, int):
            problems.append(f"source entry {idx} has non-integer n: {n!r}")
            continue
        if n in source_by_n:
            problems.append(f"duplicate source number n={n}")
        source_by_n[n] = s

        url = s.get("url")
        if not url or not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source [{n}] has invalid url: {url!r}")
        else:
            clean_url = url.rstrip("/")
            if clean_url in seen_urls:
                problems.append(f"duplicate url in sources.json: {url} (shared by [{seen_urls[clean_url]}] and [{n}])")
            else:
                seen_urls[clean_url] = n

    ref_matches = list(_REF_HEADING.finditer(report_text))
    if not ref_matches:
        problems.append("missing heading '## References'")
        return problems

    ref_heading_pos = ref_matches[-1]
    body = report_text[:ref_heading_pos.start()]
    references_text = report_text[ref_heading_pos.end():]

    segments = _CODE.split(body)
    cited_numbers = set()
    for i, seg in enumerate(segments):
        if i % 2 == 1:
            continue
        for match in _GROUP.finditer(seg):
            for num in _group_numbers(match.group(1)):
                cited_numbers.add(num)

    if not cited_numbers:
        problems.append("no citations found in report body")

    for n in sorted(cited_numbers):
        if n not in source_by_n:
            problems.append(f"[{n}] cited in body but missing from sources.json")

    for n in sorted(source_by_n.keys()):
        if n not in cited_numbers:
            problems.append(f"source [{n}] never cited in body")

    ref_lines = [line.strip() for line in references_text.splitlines() if line.strip()]
    seen_ref_n = set()
    for line in ref_lines:
        m = re.match(r"^\[(\d+)\](?:\s+(.*))?$", line)
        if not m:
            problems.append(f"invalid reference line (must start with '[n]'): {line}")
            continue
        ref_n = int(m.group(1))
        if ref_n in seen_ref_n:
            problems.append(f"duplicate reference entry for [{ref_n}] in References")
        seen_ref_n.add(ref_n)

        if ref_n not in source_by_n:
            problems.append(f"reference [{ref_n}] in References does not exist in sources.json")
            continue

        raw_urls = _URL.findall(line)
        cleaned_urls = [u.rstrip(".,;)") for u in raw_urls]
        if len(cleaned_urls) != 1:
            problems.append(f"reference [{ref_n}] line must have exactly ONE URL (found {len(cleaned_urls)})")
        else:
            expected_url = source_by_n[ref_n].get("url", "")
            if cleaned_urls[0] != expected_url and cleaned_urls[0].rstrip("/") != expected_url.rstrip("/"):
                problems.append(f"reference [{ref_n}] URL mismatch: expected {expected_url}, found {cleaned_urls[0]}")

    for n in sorted(source_by_n.keys()):
        if n not in seen_ref_n:
            problems.append(f"missing reference line for source [{n}] in References")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
