"""check_citations.py.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import sys
import re
from collections import Counter
from urllib.parse import urlsplit

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def _without_code(text):
    lines, fence = [], None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence is None:
            if marker:
                fence = (marker.group(1)[0], len(marker.group(1)))
            else:
                lines.append(line)
        else:
            closer = re.match(r"^ {0,3}(`{3,}|~{3,})\s*$", line)
            if closer and closer.group(1)[0] == fence[0] and len(closer.group(1)) >= fence[1]:
                fence = None
    return re.sub(r"(`+)(.*?)\1", "", "".join(lines), flags=re.DOTALL)


def check(report_text, sources):
    """Validate sources, body citations and one exact-URL reference per source. Ignore code and Markdown citation links."""
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json (expected a non-empty array)"]
    by_number, seen_urls = {}, set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            problems.append(f"source entry {index} must be an object")
            continue
        n, url = source.get("n"), source.get("url")
        valid_n = type(n) is int and n > 0
        if not valid_n:
            problems.append(f"source entry {index}: n must be a positive integer")
        elif n in by_number:
            problems.append(f"duplicate source number [{n}]")
        else:
            by_number[n] = source
        try:
            valid_url = isinstance(url, str) and url.startswith(("http://", "https://")) and bool(urlsplit(url).netloc)
        except ValueError:
            valid_url = False
        if not valid_url:
            problems.append(f"source entry {index}: invalid URL")
        elif url in seen_urls:
            problems.append(f"duplicate source URL: {url}")
        else:
            seen_urls.add(url)

    # Code examples and Markdown link labels are not evidence citations.
    clean = _without_code(report_text)
    headings = list(re.finditer(r"(?m)^## References\s*$", clean))
    if not headings:
        problems.append("missing ## References heading")
        body, references = clean, ""
    else:
        if len(headings) != 1:
            problems.append("expected exactly one ## References heading")
        body, references = clean[:headings[0].start()], clean[headings[0].end():]
    cited = set()
    for match in re.finditer(r"(?<!!)\[(\d+(?:\s*(?:,|-)\s*\d+)*)\](?!\s*\()", body):
        for part in match.group(1).split(","):
            bounds = [int(value.strip()) for value in part.split("-")]
            if len(bounds) == 1:
                cited.add(bounds[0])
            elif bounds[0] <= bounds[1] and bounds[1] - bounds[0] <= 10000:
                cited.update(range(bounds[0], bounds[1] + 1))
            else:
                problems.append(f"invalid citation range [{part}]")
    for n in sorted(cited - by_number.keys()):
        problems.append(f"[{n}] cited but missing from sources.json")
    for n in sorted(by_number.keys() - cited):
        problems.append(f"source [{n}] never cited")
    counts = Counter()
    for match in re.finditer(r"(?m)^\s*\[(\d+)\]\s*(.*)$", references):
        n, line = int(match.group(1)), match.group(2)
        counts[n] += 1
        if n not in by_number:
            problems.append(f"reference [{n}] missing from sources.json")
        urls = re.findall(r"https?://[^\s<>]+", line)
        if len(urls) != 1:
            problems.append(f"reference [{n}] must contain exactly one URL")
        elif n in by_number and urls[0] != by_number[n].get("url") and urls[0].rstrip(").,;") != by_number[n].get("url"):
            problems.append(f"reference [{n}] URL does not match sources.json")
    for n in sorted(by_number):
        if counts[n] != 1:
            problems.append(f"source [{n}] needs exactly one reference line (found {counts[n]})")
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
