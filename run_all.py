"""Run the five lab topics, preserving completed outputs and failing honestly."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from check_citations import check

ROOT = Path(__file__).parent


def completed(topic):
    from research import slugify, report_quality_errors
    prefix = ROOT / "reports" / slugify(topic)
    try:
        report = prefix.with_suffix(".md").read_text(encoding="utf-8")
        sources = json.loads(prefix.with_suffix(".sources.json").read_text(encoding="utf-8"))
        meta = json.loads(prefix.with_suffix(".meta.json").read_text(encoding="utf-8"))
        return (not check(report, sources) and not report_quality_errors(report, sources) and meta.get("topic") == topic
                and meta.get("subagent_calls", 0) >= 3 and len(meta.get("source_families", [])) >= 3)
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Regenerate even valid completed topics")
    args = parser.parse_args()
    topics = re.findall(r"(?m)^\d+\.\s+(.+)$", (ROOT / "topics.md").read_text(encoding="utf-8"))
    if len(topics) != 5:
        print("Expected exactly five topics in topics.md", file=sys.stderr)
        return 1
    for topic in topics:
        if not args.force and completed(topic):
            print(f"Already complete: {topic}", flush=True)
            continue
        result = subprocess.run([sys.executable, str(ROOT / "research.py"), topic], cwd=ROOT)
        if result.returncode:
            print(f"Stopped at failed topic: {topic}; rerun to resume", file=sys.stderr)
            return result.returncode
    return subprocess.run([sys.executable, str(ROOT / "self_check.py")], cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
