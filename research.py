"""research.py.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json  # noqa: F401
import os  # noqa: F401
import re  # noqa: F401
import sys
import time  # noqa: F401
import tempfile
import ast
import threading
from datetime import date, timedelta
from collections import Counter  # noqa: F401
from pathlib import Path
from langchain_core.callbacks import BaseCallbackHandler

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent  # noqa: F401
from model import make_model  # noqa: F401
from sandbox import download, open_sandbox, upload  # noqa: F401

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def report_quality_errors(report, sources):
    """Check the template and the rubric's older/recent source coverage."""
    headings = re.findall(r'(?m)^##\s+(.+?)\s*$', report)
    normalized = [heading.casefold() for heading in headings]
    required = {'tl;dr', 'background', 'trends and open problems', 'references'}
    errors = [f'Missing required heading: {heading}' for heading in sorted(required - set(normalized))]
    themes = [heading for heading in normalized if heading not in required]
    if not 3 <= len(themes) <= 6:
        errors.append('The report needs 3-6 thematic sections.')
    dates = []
    for source in sources:
        try:
            dates.append(date.fromisoformat(source.get('date', '')[:10]))
        except (ValueError, TypeError):
            pass
    cutoff = date.today() - timedelta(days=730)
    if not any(published < cutoff for published in dates):
        errors.append(f'No earlier foundational source dated before {cutoff}; retrieve a relevant foundational paper, read it, and cite it in Background.')
    if not any(cutoff <= published <= date.today() for published in dates):
        errors.append('No recent source from the last two years.')
    return errors


class Progress(BaseCallbackHandler):
    """Show bounded, redacted tool arguments for diagnosing failed runs."""
    def __init__(self, audit_path=None):
        self.lead_records = []
        self._lead_model_runs = set()
        self._source_runs = {}
        self.retrieved_sources = set()
        self.audit_path = audit_path
        self._audit_lock = threading.Lock()

    @staticmethod
    def _is_lead(metadata):
        return (metadata or {}).get("lc_agent_name") not in {"researcher", "citation-checker", "general-purpose"}

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get('name', 'unknown')
        if name in {'arxiv_search', 'hf_daily_papers', 'hf_search_papers', 'web_search', 'web_fetch'}:
            arguments = kwargs.get('inputs')
            if not isinstance(arguments, dict):
                try:
                    arguments = ast.literal_eval(input_str)
                except (ValueError, SyntaxError):
                    arguments = {}
            self._source_runs[kwargs.get('run_id')] = (name, arguments)
        if self._is_lead(kwargs.get("metadata")):
            self.lead_records.append({"tool_calls": [{"name": serialized.get("name", "unknown")} ]})
        print(f"  tool: {serialized.get('name', 'unknown')}", flush=True)
        from tools import _redact
        print(_redact(f"    args: {input_str[:500]}"), flush=True)

    def on_tool_error(self, error, **kwargs):
        from tools import _redact
        print(_redact(f"  tool failed: {error}"), flush=True)

    def on_chat_model_start(self, serialized, messages, **kwargs):
        if self._is_lead(kwargs.get("metadata")):
            self._lead_model_runs.add(kwargs.get("run_id"))
        print("  model: generating", flush=True)

    def on_llm_end(self, response, **kwargs):
        generation = response.generations[0][0]
        message = getattr(generation, "message", None)
        calls = getattr(message, "tool_calls", [])
        if kwargs.get("run_id") in self._lead_model_runs:
            self._lead_model_runs.discard(kwargs.get("run_id"))
            self.lead_records.append({"usage_metadata": getattr(message, "usage_metadata", None) or {}})
        print(f"  model: complete ({len(calls)} tool calls)", flush=True)
        if not calls and message is not None:
            from tools import _redact
            content = message.content
            if isinstance(content, list):
                content = '\n'.join(block.get('text', '') for block in content if isinstance(block, dict) and block.get('type') in {'text', 'output_text'})
            print(_redact(f"    reply: {str(content)[:300]}"), flush=True)
            if not message.content:
                print(_redact(f"    empty response metadata: {message.response_metadata}"), flush=True)

    def on_tool_end(self, output, **kwargs):
        content = getattr(output, "content", output)
        source_run = self._source_runs.pop(kwargs.get('run_id'), None)
        if source_run and isinstance(content, str) and content.strip() and not content.startswith(('ERROR:', 'NO RESULTS')):
            name, arguments = source_run
            family = {'arxiv_search': 'arxiv', 'hf_daily_papers': 'hf-daily',
                      'hf_search_papers': 'hf-search', 'web_search': 'web', 'web_fetch': 'web'}[name]
            if family != 'web':
                try:
                    records = json.loads(content)
                    urls = [record['url'] for record in records if isinstance(record, dict) and record.get('url')]
                except (ValueError, TypeError):
                    urls = []
            elif name == 'web_fetch':
                urls = [arguments['url']] if arguments.get('url') else []
            else:
                urls = [url.rstrip(').,;') for url in re.findall(r'https?://[^\s<>"\x27]+', content)]
            with self._audit_lock:
                self.retrieved_sources.update((family, url) for url in urls)
                if self.audit_path:
                    from tools import _redact
                    self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.audit_path.open('a', encoding='utf-8') as stream:
                        stream.write(_redact(json.dumps({'tool': name, 'arguments': arguments, 'output': content}, ensure_ascii=False)) + '\n')
        if isinstance(content, str) and content.startswith("ERROR:"):
            from tools import _redact
            print(_redact("  " + content[:300]), flush=True)

    def provenance_errors(self, sources):
        return [f"{source.get('url')} was not retrieved by family {source.get('source')}; actual families: "
                f"{sorted(family for family, url in self.retrieved_sources if url == source.get('url'))}"
                for source in sources if (source.get('source'), source.get('url')) not in self.retrieved_sources]


def slugify(topic):
    """Return a safe, non-empty report filename stem, limited to 60 characters."""
    return re.sub(r"[^\w]+", "-", topic.lower()).strip("-_")[:60].rstrip("-_") or "topic"


def build_prompt(topic):
    """Supply topic, current date and required deliverables to the lead."""
    return (f"Research topic: {topic}\nToday: {date.today().isoformat()}. "
            "Produce the complete English survey with foundational sources and work from the last two years. "
            "Delegate at least three independent questions, cite at least three relevant source families, "
            "and finish sandbox finalization, validation and claim spot-checks before returning. /no_think")


def summarize(messages, elapsed, model_name):
    """Count lead task/tool calls and input/output tokens; subagent tokens are not included."""
    calls = Counter()
    tokens = {"input": 0, "output": 0}
    for message in messages:
        get = message.get if isinstance(message, dict) else lambda key, default=None: getattr(message, key, default)
        for call in get("tool_calls", []) or []:
            calls[call.get("name", "unknown")] += 1
        usage = get("usage_metadata", {}) or {}
        tokens["input"] += usage.get("input_tokens", 0) or 0
        tokens["output"] += usage.get("output_tokens", 0) or 0
    return {"model": model_name, "elapsed_s": round(elapsed, 1), "subagent_calls": calls["task"],
            "tool_calls": dict(calls), "tokens": tokens}


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Validate sandbox outputs, preserve their exact bytes and publish all three files with rollback on failure."""
    from check_citations import check
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    raw_report, raw_sources = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not raw_report or not raw_report.strip() or not raw_sources:
        raise RuntimeError("sandbox report or sources missing/empty; no outputs saved")
    try:
        report = raw_report.decode("utf-8")
        sources = json.loads(raw_sources)
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError("invalid sandbox output; no outputs saved") from exc
    problems = check(report, sources)
    if problems:
        raise RuntimeError("invalid citations: " + "; ".join(problems))
    stats = summarize(messages, elapsed, model_name)
    families = sorted({source.get("source", "") for source in sources})
    if stats["subagent_calls"] < 3:
        raise RuntimeError("lead delegated fewer than three tasks; no outputs saved")
    if len(families) < 3 or not set(families) <= {"arxiv", "hf-daily", "hf-search", "web"}:
        raise RuntimeError("report must cite at least three valid source families")
    for source in sources:
        family, url = source["source"], source["url"]
        if family == "arxiv" and not url.startswith("https://arxiv.org/abs/"):
            raise RuntimeError("arxiv family URL mismatch")
        if family.startswith("hf-") and not url.startswith("https://huggingface.co/papers/"):
            raise RuntimeError("Hugging Face family URL mismatch")
    meta = {"topic": topic, **stats, "n_sources": len(sources), "source_families": families,
            "sandbox": os.getenv("SANDBOX", "daytona"), "tokens_scope": "lead only"}
    directory = Path(reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)
    # Preserve sandbox bytes exactly. Stage all three files before replacing outputs,
    # and roll back if any replacement fails (including an existing successful run).
    payloads = {directory / f"{slug}.md": raw_report,
                directory / f"{slug}.sources.json": raw_sources,
                directory / f"{slug}.meta.json": (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode()}
    previous = {path: path.read_bytes() if path.exists() else None for path in payloads}
    changed, staged = [], []
    try:
        for path, content in payloads.items():
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                stream.write(content)
                staged.append((Path(stream.name), path))
        for temporary, destination in staged:
            os.replace(temporary, destination)
            changed.append(destination)
    except Exception:
        for path in reversed(changed):
            if previous[path] is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(previous[path])
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    return directory / f"{slug}.md"


def main(topic):
    """Run research in a managed sandbox. Exit 0 on success, 1 on failure, 2 for an empty topic."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if not topic.strip():
        print(__doc__, file=sys.stderr)
        return 2
    try:
        model = make_model()
        start = time.monotonic()
        with open_sandbox() as backend:
            _execute_checked(backend, f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            seeds = {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()}
            upload(backend, seeds)
            if download(backend, list(seeds)) != seeds:
                raise RuntimeError("sandbox script upload failed")
            progress = Progress(ROOT / '.local' / f'evidence-{slugify(topic)}-{int(time.time())}.jsonl')
            try:
                agent = build_lead_agent(backend, model, lambda: progress.retrieved_sources)
                print(f"Researching: {topic}", flush=True)
                result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                      config={"recursion_limit": 1000, "max_concurrency": 3, "callbacks": [progress]})
                # Local models sometimes finish in prose before writing the deliverables.
                # Continue the same state with concrete validation feedback, preserving
                # middleware limits and real calls rather than constructing outputs here.
                for continuation in range(3):
                    validation = backend.execute(f"python3 {VALIDATOR_PATH}")
                    if validation.exit_code == 0 and validation.output.strip().startswith("OK:"):
                        current_sources = json.loads(download(backend, [SOURCES_PATH])[SOURCES_PATH])
                        current_families = {source.get('source') for source in current_sources}
                        provenance_errors = progress.provenance_errors(current_sources)
                        current_report = download(backend, [REPORT_PATH])[REPORT_PATH].decode('utf-8')
                        quality_errors = report_quality_errors(current_report, current_sources)
                        if not provenance_errors and not quality_errors and len(current_families) >= 3 and current_families <= {'arxiv', 'hf-daily', 'hf-search', 'web'}:
                            break
                        feedback = (f"Current families: {sorted(current_families, key=str)}. Three are required. "
                                    f"Provenance errors: {provenance_errors[:12]}. "
                                    f"Report quality errors: {quality_errors}. "
                                    "Use the actual retrieval family; do not infer it from the URL. "
                                    "If arxiv is unavailable, query hf_daily_papers on dates from discovered relevant papers "
                                    "and cite different relevant paper URLs from hf-search and hf-daily.")
                    else:
                        feedback = validation.output[:1500]
                    from tools import _redact
                    feedback = _redact(feedback)
                    print(f"  continuing: sandbox validation failed (attempt {continuation + 1})", flush=True)
                    result = agent.invoke({**result, "messages": [*result["messages"],
                        {"role": "user", "content":
                         f"The deliverables are incomplete. Validator output:\n{feedback}\n"
                         "Use tools now. Read researcher notes or their returned retrieved evidence. "
                         f"Write {SOURCES_PATH} and {REPORT_PATH} using only that evidence. "
                         "If evidence is insufficient, use a targeted researcher task. "
                         "Do not answer with proposed file content in prose. Call write_file or execute to create files, "
                         f"then run python3 {FINALIZER_PATH} and python3 {VALIDATOR_PATH}. "
                         "Maintain three relevant source families and complete the citation-checker task."}]},
                        config={"recursion_limit": 1000, "max_concurrency": 3, "callbacks": [progress]})
                # Require a successful validator inside the sandbox even if the lead forgot.
                _execute_checked(backend, f"python3 {VALIDATOR_PATH}", expect_ok=True)
                provenance_errors = progress.provenance_errors(json.loads(download(backend, [SOURCES_PATH])[SOURCES_PATH]))
                if provenance_errors:
                    raise RuntimeError('Source provenance mismatch: ' + '; '.join(provenance_errors[:6]))
                final_files = download(backend, [REPORT_PATH, SOURCES_PATH])
                quality_errors = report_quality_errors(final_files[REPORT_PATH].decode('utf-8'), json.loads(final_files[SOURCES_PATH]))
                if quality_errors:
                    raise RuntimeError('Report quality: ' + '; '.join(quality_errors))
                path = save_outputs(backend, topic, progress.lead_records, time.monotonic() - start,
                                    os.getenv("LAB_MODEL") or os.getenv("OPENAI_DEPLOYMENT_MODEL", "unknown"))
            finally:
                # Keep genuine sandbox deliverables for diagnosis even on failure.
                recovery = ROOT / '.local' / f'recovery-{slugify(topic)}-{int(time.time())}'
                recovery.mkdir(parents=True, exist_ok=True)
                try:
                    for remote, raw in download(backend, [REPORT_PATH, SOURCES_PATH]).items():
                        if raw:
                            (recovery / Path(remote).name).write_bytes(raw)
                except Exception:
                    pass

        print(f"Saved: {path}")
        return 0
    except Exception as exc:
        from tools import _redact
        print(_redact(f"FAILED: {type(exc).__name__}: {exc}"), file=sys.stderr)
        return 1


def _execute_checked(backend, command, expect_ok=False):
    result = backend.execute(command)
    if result.exit_code != 0 or (expect_ok and not result.output.strip().startswith("OK:")):
        raise RuntimeError(f"sandbox command failed: {result.output[:1500]}")
    return result


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
