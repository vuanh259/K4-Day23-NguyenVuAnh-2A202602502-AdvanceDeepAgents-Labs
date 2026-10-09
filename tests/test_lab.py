import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import httpx
import tools
import research
from agents import BatchToolGuard, NonStreamingOllama, ResearchEvidenceStage, LeadWorkflow, GoogleQuotaPacing, GatewayRetry, SOURCES_PATH, REPORT_PATH, VALIDATOR_PATH
from langchain_core.messages import AIMessage, ToolMessage
from check_citations import check


class ProgressTests(unittest.TestCase):
    def test_gateway_retries_transient_failure_but_not_bad_request(self):
        model = type('GatewayModel', (), {'__module__': 'langchain_openai.chat_models.base'})()
        request = SimpleNamespace(model=model)
        failure = RuntimeError("502 {'retry_after': 3}")
        failure.status_code = 502
        handler = unittest.mock.Mock(side_effect=[failure, 'recovered'])
        with patch('agents.time.sleep') as sleep:
            self.assertEqual(GatewayRetry().wrap_model_call(request, handler), 'recovered')
            self.assertEqual(handler.call_args_list[0], handler.call_args_list[1])
            sleep.assert_called_once_with(3)
        failure.status_code = 400
        with patch('agents.time.sleep') as sleep, self.assertRaises(RuntimeError):
            GatewayRetry().wrap_model_call(request, unittest.mock.Mock(side_effect=failure))
        sleep.assert_not_called()

    def test_google_pacing_is_shared_and_daily_quota_fails_without_retry(self):
        model = type('GoogleModel', (), {'__module__': 'langchain_google_genai.chat_models'})()
        request = SimpleNamespace(model=model)
        one, two = GoogleQuotaPacing(), GoogleQuotaPacing()
        with patch('agents.time.monotonic', side_effect=[100, 100, 105, 113]), patch('agents.time.sleep') as sleep:
            GoogleQuotaPacing._last_start = 0
            self.assertEqual(one.wrap_model_call(request, lambda _: 'first'), 'first')
            self.assertEqual(two.wrap_model_call(request, lambda _: 'second'), 'second')
            sleep.assert_called_once_with(8)
        with patch('agents.time.sleep') as sleep, patch('agents.time.monotonic', return_value=1000):
            GoogleQuotaPacing._last_start = 0
            calls = []
            def exhausted(_):
                calls.append(1)
                raise RuntimeError('429 GenerateRequestsPerDayPerProjectPerModel-FreeTier')
            with self.assertRaises(RuntimeError):
                one.wrap_model_call(request, exhausted)
            self.assertEqual(len(calls), 1)
            sleep.assert_not_called()

    def test_lead_moves_from_research_to_files_validation_and_checker(self):
        workflow = LeadWorkflow(lambda: {(family, 'https://example.com/' + family) for family in ['arxiv', 'hf-search', 'web']})
        messages = []
        available = [SimpleNamespace(name=name) for name in
                     ['write_todos', 'task', 'read_file', 'ls', 'write_file', 'edit_file', 'execute']]
        def stage():
            request = SimpleNamespace(messages=messages, tools=available, model=None)
            request.override = lambda **changes: SimpleNamespace(**changes)
            return workflow.wrap_model_call(request, lambda value: value)
        def completed(name, args, text):
            identifier = str(len(messages))
            messages.append(AIMessage(content='', tool_calls=[{'id': identifier, 'name': name, 'args': args}]))
            messages.append(ToolMessage(content=text, name=name, tool_call_id=identifier))
        self.assertEqual([tool.name for tool in stage().tools], ['write_todos'])
        completed('write_todos', {'todos': [{'content': 'Research', 'status': 'in_progress'}]}, 'Updated todo list')
        self.assertEqual([tool.name for tool in stage().tools], ['task'])
        for i in range(3):
            completed('task', {'subagent_type': 'researcher', 'description': f'Research /tmp/work/research/notes/0{i}-topic.md'}, 'Notes ready')
        self.assertEqual({tool.name for tool in stage().tools}, {'read_file', 'ls'})
        for i in range(3):
            completed('read_file', {'file_path': f'/tmp/work/research/notes/0{i}-topic.md'}, 'Retrieved evidence')
        self.assertIn(SOURCES_PATH, stage().messages[-1].content)
        completed('write_file', {'file_path': SOURCES_PATH, 'content': json.dumps([{'source': family, 'url': 'https://example.com/' + family} for family in ['arxiv', 'hf-search', 'web']])}, f'Updated file {SOURCES_PATH}')
        self.assertIn(REPORT_PATH, stage().messages[-1].content)
        completed('write_file', {'file_path': REPORT_PATH}, f'Updated file {REPORT_PATH}')
        self.assertEqual([tool.name for tool in stage().tools], ['execute'])
        completed('execute', {'command': f'python3 {VALIDATOR_PATH}'}, 'OK: citations valid')
        self.assertEqual({tool.name for tool in stage().tools}, {'read_file', 'task'})
        completed('task', {'subagent_type': 'citation-checker', 'description': 'Three claims'}, '1 SUPPORTED; 2 SUPPORTED; 3 SUPPORTED')
        self.assertEqual([tool.name for tool in stage().tools], ['write_todos'])
        completed('write_todos', {'todos': [{'content': 'Research', 'status': 'completed'}]}, 'Updated todo list')
        self.assertEqual(stage().tools, [])

    def test_provenance_uses_successful_tool_not_url_domain(self):
        progress = research.Progress()
        url = 'https://arxiv.org/abs/1803.10122'
        with patch('builtins.print'):
            progress.on_tool_start({'name': 'arxiv_search'}, '{}', run_id='failed')
            progress.on_tool_end('ERROR: HTTP 429', run_id='failed')
            progress.on_tool_start({'name': 'web_fetch'}, repr({'url': url}), run_id='fetch')
            progress.on_tool_end('SOURCE_FAMILY: web\nActual abstract', run_id='fetch')
        self.assertEqual(progress.provenance_errors([{'url': url, 'source': 'web'}]), [])
        self.assertEqual(len(progress.provenance_errors([{'url': url, 'source': 'arxiv'}])), 1)

    def test_research_stage_removes_failed_source_then_requires_notes(self):
        middleware = ResearchEvidenceStage()
        source_tools = [SimpleNamespace(name=name) for name in
                        ['arxiv_search', 'hf_search_papers', 'web_fetch', 'write_file']]
        def request(messages):
            value = SimpleNamespace(messages=messages, tools=source_tools)
            value.override = lambda **changes: SimpleNamespace(**changes)
            return value
        messages = [ToolMessage(content='ERROR: HTTP 429', name='arxiv_search', tool_call_id='a')]
        narrowed = middleware.wrap_model_call(request(messages), lambda value: value)
        self.assertEqual([tool.name for tool in narrowed.tools], ['hf_search_papers', 'web_fetch'])
        messages.append(ToolMessage(content='[{"id":"real-paper"}]', name='hf_search_papers', tool_call_id='b'))
        one_family = middleware.wrap_model_call(request(messages), lambda value: value)
        self.assertNotIn('hf_search_papers', [tool.name for tool in one_family.tools])
        messages.append(ToolMessage(content='URL: https://example.com/paper\nActual abstract.', name='web_fetch', tool_call_id='c'))
        narrowed = middleware.wrap_model_call(request(messages), lambda value: value)
        self.assertEqual([tool.name for tool in narrowed.tools], ['write_file'])
        messages.append(ToolMessage(content='Updated file /tmp/work/research/notes/01-topic.md',
                                    name='write_file', tool_call_id='d'))
        finished = middleware.wrap_model_call(request(messages), lambda value: value)
        self.assertEqual(finished.tools, [])

    def test_counts_executed_lead_calls_and_excludes_researcher_tokens(self):
        progress = research.Progress()
        with patch('builtins.print'):
            for _ in range(3):
                progress.on_tool_start({'name': 'task'}, '{}', metadata={'lc_agent_name': None})
            progress.on_tool_start({'name': 'arxiv_search'}, '{}', metadata={'lc_agent_name': 'researcher'})
            for run_id, name, amount in [('lead', None, 5), ('child', 'researcher', 100)]:
                progress.on_chat_model_start({}, [], run_id=run_id, metadata={'lc_agent_name': name})
                message = AIMessage(content='', usage_metadata={'input_tokens': amount, 'output_tokens': amount, 'total_tokens': amount * 2})
                response = SimpleNamespace(generations=[[SimpleNamespace(message=message)]])
                progress.on_llm_end(response, run_id=run_id)
        stats = research.summarize(progress.lead_records, 1, 'local')
        self.assertEqual(stats['subagent_calls'], 3)
        self.assertEqual(stats['tokens'], {'input': 5, 'output': 5})


class CitationsTests(unittest.TestCase):
    def setUp(self):
        self.sources = [{"n": 1, "url": "https://example.com/a"}, {"n": 2, "url": "https://example.com/b"}]
        self.refs = "\n## References\n[1] A https://example.com/a\n[2] B https://example.com/b\n"

    def test_adjacent_grouped_and_range(self):
        for body in ("Claim [1][2].", "Claim [1, 2].", "Claim [1-2]."):
            self.assertEqual(check(body + self.refs, self.sources), [])

    def test_code_and_links_do_not_count(self):
        for body in ("```python\nx = [1]\n```\n[2](https://example.com/b)\n",
                     "~~~python\nx = [1]\n~~~~\n``[2]``\n"):
            self.assertEqual(sum("never cited" in problem for problem in check(body + self.refs, self.sources)), 2)

    def test_reference_url_parentheses_are_preserved(self):
        sources = [{"n": 1, "url": "https://example.com/World_(model)"}]
        self.assertEqual(check("A claim [1].\n## References\n[1] A https://example.com/World_(model) (2026)\n", sources), [])

    def test_reference_rules(self):
        cases = ["[1] A https://example.com/a https://example.com/b\n[2] B https://example.com/b",
                 "[1] A https://example.com/wrong\n[2] B https://example.com/b",
                 "[1] A https://example.com/a\n[1] A https://example.com/a",
                 "[1] A https://example.com/a\n[3] C https://example.com/c"]
        for refs in cases:
            self.assertTrue(check("Claim [1][2].\n## References\n" + refs, self.sources))

    def test_missing_unknown_and_malformed(self):
        self.assertTrue(check("Claim [3]." + self.refs, self.sources))
        self.assertTrue(check("Claim [1][2].", self.sources))
        for sources in ([], {}, [None], [{"n": True, "url": None}], [{"n": 1, "url": "http://["}], self.sources + [self.sources[0]]):
            self.assertTrue(check("Claim [1][2]." + self.refs, sources))


class RetryTests(unittest.TestCase):
    @patch("tools.time.sleep")
    @patch("tools.random.uniform", return_value=0.5)
    def test_backoff_cap_and_last_attempt(self, jitter, sleep):
        fn = unittest.mock.Mock(side_effect=tools.RetryableError("busy"))
        with self.assertRaises(tools.RetryableError):
            tools.with_retry(fn, attempts=4, base=2, cap=5)
        self.assertEqual(fn.call_count, 4)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2.5, 4.5, 5])

    @patch("tools.time.sleep")
    def test_retry_after_success_and_nonretry(self, sleep):
        fn = unittest.mock.Mock(side_effect=[tools.RetryableError("busy", 9), "done"])
        self.assertEqual(tools.with_retry(fn, cap=6), "done")
        sleep.assert_called_once_with(6)
        sleep.reset_mock()
        with self.assertRaises(ValueError):
            tools.with_retry(unittest.mock.Mock(side_effect=ValueError("bug")))
        sleep.assert_not_called()

    @patch("tools.httpx.request")
    def test_transport_and_http_status(self, request):
        request.side_effect = httpx.ConnectError("offline")
        with self.assertRaises(tools.RetryableError):
            tools._request("GET", "https://example.com")
        request.side_effect = None
        for status in (429, 500, 502, 503, 504):
            request.return_value = httpx.Response(status, headers={"Retry-After": "7"})
            with self.assertRaises(tools.RetryableError) as caught:
                tools._request("GET", "https://example.com")
            self.assertEqual(caught.exception.retry_after, 7)
        request.return_value = httpx.Response(401, request=httpx.Request("GET", "https://example.com"))
        with self.assertRaises(httpx.HTTPStatusError):
            tools._request("GET", "https://example.com")


class SourceToolsTests(unittest.TestCase):
    @patch("tools._mcp_call", side_effect=tools.RetryableError("Exa unavailable"))
    @patch("tools._request")
    @patch("tools.time.sleep")
    def test_direct_fetch_fallback_has_real_evidence(self, sleep, request, mcp):
        request.return_value = httpx.Response(200, headers={"content-type": "text/html"},
                                             text="<h1>Title</h1><script>ignore me</script><p>Evidence</p>")
        with patch.dict("os.environ", {"EXA_RETRY_ATTEMPTS": "1", "WEB_FETCH_DIRECT_FALLBACK": "1"}):
            result = tools.web_fetch.invoke({"url": "https://example.com/paper"})
        self.assertIn("FETCH_BACKEND: direct-http", result)
        self.assertIn("Evidence", result)
        self.assertNotIn("ignore me", result)

    @patch("tools._request")
    def test_arxiv_sanitization_and_xml(self, request):
        request.return_value = httpx.Response(200, text='''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
        <id>http://arxiv.org/abs/2501.00001v2</id><title> A\n title </title>
        <published>2025-01-01T00:00:00Z</published><summary> evidence </summary></entry></feed>''')
        with patch("tools._last_arxiv_call", None):
            result = json.loads(tools.arxiv_search.invoke({"query": 'all:"world" AND model'}))
        self.assertEqual(result[0]["url"], "https://arxiv.org/abs/2501.00001")
        self.assertEqual(result[0]["title"], "A title")
        self.assertEqual(request.call_args.kwargs["params"]["search_query"], "all:world AND all:model")
        request.reset_mock()
        self.assertEqual(tools.arxiv_search.invoke({"query": '" : AND OR'}), "NO RESULTS")
        request.assert_not_called()

    @patch("tools._request")
    def test_hf_mapping_filter_and_sort(self, request):
        payload = [{"paper": {"id": "1", "title": "World models", "summary": "long", "ai_summary": "short", "upvotes": 2}},
                   {"paper": {"id": "2", "title": "World models", "upvotes": 9}}, {"paper": {"title": "missing"}}]
        request.return_value = httpx.Response(200, json=payload)
        items = json.loads(tools.hf_daily_papers.invoke({"keyword": "WORLD"}))
        self.assertEqual([item["id"] for item in items], ["2", "1"])
        self.assertEqual(json.loads(tools.hf_search_papers.invoke({"query": "world"}))[0]["summary"], "short")

    @patch('tools._request')
    def test_daily_query_alias_filters_instead_of_returning_unrelated_papers(self, request):
        request.return_value = httpx.Response(200, json=[{'paper': {'id': '1', 'title': 'World model'}},
                                                        {'paper': {'id': '2', 'title': 'Unrelated study'}}])
        message = AIMessage(content='', tool_calls=[{'id': 'daily', 'name': 'hf_daily_papers',
                                                     'args': {'query': 'world model', 'limit': 3}}])
        corrected = BatchToolGuard().after_model({'messages': [message]}, None)['messages'][0].tool_calls[0]
        results = json.loads(tools.hf_daily_papers.invoke(corrected['args']))
        self.assertEqual([record['id'] for record in results], ['1'])

    @patch("tools._request")
    def test_exa_sse_errors_rate_limits_and_redaction(self, request):
        def respond(envelope):
            request.return_value = httpx.Response(200, text="event: message\ndata: " + json.dumps(envelope) + "\n\n")
        respond({"result": {"content": [{"type": "text", "text": "evidence"}]}})
        self.assertEqual(tools._mcp_call("web_fetch_exa", {"urls": ["https://example.com"]}), "evidence")
        respond({"result": {"content": [{"type": "text", "text": "Serving systems implement rate limits."}]}})
        self.assertEqual(tools._mcp_call("web_fetch_exa", {"urls": ["https://example.com"]}), "Serving systems implement rate limits.")
        for envelope in ({"result": {"_meta": {"rateLimitExceeded": True}, "content": []}},
                         {"result": {"content": [{"type": "text", "text": "Rate limit exceeded"}]}},
                         {"error": {"message": "too many requests"}}):
            respond(envelope)
            with self.assertRaises(tools.RetryableError):
                tools._mcp_call("web_search_exa", {})
        respond({"error": {"message": "bad request"}})
        with self.assertRaises(RuntimeError):
            tools._mcp_call("web_search_exa", {})
        with patch.dict("os.environ", {"EXA_API_KEY": "private-test-value"}):
            request.side_effect = ValueError("https://mcp.exa.ai/mcp?exaApiKey=private-test-value")
            self.assertNotIn("private-test-value", tools.web_search.invoke({"query": "world"}))


class ResearchTests(unittest.TestCase):
    def test_structured_sources_are_serialized_without_changing_records(self):
        records = [{'n': 1, 'title': 'A paper', 'url': 'https://example.com', 'source': 'web'}]
        message = AIMessage(content='', tool_calls=[{'id': 'write', 'name': 'write_file',
            'args': {'file_path': SOURCES_PATH, 'content': records}}])
        corrected = BatchToolGuard().after_model({'messages': [message]}, None)['messages'][0].tool_calls[0]
        self.assertEqual(json.loads(corrected['args']['content']), records)

    def test_task_arguments_preserve_instructions_in_supported_schema(self):
        original = {'topic': 'World models', 'question': 'Planning', 'sources': ['arxiv', 'web'],
                    'notes_path': '/tmp/work/research/notes/01-planning.md'}
        message = AIMessage(content='', tool_calls=[{'id': 'task1', 'name': 'task', 'args': original}])
        result = BatchToolGuard().after_model({'messages': [message]}, None)
        args = result['messages'][0].tool_calls[0]['args']
        self.assertEqual(set(args), {'description', 'subagent_type'})
        self.assertEqual(args['subagent_type'], 'researcher')
        for value in ('World models', 'Planning', '/tmp/work/research/notes/01-planning.md', 'arxiv', 'web'):
            self.assertIn(value, args['description'])

    def test_planning_does_not_drop_third_parallel_researcher(self):
        calls = [{'name': 'write_todos', 'args': {'todos': []}, 'id': 'plan'}]
        calls.extend({'name': 'task', 'args': {'description': str(i), 'subagent_type': 'researcher'}, 'id': str(i)} for i in range(3))
        message = AIMessage(content='', tool_calls=calls)
        self.assertIsNone(BatchToolGuard().after_model({'messages': [message]}, None))

    def test_ollama_receives_complete_tool_responses(self):
        model = NonStreamingOllama(model="local-test", num_ctx=16384)
        params = model._chat_params([], stream=True)
        self.assertFalse(params["stream"])

    def test_repeated_tool_batch_is_bounded_before_execution(self):
        calls = [{"name": "task", "args": {"description": f"Question {i % 5}"}, "id": str(i)} for i in range(47)]
        message = AIMessage(content="", tool_calls=calls, id="message-id")
        result = BatchToolGuard().after_model({"messages": [message]}, None)
        updated = result["messages"][0]
        self.assertEqual(len(updated.tool_calls), 3)
        self.assertEqual(updated.id, message.id)
        self.assertEqual(len({json.dumps(call["args"], sort_keys=True) for call in updated.tool_calls}), 3)

    def test_slug_and_usage(self):
        self.assertEqual(research.slugify("../../x"), "x")
        self.assertEqual(research.slugify("!!!"), "topic")
        self.assertLessEqual(len(research.slugify("a" * 100)), 60)
        result = research.summarize([{"tool_calls": [{"name": "task"}, {"name": "execute"}],
                                     "usage_metadata": {"input_tokens": 12, "output_tokens": 4}}], 1.234, "local")
        self.assertEqual(result["subagent_calls"], 1)
        self.assertEqual(result["tokens"], {"input": 12, "output": 4})

    @patch("research.download", return_value={})
    def test_missing_outputs_write_nothing(self, download):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                research.save_outputs(None, "topic", [], 0, "local", directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_valid_output_bytes_and_failed_run_preserves_previous(self):
        sources = [{"n": 1, "url": "https://arxiv.org/abs/1", "source": "arxiv"},
                   {"n": 2, "url": "https://huggingface.co/papers/2", "source": "hf-search"},
                   {"n": 3, "url": "https://example.com/3", "source": "web"}]
        report = b"Claims [1][2][3].\n## References\n[1] A https://arxiv.org/abs/1\n[2] B https://huggingface.co/papers/2\n[3] C https://example.com/3\n"
        files = {research.REPORT_PATH: report, research.SOURCES_PATH: json.dumps(sources).encode()}
        messages = [{"tool_calls": [{"name": "task"}] * 3}]
        with tempfile.TemporaryDirectory() as directory, patch("research.download", return_value=files):
            path = research.save_outputs(None, "topic", messages, 0, "local", directory)
            self.assertEqual(path.read_bytes(), report)
            before = {p.name: p.read_bytes() for p in Path(directory).iterdir()}
            real_replace = research.os.replace
            calls = 0
            def fail_second_replace(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("disk failure after first replacement")
                return real_replace(source, destination)
            files[research.REPORT_PATH] = b"# Updated survey\n" + report
            with patch("research.os.replace", side_effect=fail_second_replace):
                with self.assertRaises(OSError):
                    research.save_outputs(None, "topic", messages, 1, "different-model", directory)
            self.assertEqual(before, {p.name: p.read_bytes() for p in Path(directory).iterdir()})


if __name__ == "__main__":
    unittest.main()
