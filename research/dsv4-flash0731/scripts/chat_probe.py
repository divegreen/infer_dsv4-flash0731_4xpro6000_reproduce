#!/usr/bin/env python3
"""Finite localhost chat checks using only the Python standard library.

This is a small protocol/content smoke test, not an accuracy benchmark.
--self-test and --prepare-only never contact a server or import GPU libraries.
"""
from __future__ import annotations

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output

import argparse
from datetime import datetime, timezone
import hashlib
from http.client import HTTPException, IncompleteRead
import json
from pathlib import Path
import sys
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


MAX_RESPONSE_BYTES = 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def endpoint(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") not in ("", "/v1")
            or port is None or not 1 <= port <= 65535):
        raise argparse.ArgumentTypeError("Use http://127.0.0.1:PORT[/v1] only.")
    return f"http://127.0.0.1:{port}/v1/chat/completions"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "Redirects are disabled", headers, fp)


def make_cases(model: str, history_reasoning_content: bool) -> list[dict]:
    simple = [
        ("arithmetic", "What is 7 + 5? Reply with only the number.", ["12", "十二"]),
        ("france_capital", "What is the capital of France? Reply with only the city name.",
         ["Paris", "巴黎"]),
        ("chinese_week", "一个星期有几天？只回答天数。", ["7", "七", "7天", "七天"]),
    ]
    cases = []
    for name, prompt, answers in simple:
        cases.append(dict(name=name, kind="required_answer", expected_answers=answers,
                          messages=[dict(role="user", content=prompt)]))
    history = dict(role="assistant", content="记住了，你的编号是23。")
    if history_reasoning_content:
        history["reasoning_content"] = "这是一条用于历史消息兼容性检查的测试记录。"
    cases.append(dict(
        name="multi_turn_history", kind="required_answer", expected_answers=["25", "二十五"],
        messages=[dict(role="user", content="我的编号是23，请记住。"), history,
                  dict(role="user", content="我的编号加2是多少？只回答数字。")],
    ))
    cases.append(dict(
        name="low_reasoning_observation", kind="observation", expected_answers=["12", "十二"],
        messages=[dict(role="user", content="What is 7 + 5? Give a brief final answer.")],
    ))
    for case in cases:
        low = case["kind"] == "observation"
        case["request"] = dict(
            model=model, messages=case.pop("messages"), temperature=0, stream=False,
            max_tokens=128 if low else 64, reasoning_effort="low" if low else "none",
            chat_template_kwargs={"enable_thinking": low},
        )
    return cases


def normalized_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip().casefold()
    return text.strip(" \t\r\n`*\"'“”‘’。.!！").strip()


def check_response(case: dict, response: object) -> dict:
    """Separate HTTP/JSON shape, strict short-answer check, and low observation."""
    if not isinstance(response, dict):
        raise ValueError("Response is not a JSON object.")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Expected exactly one choice.")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ValueError("Choice has no message object.")
    message = choice["message"]
    if message.get("role") != "assistant":
        raise ValueError("Response message role is not assistant.")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise ValueError("Expected string or null final content.")
    reasoning = message.get("reasoning")
    if reasoning is None:
        reasoning = message.get("reasoning_content")
    if reasoning is not None and not isinstance(reasoning, str):
        raise ValueError("Expected string or null reasoning.")
    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str):
        raise ValueError("Missing finish_reason.")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("Missing token usage object.")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if type(usage.get(key)) is not int or usage[key] < 0:
            raise ValueError(f"Missing or invalid usage.{key}.")
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise ValueError("Token usage total does not match prompt + completion.")
    final_text = content or ""
    reasoning_text = reasoning or ""
    if not final_text.strip() and not reasoning_text.strip():
        raise ValueError("Both final content and reasoning are empty.")
    matches = normalized_answer(final_text) in {
        normalized_answer(answer) for answer in case["expected_answers"]
    }
    truncated = finish_reason == "length"
    required = case["kind"] == "required_answer"
    passed = matches and finish_reason == "stop" if required else True
    return dict(
        status=("PASS" if passed else "FAIL") if required else
               ("OBSERVED_TRUNCATED" if truncated else "OBSERVED"),
        passed=passed, protocol_valid=True, final_content=content, reasoning=reasoning,
        finish_reason=finish_reason, truncated=truncated, usage=usage,
        answer_check=dict(
            method="NFKC/casefold and outer punctuation removal, then exact allowed-answer match",
            expected_answers=case["expected_answers"], matched=matches,
            required=required,
            note="Low-reasoning content and truncation are observations, not accuracy failures."
                 if not required else "A mismatch may be formatting; inspect the saved raw response.",
        ),
    )


def request_case(opener, url: str, case: dict, timeout: float) -> dict:
    payload = json.dumps(case["request"], ensure_ascii=False).encode("utf-8")
    request = Request(url, data=payload, method="POST",
                      headers={"Content-Type": "application/json", "Accept": "application/json"})
    result = dict(name=case["name"], kind=case["kind"], started_at=utc_now(),
                  request=case["request"], status="ERROR", passed=False,
                  protocol_valid=False, http_status=None, raw_response=None, response=None)
    start = time.perf_counter()
    try:
        with opener.open(request, timeout=timeout) as reply:
            result["http_status"] = reply.status
            result["response_headers"] = dict(reply.headers.items())
            raw = reply.read(MAX_RESPONSE_BYTES + 1)
        result["raw_response"] = raw[:MAX_RESPONSE_BYTES].decode("utf-8", errors="replace")
        if len(raw) > MAX_RESPONSE_BYTES:
            result["raw_response_truncated"] = True
            raise ValueError("Response exceeds the 1 MiB capture limit.")
        response = json.loads(raw.decode("utf-8"))
        result["response"] = response
        if result["http_status"] != 200:
            raise ValueError(f"Unexpected HTTP status {result['http_status']}.")
        result.update(check_response(case, response))
    except HTTPError as exc:
        result["http_status"] = exc.code
        result["response_headers"] = dict(exc.headers.items()) if exc.headers else {}
        result["error"] = f"HTTPError: {exc}"
        try:
            raw = exc.read(MAX_RESPONSE_BYTES + 1)
            result["raw_response"] = raw[:MAX_RESPONSE_BYTES].decode("utf-8", errors="replace")
            result["raw_response_truncated"] = len(raw) > MAX_RESPONSE_BYTES
        except (URLError, OSError, HTTPException) as read_error:
            result["body_read_error"] = f"{type(read_error).__name__}: {read_error}"
            if isinstance(read_error, IncompleteRead):
                result["raw_response"] = read_error.partial[:MAX_RESPONSE_BYTES].decode("utf-8", errors="replace")
                result["raw_response_truncated"] = len(read_error.partial) > MAX_RESPONSE_BYTES
        finally:
            exc.close()
    except IncompleteRead as exc:
        result["error"] = result["body_read_error"] = f"IncompleteRead: {exc}"
        result["raw_response"] = exc.partial[:MAX_RESPONSE_BYTES].decode("utf-8", errors="replace")
        result["raw_response_truncated"] = len(exc.partial) > MAX_RESPONSE_BYTES
    except (URLError, OSError, HTTPException, UnicodeError, ValueError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["elapsed_s"] = time.perf_counter() - start
        result["completed_at"] = utc_now()
    return result


def save_report(handle, report: dict) -> None:
    """Rewrite only the handle exclusively created by this invocation."""
    handle.seek(0)
    json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    handle.write("\n")
    handle.truncate()
    handle.flush()


def run(args: argparse.Namespace) -> int:
    cases = make_cases(args.model, args.history_reasoning_content)
    output = args.output.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, status='PREPARED' if args.prepare_only else 'RUNNING', created_at=utc_now(), endpoint=args.base_url, model=args.model, timeout_seconds=args.timeout, history_reasoning_content=args.history_reasoning_content, client=dict(path=str(Path(__file__).resolve()), python=sys.version, sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), scope='Five serial nonstreaming chat probes; no statistical accuracy or throughput claim.', latency_definition='HTTP POST start through full response read and client JSON/check processing; no TTFT.', planned_cases=cases, results=[], caveats=['Four short-answer checks are strict rules; inspect raw responses on mismatch.', 'The low-reasoning probe is observational; length truncation does not fail accuracy.', 'Finite token limits and socket timeout are used; urllib timeout is not a hard overall deadline.', 'Requires an already-running compatible server; does not start or change the server.', 'Reasoning extraction depends on server parser configuration.'])
    with open_output(output, 'x', encoding='utf-8') as handle:
        save_report(handle, report)
        if args.prepare_only:
            print(json.dumps(dict(status='PREPARED', output=str(output), requests_sent=0)))
            return 0
        opener = build_opener(ProxyHandler({}), NoRedirect())
        for case in cases:
            result = request_case(opener, args.base_url, case, args.timeout)
            report['results'].append(result)
            save_report(handle, report)
            print(json.dumps(dict(name=case['name'], status=result['status'], elapsed_s=result['elapsed_s'], usage=result.get('usage')), ensure_ascii=False), flush=True)
        results = report['results']
        required = [row for row in results if row['kind'] == 'required_answer']
        report['summary'] = dict(requests_sent=len(results), required_checks=len(required), required_passed=sum((row['passed'] for row in required)), protocol_errors=sum((not row['protocol_valid'] for row in results)), low_reasoning_status=results[-1]['status'])
        success = all((row['passed'] for row in results))
        report['status'] = 'PASS' if success else 'FAIL'
        report['completed_at'] = utc_now()
        save_report(handle, report)
    print(json.dumps(dict(status=report['status'], output=str(output), **report['summary'])))
    return 0 if success else 1


def self_test() -> None:
    """Offline mocks exercise response handling; no sockets or subprocesses."""
    from io import BytesIO
    from tempfile import TemporaryDirectory
    from unittest.mock import Mock, patch
    for invalid in ('http://localhost:8000', 'http://127.0.0.2:8000/v1', 'https://127.0.0.1:8000', 'http://127.0.0.1/v1', 'http://127.0.0.1:0', 'http://127.0.0.1:8000/x', 'http://u@127.0.0.1:8000/v1', 'http://127.0.0.1:8000/v1?q=1'):
        try:
            endpoint(invalid)
        except argparse.ArgumentTypeError:
            pass
        else:
            raise AssertionError(f'Accepted invalid endpoint: {invalid}')
    url = endpoint('http://127.0.0.1:8000/v1')
    assert url.endswith('/v1/chat/completions')
    cases = make_cases('dsv4', True)
    assert len(cases) == 5 and cases[3]['request']['messages'][1]['reasoning_content']
    assert 'reasoning_content' not in make_cases('dsv4', False)[3]['request']['messages'][1]

    def response(content='12', reason=None, finish='stop'):
        return dict(choices=[dict(message=dict(role='assistant', content=content, reasoning=reason), finish_reason=finish)], usage=dict(prompt_tokens=10, completion_tokens=2, total_tokens=12))
    for case, text in zip(cases[:4], ('12.', 'Paris', '七天。', '25')):
        assert check_response(case, response(text))['passed']
    assert not check_response(cases[0], response('112'))['passed']
    assert not check_response(cases[0], response('12', finish='length'))['passed']
    truncated = check_response(cases[-1], response(None, 'Still considering the answer', 'length'))
    assert truncated['passed'] and truncated['status'] == 'OBSERVED_TRUNCATED'
    old = response(None, finish='length')
    old['choices'][0]['message']['reasoning_content'] = 'Legacy field'
    assert check_response(cases[-1], old)['reasoning'] == 'Legacy field'
    for bad in ({}, response(None), {**response(), 'usage': {'total_tokens': 12}}):
        try:
            check_response(cases[0], bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Accepted malformed or empty response')

    class FakeReply(BytesIO):
        status = 200
        headers = {'Content-Type': 'application/json'}
    opener = Mock()
    opener.open.return_value = FakeReply(json.dumps(response()).encode())
    assert request_case(opener, url, cases[0], 1)['passed']
    opener.open.return_value = FakeReply(b'not json')
    assert request_case(opener, url, cases[0], 1)['status'] == 'ERROR'
    opener.open.side_effect = HTTPError(url, 400, 'Bad Request', {}, BytesIO(b'{"error":"bad"}'))
    failed = request_case(opener, url, cases[0], 1)
    assert failed['http_status'] == 400 and failed['raw_response'] == '{"error":"bad"}'
    opener.open.side_effect = URLError('offline mock')
    assert request_case(opener, url, cases[0], 1)['status'] == 'ERROR'
    broken = FakeReply()
    broken.read = Mock(side_effect=IncompleteRead(b'partial', 7))
    opener.open.side_effect = None
    opener.open.return_value = broken
    incomplete = request_case(opener, url, cases[0], 1)
    assert incomplete['status'] == 'ERROR' and incomplete['raw_response'] == 'partial'
    broken = FakeReply()
    broken.read = Mock(side_effect=TimeoutError('body timeout'))
    opener.open.side_effect = HTTPError(url, 400, 'Bad Request', {}, broken)
    incomplete = request_case(opener, url, cases[0], 1)
    assert incomplete['http_status'] == 400 and 'TimeoutError' in incomplete['body_read_error']
    try:
        NoRedirect().redirect_request(Request(url), BytesIO(), 302, 'Found', {}, 'http://example.invalid')
    except HTTPError as exc:
        assert exc.code == 302
        exc.close()
    else:
        raise AssertionError('Redirect was not rejected')
    with TemporaryDirectory(prefix='dsv4-chat-probe-selftest-') as temp:
        output = Path(temp) / 'result.json'
        with open_output(output, 'x', encoding='utf-8') as handle:
            save_report(handle, {'status': 'RUNNING', 'results': ['long']})
            save_report(handle, {'status': 'PASS'})
        assert load_document(output) == {'status': 'PASS'}
        try:
            open_output(output, 'x').close()
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing output was overwritten')
        from contextlib import redirect_stdout
        from io import StringIO
        args = argparse.Namespace(model='dsv4', history_reasoning_content=True, output=Path(temp) / 'complete.json', prepare_only=False, base_url=url, timeout=1)
        opener = Mock()
        opener.open.side_effect = [FakeReply(json.dumps(response(answer)).encode()) for answer in ('12', 'Paris', '7', '25')] + [FakeReply(json.dumps(response(None, 'Thinking', 'length')).encode())]
        with patch(__name__ + '.build_opener', return_value=opener), redirect_stdout(StringIO()):
            assert run(args) == 0
        report = load_document(args.output)
        assert report['status'] == 'PASS' and report['summary']['required_passed'] == 4
        assert report['summary']['low_reasoning_status'] == 'OBSERVED_TRUNCATED'
        assert opener.open.call_count == 5
        with patch(__name__ + '.build_opener') as blocked:
            try:
                run(args)
            except FileExistsError:
                pass
            else:
                raise AssertionError('Run accepted an existing report')
            blocked.assert_not_called()
            args.output = Path(temp) / 'prepared.json'
            args.prepare_only = True
            with redirect_stdout(StringIO()):
                assert run(args) == 0
            blocked.assert_not_called()
        assert load_document(args.output)['status'] == 'PREPARED'
    print('PASS: offline URL, answer, truncation, history, HTTP/JSON error, redirect and exclusive-output checks; no network/GPU access.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", type=endpoint)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="dsv4")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--history-reasoning-content", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.base_url is None or args.output is None:
        parser.error("--base-url and --output are required unless --self-test is used.")
    if not 0 < args.timeout <= 300:
        parser.error("--timeout must be in (0, 300] seconds.")
    try:
        return run(args)
    except OSError as exc:
        print(f"chat_probe: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
