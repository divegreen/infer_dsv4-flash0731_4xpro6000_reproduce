#!/usr/bin/env python3
"""Read-only, secondary JSON-content reference; strict primary never changes.

Extracts exactly one maximal balanced JSON container without looking at the
answer/schema. Uses frozen source-snapshot fixtures for objective checks. No
model, tool, generated code, or primary grader execution occurs.
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
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import zipfile


def need(value,message):
    if not value:raise ValueError(message)


def strict(text):
    def unique(pairs):
        value={}
        for key,item in pairs:
            if key in value:raise ValueError('Duplicate JSON key')
            value[key]=item
        return value
    def invalid(value):raise ValueError('Nonfinite JSON number: '+value)
    return json.loads(text,object_pairs_hook=unique,parse_constant=invalid)


def equal(a,b):
    if type(a)is not type(b):return False
    if isinstance(b,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in b)
    if isinstance(b,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def extract(text):
    need(isinstance(text,str) and 0<len(text.encode())<=65536,'Missing or unbounded final text')
    stack=[];spans=[];start=None;quoted=False;escaped=False
    for index,char in enumerate(text):
        if not stack:
            if char in '{[':stack.append(char);start=index
            elif char in '}]':raise ValueError('Unmatched closing container outside candidate')
            continue
        if quoted:
            if escaped:escaped=False
            elif char=='\\':escaped=True
            elif char=='"':quoted=False
            continue
        if char=='"':quoted=True
        elif char in '{[':stack.append(char)
        elif char in '}]':
            need(stack.pop()==('{' if char=='}' else '['),'Mismatched container brackets')
            if not stack:spans.append((start,index+1))
    need(not stack and not quoted,'Incomplete maximal JSON container')
    need(len(spans)==1,'Require exactly one maximal container; never choose among candidates')
    left,right=spans[0];candidate=text[left:right];value=strict(candidate)
    need(isinstance(value,dict),'Final candidate must be an object')
    return dict(value=value,start_char=left,end_char=right,candidate_text=candidate,
                candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
                extraction='IDENTITY' if not (text[:left]+text[right:]).strip() else 'SINGLE_CONTAINER_WITH_WRAPPER')


def content_grade(value,task,answers,source_bytes,prefix):
    need(isinstance(value,dict),'Expected object')
    checks={}
    if task['kind']=='research':
        need(set(value)=={'answers'} and isinstance(value['answers'],dict) and set(value['answers'])==set(answers),'Research answer keys')
        for key,expected in answers.items():
            item=value['answers'][key]
            need(isinstance(item,dict) and set(item)=={'value','citations'},'Research item schema')
            citations=item['citations'];need(isinstance(citations,list) and 1<=len(citations)<=12,'Citation count')
            coverage={}
            for citation in citations:
                need(isinstance(citation,dict) and set(citation)=={'doc_id','start_line','end_line'},'Citation schema')
                path='docs/'+str(citation['doc_id']);need(path in task['readable'],'Unknown cited document')
                length=len(source_bytes[prefix+'public/'+path].decode().splitlines())
                low,high=citation['start_line'],citation['end_line']
                need(type(low)is int and type(high)is int and 1<=low<=high<=length,'Invalid citation lines')
                coverage.setdefault(citation['doc_id'],set()).update(range(low,high+1))
            correct=equal(item['value'],expected['value'])
            supported=all(set(range(low,high+1))<=coverage.get(doc,set()) for doc,low,high in expected['support'])
            checks[key]=dict(value_correct=correct,evidence_supported=supported,passed=correct and supported)
    else:
        need(task['kind']=='data','Only research/data structured outputs supported')
        need(set(value)=={'answers','method'} and isinstance(value['answers'],dict) and set(value['answers'])==set(answers['answers']),'Data answer keys')
        checks['method']=dict(passed=equal(value['method'],answers['method']))
        for key,expected in answers['answers'].items():
            actual=value['answers'][key]
            numeric=type(actual)is int if type(expected)is int else type(actual) in (int,float)
            checks[key]=dict(passed=bool(numeric and math.isfinite(actual) and abs(actual-expected)<=1e-6))
    return dict(passed=all(row['passed'] for row in checks.values()),checks=checks)


def self_test():
    checks=[]
    for label,text in [('plain','{"a":1}'),('fence','Answer:\n```json\n{"a":[{"b":2}]}\n```'),
                       ('quoted_braces','Text {"a":"braces {} and escaped \\\" quote"} suffix')]:
        need(extract(text)['value'] is not None,label);checks.append(label)
    for label,text in [('multiple','{"a":1} {"a":1}'),('array','[1,2]'),('duplicate','{"a":1,"a":2}'),
                       ('nan','{"a":NaN}'),('infinity','{"a":Infinity}'),('malformed_outer','{bad: {"a":1}}'),
                       ('incomplete','{"a":1'),('unmatched','{"a":1}]'),('trailing_comma','{"a":1,}'),
                       ('no_candidate','the answer is 2'),('nested_decoy','Example {} then {"a":1}'),('mismatch','{"a":[1,2}}')]:
        try:extract(text)
        except ValueError:checks.append('reject_'+label)
        else:raise AssertionError('Accepted '+label)
    need(not equal(True,1) and not equal(1,1.0) and equal({'a':[1]}, {'a':[1]}),'Type-preserving equality')
    checks.append('type_preserving_equality')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    need(not args.output.exists(), 'Use a new reference output ID')
    ledger = {}

    def read(path):
        raw = path.read_bytes()
        ledger[str(path.resolve())] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        return raw
    result = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), reference_version='single-maximal-json-v1', primary_unchanged=True, applied_to_runtime=False, feedback_to_model=False, sessions=[])
    read(Path(__file__))
    try:
        if args.self_test:
            result['self_tests'] = self_test()
        if args.result_dir:
            directory = args.result_dir.resolve()
            need(directory not in args.output.resolve().parents, 'Output must not change original results')
            status = json.loads(read(directory / 'status.json'))
            report = json.loads(read(directory / 'sessions.json'))
            command = json.loads(read(directory / 'command.json'))
            need(status.get('end_utc') and type(status.get('exit_code')) is int, 'Experiment must be terminal')
            snapshot = directory / 'source-snapshot.zip'
            read(snapshot)
            with zipfile.ZipFile(snapshot) as archive:
                source = {name: archive.read(name) for name in archive.namelist()}
            base = 'scenarios/workloads/sessions/tasks/'
            catalog = json.loads(source[base + 'catalog.json'])['tasks']
            tasks = {t['id']: t for t in catalog}
            used = {base + 'catalog.json'}
            for phase in report['phases']:
                if phase['phase']['kind'] != 'agent':
                    continue
                for row in phase['sessions']:
                    task = tasks[row['task_id']]
                    entry = dict(phase=phase['phase']['id'], session_index=row['index'], task_id=row['task_id'], kind=task['kind'], original_primary_success=row.get('task_success'), terminal_reason=row['terminal_reason'], reference_eligible=False, reference_passed=None)
                    if task['kind'] == 'coding':
                        entry['exclusion'] = 'Code task: preserve original test result; no JSON reference scoring'
                    elif row['terminal_reason'] != 'natural_stop' or not row['protocol_ok'] or (not isinstance(row.get('final_text'), str)):
                        entry['exclusion'] = 'No complete natural final answer; do not grade partial/budget-exhausted text'
                    else:
                        entry['reference_eligible'] = True
                        entry['full_final_sha256'] = hashlib.sha256(row['final_text'].encode()).hexdigest()
                        try:
                            extracted = extract(row['final_text'])
                            entry['extraction'] = {k: v for k, v in extracted.items() if k != 'value'}
                            prefix = base + task['id'] + '/'
                            answer_path = prefix + 'private/answers.json'
                            answers = strict(source[answer_path].decode())
                            used.add(answer_path)
                            used.update((prefix + 'public/' + p for p in task['readable']))
                            grade = content_grade(extracted['value'], task, answers, source, prefix)
                            entry['reference_grade'] = grade
                            entry['reference_passed'] = grade['passed']
                        except (ValueError, KeyError, TypeError) as exc:
                            entry.update(reference_passed=False, reference_error=dict(type=type(exc).__name__, message=str(exc)))
                    result['sessions'].append(entry)
            for name in used:
                need(hashlib.sha256(source[name]).hexdigest() == command['source_sha256'][name], 'Frozen fixture SHA mismatch')
            result.update(experiment=directory.name, supervisor_exit_code=status['exit_code'], original_primary_successes=sum((s['original_primary_success'] is True for s in result['sessions'])), original_session_denominator=len(result['sessions']), structured_reference_eligible=sum((s['reference_eligible'] for s in result['sessions'])), structured_reference_passes=sum((s['reference_passed'] is True for s in result['sessions'])), fixture_source_sha256={name: command['source_sha256'][name] for name in sorted(used)})
        need(args.self_test or args.result_dir, 'Provide input or self-test')
        for path, info in ledger.items():
            need(hashlib.sha256(Path(path).read_bytes()).hexdigest() == info['sha256'], 'Input changed')
    except Exception as exc:
        import traceback
        result.update(status='FAILED', error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
    result['input_ledger'] = ledger
    result['limitations'] = ['Post-pilot diagnostic reference, not the frozen primary task score or successful-task throughput numerator.', 'Surrounding prose is not semantically scored; a correct JSON object does not establish that all narration is correct.', 'Conservative lexical scanning can reject prose containing extra bracket notation; no candidate is selected using answers or schema.', 'No repair, type conversion, schema-based candidate selection, or model feedback; apply identically to all eligible future outputs.', 'Reuses frozen objective answer/citation rubric; does not establish rubric validity or broad task capability.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, 'x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'experiment', 'original_primary_successes', 'original_session_denominator', 'structured_reference_eligible', 'structured_reference_passes', 'error') if k in result}, indent=2))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__=='__main__':main()
