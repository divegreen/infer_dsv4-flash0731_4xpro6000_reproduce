#!/usr/bin/env python3
"""Local Stage C task environments, with no HTTP or model policy.

Six authored fixtures are not SWE-bench, internet research, or customer data.
The coding worker accepts a restricted pure-function subset, not arbitrary
Python: no shell, network, host-file API, Python imports or introspection is
exposed to candidate functions. Resource-limited workers are defense in depth,
not a claim of a general-purpose OS/container security sandbox.
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
import ast
import builtins
import copy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import time
import uuid

FIXTURES = path_at(Path(__file__).resolve().parents[1], 'workloads/sessions/tasks')
MAX_SOURCE = 32768
MAX_TOOL_TEXT = 65536
MAX_WORKER_OUTPUT = 1024*1024
SAFE_NAMES = ('len', 'range', 'sum', 'min', 'max', 'abs', 'all', 'any', 'sorted',
              'list', 'tuple', 'dict', 'set', 'int', 'float', 'str', 'bool',
              'enumerate', 'zip', 'reversed', 'isinstance', 'ValueError', 'TypeError')
SAFE_BUILTINS = {name: getattr(builtins, name) for name in SAFE_NAMES}
SAFE_METHODS = {'get', 'items', 'keys', 'values', 'copy', 'append', 'extend', 'count',
                'index', 'sort', 'reverse', 'pop', 'insert', 'remove', 'strip',
                'split', 'join', 'startswith', 'endswith', 'lower', 'upper',
                'add', 'discard', 'setdefault', 'update'}
NODE_NAMES = '''Module FunctionDef arguments arg Return Assign AnnAssign AugAssign Expr
If For While Break Continue Pass Raise Try ExceptHandler Assert Delete
Name Load Store Del Constant List Tuple Dict Set Subscript Slice Attribute Call keyword
BinOp UnaryOp BoolOp Compare IfExp ListComp SetComp DictComp GeneratorExp comprehension
Add Sub Mult Div FloorDiv Mod Pow UAdd USub Not And Or Eq NotEq Lt LtE Gt GtE Is IsNot
In NotIn BitOr BitAnd BitXor LShift RShift Invert ImportFrom alias'''.split()
ALLOWED_NODES = tuple(getattr(ast, name) for name in NODE_NAMES)


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: '+key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Non-finite JSON number: '+value)
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def strict_equal(actual, expected):
    """JSON structural equality without Python bool/int coercion."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual)==set(expected) and all(strict_equal(actual[k],expected[k]) for k in expected)
    if isinstance(expected, (list,tuple)):
        return len(actual)==len(expected) and all(strict_equal(a,b) for a,b in zip(actual,expected))
    return actual==expected


def catalog_private():
    return strict_json((FIXTURES/'catalog.json').read_text())['tasks']


def task_catalog():
    """Only public metadata; no reference code, answers or hidden cases."""
    return [{key: task[key] for key in ('id', 'kind', 'title', 'provenance')}
            for task in catalog_private()]


def lookup(task_id):
    matches = [task for task in catalog_private() if task['id'] == task_id]
    if len(matches) != 1:
        raise ValueError('Unknown task ID')
    return matches[0]


def file_manifest(directory):
    return {str(path.relative_to(directory)): digest(path.read_bytes())
            for path in sorted(directory.rglob('*')) if path.is_file()}


def checked_path(workspace, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Invalid relative path')
    path = workspace/relative
    if path.resolve() != path.absolute() or path.is_symlink() or not path.is_file():
        raise ValueError('Missing file or symbolic link is not permitted')
    if path.stat().st_size > MAX_TOOL_TEXT:
        raise ValueError('Fixture file exceeds bound')
    return path


def create_session(task_id, owned_dir):
    """Create a new owned directory. Returns a JSON-serializable mutable state.

    No existing directory is reused or deleted. Only public files are copied.
    The caller owns lifecycle/retention and should execute one tool at a time
    within a session. Different sessions share no writable files.
    """
    task = lookup(task_id)
    workspace = Path(owned_dir).absolute()
    if workspace.exists() or workspace.is_symlink() or workspace.resolve() != workspace:
        raise ValueError('Session workspace must be a new path without symbolic-link ancestors')
    workspace.mkdir(parents=True, exist_ok=False)
    token = uuid.uuid4().hex
    source_manifest = file_manifest(FIXTURES / task_id)
    for relative in task['readable']:
        original = checked_path(FIXTURES / task_id / 'public', relative)
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open_output(destination, 'xb') as handle:
            handle.write(original.read_bytes())
    marker = {'task_id': task_id, 'token': token}
    with open_output(workspace / '.session-owner.json', 'x') as handle:
        json.dump(marker, handle)
    return dict(schema_version=1, task_id=task_id, workspace=str(workspace), kind=task['kind'], owner_token=token, source_manifest_sha256=digest(compact(source_manifest)), initial_public_sha256={name: source_manifest['public/' + name] for name in task['readable']}, tool_sequence=0)


def context(session):
    task = lookup(session['task_id'])
    workspace = Path(session['workspace'])
    if not workspace.is_absolute() or workspace.resolve() != workspace or not workspace.is_dir():
        raise ValueError('Invalid owned workspace')
    marker = strict_json(checked_path(workspace, '.session-owner.json').read_text())
    if marker != {'task_id': task['id'], 'token': session['owner_token']}:
        raise ValueError('Session ownership mismatch')
    if digest(compact(file_manifest(FIXTURES/task['id']))) != session['source_manifest_sha256']:
        raise ValueError('Task source fixture changed during session')
    for path in task['readable']:
        local = checked_path(workspace, path)
        if path not in task['writable'] and digest(local.read_bytes()) != session['initial_public_sha256'][path]:
            raise ValueError('Immutable public fixture changed')
    return task, workspace


def state_hash(task, workspace):
    return digest(compact({name: digest(checked_path(workspace, name).read_bytes()) for name in task['readable']}))


def initial_messages(session):
    task, _ = context(session)
    extra = ''
    if task['kind'] == 'coding':
        extra = (' Candidate source must stay in the supported pure-function Python subset: only top-level function '
                 'definitions and exact from-imports of functions from the listed local modules. No aliases, '
                 'decorators, annotations, classes, private names, dynamic calls, filesystem/process/network API '
                 'or external imports. Common container/string methods and basic builtins are available. '
                 'Do not edit tests. Hidden grading is separate from public tool tests.')
    system = ('You are solving a finite local benchmark task with real tools. Choose which tools to call and '
              'their arguments yourself; no tool sequence is prescribed. Use only tool results and task evidence. '
              'The task corpus, repository and data were manually constructed for this research. '
              'No internet or arbitrary shell is available. Complete within the caller-provided turn/tool budget.'+extra)
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': task['prompt']}]


def schema(name, description, properties, required=()):
    return dict(type='function', function=dict(name=name, description=description,
                parameters=dict(type='object', properties=properties, required=list(required), additionalProperties=False)))


TEXT = {'type': 'string', 'maxLength': MAX_SOURCE}


def tools_for(session):
    task, _ = context(session)
    if task['kind'] == 'coding':
        return [schema('list_files', 'List the readable local repository files and writable source files.', {}),
                schema('read_file', 'Read an allowed repository file, including immutable public cases.', {'path': TEXT}, ('path',)),
                schema('search', 'Literal case-sensitive search across the readable repository files.', {'query': TEXT}, ('query',)),
                schema('replace_text', 'Replace one exact occurrence in a writable source file. Validation is atomic; invalid code is not installed.',
                       {'path': TEXT, 'old': TEXT, 'new': TEXT}, ('path', 'old', 'new')),
                schema('run_tests', 'Run only immutable public cases in a resource-limited worker. No hidden grading feedback.', {})]
    if task['kind'] == 'research':
        docs = [name.removeprefix('docs/') for name in task['readable'] if name.startswith('docs/')]
        return [schema('search_docs', 'Search the fixed local corpus with case-insensitive whitespace-separated terms; return ranked matching numbered lines.',
                       {'query': {'type': 'string', 'maxLength': 256}}, ('query',)),
                schema('read_doc', 'Read all numbered lines of one fixed corpus document.',
                       {'doc_id': {'type': 'string', 'enum': docs}}, ('doc_id',))]
    table = {'type': 'string', 'enum': list(task['tables'])}
    column = {'type': 'string', 'maxLength': 64}
    scalar = {'type': ['integer', 'string', 'null']}
    filter_value = {'type': ['integer', 'string', 'null', 'array'], 'items': scalar,
                    'description': 'Match the declared CSV column type: integer literal for integer columns, string for string columns, null for missing values. For op=in, provide an array of values of that column type.'}
    filters = {'type': 'array', 'maxItems': 8, 'items': {'type': 'object', 'properties': {
        'column': column, 'op': {'type': 'string', 'enum': ['eq', 'ne', 'lt', 'le', 'gt', 'ge', 'in']},
        'value': filter_value}, 'required': ['column', 'op', 'value'], 'additionalProperties': False}}
    dedup = {'type': 'object', 'properties': {'key': column, 'latest_by': column},
             'required': ['key', 'latest_by'], 'additionalProperties': False}
    common = dict(table=table, filters=filters, deduplicate=dedup)
    return [schema('describe_table', 'Show immutable CSV schema, row count, null conventions and sample rows.', {'table': table}, ('table',)),
            schema('query_rows', 'Select rows after optional latest-revision deduplication, then AND filters. Return a bounded prefix with explicit matched/returned counts.',
                   dict(common, columns={'type': 'array', 'items': column, 'maxItems': 16}, limit={'type': 'integer', 'minimum': 1, 'maximum': 50}), ('table',)),
            schema('aggregate_table', 'Deduplicate first, then filter, group and aggregate. sum/mean/min/max/p95 skip null numeric cells; count counts rows. p95 uses linear interpolation at (n-1)*.95.',
                   dict(common, group_by={'type': 'array', 'items': column, 'maxItems': 3}, metrics={'type': 'array', 'minItems': 1, 'maxItems': 8,
                       'items': {'type': 'object', 'properties': {'name': column, 'op': {'type': 'string', 'enum': ['count','sum','mean','min','max','p95']}, 'column': column},
                                 'required': ['name','op'], 'additionalProperties': False}}), ('table','metrics'))]


def validate_source(source, module, task):
    if len(source.encode()) > MAX_SOURCE:
        raise ValueError('Source exceeds 32 KiB')
    tree = ast.parse(source)
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if not functions or len(functions) != sum(isinstance(n, ast.FunctionDef) for n in tree.body):
        raise ValueError('At least one unique top-level function required')
    imports = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in task['modules'] or node.module == module:
                raise ValueError('Only explicit imports from other local fixture modules are permitted')
            for alias in node.names:
                if alias.asname or alias.name == '*' or alias.name.startswith('_') or alias.name in imports or alias.name in functions or alias.name in SAFE_BUILTINS:
                    raise ValueError('Invalid or conflicting local function import')
                imports[alias.name] = node.module
        elif not isinstance(node, ast.FunctionDef):
            raise ValueError('Only top-level functions and approved local imports are supported')
    nodes = list(ast.walk(tree))
    if len(nodes) > 10000:
        raise ValueError('AST exceeds node bound')
    for node in nodes:
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError('Unsupported Python syntax: '+type(node).__name__)
        if isinstance(node, ast.ImportFrom) and node not in tree.body:
            raise ValueError('Nested imports are not allowed')
        if isinstance(node, ast.FunctionDef):
            if node not in tree.body or node.name.startswith('_') or node.name in SAFE_BUILTINS or node.decorator_list or node.returns:
                raise ValueError('No nested/private functions, builtin shadowing, decorators or annotations')
            for default in node.args.defaults+list(filter(None,node.args.kw_defaults)):
                if not isinstance(default, ast.Constant) or not isinstance(default.value, (type(None),bool,int,float,str)):
                    raise ValueError('Defaults must be simple scalar literals')
        if isinstance(node, ast.Name) and node.id.startswith('_'):
            raise ValueError('Private or introspective names are not allowed')
        if isinstance(node, ast.arg) and (node.arg.startswith('_') or node.annotation):
            raise ValueError('No private or annotated arguments')
        if isinstance(node, ast.Attribute) and (node.attr not in SAFE_METHODS or not isinstance(node.ctx, ast.Load)):
            raise ValueError('Only listed container/string methods are supported')
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in SAFE_BUILTINS and node.func.id not in functions and node.func.id not in imports:
                    raise ValueError('Dynamic or unknown function call: '+node.func.id)
            elif not isinstance(node.func, ast.Attribute):
                raise ValueError('Indirect calls are not allowed')
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (str, bytes)) and len(node.value) > 8192:
                raise ValueError('Literal exceeds bound')
            if isinstance(node.value, int) and node.value.bit_length() > 4096:
                raise ValueError('Integer literal exceeds bound')
    return tree, functions, imports


def validate_sources(task, sources):
    parsed = {module: validate_source(sources[path], module, task) for module,path in task['modules'].items()}
    for module, (_, _, imports) in parsed.items():
        for name, origin in imports.items():
            if name not in parsed[origin][1]:
                raise ValueError('Imported local function is undefined: '+origin+'.'+name)
    return parsed


def load_functions(task, sources):
    parsed = validate_sources(task, sources)
    loaded, pending = {}, set()
    def load(module):
        if module in loaded:
            return loaded[module]
        if module in pending:
            raise ValueError('Local module dependency cycle')
        pending.add(module)
        tree, functions, imports = parsed[module]
        namespace = {'__builtins__': SAFE_BUILTINS.copy()}
        for name, origin in imports.items():
            namespace[name] = load(origin)[name]
        pure = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
        exec(compile(pure, task['modules'][module], 'exec'), namespace)
        loaded[module] = {name: namespace[name] for name in functions}
        pending.remove(module)
        return loaded[module]
    for module in parsed:
        load(module)
    return loaded


def worker(workspace, task_id, mode):
    # Limits precede candidate parsing, validation, compilation and execution.
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (256*1024*1024, 256*1024*1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_WORKER_OUTPUT, MAX_WORKER_OUTPUT))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    task = lookup(task_id)
    if task['kind'] != 'coding' or mode not in ('public','grading'):
        raise ValueError('Invalid worker target')
    workspace = Path(workspace).resolve()
    sources = {path: checked_path(workspace,path).read_text() for path in task['writable']}
    modules = load_functions(task,sources)
    cases = strict_json((FIXTURES/task_id/'public/tests/public_cases.json').read_text())
    if mode == 'grading':
        cases += strict_json((FIXTURES/task_id/'private/grading_cases.json').read_text())
    results = []
    for case in cases:
        args = copy.deepcopy(case['args'])
        before = copy.deepcopy(args)
        module,name = case['target'].rsplit('.',1)
        row = {'id':case['id'],'passed':False}
        try:
            actual = modules[module][name](*args)
            # JSON normalizes tuple/list API return conventions in test fixtures.
            actual = strict_json(compact(actual))
            row.update(actual=actual, passed='error' not in case and strict_equal(actual,case.get('expected')))
            if mode == 'public':
                row['expected'] = case.get('expected')
        except BaseException as exc:
            row.update(error_type=type(exc).__name__,error_message=str(exc)[:2048],passed=type(exc).__name__==case.get('error'))
        row['inputs_unchanged'] = strict_equal(args,before)
        row['passed'] = row['passed'] and row['inputs_unchanged']
        if mode == 'grading':
            row = {key:row[key] for key in ('id','passed','inputs_unchanged')}
        results.append(row)
    print(compact(dict(executed=True,passed=all(r['passed'] for r in results),tests_run=len(results),cases=results)))


def run_code_tests(task, workspace, grading=False, timeout_s=5.0):
    start = time.perf_counter()
    command = [sys.executable, '-I', '-S', '-B', str(Path(__file__).resolve()), '--worker', str(workspace), task['id'], 'grading' if grading else 'public']
    record = dict(command=command, grading=grading, timeout_s=timeout_s, executed=False, passed=False, timed_out=False)
    env = {'PATH': os.defpath, 'LANG': 'C.UTF-8', 'CUDA_VISIBLE_DEVICES': ''}
    with tempfile.TemporaryDirectory(prefix='.worker-', dir=workspace) as owned:
        with open_output(Path(owned) / 'stdout', 'w+b') as stdout, open_output(Path(owned) / 'stderr', 'w+b') as stderr:
            proc = subprocess.Popen(command, cwd=workspace, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True, close_fds=True)
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                record['timed_out'] = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=2)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=2)
            record['returncode'] = proc.returncode
            for key, stream in (('stdout', stdout), ('stderr', stderr)):
                stream.seek(0)
                raw = stream.read(MAX_WORKER_OUTPUT + 1)
                record[key] = raw[:MAX_WORKER_OUTPUT].decode('utf-8', errors='replace')
                record[key + '_truncated'] = len(raw) > MAX_WORKER_OUTPUT
            if proc.returncode == 0 and (not record['timed_out']) and (not record['stdout_truncated']):
                try:
                    result = strict_json(record['stdout'])
                    if not isinstance(result, dict) or result.get('executed') is not True:
                        raise ValueError('Malformed worker result')
                    record.update(result)
                except ValueError as exc:
                    record['parse_error'] = str(exc)
    record['wall_s'] = time.perf_counter() - start
    return record


def require_keys(args, required, optional=()):
    if not isinstance(args,dict) or not set(required)<=set(args) or set(args)-set(required)-set(optional):
        raise ValueError('Unknown or missing tool argument keys')


def text_arg(value, max_bytes=MAX_SOURCE, nonempty=True):
    if not isinstance(value,str) or len(value.encode())>max_bytes or (nonempty and not value):
        raise ValueError('Expected bounded text argument')
    return value


def read_table(task,workspace,name):
    if name not in task['tables']:
        raise ValueError('Unknown table')
    info = task['tables'][name]
    reader = csv.DictReader(io.StringIO(checked_path(workspace,info['path']).read_text()))
    if reader.fieldnames != list(info['columns']):
        raise ValueError('CSV schema changed')
    rows=[]
    for row in reader:
        parsed={}
        for key,kind in info['columns'].items():
            value=row[key]
            parsed[key] = None if value=='' else int(value) if kind=='integer' else value
        rows.append(parsed)
    return info,rows


def filtered_table(task,workspace,args):
    info,rows = read_table(task,workspace,args['table'])
    input_rows=len(rows)
    if 'deduplicate' in args:
        spec=args['deduplicate'];require_keys(spec,('key','latest_by'))
        key,latest=spec['key'],spec['latest_by']
        if key not in info['columns'] or latest not in info['columns'] or info['columns'][latest]!='integer':
            raise ValueError('Invalid deduplication columns')
        grouped={}
        for row in rows:
            old=grouped.get(row[key])
            if old is not None and row[latest]==old[latest] and old!=row:
                raise ValueError('Conflicting duplicate with equal revision')
            if old is None or row[latest]>old[latest]:
                grouped[row[key]]=row
        rows=list(grouped.values())
    after_dedup=len(rows)
    filters=args.get('filters',[])
    if not isinstance(filters,list) or len(filters)>8:
        raise ValueError('At most eight filters')
    for condition in filters:
        require_keys(condition,('column','op','value'))
        key,op,value=condition['column'],condition['op'],condition['value']
        if key not in info['columns'] or op not in ('eq','ne','lt','le','gt','ge','in'):
            raise ValueError('Invalid filter column/operator')
        values=value if op=='in' else [value]
        if not isinstance(values,list) or len(values)>32 or any(v is not None and
            (type(v) is not int if info['columns'][key]=='integer' else not isinstance(v,str)) for v in values):
            raise ValueError('Filter value type differs from column')
        def matches(row):
            actual=row[key]
            if op=='in':return actual in value
            if op=='eq':return actual==value
            if op=='ne':return actual!=value
            if actual is None or value is None:return False
            return {'lt':lambda:actual<value,'le':lambda:actual<=value,'gt':lambda:actual>value,'ge':lambda:actual>=value}[op]()
        rows=[r for r in rows if matches(r)]
    return info,rows,dict(input_rows=input_rows,after_dedup_rows=after_dedup,matched_rows=len(rows),operation_order='deduplicate_then_filter')


def data_tool(task,workspace,name,args):
    if name=='describe_table':
        require_keys(args,('table',));info,rows=read_table(task,workspace,args['table'])
        return dict(table=args['table'],schema=info['columns'],description=info['description'],rows=len(rows),sample_rows=rows[:5],null='Empty CSV numeric cells are null; aggregations skip null numeric cells.')
    optional=('filters','deduplicate','columns','limit') if name=='query_rows' else ('filters','deduplicate','group_by')
    required=('table',) if name=='query_rows' else ('table','metrics')
    if name not in ('query_rows','aggregate_table'):
        raise ValueError('Unknown tool for data task')
    require_keys(args,required,optional);info,rows,counts=filtered_table(task,workspace,args)
    if name=='query_rows':
        columns=args.get('columns',list(info['columns']));limit=args.get('limit',20)
        if not isinstance(columns,list) or not columns or len(columns)>16 or len(set(columns))!=len(columns) or any(c not in info['columns'] for c in columns):
            raise ValueError('Invalid selected columns')
        if type(limit) is not int or not 1<=limit<=50:
            raise ValueError('Row limit must be 1..50')
        return dict(counts,rows=[{c:r[c] for c in columns} for r in rows[:limit]],returned_rows=min(len(rows),limit),truncated=len(rows)>limit)
    groups=args.get('group_by',[]);metrics=args['metrics']
    if not isinstance(groups,list) or len(groups)>3 or len(set(groups))!=len(groups) or any(c not in info['columns'] for c in groups):
        raise ValueError('Invalid grouping columns')
    if not isinstance(metrics,list) or not 1<=len(metrics)<=8:
        raise ValueError('Expected 1..8 metrics')
    aliases=set(groups)
    for metric in metrics:
        require_keys(metric,('name','op'),('column',))
        alias=text_arg(metric['name'],64)
        if not alias.isidentifier() or alias.startswith('_') or alias in aliases:
            raise ValueError('Metric names must be unique public identifiers')
        aliases.add(alias)
        if metric['op'] not in ('count','sum','mean','min','max','p95'):
            raise ValueError('Unknown aggregate operation')
        if metric['op']!='count' and info['columns'].get(metric.get('column'))!='integer':
            raise ValueError('Numeric aggregate requires an integer column')
        if metric['op']=='count' and 'column' in metric:
            raise ValueError('count takes no column and counts all filtered rows')
    buckets={}
    for row in rows:buckets.setdefault(tuple(row[c] for c in groups),[]).append(row)
    if not groups and not rows:buckets[()]=[]
    output=[]
    for key,members in sorted(buckets.items(),key=lambda pair:compact(pair[0])):
        row=dict(zip(groups,key))
        for metric in metrics:
            op=metric['op'];values=sorted(r[metric['column']] for r in members if r[metric['column']] is not None) if op!='count' else []
            if op=='count':value=len(members)
            elif op=='sum':value=sum(values)
            elif not values:value=None
            elif op=='mean':value=sum(values)/len(values)
            elif op=='min':value=values[0]
            elif op=='max':value=values[-1]
            else:
                position=(len(values)-1)*.95;lo=math.floor(position);hi=math.ceil(position)
                value=values[lo]+(values[hi]-values[lo])*(position-lo)
            row[metric['name']]=value
        output.append(row)
    return dict(counts,group_count=len(output),rows=output)


def execute_tool(session, call):
    """Execute exactly the supplied model call. No retries or next-step policy.

    result is the public JSON value to put in a tool message. The full record
    additionally retains worker execution details, errors and state hashes.
    """
    start=time.perf_counter()
    record=dict(call=copy.deepcopy(call),valid=False,action_ok=False,result=None)
    try:
        task,workspace=context(session)
        record['state_before_sha256']=state_hash(task,workspace)
        if not isinstance(call,dict) or call.get('type')!='function' or not isinstance(call.get('function'),dict):
            raise ValueError('Expected OpenAI function tool call')
        fn=call['function'];name=text_arg(fn.get('name'),64)
        arguments=text_arg(fn.get('arguments'),MAX_TOOL_TEXT,False)
        args=strict_json(arguments)
        names={tool['function']['name'] for tool in tools_for(session)}
        if name not in names or not isinstance(args,dict):
            raise ValueError('Tool or argument object is invalid for this task')
        record['valid']=True
        if task['kind']=='coding':
            expected={'list_files':(),'read_file':('path',),'search':('query',),'replace_text':('path','old','new'),'run_tests':()}
            require_keys(args,expected[name])
            for value in args.values():text_arg(value,MAX_SOURCE,False)
            if name=='list_files':result=dict(readable=task['readable'],writable=task['writable'])
            elif name=='run_tests':
                test=run_code_tests(task,workspace);record['test_execution']=test
                result={k:test[k] for k in ('executed','passed','tests_run','cases','timed_out','returncode','parse_error') if k in test}
                if not test['executed']:result['error']='Worker did not complete valid public tests; inspect and repair source.'
                record['action_ok']=test['executed']
            elif name=='search':
                query=text_arg(args['query'],256)
                result=dict(matches=[dict(path=path,line=i,text=line) for path in task['readable'] for i,line in enumerate(checked_path(workspace,path).read_text().splitlines(),1) if query in line])
            else:
                path=args['path']
                if path not in task['readable']:raise ValueError('File is not in the public allowlist')
                local=checked_path(workspace,path);source=local.read_text()
                if name=='read_file':result=dict(path=path,content=source)
                else:
                    if path not in task['writable']:raise ValueError('Only named source files may be changed')
                    if not args['old'] or source.count(args['old'])!=1 or args['old']==args['new']:
                        raise ValueError('old must match exactly once and new must differ')
                    changed=source.replace(args['old'],args['new'],1)
                    sources={p:checked_path(workspace,p).read_text() for p in task['writable']};sources[path]=changed
                    validate_sources(task,sources)
                    # No model code executes here. A same-directory replace is
                    # atomic; private grading data is never copied into session.
                    with tempfile.NamedTemporaryFile('w',dir=local.parent,delete=False) as tmp:
                        tmp.write(changed);temporary=Path(tmp.name)
                    try:os.replace(temporary,local)
                    finally:temporary.unlink(missing_ok=True)
                    result=dict(path=path,replaced=1,before_sha256=digest(source),after_sha256=digest(changed))
        elif task['kind']=='research':
            docs={p.removeprefix('docs/'):p for p in task['readable'] if p.startswith('docs/')}
            if name=='read_doc':
                require_keys(args,('doc_id',))
                if args['doc_id'] not in docs:raise ValueError('Unknown public document')
                result=dict(doc_id=args['doc_id'],lines=[dict(line=i,text=line) for i,line in enumerate(checked_path(workspace,docs[args['doc_id']]).read_text().splitlines(),1)])
            else:
                require_keys(args,('query',));query=text_arg(args['query'],256);terms=query.lower().split()
                if not terms:raise ValueError('Search requires at least one term')
                matches=[]
                for doc_id,path in docs.items():
                    lines=[dict(line=i,text=line) for i,line in enumerate(checked_path(workspace,path).read_text().splitlines(),1) if any(term in line.lower() for term in terms)]
                    if lines:matches.append(dict(doc_id=doc_id,matching_lines=len(lines),lines=lines))
                result=dict(query=query,matches=sorted(matches,key=lambda row:(-row['matching_lines'],row['doc_id'])))
        else:result=data_tool(task,workspace,name,args)
        if name!='run_tests':record['action_ok']=True
        record['result']=result
    except (KeyError,TypeError,ValueError,SyntaxError,OSError,RecursionError) as exc:
        record['result']=dict(error=type(exc).__name__,message=str(exc)[:4096])
    finally:
        session['tool_sequence']=session.get('tool_sequence',0)+1
        record['sequence']=session['tool_sequence']
        record['wall_s']=time.perf_counter()-start
        try:
            task,workspace=context(session);record['state_after_sha256']=state_hash(task,workspace)
        except (KeyError,TypeError,ValueError,OSError) as exc:
            record['state_validation_error']=str(exc)
            record['action_ok']=False
    return record


def grade(session, final_text):
    """Private terminal scoring. Never add this return value to model history."""
    start=time.perf_counter();result={'success':False,'details':{}}
    try:
        task,workspace=context(session)
        if not isinstance(final_text,str) or not final_text.strip() or len(final_text.encode())>MAX_TOOL_TEXT:
            raise ValueError('A nonempty bounded final answer is required')
        if task['kind']=='coding':
            test=run_code_tests(task,workspace,grading=True)
            result.update(success=test['executed'] and test['passed'],details=test)
        else:
            answer=strict_json(final_text)
            key=strict_json((FIXTURES/task['id']/'private/answers.json').read_text())
            checks={}
            if task['kind']=='research':
                require_keys(answer,('answers',));require_keys(answer['answers'],tuple(key))
                for field,expected in key.items():
                    item=answer['answers'][field];require_keys(item,('value','citations'))
                    citations=item['citations']
                    if not isinstance(citations,list) or not 1<=len(citations)<=12:
                        raise ValueError('Every answer needs 1..12 citations')
                    covered={}
                    for citation in citations:
                        require_keys(citation,('doc_id','start_line','end_line'))
                        path='docs/'+str(citation['doc_id'])
                        if path not in task['readable']:raise ValueError('Citation names an unknown document')
                        lo,hi=citation['start_line'],citation['end_line']
                        count=len(checked_path(workspace,path).read_text().splitlines())
                        if type(lo) is not int or type(hi) is not int or not 1<=lo<=hi<=count:
                            raise ValueError('Citation line range is invalid')
                        covered.setdefault(citation['doc_id'],set()).update(range(lo,hi+1))
                    correct=strict_equal(item['value'],expected['value'])
                    supported=all(set(range(lo,hi+1))<=covered.get(doc,set()) for doc,lo,hi in expected['support'])
                    checks[field]=dict(value_correct=correct,evidence_supported=supported,passed=correct and supported)
            else:
                require_keys(answer,('answers','method'));require_keys(answer['answers'],tuple(key['answers']))
                checks['method']=dict(passed=strict_equal(answer['method'],key['method']))
                for field,expected in key['answers'].items():
                    value=answer['answers'][field]
                    numeric_type_ok = type(value) is int if type(expected) is int else type(value) in (int,float)
                    good=numeric_type_ok and math.isfinite(value) and abs(value-expected)<=1e-6
                    checks[field]=dict(passed=good)
            result.update(success=all(c['passed'] for c in checks.values()),details=dict(checks=checks,passed_fields=sum(c['passed'] for c in checks.values()),total_fields=len(checks)))
    except (KeyError,TypeError,ValueError,OSError) as exc:
        result['details']=dict(error=type(exc).__name__,message=str(exc)[:4096])
    result['wall_s']=time.perf_counter()-start
    result['grading_scope']='Local objective task checks; not a general capability or production-quality benchmark'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',nargs=3,metavar=('WORKSPACE','TASK','MODE'))
    args=parser.parse_args()
    if not args.worker:parser.error('Use the Python task API; this module does not call a model service')
    worker(*args.worker)


if __name__=='__main__':main()
