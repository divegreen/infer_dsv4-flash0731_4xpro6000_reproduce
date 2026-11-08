#!/usr/bin/env python3
"""Portable research entry point; see doc/reproduction.md for the run protocol."""
from __future__ import annotations
import argparse
import ast
import contextlib
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
RESEARCH = PROJECT / 'research/dsv4-flash0731'
sys.path.insert(0, str(RESEARCH))
import project_layout as layout
CATALOG = RESEARCH / 'experiments/catalog.json'
RESULTS = PROJECT / 'results'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n'); temporary = Path(f.name)
    temporary.replace(path)


def catalog():
    return read(CATALOG)


def python_binary():
    path = Path(os.environ.get('VLLM_PYTHON', str(PROJECT / '.venv-vllm/bin/python'))).expanduser()
    return (path if path.is_absolute() else PROJECT / path).absolute()


def sources():
    paths = [p for p in RESEARCH.rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.csv', '.md')
             and 'doc' not in p.relative_to(RESEARCH).parts and '__pycache__' not in p.parts]
    paths += [Path(__file__), PROJECT / 'requirements-vllm.lock.txt']
    return sorted(set(paths))


def runtime_fingerprint():
    # Use package install metadata and small model files, without reading 167 GB
    # of weight payload on every resume check. Weight size/mtime changes invalidate reuse.
    binary = python_binary()
    identity = {'python': str(binary), 'files': {}}
    candidates = [binary, binary.parent.parent / 'pyvenv.cfg']
    for pattern in ('lib/python*/site-packages/*.dist-info/METADATA',
                    'lib/python*/site-packages/*.dist-info/RECORD'):
        candidates.extend(binary.parent.parent.glob(pattern))
    for path in sorted(set(candidates)):
        if path.is_file():
            identity['files'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    model = layout.model_path()
    identity['model'] = {'path': str(model), 'files': {}}
    if model.is_dir():
        for path in sorted(model.rglob('*')):
            relative = path.relative_to(model)
            if any(part in {'__pycache__', '.cache', '.git', '.pytest_cache', '.mypy_cache', '.ruff_cache'} for part in relative.parts): continue
            if path.suffix in {'.pyc', '.pyo', '.lock', '.tmp', '.part', '.incomplete'} or not path.is_file(): continue
            stat = path.stat()
            row = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
            if path.suffix in ('.json', '.jinja', '.py') and stat.st_size < 32*1024*1024:
                row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            identity['model']['files'][str(path.relative_to(model))] = row
    return identity


def signature():
    h = hashlib.sha256()
    for p in sources():
        h.update(str(p.relative_to(PROJECT)).encode()); h.update(b'\0'); h.update(p.read_bytes())
    # Relocated runs may be analyzed, but may not be silently treated as newly executed evidence.
    h.update(str(PROJECT).encode()); h.update(json.dumps(runtime_fingerprint(), sort_keys=True).encode())
    return h.hexdigest()


def command(entry):
    binary = python_binary()
    def arg(value):
        if value.startswith('.venv-vllm/bin/'):
            return str(binary.parent / value.rsplit('/', 1)[-1])
        if value.startswith('research/'):
            return str(PROJECT / value)
        return value
    cmd = [str(binary), str(PROJECT / entry['runner']), '--name', entry['id'],
           '--timeout', str(entry['timeout']), '--gpus', entry['gpus'], '--reference', entry['reference']]
    if entry.get('hypothesis'):
        cmd += ['--hypothesis', entry['hypothesis']]
    return cmd + ['--'] + [arg(a) for a in entry['child_argv']]


def register_scheduler_pairs(cat):
    template = RESEARCH / 'scheduler/decisions/normal-pair-blocks-r1.json'
    decision = read(template)
    entries = {e['id']: e for e in cat['experiments']}
    hashes = {}
    for group in decision['groups']:
        for name in group['experiments']:
            for field in ('config', 'plan'):
                path = PROJECT / entries[name][field]
                hashes[str(path.relative_to(RESEARCH))] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(RESULTS / 'setup/scheduler-pair-blocks.json', {
        'schema_version': 1, 'generated': True, 'status': 'PREREGISTERED_NOT_EXECUTED',
        'groups': decision['groups'], 'input_sha256': hashes, 'rule': decision['rule'],
        'historical_reference': {'archive': 'archive/initial-project.tar.gz',
                                'member': 'research/dsv4-flash0731/phase-d-20261107/decisions/normal-pair-blocks-r1.json',
                                'migrated_template_sha256': hashlib.sha256(template.read_bytes()).hexdigest()},
        'purpose': 'Same registered restart blocks/order; bind current portable input bytes before launch, never fabricate prior execution.'})


def prepare(args):
    layout.ensure_layout()
    # Catalog is authoritative; process files are deterministic, disposable views.
    cat = catalog()
    for stage in cat['stages']:
        folder = RESEARCH / 'experiments' / stage['id']
        save(folder / 'stage.json', dict(stage, generated=True, source='experiments/catalog.json'))
        for entry in cat['experiments']:
            if entry['process_key'] == stage['id']:
                save(folder / (entry['id'] + '.json'), dict(entry, generated=True, source='experiments/catalog.json'))
    register_scheduler_pairs(cat)
    groups = {'core': [], 'cubin': [], 'jit': []}
    for line in (PROJECT / 'requirements-vllm.lock.txt').read_text().splitlines():
        if not line or line.startswith('#'): continue
        group = 'cubin' if line.startswith('flashinfer-cubin==') else 'jit' if line.startswith('flashinfer-jit-cache') else 'core'
        groups[group].append(line)
    for name, rows in groups.items():
        layout.write_text(RESULTS / 'setup' / (name + '.requirements.txt'), '\n'.join(rows) + '\n')
    versions = {}
    for name in ('vllm', 'torch', 'aiohttp', 'tokenizers'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    environment = {'generated': True, 'scope': 'CPU setup metadata; no GPU experiment has run',
                   'platform': platform.platform(), 'python': sys.version, 'observed_interpreter': sys.executable,
                   'observed_packages': versions, 'planned_runtime_python': str(python_binary()),
                   'project_root': str(PROJECT), 'runtime_environment_exists': python_binary().is_file()}
    save(RESULTS / 'setup/environment.json', environment)
    model = layout.model_path()
    inventory = {'generated': True, 'model_path': str(model), 'exists': model.is_dir(), 'files': {}}
    for p in sorted(model.glob('*')) if model.is_dir() else []:
        if not p.is_file(): continue
        row = {'bytes': p.stat().st_size}
        if p.suffix in ('.json', '.jinja', '.py') and p.stat().st_size < 32 * 1024 * 1024:
            row['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
        inventory['files'][p.name] = row
    save(RESULTS / 'setup/model-inventory.json', inventory)
    print(json.dumps({'status': 'PREPARED', 'requirements': {k: len(v) for k,v in groups.items()},
                      'model': str(model), 'model_exists': model.is_dir(), 'python': str(python_binary()),
                      'gpu_execution': 'NOT RUN'}, ensure_ascii=False, indent=2))


def validate(args):
    errors = []; cat = catalog(); ids = {e['id'] for e in cat['experiments']}
    if len(ids) != len(cat['experiments']): errors.append('Repeated experiment IDs')
    for e in cat['experiments']:
        for k in ('runner', 'config', 'plan'):
            if e.get(k) and not (PROJECT / e[k]).is_file(): errors.append(e['id'] + ': missing ' + e[k])
        for name in e.get('requires_success', []):
            if name not in ids: errors.append('Unknown dependency ' + name)
        for a in e.get('child_argv', []):
            if a.startswith('research/') and not (PROJECT / a).is_file(): errors.append('Missing command input ' + a)
    stages = {s['id']: s for s in cat['stages']}
    def visit(name, visiting, done):
        if name in visiting: raise ValueError('Cyclic stage dependency ' + name)
        if name in done: return
        for dep in stages[name]['depends_on']: visit(dep, visiting | {name}, done)
        done.add(name)
    try:
        done = set()
        for name in stages: visit(name, set(), done)
    except (KeyError, ValueError) as e: errors.append(str(e))
    for p in sources():
        try:
            if p.suffix == '.py': ast.parse(p.read_text(), filename=str(p))
            if p.suffix == '.json': read(p)
        except (SyntaxError, json.JSONDecodeError) as e: errors.append(str(e))
        if p.suffix in ('.py', '.json') and re.search(r'/(?:home|Users)/[A-Za-z0-9_.-]+/', p.read_text()):
            errors.append('Machine-specific native input: ' + str(p.relative_to(PROJECT)))
    report = {'generated': True, 'status': 'PASS' if not errors else 'FAILED', 'experiments': len(ids),
              'stages': len(stages), 'errors': errors, 'scope': 'Static source/input/CLI references; not GPU execution'}
    save(RESULTS / 'validation/layout.json', report); print(json.dumps(report, indent=2))
    return bool(errors)


def listing(args):
    cat = catalog()
    if args.json: print(json.dumps(cat, indent=2)); return
    for stage in cat['stages']:
        print(stage['id'] + ' <- ' + (', '.join(stage['depends_on']) or 'prepare'))
        for e in cat['experiments']:
            if e['process_key'] == stage['id']:
                print('  ' + e['id'] + (' [historical input only]' if not e['executable_definition'] else ' [optional]' if e['optional'] else ''))


def preserve(path, category):
    if not path.exists(): return
    target = RESULTS / 'preserved' / category / path.name
    target.mkdir(parents=True, exist_ok=True)
    n = 1
    while (target / ('attempt-' + str(n))).exists(): n += 1
    shutil.move(str(path), str(target / ('attempt-' + str(n))))


def evidence_manifest(folder):
    manifest = {}
    for p in sorted(folder.rglob('*')):
        if not p.is_file(): continue
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
        manifest[str(p.relative_to(folder))] = h.hexdigest()
    return manifest


def terminal(entry):
    path = RESULTS / 'runs' / entry['id'] / 'status.json'
    if not path.is_file(): return None
    obj = read(path)
    if obj.get('experiment_id') != entry['id']: return None
    if not obj.get('end_utc'): return None
    code = obj.get('supervisor_exit_code', obj.get('exit_code'))
    if type(code) is not int: return None
    return {'exit_code': code, 'successful': code == 0 and not obj.get('timed_out') and not obj.get('supervisor_error'),
            'status_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'evidence': evidence_manifest(path.parent)}


def check_stage(name, fingerprint):
    receipt = RESULTS / 'stage-state' / (name + '.json')
    if not receipt.is_file(): raise ValueError('Run prerequisite stage first: ' + name)
    obj = read(receipt)
    if obj.get('signature') != fingerprint or obj.get('status') != 'COMPLETED':
        raise ValueError('Prerequisite is incomplete or its code/inputs changed: ' + name)
    entries = {e['id']: e for e in catalog()['experiments']}
    for run in obj['runs']:
        if run.get('status') == 'EXCLUDED_SCREEN_FAILED': continue
        current = terminal(entries[run['id']])
        if not current or current != run.get('terminal'):
            raise ValueError('Prerequisite evidence changed/missing: ' + run['id'])


@contextlib.contextmanager
def gpu_reservation(gpus):
    """Reserve the registered four-GPU workflow across copies; never stop other jobs.

    This outer mutex has separate names from the supervisors' internal locks.
    Call only immediately before a real launch, never from prepare/preflight/dry-run.
    """
    devices = ('0', '1', '2', '3')
    selected = gpus.split(',')
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(devices):
        raise ValueError('Registered experiments may select GPU indices 0,1,2,3 only')
    lock_paths = []
    with contextlib.ExitStack() as locks:
        for device in devices:
            path = Path(tempfile.gettempdir()) / ('portable-inference-gpu-' + device + '.lock')
            handle = locks.enter_context(path.open('a+'))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError('GPU ' + device + ' is reserved by another reproduction process; no experiment started') from error
            lock_paths.append(str(path))
        output = subprocess.check_output(
            ['nvidia-smi', '--id=0,1,2,3', '--query-gpu=index,uuid,memory.used,utilization.gpu',
             '--format=csv,noheader,nounits'], text=True, timeout=15)
        inventory = {}
        for row in csv.reader(output.splitlines()):
            if not row: continue
            if len(row) != 4: raise RuntimeError('Incomplete GPU inventory; no experiment started')
            index, uuid, memory, utilization = [value.strip() for value in row]
            if index in inventory or not uuid.startswith('GPU-'):
                raise RuntimeError('Invalid GPU inventory; no experiment started')
            try:
                used, busy = int(memory), int(utilization)
            except ValueError as error:
                raise RuntimeError('GPU idle metrics unavailable; no experiment started') from error
            if used < 0 or not 0 <= busy <= 100:
                raise RuntimeError('Invalid GPU idle metrics; no experiment started')
            inventory[index] = {'uuid': uuid, 'memory_used_mib': used, 'utilization_pct': busy}
        if set(inventory) != set(devices) or len({row['uuid'] for row in inventory.values()}) != len(devices):
            raise RuntimeError('GPU 0,1,2,3 inventory incomplete; no experiment started')
        busy_devices = [index for index, row in inventory.items() if row['memory_used_mib'] > 512 or row['utilization_pct'] != 0]
        if busy_devices:
            raise RuntimeError('GPUs are busy: ' + ','.join(busy_devices) + '; no experiment started')
        processes = subprocess.check_output(
            ['nvidia-smi', '--id=0,1,2,3', '--query-compute-apps=gpu_uuid,pid',
             '--format=csv,noheader,nounits'], text=True, timeout=15)
        if processes.strip():
            raise RuntimeError('GPU compute processes are present; no experiment started')
        yield {'selected': selected, 'inventory': inventory, 'locks': lock_paths,
               'scope': 'GPU 0-3 reserved; <=512 MiB each, zero sampled utilization, no compute process'}


def execute_one(e, args, fingerprint):
    if not e['executable_definition']: raise ValueError('Historical input has no complete launch definition: ' + e['id'])
    cmd = command(e)
    if args.dry_run:
        print(shlex.join(cmd)); return {'id': e['id'], 'status': 'DRY RUN'}
    for stage in e['depends_on']: check_stage(stage, fingerprint)
    byid = {x['id']: x for x in catalog()['experiments']}
    for dep in e['requires_success']:
        prior = terminal(byid[dep])
        receipt = RESULTS / 'stage-state/runs' / (dep + '.json')
        if not prior or not prior['successful'] or not receipt.is_file() or read(receipt).get('signature') != fingerprint or read(receipt).get('terminal') != prior:
            return {'id': e['id'], 'status': 'EXCLUDED_SCREEN_FAILED', 'reason': 'Required successful run: ' + dep}
    folder = RESULTS / 'runs' / e['id']; state = RESULTS / 'stage-state/runs' / (e['id'] + '.json')
    log = RESULTS / 'stage-state/logs' / (e['id'] + '.log')
    previous = folder.exists() or state.exists() or log.exists()
    if previous:
        old = terminal(e)
        if not args.overwrite:
            if old and state.is_file() and read(state).get('signature') == fingerprint and read(state).get('terminal') == old:
                return {'id': e['id'], 'status': 'REUSED_TERMINAL', 'terminal': old}
            raise ValueError('An unverified/stale attempt exists; use --overwrite to preserve it and rerun: ' + e['id'])
        preserve(folder, 'runs')
        preserve(state, 'run-state')
        preserve(log, 'launcher-logs')
    env = os.environ.copy(); env.update(e.get('environment', {})); env['PYTHONDONTWRITEBYTECODE'] = '1'
    log.parent.mkdir(parents=True, exist_ok=True)
    print('RUN ' + e['id'] + '\n' + shlex.join(cmd), flush=True)
    save(state, {'id': e['id'], 'signature': fingerprint, 'command': cmd, 'status': 'RUNNING'})
    guard = None; launched = False; failure = None; returncode = None
    with log.open('w') as f:
        try:
            with gpu_reservation(e['gpus']) as guard:
                f.write(json.dumps({'gpu_guard': guard}) + '\n'); f.flush()
                launched = True
                completed = subprocess.run(cmd, cwd=PROJECT, env=env, stdout=f, stderr=subprocess.STDOUT)
                returncode = completed.returncode
        except Exception as error:
            failure = type(error).__name__ + ': ' + str(error)
            f.write(failure + '\n'); f.flush()
    end = terminal(e)
    row = {'id': e['id'], 'signature': fingerprint, 'command': cmd, 'returncode': returncode,
           'status': 'FAILED_BEFORE_LAUNCH' if not launched else 'TERMINAL' if end else 'FAILED_WITHOUT_TERMINAL',
           'terminal': end, 'log': str(log.relative_to(PROJECT)), 'gpu_guard': guard, 'launcher_error': failure}
    save(state, row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def run(args):
    cat = catalog(); entries = {e['id']: e for e in cat['experiments']}
    if args.stage:
        stage = next((s for s in cat['stages'] if s['id'] == args.stage), None)
        if stage is None: raise ValueError('Unknown stage: ' + args.stage)
        names = stage['experiments'] + (stage['optional_experiments'] if args.include_optional else [])
        names = [n for n in names if entries[n]['executable_definition']]
    else:
        if args.experiment not in entries: raise ValueError('Choose an experiment ID from list or use --stage')
        stage = None; names = [args.experiment]
    if not names: raise ValueError('This stage only contains optional/history inputs; choose --include-optional explicitly')
    if args.dry_run:
        for name in names: execute_one(entries[name], args, '')
        return 0
    if not python_binary().is_file(): raise ValueError('Runtime environment missing; install it using doc/reproduction.md')
    if not (layout.model_path() / 'config.json').is_file(): raise ValueError('Model config missing: ' + str(layout.model_path()))
    # Preparation precedes use of lock files and output metadata. Never creates fake model weights.
    prepare(args)
    with (RESULTS / '.reproduction.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fingerprint = signature()
        if stage:
            for dep in stage['depends_on']: check_stage(dep, fingerprint)
        rows = []
        for name in names:
            if signature() != fingerprint: raise ValueError('Code or workload changed during stage execution')
            rows.append(execute_one(entries[name], args, fingerprint))
        complete = all(r['status'] in ('TERMINAL', 'REUSED_TERMINAL', 'EXCLUDED_SCREEN_FAILED') for r in rows)
        if stage:
            save(RESULTS / 'stage-state' / (stage['id'] + '.json'),
                 {'generated': True, 'stage': stage['id'], 'status': 'COMPLETED' if complete else 'FAILED',
                  'signature': fingerprint, 'runs': rows,
                  'meaning': 'All scheduled attempts accounted for; this does not mean every experiment passed or met SLO.'})
        return 0 if complete else 1


def preflight(args):
    prepare(args); errors = []; selected = [e for e in catalog()['experiments'] if e['executable_definition']]
    if args.experiment: selected = [e for e in selected if e['id'] == args.experiment]
    if not selected: raise ValueError('Unknown experiment')
    for e in selected:
        argv = command(e); child = argv[argv.index('--')+1:]
        if child[0].endswith('/torchrun'): continue
        cmd = child + ['--validate-only']
        env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', RESEARCH_RUN_DIR=str(RESULTS/'runs'/e['id']))
        output = subprocess.run(cmd, cwd=PROJECT, env=env, capture_output=True, text=True)
        save(RESULTS / 'validation/preflight' / (e['id']+'.json'),
             {'generated': True, 'command': cmd, 'exit_code': output.returncode, 'stdout': output.stdout, 'stderr': output.stderr})
        if output.returncode: errors.append(e['id'])
    print(json.dumps({'status': 'FAILED' if errors else 'PASS', 'errors': errors, 'scope': 'CPU launch validation; no model inference'}))
    return bool(errors)


def analyze(args):
    topic = args.topic; cat = catalog()
    entries = [e for e in cat['experiments'] if e['topic'] == topic and (RESULTS/'runs'/e['id']).is_dir()]
    if not entries: raise ValueError('No run results for ' + topic)
    out = RESULTS / 'analysis' / topic
    if out.exists():
        if not args.overwrite: raise ValueError('Analysis exists; use --overwrite')
        preserve(out, 'analysis')
    out.mkdir(parents=True, exist_ok=True); prefix = out/'doc/summary'
    prefix.parent.mkdir(parents=True, exist_ok=True)
    dirs = [str(RESULTS/'runs'/e['id']) for e in entries]
    py = str(python_binary()); cmds = []
    if topic == 'architecture':
        cmds = [[py, str(RESEARCH/'scripts/summarize_results.py')]]
    elif topic == 'scenarios':
        cmds = [[py,str(RESEARCH/'scenarios/scripts/summarize_scenarios.py'),'--results',str(RESULTS/'runs'),'--output',str(out/'doc')]
                + [v for e in entries for v in ('--experiment',e['id'])]]
    elif topic == 'capacity':
        paths = [str(Path(d)/'capacity.json') for d in dirs if (Path(d)/'capacity.json').is_file()]
        if not paths: raise ValueError('No capacity.json outputs; run failures are preserved')
        cmds = [[py,str(RESEARCH/'scenarios/scripts/summarize_capacity.py'),'--files',*paths,'--output-prefix',str(prefix)]]
    elif topic == 'long-context':
        for profile in (False, True):
            group = [str(RESULTS/'runs'/e['id']) for e in entries if ('profile' in e['id']) == profile]
            if group:
                script = 'analyze_long_context_profiles.py' if profile else 'analyze_long_context.py'
                cmds.append([py,str(RESEARCH/'long-context/scripts'/script),'--result-dirs',*group,'--output-prefix',str(prefix.parent/('profiles' if profile else 'summary'))])
    elif topic == 'sessions':
        cmds = [[py,str(RESEARCH/'sessions/scripts/analyze_sessions.py'),'--result-dirs',*dirs,'--output-prefix',str(prefix)]]
    elif topic == 'scheduler':
        for profile in (False,True):
            group = [str(RESULTS/'runs'/e['id']) for e in entries if ('profile' in e['id']) == profile]
            if group:
                script='analyze_profiles.py' if profile else 'analyze_performance.py'
                cmds.append([py,str(RESEARCH/'scheduler/scripts'/script),'--result-dirs',*group,'--output-prefix',str(prefix.parent/('profiles' if profile else 'summary'))])
    codes=[]
    for i,cmd in enumerate(cmds):
        print(shlex.join(cmd),flush=True)
        with (out/f'analysis-{i+1}.log').open('w') as f:
            code=subprocess.run(cmd,cwd=PROJECT,stdout=f,stderr=subprocess.STDOUT).returncode
        codes.append(code)
    save(out/'analysis.json',{'generated':True,'topic':topic,'commands':cmds,'exit_codes':codes,'experiments':[e['id'] for e in entries]})
    for p in out.rglob('*.md'):
        body=p.read_text(); p.write_text('> 自动生成结果：可删除，使用 tools/reproduce.py analyze 重建。\n\n'+body)
    print(json.dumps({'status': 'FAILED' if any(codes) else 'PASS', 'topic': topic, 'outputs': str(out), 'exit_codes': codes}))
    return int(any(codes))


SOURCE_DIRECTORIES = {'research', 'tools', 'doc'}
SOURCE_ROOT_FILES = {'README.md', 'AGENTS.md', 'progress.md', 'activate-vllm.sh',
                     'requirements-vllm.lock.txt', '.gitignore'}
LOCAL_DIRECTORIES = {'.git', '.cache', '__pycache__', '.pytest_cache', '.mypy_cache',
                     '.ruff_cache', '.ipynb_checkpoints'}
HISTORICAL_DOCUMENTS = {'doc/validation.json', 'doc/migration-map.json', 'doc/review.md',
    'research/dsv4-flash0731/doc/framework-capabilities.md',
    'research/dsv4-flash0731/doc/model-analysis.md',
    'research/dsv4-flash0731/doc/roadmap.md'}


def excluded_from_source(relative):
    """Recognize local outputs without confusing frozen workload inputs with them."""
    parts = relative.parts
    if not parts:
        return False
    if any(part in LOCAL_DIRECTORIES or part.startswith('.venv') for part in parts):
        return True
    if parts[0] in {'results', 'archive', 'logs', 'reports', 'build', 'dist', 'orders_history'}:
        return True
    if relative.as_posix() in HISTORICAL_DOCUMENTS:
        return True
    if parts[:2] in (('doc', 'validation'), ('doc', 'history')):
        return True
    if any(parts[i:i+2] == ('doc', 'history') for i in range(len(parts)-1)):
        return True
    if len(parts) >= 5 and parts[:3] == ('research', 'dsv4-flash0731', 'experiments') and parts[4].startswith('process'):
        return True
    if relative.name in {'.research-gpu.lock', '.reproduction.lock', '.DS_Store'}:
        return True
    return relative.suffix in {'.pyc', '.pyo', '.log', '.tmp'}


def native_files(root):
    """Return publishable files, pruning environments before directory traversal.

    A generated document placed among native inputs is an error, not permission
    to silently drop an unknown dependency. Standard output locations are ignored.
    """
    root = Path(root)
    selected = []
    for current, directories, names in os.walk(root, followlinks=False):
        parent = Path(current)
        kept = []
        for name in sorted(directories):
            path = parent / name; relative = path.relative_to(root)
            if parent == root and name not in SOURCE_DIRECTORIES:
                continue
            if excluded_from_source(relative):
                continue
            if path.is_symlink():
                raise ValueError('Source package requires ordinary directories: ' + str(relative))
            kept.append(name)
        directories[:] = kept
        for name in sorted(names):
            path = parent / name; relative = path.relative_to(root)
            if parent == root and name not in SOURCE_ROOT_FILES:
                continue
            if excluded_from_source(relative):
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError('Source package requires ordinary files: ' + str(relative))
            if path.suffix == '.json':
                value = read(path)
                if isinstance(value, dict) and value.get('generated') is True:
                    raise ValueError('Generated JSON found among native inputs: ' + str(relative))
            if path.suffix == '.ipynb':
                notebook = read(path)
                if any(c.get('outputs') or c.get('execution_count') is not None for c in notebook.get('cells', [])):
                    raise ValueError('Notebook contains execution output: ' + str(relative))
            selected.append(path)
    return sorted(selected)


def package(args):
    target = Path(args.output).expanduser().resolve(strict=False)
    manifest_path = target.with_suffix(target.suffix + '.manifest.json')
    if target.exists() or manifest_path.exists():
        raise ValueError('Package or manifest already exists; select another output name')
    # Publish outside native directories so subsequent packages cannot capture a
    # previous package/manifest as an accidental input.
    try:
        relative = target.relative_to(PROJECT.resolve())
    except ValueError:
        relative = None
    if relative is not None and relative.parts[0] != 'results':
        raise ValueError('Write packages under results/ or outside the project')
    files = native_files(PROJECT)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.' + target.name, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with tarfile.open(temporary_path, 'w:gz') as tar:
            for path in files:
                tar.add(path, arcname=str(Path('portable-inference') / path.relative_to(PROJECT)), recursive=False)
        temporary_path.replace(target)
    finally:
        temporary_path.unlink(missing_ok=True)
    save(manifest_path, {'generated': True, 'members': manifest,
                        'archive_sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    print(json.dumps({'package': str(target), 'files': len(files),
                      'model_and_virtualenv_included': False, 'generated_outputs_included': False}))


def main():
    parser=argparse.ArgumentParser(description=__doc__); subs=parser.add_subparsers(dest='action',required=True)
    p=subs.add_parser('prepare');p.set_defaults(function=prepare)
    p=subs.add_parser('validate');p.set_defaults(function=validate)
    p=subs.add_parser('list');p.add_argument('--json',action='store_true');p.set_defaults(function=listing)
    p=subs.add_parser('preflight');p.add_argument('experiment',nargs='?');p.set_defaults(function=preflight)
    p=subs.add_parser('run');p.add_argument('experiment',nargs='?');p.add_argument('--stage');p.add_argument('--dry-run',action='store_true');p.add_argument('--overwrite',action='store_true');p.add_argument('--include-optional',action='store_true');p.set_defaults(function=run)
    p=subs.add_parser('analyze');p.add_argument('topic',choices=['architecture','scenarios','capacity','long-context','sessions','scheduler']);p.add_argument('--overwrite',action='store_true');p.set_defaults(function=analyze)
    p=subs.add_parser('package');p.add_argument('--output',required=True);p.set_defaults(function=package)
    args=parser.parse_args()
    try:return args.function(args) or 0
    except (ValueError,OSError,KeyError) as e:
        print('ERROR: '+str(e),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
