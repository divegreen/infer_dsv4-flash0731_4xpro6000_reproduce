#!/usr/bin/env python3
"""Independent read-only CPU audit of phase-A capacity evidence.

Does not import clients/snapshots, query GPUs, connect HTTP, or rewrite raw data.
PASS means saved evidence is internally consistent, not stable/SLO/model PASS.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('ttft_s','scheduled_ttft_s','tpot_s','e2e_s','scheduled_e2e_s',
           'dispatch_lag_s','ttft_ratio','e2e_ratio')
MIXES = {
 'short_mixed':[dict(name='p512_o128',input_tokens=512,output_tokens=128,weight=70),
                dict(name='p4096_o256',input_tokens=4096,output_tokens=256,weight=20),
                dict(name='p16384_o128',input_tokens=16384,output_tokens=128,weight=10)],
 'long_generation':[dict(name='p512_o128',input_tokens=512,output_tokens=128,weight=70),
                    dict(name='p1024_o1024',input_tokens=1024,output_tokens=1024,weight=30)]}


def digest(data):return hashlib.sha256(data).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def finite(x):return type(x) in (int,float) and math.isfinite(x)


def percentile(values,p):
    if not values:return None
    v=sorted(values);at=(len(v)-1)*p;a,b=math.floor(at),math.ceil(at)
    return v[a]+(v[b]-v[a])*(at-a)


def stats(values):
    values=[float(v) for v in values if v is not None and finite(v)]
    return dict(n=len(values),p50=percentile(values,.5),p95=percentile(values,.95),
                p99=percentile(values,.99),max=max(values) if values else None)


def tier(n):
    for bound,name,limit in [(1024,'le1k',1.),(4096,'le4k',3.),(16384,'le16k',8.)]:
        if n<=bound:return name,limit
    return 'gt16k',None


def row_slo(row,mode):
    name,limit=tier(row['input_tokens']);o=(row.get('usage') or {}).get('completion_tokens')
    o=o if type(o) is int and o>=1 else None
    deadline=limit+.05*(o-1) if limit is not None and o else None
    prefix='scheduled_' if mode in ('poisson','burst','trace') else ''
    ttft,e2e=row.get(prefix+'ttft_s'),row.get(prefix+'e2e_s')
    tr=ttft/limit if ttft is not None and limit else None
    er=e2e/deadline if e2e is not None and deadline else None
    tp=o==1 or row.get('tpot_s') is not None and row['tpot_s']<=.05
    return dict(input_tier=name,ttft_limit_s=limit,e2e_limit_s=deadline,slo_applicable=limit is not None,
                slo_latency_basis='scheduled' if prefix else 'request',ttft_ratio=tr,e2e_ratio=er,
                slo_pass=bool(row.get('ok') and tr is not None and tr<=1 and er is not None and er<=1 and tp) if limit is not None else None,
                legacy_slo_pass=bool(row.get('ok') and ttft is not None and ttft<=1 and e2e is not None and e2e<=10 and tp))


def group(rows):
    ok=[r for r in rows if r['ok']];eligible=[r for r in rows if r['slo_applicable']]
    es=[r for r in ok if r['slo_applicable']];passes=sum(r['slo_pass'] is True for r in rows)
    legacy=sum(r['legacy_slo_pass'] for r in rows);p95=None
    if es:
        tr=stats([r['ttft_ratio'] for r in es])['p95'];er=stats([r['e2e_ratio'] for r in es])['p95']
        tp=stats([r['tpot_s'] for r in es])['p95']
        p95=tr is not None and tr<=1 and er is not None and er<=1 and (tp is None or tp<=.05)
    return dict(requests=len(rows),admitted=sum(r['admitted'] for r in rows),successful=len(ok),
        failed=sum(r['admitted'] and not r['ok'] for r in rows),censored=sum(r['censored'] for r in rows),
        http_accepted=sum(r['http_status']==200 for r in rows),timeouts=sum(r['timed_out'] for r in rows),
        http_status_counts=dict(Counter(str(r['http_status']) for r in rows if r['http_status'])),
        metrics={k:stats([r.get(k) for r in ok]) for k in METRICS},slo_eligible_requests=len(eligible),
        slo_passes=passes,slo_joint_pass_rate=passes/len(eligible) if eligible else None,
        slo_joint_pass_rate_success_only=passes/len(es) if es else None,p95_slo_pass=p95,
        legacy_slo_passes=legacy,legacy_slo_joint_pass_rate=legacy/len(rows) if rows else None)


def summarize(rows,phase,wall,samples,peak):
    d=group(rows);window=phase['duration_s'] if phase['mode'] in ('poisson','trace') else wall
    ok=[r for r in rows if r['ok']];within=[r for r in ok if r['end_offset_s']<=window]
    good=[r for r in ok if r['slo_pass'] is True];gw=[r for r in within if r['slo_pass'] is True]
    output=lambda seq:sum(r['usage']['completion_tokens'] for r in seq)
    d.update(measurement_window_s=window,total_wall_s=wall,drain_s=max(0.,wall-window),
        planned_arrivals=len(rows),planned_rate_s=phase.get('rate'),realized_offered_rate_s=len(rows)/window if window else None,
        successful_output_tokens=output(ok),peak_inflight=peak,
        in_flight_at_window_end=sum(r['admitted'] and r['dispatch_offset_s']<=window<r['end_offset_s'] for r in rows),
        throughput=dict(cohort_requests_s=len(ok)/wall,cohort_output_tokens_s=output(ok)/wall,
            completed_in_window_requests_s=len(within)/window,completed_in_window_output_tokens_s=output(within)/window,
            cohort_slo_goodput_requests_s=len(good)/wall,cohort_slo_goodput_output_tokens_s=output(good)/wall,
            completed_in_window_slo_goodput_requests_s=len(gw)/window,offered_window_slo_goodput_requests_s=len(good)/window),
        metrics_poll_errors=sum('error' in s for s in samples),
        by_shape={s:group([r for r in rows if r['shape_name']==s]) for s in sorted({r['shape_name'] for r in rows})},
        by_input_tier={s:group([r for r in rows if r['input_tier']==s]) for s in sorted({r['input_tier'] for r in rows})})
    bins=[];width=max(1.,window/4)
    for i in range(4):
        left,right=i*width,min(window,(i+1)*width)
        if left>=window:break
        cohort=[r for r in rows if left<=r['planned_offset_s']<right]
        bins.append(dict(start_s=left,end_s=right,arrivals=len(cohort),
            completed=sum(r['ok'] and left<=r['end_offset_s']<right for r in rows),
            outstanding_at_end=sum(r['admitted'] and r['dispatch_offset_s']<right<r['end_offset_s'] for r in rows),
            cohort_scheduled_ttft_s=stats([r['scheduled_ttft_s'] for r in cohort if r['ok']]),
            cohort_dispatch_lag_s=stats([r['dispatch_lag_s'] for r in cohort if r['admitted']])))
    d['time_bins']=bins;tiers=[x for x in d['by_input_tier'].values() if x['slo_eligible_requests']]
    d['overall_ratio_p95_slo_pass']=d['p95_slo_pass']
    d['p95_slo_pass']=all(x['p95_slo_pass'] is True for x in tiers) if tiers else None
    d.update(p95_slo_scope='each eligible input-length tier; aggregate normalized ratios are auxiliary',
             client_uncensored=d['censored']==0,transport_success=d['failed']==0,stability_classification='NOT_ASSESSED_BY_CLIENT')
    return d


def snapshot_source_key(saved, root, category):
    """Resolve a recorded launch path to its captured physical research key.

    Supports portable centralized inputs and original self-contained fixtures;
    never reads or executes source bytes from the archive.
    """
    parts = PurePosixPath(str(saved)).parts
    prefixes = [('experiments', root.name, category), (root.name, category)]
    for prefix in prefixes:
        for index in range(len(parts) - len(prefix) + 1):
            if parts[index:index + len(prefix)] == prefix:
                return str(PurePosixPath(*parts[index:]))
    return None


def expand_captured_document(value, context):
    """Expand only recorded path placeholders, leaving numerical inputs intact."""
    if isinstance(value, dict):
        return {key: expand_captured_document(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_captured_document(item, context) for item in value]
    if not isinstance(value, str):
        return value
    for token, field in (('${PROJECT_ROOT}', 'project_root'), ('${MODEL_PATH}', 'model_path'),
                         ('${RUN_DIR}', 'run_dir')):
        if token in value:
            if field not in context:
                raise ValueError('Missing recorded path context: ' + field)
            value = value.replace(token, context[field])
    prefix = '~/models/dsv4-flash0731'
    if value.startswith(prefix):
        if 'model_path' not in context:
            raise ValueError('Missing recorded model path context')
        value = context['model_path'] + value[len(prefix):]
    return value


def normalize_plan(raw):
    plan = deepcopy(raw)
    url = plan.get('base_url', 'http://127.0.0.1:8260').rstrip('/')
    plan['base_url'] = url[:-3] if url.endswith('/v1') else url
    for key, val in dict(model='dsv4', tokenizer_path=str(model_path()), seed=20261106, timeout_s=300.0, max_inflight=256, metrics_interval_s=1.0).items():
        plan.setdefault(key, val)
    for i, phase in enumerate(plan['phases']):
        for key, val in dict(seed=plan['seed'] + i, ignore_eos=True, role='measure', max_inflight=plan['max_inflight']).items():
            phase.setdefault(key, val)
        if phase['mode'] in ('burst', 'closed-loop'):
            phase.setdefault('concurrency', phase['requests'] if phase['mode'] == 'burst' else 1)
        mix = phase.get('shape_mix', 'short_mixed')
        phase['shape_mix'] = deepcopy(MIXES[mix] if isinstance(mix, str) else mix)
        if phase['mode'] == 'trace':
            duration = phase.get('duration_s')
            arrivals = phase.get('arrivals')
            if not finite(duration) or not 0 < duration <= 7200:
                raise ValueError('Invalid trace duration')
            if any((k in phase for k in ('requests', 'rate', 'concurrency'))):
                raise ValueError('Conflicting trace arrival controls')
            if not isinstance(arrivals, list) or not 1 <= len(arrivals) <= 100000:
                raise ValueError('Invalid trace arrivals')
            if not isinstance(mix, list) or not mix:
                raise ValueError('Trace shape_mix must be an explicit list')
            names = [s.get('name') for s in mix]
            if any((not isinstance(n, str) or not n for n in names)) or len(set(names)) != len(names):
                raise ValueError('Invalid trace shape names')
            previous = -1.0
            for arrival in arrivals:
                if not isinstance(arrival, dict):
                    raise ValueError('Invalid trace arrival object')
                offset = arrival.get('offset_s')
                if not finite(offset) or not 0 <= offset < duration or offset < previous:
                    raise ValueError('Invalid trace offset')
                if arrival.get('shape_name') not in names:
                    raise ValueError('Unknown trace shape')
                previous = offset
    return plan


def regenerate(phase):
    offsets=[]
    if phase['mode']=='trace':
        offsets=[arrival['offset_s'] for arrival in phase['arrivals']]
    elif phase['mode']=='poisson':
        rng=random.Random(phase['seed']);time=0.
        while True:
            time+=rng.expovariate(phase['rate'])
            if time>=phase['duration_s']:break
            offsets.append(time)
    else:offsets=[0.]*phase['requests']
    if phase['mode']=='trace':
        shapes={s['name']:s for s in phase['shape_mix']}
        selected=[shapes[arrival['shape_name']] for arrival in phase['arrivals']]
    else:
        rng=random.Random(phase['seed'] ^ 0x51A9)
        selected=rng.choices(phase['shape_mix'],weights=[s.get('weight',1) for s in phase['shape_mix']],k=len(offsets))
    items=[dict(index=i,shape_name=s['name'],input_tokens=s['input_tokens'],requested_output_tokens=s['output_tokens'],
                seed=phase['seed']+i,ignore_eos=phase['ignore_eos']) for i,s in enumerate(selected)]
    return offsets,items


def parse_metrics(text):
    result={}
    for line in text.splitlines():
        if line.startswith('#') or not line.strip():continue
        try:
            key,raw=line.rsplit(None,1);name=key.split('{')[0];value=float(raw)
            if math.isfinite(value) and any(x in name for x in ('num_requests_running','num_requests_waiting',
                    'kv_cache_usage_perc','prefix_cache_hits_total','prefix_cache_queries_total')):result[key]=value
        except (ValueError,IndexError):pass
    return result


class Auditor:
    def __init__(self,root):
        self.root=root.resolve();self.errors=[];self.warnings=[];self.incomplete=[]
        self.files={};self.states={};self.reports=[];self.pairs={};self.totals=Counter()
    def check(self,test,where,detail):
        if not test:self.errors.append(dict(location=where,detail=detail))
    def equal(self,actual,expected,where):
        if isinstance(expected,dict):
            if not isinstance(actual,dict):self.check(False,where,'Expected object');return
            self.check(set(actual)==set(expected),where,'Object key set differs')
            for k,v in expected.items():self.equal(actual.get(k),v,where+'.'+k)
        elif isinstance(expected,list):
            if not isinstance(actual,list):self.check(False,where,'Expected list');return
            self.check(len(actual)==len(expected),where,'List length differs')
            for i,(a,e) in enumerate(zip(actual,expected)):self.equal(a,e,f'{where}[{i}]')
        elif type(expected) is float:
            self.check(finite(actual) and math.isclose(actual,expected,rel_tol=1e-8,abs_tol=1e-8),where,f'Expected {expected!r}, got {actual!r}')
        else:self.check(type(actual) is type(expected) and actual==expected,where,f'Expected {expected!r}, got {actual!r}')
    def subset(self,actual,expected,where):
        for key,val in expected.items():self.equal(actual.get(key),val,where+'.'+key)
    def read(self,path):
        path=path.resolve();before=path.stat();raw=path.read_bytes();after=path.stat()
        self.check((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),str(path),'File changed while reading')
        self.states[path]=(after.st_size,after.st_mtime_ns);self.files[str(path)]=digest(raw)
        return raw
    def json(self,path):return json.loads(self.read(path))
    def pair(self,key,value,where):
        h=digest(canonical(value).encode())
        if key in self.pairs:self.check(self.pairs[key]['sha256']==h,where,'Matched workload differs from '+self.pairs[key]['first'])
        else:self.pairs[key]=dict(sha256=h,first=where)
    def local(self,parent,saved,expected):
        self.check(Path(saved).name==Path(expected).name,str(parent),'Saved artifact name differs: '+str(saved))
        return parent/expected
    def snapshot_document(self, raw, key, command):
        value = json.loads(raw)
        if key and key.startswith('experiments/'):
            context = command.get('path_context', {})
            self.check(all(isinstance(context.get(field), str) and Path(context[field]).is_absolute()
                           for field in ('project_root', 'model_path', 'run_dir')),
                       key, 'Centralized input lacks absolute recorded path context')
            value = expand_captured_document(value, context)
            if '/workloads/' in key:
                value.setdefault('tokenizer_path', context.get('model_path'))
        return value

    def path_sources(self, snapshot, command, status, where):
        if 'path_context' not in command:
            return  # Original archive/explicit legacy fixtures have no resolver.
        required = ('project_layout.py', 'experiments/catalog.json', 'project/tools/reproduce.py',
                    'metadata/environment.json', 'metadata/model-inventory.json')
        for key in required:
            self.check(key in snapshot, where, 'Runtime dependency missing from snapshot: ' + key)
        if status is not None:
            self.equal(status.get('path_context'), command['path_context'], where + '.terminal_path_context')
    def metadata(self,path,report):
        directory=path.parent;snapshot={}
        if not (directory/'command.json').exists():
            self.incomplete.append(str(path)+': no supervisor/source snapshot; runtime provenance not verified')
            for field,key in [('client_sha256',self.root/'scripts/capacity_client.py'),
                              ('base_helper_sha256',self.root.parent/'scripts/benchmark_client.py')]:
                current=digest(self.read(key))
                self.warnings.append(str(path)+': '+field+(' matches current source, but historical runtime is not independently proven'
                    if report[field]==current else ' differs from current source; no historical snapshot is available'))
            return snapshot,None
        command=self.json(directory/'command.json');zpath=directory/'source-snapshot.zip'
        self.read(zpath)
        with zipfile.ZipFile(zpath) as z:
            names=z.namelist();self.check(len(names)==len(set(names)),str(zpath),'Duplicate ZIP entries')
            self.equal(sorted(names),sorted(command['source_sha256']),str(zpath)+'.members')
            for name in names:
                p=PurePosixPath(name);self.check(not p.is_absolute() and '..' not in p.parts,str(zpath),'Unsafe member path')
                snapshot[name]=z.read(name);self.equal(digest(snapshot[name]),command['source_sha256'][name],name+'.hash')
        self.equal(report['client_sha256'],digest(snapshot[self.root.name+'/scripts/capacity_client.py']),str(path)+'.client_hash')
        self.equal(report['base_helper_sha256'],digest(snapshot['scripts/benchmark_client.py']),str(path)+'.helper_hash')
        self.equal(command['experiment_id'],directory.name,str(path)+'.experiment_id')
        self.equal(report['plan'].get('experiment_id'),directory.name,str(path)+'.plan_id')
        env=command['environment'];self.equal(env.get('CUDA_DEVICE_ORDER'),'PCI_BUS_ID',str(path)+'.device_order')
        self.equal(env.get('CUDA_VISIBLE_DEVICES'),'0,1,2,3',str(path)+'.visible_devices')
        self.equal(self.json(directory/'capacity-plan.json'),report['plan'],str(path)+'.saved_plan')
        config=self.json(directory/'config.json');server=self.json(directory/'server-command.json')
        cmd=server['command'];args=config['args'];offset=cmd.index('--port')+2
        self.equal(cmd[offset:],args,str(path)+'.server_args')
        self.check('--no-enable-prefix-caching' in args and '--enable-prefix-caching' not in args,str(path),'Capacity APC must be off')
        self.equal(cmd[cmd.index('--host')+1],'127.0.0.1',str(path)+'.host')
        self.equal(cmd[cmd.index('--served-model-name')+1],report['plan']['model'],str(path)+'.model_alias')
        self.equal(report['plan']['base_url'],'http://127.0.0.1:'+cmd[cmd.index('--port')+1],str(path)+'.url')
        config_key=next((k for k,v in snapshot.items()
                         if (k.startswith(self.root.name+'/configs/')
                             or k.startswith('experiments/'+self.root.name+'/configs/'))
                         and self.snapshot_document(v,k,command)==config),None)
        self.check(config_key is not None,str(path),'Config not present in run snapshot')
        self.totals['source_snapshot_members']+=len(snapshot)
        if not (directory/'status.json').exists():self.incomplete.append(str(directory)+': supervisor running');return snapshot,None
        status=self.json(directory/'status.json');self.equal(status.get('experiment_id'),directory.name,str(path)+'.status_id')
        self.path_sources(snapshot,command,status,str(path))
        self.check(bool(status.get('end_utc')),str(path),'Supervisor lacks terminal time')
        self.equal(status.get('source_sha256'),command['source_sha256'],str(path)+'.status_sources')
        self.equal(status.get('command'),command['command'],str(path)+'.status_command')
        experiment=self.json(directory/'experiment.json');success=status.get('exit_code')==0 and not status.get('timed_out') and not status.get('supervisor_error')
        if success:
            self.equal(experiment['status'],'PASS',str(path)+'.experiment_status')
            self.equal(report['status'],'PASSED',str(path)+'.successful_supervisor_report')
            self.check(all(c.get('exit_code')==0 for c in experiment['clients']),str(path),'Successful supervisor with failed client')
        return snapshot,success
    def workload(self,path,report,snapshot):
        wpath=self.local(path.parent,report['workload_file'],path.stem+'-workload.json')
        raw=self.read(wpath);self.equal(digest(raw),report['workload_sha256'],str(wpath)+'.hash');prepared=json.loads(raw)
        phases=report['plan']['phases'];self.equal(len(prepared),len(phases),str(wpath)+'.phases')
        self.check(len({p['id'] for p in phases})==len(phases),str(path),'Duplicate phase IDs')
        # Derive the snapshot key from the recorded plan/command, rather than
        # assuming all later research plans live in the phase-A capacity folder.
        # Metadata has already checked ZIP members against source_sha256.
        def source_key(saved):
            return snapshot_source_key(saved,self.root,'workloads')
        recorded_key=source_key(report['plan_path']);keys=[recorded_key] if recorded_key else []
        command_path=path.parent/'command.json';launch={}
        if command_path.is_file():
            launch=self.json(command_path);command=launch['command']
            if '--plan' in command and command.index('--plan')+1<len(command):
                command_key=source_key(command[command.index('--plan')+1])
                if command_key:
                    if recorded_key:self.equal(recorded_key,command_key,str(path)+'.command_plan_source')
                    keys.append(command_key)
        # Preserve lookup compatibility for older saved paths lacking the
        # project-relative anchor. A wrong matching key must fail its hash check.
        keys.append(self.root.name+'/workloads/capacity/'+Path(report['plan_path']).name)
        plan_key=next((key for key in keys if key in snapshot),None)
        if plan_key is not None:raw_plan=snapshot[plan_key]
        else:
            raw_plan=self.read(Path(report['plan_path']))
            self.warnings.append(str(path)+': plan read from recorded source path, not snapshot')
        self.equal(digest(raw_plan),report['plan_sha256'],str(path)+'.plan_hash')
        self.equal(report['plan'],normalize_plan(self.snapshot_document(raw_plan,plan_key,launch)),str(path)+'.normalized_plan')
        self.equal(report['prepared_requests'],{p['spec']['id']:len(p['items']) for p in prepared},str(path)+'.prepared_counts')
        self.pair('tokenizer',report['tokenizer_sha256'],str(path))
        by_id={}
        for item,phase in zip(prepared,phases):
            where=str(wpath)+'/'+phase['id'];self.equal(item['spec'],phase,where+'.spec')
            offsets,expected=regenerate(phase);self.equal(item['planned_offsets_s'],offsets,where+'.offsets')
            self.equal(len(item['items']),len(expected),where+'.count')
            for actual,e in zip(item['items'],expected):
                self.subset(actual,e,where+'.item');p=item['prompts'][actual['prompt_sha256']]
                self.equal(p['input_tokens'],e['input_tokens'],where+'.input_length')
            for h,p in item['prompts'].items():
                ids=p['token_ids'];self.check(all(type(i) is int and i>=0 for i in ids),where,'Invalid token ID')
                self.equal(len(ids),p['input_tokens'],where+'.token_length');self.equal(p['sha256'],h,where+'.embedded_hash')
                self.equal(digest(json.dumps(ids,separators=(',',':')).encode()),h,where+'.prompt_hash')
            self.equal(set(item['prompts']),{i['prompt_sha256'] for i in item['items']},where+'.prompt_set')
            identity=dict(mode=phase['mode'],seed=phase['seed'],duration_s=phase.get('duration_s'),rate=phase.get('rate'),
                          requests=phase.get('requests'),shape_mix=phase['shape_mix'])
            if phase['mode']=='trace':identity['arrivals']=phase['arrivals']
            self.pair('workload/'+digest(canonical(identity).encode()),dict(offsets=offsets,items=item['items'],prompts=item['prompts']),where)
            self.totals['prompts']+=len(item['prompts']);self.totals['planned_requests']+=len(item['items']);by_id[phase['id']]=item
        return by_id
    def metrics(self,path,run):
        name=run['phase']['id'];directory=path.parent/f'{path.stem}-{name}-metrics'
        seq=[('before.prom',run['metrics_before']),('after.prom',run['metrics_after'])]
        seq += [(f'{i:05d}.prom',s) for i,s in enumerate(run['metric_samples'])]
        last=-1.
        for filename,row in seq:
            where=str(directory/filename)
            if 'raw_file' in row:
                source=self.local(directory,row['raw_file'],filename);raw=self.read(source)
                self.equal(digest(raw),row['sha256'],where+'.hash');self.equal(row['values'],parse_metrics(raw.decode()),where+'.values')
                self.equal(row['http_status'],200,where+'.http_status')
            elif 'error' not in row:self.check(False,where,'Successful metrics sample has no raw file')
            if 'offset_s' in row:
                self.check(finite(row['offset_s']) and row['offset_s']>=last,where,'Nonmonotonic/nonfinite metric time');last=row['offset_s']
        valid=[s for s in run['metric_samples'] if 'error' not in s and 'values' in s]
        if not valid or not all(any('num_requests_waiting' in k for k in s['values']) for s in valid):
            self.warnings.append(str(path)+'/'+name+': waiting gauge coverage incomplete; stability cannot be certified from client data alone')
        self.totals['metrics_samples']+=len(run['metric_samples'])
    def row(self,row,item,phase,offset,start,where):
        self.subset(row,item,where+'.identity')
        self.equal(row['request_id'],phase['id']+'/'+str(item['index']),where+'.id')
        self.equal(row['phase_id'],phase['id'],where+'.phase_id')
        if phase['mode']!='closed-loop':self.equal(row['planned_offset_s'],offset,where+'.planned_offset')
        self.equal(row['planned_unix'],start+row['planned_offset_s'],where+'.planned_unix')
        for key in ('ok','admitted','censored','timed_out','done_received'):self.check(type(row[key]) is bool,where,'Invalid flag '+key)
        self.check(row['admitted'] != row['censored'],where,'Request must be either admitted or censored')
        self.check(finite(row['end_offset_s']) and row['end_offset_s']>=0,where,'Invalid end time')
        if row['admitted']:
            dispatch=row['dispatch_offset_s'];end=row['end_offset_s'];planned=row['planned_offset_s']
            self.check(finite(dispatch) and end>=dispatch>=planned-1e-5,where,'Invalid dispatch/end ordering')
            self.equal(row['dispatch_lag_s'],max(0.,dispatch-planned),where+'.dispatch_lag')
            self.equal(digest(row['full_text'].encode()),row['text_sha256'],where+'.text_hash')
            events=row['text_event_seconds'];self.check(all(finite(t) and t>=0 for t in events),where,'Invalid text event')
            self.check(events==sorted(events),where,'Text times not ordered')
            if events:self.equal(row['ttft_s'],events[0],where+'.first_text')
            for key in ('ttft_s','e2e_s'):
                value=row.get(key)
                if value is not None:self.check(finite(value) and value>=0,where,'Invalid '+key)
                self.equal(row.get('scheduled_'+key),row['dispatch_lag_s']+value if value is not None else None,where+'.scheduled_'+key)
            self.check(row['elapsed_until_return_s']>=0 and abs(row['elapsed_until_return_s']-(end-dispatch))<.05,where,'Elapsed/end time disagreement')
            if row['e2e_s'] is not None:
                self.check(row['e2e_s']<=end-dispatch+1e-5,where,'DONE later than completion')
                self.check(not events or events[-1]<=row['e2e_s']+1e-8,where,'Text after DONE')
        else:
            self.check(not row['ok'] and not row['timed_out'] and row['http_status'] is None and row['usage'] is None,
                       where,'Censored row contains model response/success')
            self.check(row['error']['type']=='ClientSafetyCap',where,'Unexpected censor reason')
            self.equal(row['dispatch_offset_s'],None,where+'.censored_dispatch')
        if row['ok']:
            self.check(row['admitted'] and row['http_status']==200 and row['done_received'] and not row['timed_out'] and row['error'] is None,
                       where,'Successful row has transport error')
            u=row['usage'];p,o=u.get('prompt_tokens'),u.get('completion_tokens')
            self.check(type(p) is int and p==item['input_tokens'] and type(o) is int and 1<=o<=item['requested_output_tokens'],where,'Actual usage invalid')
            self.equal(u.get('total_tokens',p+o),p+o,where+'.total_tokens')
            if item['ignore_eos']:
                self.equal(o,item['requested_output_tokens'],where+'.fixed_output');self.equal(row['finish_reason'],'length',where+'.finish')
            else:self.check(row['finish_reason'] in ('length','stop'),where,'Invalid finish reason')
            self.check(row['e2e_s']>=row['ttft_s']>=0,where,'Invalid successful timing')
            self.equal(row['tpot_s'],(row['e2e_s']-row['ttft_s'])/(o-1) if o>1 else None,where+'.tpot')
            self.check(bool(row['full_text']) and bool(row['text_event_seconds']),where,'Success has empty text/events')
        elif row['admitted']:self.check(row['error'] is not None,where,'Failed admitted request without error')
        self.subset(row,row_slo(row,phase['mode']),where+'.slo')
    def run(self,path,run,prepared):
        phase=run['phase'];name=phase['id'];where=str(path)+'/'+name
        self.equal(phase,prepared['spec'],where+'.phase');rows=run['requests'];items=prepared['items']
        self.equal(len(rows),len(items),where+'.request_count')
        self.equal([r['index'] for r in rows],list(range(len(items))),where+'.row_order')
        for i,(row,item) in enumerate(zip(rows,items)):
            self.row(row,item,phase,prepared['planned_offsets_s'][i],run['start_unix'],where+'/'+str(i))
        raw_path=self.local(path.parent,run['raw_requests_jsonl'],f'{path.stem}-{name}-requests.jsonl')
        raw_rows=[json.loads(line) for line in self.read(raw_path).splitlines() if line.strip()]
        self.equal(sorted(raw_rows,key=lambda r:r['index']),rows,where+'.jsonl')
        self.metrics(path,run)
        wall=run['summary']['total_wall_s'];self.check(finite(wall) and wall>0,where,'Invalid wall')
        self.check(abs((run['end_unix']-run['start_unix'])-wall)<.1,where,'Monotonic/unix wall mismatch')
        self.check(all(r['end_offset_s']<=wall+1e-5 for r in rows),where,'Request ends after phase wall')
        if phase['mode'] in ('poisson','trace'):self.check(wall>=phase['duration_s'],where,'Window terminated early')
        peak=run['summary']['peak_inflight'];self.check(type(peak) is int and 0<=peak<=len(rows),where,'Invalid peak inflight')
        events=[]
        for r in rows:
            if r['admitted']:events.extend([(r['dispatch_offset_s'],1),(r['end_offset_s'],-1)])
        active=0;actual_peak=0
        for t,delta in sorted(events):active+=delta;actual_peak=max(actual_peak,active)
        self.check(peak>=actual_peak,where,'Reported task peak below observed HTTP overlap')
        if phase['mode']!='closed-loop':self.check(peak<=phase['max_inflight'],where,'Task peak exceeds safety cap')
        else:self.equal(peak,min(phase['concurrency'],len(items)),where+'.closed_peak')
        self.equal(run['summary'],summarize(rows,phase,wall,run['metric_samples'],peak),where+'.summary')
        self.totals['requests']+=len(rows);self.totals['phases']+=1
        return dict(id=name,mode=phase['mode'],role=phase['role'],requests=len(rows),failed=run['summary']['failed'],
                    censored=run['summary']['censored'],p95_slo_pass=run['summary']['p95_slo_pass'],
                    stability='NOT AUDITED: independent capacity analysis is a separate step')
    def report(self,path):
        d=self.json(path);where=str(path)
        if d.get('status') in ('PREPARING','PREPARED','RUNNING'):
            self.incomplete.append(where+': '+d['status'])
        self.equal(d['schema_version'],1,where+'.schema')
        self.equal(d['transport'],dict(connection_policy='fresh_per_request',retries=0),where+'.transport')
        self.subset(d['research_slo'],dict(input_bounds=[1024,4096,16384],ttft_s=[1.,3.,8.],tpot_s=.05),where+'.slo_definition')
        if 'workload_file' not in d:
            self.incomplete.append(where+': workload not prepared');return
        snapshot,terminal_success=self.metadata(path,d);prepared=self.workload(path,d,snapshot)
        ids=[p['id'] for p in d['plan']['phases']];observed=[r['phase']['id'] for r in d['runs']]
        self.equal(observed,ids[:len(observed)],where+'.phase_order')
        if d['status']=='PASSED':self.equal(observed,ids,where+'.complete_phases')
        results=[]
        for run in d['runs']:
            results.append(self.run(path,run,prepared[run['phase']['id']]))
        if d['status']=='PASSED':self.check(all(r['failed']==0 and r['censored']==0 for r in results),where,'Passed report contains failures/censoring')
        elif d['status']=='CENSORED':self.check(bool(results) and results[-1]['censored']>0,where,'CENSORED report has no censored last phase')
        self.reports.append(dict(path=where,status=d['status'],supervisor_success=terminal_success,phases=results))
    def finish(self):
        for path,state in self.states.items():
            now=path.stat();self.check(state==(now.st_size,now.st_mtime_ns),str(path),'Source changed during audit')
        return dict(status='FAIL' if self.errors else 'INCOMPLETE' if self.incomplete or not self.reports else 'PASS',
            scope='Read-only independent evidence/formula consistency; not stable capacity, model quality, or SLO certification',
            created_utc=datetime.now(timezone.utc).isoformat(),auditor_sha256=digest(Path(__file__).read_bytes()),
            totals=dict(self.totals),errors=self.errors,warnings=self.warnings,incomplete=self.incomplete,
            reports=self.reports,source_sha256=self.files,matched_workloads=self.pairs,
            limits=['Uses original measured wall/timestamps; cannot independently re-time historical requests.',
                'Task peak checked against HTTP overlap and caps, not reconstructible scheduler microtiming.',
                'Prepared token IDs hashed/length-checked; tokenizer encoding not independently replayed.',
                'P95 uses successful timings; failures/censoring must separately block capacity certification.',
                'Stability classification and repeated long-window evidence are separate from this data audit.'])


def fixture_rows():
    phase=dict(id='fixture',mode='burst',role='screen',requests=2,concurrency=2,max_inflight=2,
               seed=71,ignore_eos=False,shape_mix=[dict(name='toy',input_tokens=512,output_tokens=4,weight=1)])
    rows=[]
    for i in range(2):
        o=i+1;ttft=.1+i*.2;e2e=ttft+.02*(o-1);dispatch=.01+i*.01
        row=dict(index=i,shape_name='toy',input_tokens=512,requested_output_tokens=4,seed=71+i,
                 ignore_eos=False,prompt_sha256='fixture',request_id=f'fixture/{i}',phase_id='fixture',
                 planned_offset_s=0.,planned_unix=100.,dispatch_offset_s=dispatch,dispatch_lag_s=dispatch,
                 end_offset_s=dispatch+e2e+.001,admitted=True,censored=False,timed_out=False,http_status=200,
                 ok=True,usage=dict(prompt_tokens=512,completion_tokens=o,total_tokens=512+o),
                 ttft_s=ttft,e2e_s=e2e,tpot_s=.02 if o>1 else None,scheduled_ttft_s=dispatch+ttft,
                 scheduled_e2e_s=dispatch+e2e,error=None,full_text='x',text_sha256=digest(b'x'),done_received=True,
                 text_event_seconds=[ttft],finish_reason='stop',elapsed_until_return_s=e2e+.001)
        row.update(row_slo(row,phase['mode']));rows.append(row)
    return phase,rows


def full_fixture(root):
    """Tiny explicitly synthetic complete evidence tree, never executed as model code."""
    phase, rows = fixture_rows()
    directory = path_at(root, 'results/fixture')
    directory.mkdir(parents=True)
    path = directory / 'capacity.json'
    plan_path = path_at(root, 'workloads/capacity/fixture.json')
    raw_plan = dict(schema_version=1, experiment_id='fixture', phases=[phase])
    plan = normalize_plan(raw_plan)
    phase = plan['phases'][0]

    def put(p, value, raw=False):
        p.parent.mkdir(parents=True, exist_ok=True)
        write_bytes(p, value if raw else (json.dumps(value, sort_keys=True) + '\n').encode())
    put(plan_path, raw_plan)
    prompts = {}
    items = []
    for r in rows:
        ids = [r['index'] + 1] + [7] * 511
        h = digest(json.dumps(ids, separators=(',', ':')).encode())
        r['prompt_sha256'] = h
        prompts[h] = dict(token_ids=ids, input_tokens=512, sha256=h)
        items.append({k: r[k] for k in ('index', 'shape_name', 'input_tokens', 'requested_output_tokens', 'seed', 'ignore_eos', 'prompt_sha256')})
    workload = [dict(spec=phase, items=items, prompts=prompts, planned_offsets_s=[0.0, 0.0])]
    wpath = directory / 'capacity-workload.json'
    put(wpath, workload)
    mdir = directory / 'capacity-fixture-metrics'
    text = b'vllm:num_requests_waiting{model_name="fixture"} 0\n'

    def metric(filename, offset=None):
        p = mdir / filename
        put(p, text, True)
        value = dict(unix=100.0, http_status=200, values=parse_metrics(text.decode()), raw_file=str(p), sha256=digest(text))
        if offset is not None:
            value['offset_s'] = offset
        return value
    before = metric('before.prom')
    after = metric('after.prom')
    samples = [metric('00000.prom', 0.0)]
    jsonl = directory / 'capacity-fixture-requests.jsonl'
    put(jsonl, ('\n'.join((json.dumps(r) for r in rows)) + '\n').encode(), True)
    run = dict(phase=phase, start_unix=100.0, end_unix=101.0, requests=rows, metric_samples=samples, metrics_before=before, metrics_after=after, raw_requests_jsonl=str(jsonl), summary=summarize(rows, phase, 1.0, samples, 2))
    config = dict(name='fixture', args=['--no-enable-prefix-caching'])
    snapshot = {root.name + '/scripts/capacity_client.py': b'# SYNTHETIC FIXTURE SOURCE - NOT EXECUTED', 'scripts/benchmark_client.py': b'# SYNTHETIC FIXTURE HELPER - NOT EXECUTED', root.name + '/workloads/capacity/fixture.json': plan_path.read_bytes(), root.name + '/configs/fixture.json': json.dumps(config).encode()}
    command = dict(experiment_id='fixture', command=['SYNTHETIC_NOT_EXECUTED'], source_sha256={k: digest(v) for k, v in snapshot.items()}, environment=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='0,1,2,3'))
    put(directory / 'command.json', command)
    put(directory / 'status.json', dict(command, exit_code=0, timed_out=False, supervisor_error=None, end_utc='synthetic terminal'))
    put(directory / 'experiment.json', dict(status='PASS', clients=[dict(exit_code=0)]))
    put(directory / 'config.json', config)
    put(directory / 'capacity-plan.json', plan)
    put(directory / 'server-command.json', dict(command=['SYNTHETIC_NOT_EXECUTED', 'serve', 'fixture', '--served-model-name', 'dsv4', '--host', '127.0.0.1', '--port', '8260', *config['args']]))
    with zipfile.ZipFile(directory / 'source-snapshot.zip', 'x') as z:
        for name, data in snapshot.items():
            z.writestr(name, data)
    report = dict(schema_version=1, status='PASSED', synthetic_fixture=True, plan=plan, plan_path=str(plan_path), plan_sha256=digest(plan_path.read_bytes()), client_sha256=digest(snapshot[root.name + '/scripts/capacity_client.py']), base_helper_sha256=digest(snapshot['scripts/benchmark_client.py']), workload_file=str(wpath), workload_sha256=digest(wpath.read_bytes()), prepared_requests=dict(fixture=2), tokenizer_sha256='synthetic tokenizer', transport=dict(connection_policy='fresh_per_request', retries=0), research_slo=dict(input_bounds=[1024, 4096, 16384], ttft_s=[1.0, 3.0, 8.0], tpot_s=0.05), runs=[run])
    put(path, report)
    return path


def self_test():
    phase, rows = fixture_rows()
    checks = []
    auditor = Auditor(ROOT)
    for i, r in enumerate(rows):
        item = {k: r[k] for k in ('index', 'shape_name', 'input_tokens', 'requested_output_tokens', 'seed', 'ignore_eos', 'prompt_sha256')}
        auditor.row(r, item, phase, 0.0, 100.0, str(i))
    assert not auditor.errors, auditor.errors
    expected = summarize(rows, phase, 1.0, [], 2)
    auditor.equal(expected, summarize(rows, phase, 1.0, [], 2), 'summary')
    assert not auditor.errors
    checks.append('valid natural-EOS usage, single-token TPOT null and summary')
    for label, mutate in [('actual usage', lambda r: r['usage'].__setitem__('prompt_tokens', 513)), ('text hash', lambda r: r.__setitem__('text_sha256', '0' * 64)), ('scheduled latency', lambda r: r.__setitem__('scheduled_ttft_s', 0.0001)), ('E2E actual-output budget', lambda r: r.__setitem__('e2e_limit_s', 999.0)), ('TPOT formula', lambda r: r.__setitem__('tpot_s', 0.001)), ('censored response contradiction', lambda r: r.__setitem__('censored', True))]:
        bad = deepcopy(rows[1])
        mutate(bad)
        a = Auditor(ROOT)
        item = {k: rows[1][k] for k in ('index', 'shape_name', 'input_tokens', 'requested_output_tokens', 'seed', 'ignore_eos', 'prompt_sha256')}
        a.row(bad, item, phase, 0.0, 100.0, label)
        assert a.errors, label
        checks.append(label + ' corruption detected')
    bad = deepcopy(expected)
    bad['by_input_tier']['le1k']['metrics']['ttft_s']['p99'] += 1
    a = Auditor(ROOT)
    a.equal(bad, expected, 'p99')
    assert a.errors
    checks.append('tier P99 corruption detected')
    mixed = [deepcopy(rows[0]) for _ in range(99)] + [deepcopy(rows[0])]
    mixed[-1].update(input_tokens=16384, scheduled_ttft_s=9.0, scheduled_e2e_s=9.0)
    for r in mixed:
        r.update(row_slo(r, 'burst'))
    s = summarize(mixed, phase, 10.0, [], 2)
    assert s['overall_ratio_p95_slo_pass'] and (not s['p95_slo_pass'])
    checks.append('rare failing tier is not hidden by aggregate P95')
    late = deepcopy(rows)
    late[-1].update(dispatch_offset_s=3.0, end_offset_s=4.0)
    p = deepcopy(phase)
    p.update(mode='poisson', duration_s=2.0, rate=1.0)
    assert summarize(late, p, 4.0, [], 2)['in_flight_at_window_end'] == 0
    checks.append('post-window dispatch excluded from window-end inflight')
    p.update(seed=72)
    o, items = regenerate(p)
    o2, i2 = regenerate(p)
    assert o == o2 and items == i2 and all((0 < x < 2 for x in o))
    checks.append('seeded finite-horizon Poisson and shape identity')
    assert stats([0, 10])['p99'] == 9.9
    checks.append('linear P99 interpolation')
    with tempfile.TemporaryDirectory(prefix='capacity-audit-') as temp:
        root = Path(temp) / 'research/model/scenarios-fixture'
        path = full_fixture(root)
        a = Auditor(root)
        a.report(path)
        result = a.finish()
        assert result['status'] == 'PASS', result['errors']
        checks.append('full synthetic terminal snapshot/plan/workload/JSONL/metrics/report roundtrip')
        raw = path.read_bytes()
        bad = json.loads(raw)
        bad['runs'][0]['summary']['throughput']['cohort_requests_s'] = 999
        write_text(path, json.dumps(bad))
        a = Auditor(root)
        a.report(path)
        assert a.finish()['status'] == 'FAIL'
        checks.append('full-report throughput corruption detected')
        write_bytes(path, raw)
        wpath = path.with_name('capacity-workload.json')
        original = wpath.read_bytes()
        bad = json.loads(original)
        key = next(iter(bad[0]['prompts']))
        bad[0]['prompts'][key]['token_ids'][0] += 1
        write_text(wpath, json.dumps(bad))
        a = Auditor(root)
        a.report(path)
        assert a.finish()['status'] == 'FAIL'
        checks.append('full-workload hash and prompt-ID corruption detected')
        write_bytes(wpath, original)
        metric = path.parent / 'capacity-fixture-metrics/00000.prom'
        original = metric.read_bytes()
        write_bytes(metric, original + b'# mutation\n')
        a = Auditor(root)
        a.report(path)
        assert a.finish()['status'] == 'FAIL'
        checks.append('raw metrics checksum corruption detected')
    with tempfile.TemporaryDirectory(prefix='capacity-audit-phase-b-') as temp:
        root = Path(temp) / 'research/model/scenarios-fixture'
        path = full_fixture(root)
        report = load_document(path)
        old_plan = Path(report['plan_path'])
        new_plan = path_at(root, 'workloads/long-context') / old_plan.name
        old_key = root.name + '/workloads/capacity/' + old_plan.name
        new_key = root.name + '/workloads/long-context/' + old_plan.name
        report['plan_path'] = str(new_plan)
        write_text(path, json.dumps(report))
        command = load_document(path.parent / 'command.json')
        command['command'] = ['SYNTHETIC_NOT_EXECUTED', '--plan', str(new_plan)]
        command['source_sha256'][new_key] = command['source_sha256'].pop(old_key)
        write_text(path.parent / 'command.json', json.dumps(command))
        status = load_document(path.parent / 'status.json')
        status.update(command=command['command'], source_sha256=command['source_sha256'])
        write_text(path.parent / 'status.json', json.dumps(status))
        zpath = path.parent / 'source-snapshot.zip'
        with zipfile.ZipFile(zpath) as z:
            snapshot = {name: z.read(name) for name in z.namelist()}
        snapshot[new_key] = snapshot.pop(old_key)
        with zipfile.ZipFile(zpath, 'w') as z:
            for name, data in snapshot.items():
                z.writestr(name, data)
        old_plan.unlink()
        a = Auditor(root)
        a.report(path)
        result = a.finish()
        assert result['status'] == 'PASS', result['errors']
        assert not any(('plan read from recorded source path' in warning for warning in result['warnings']))
        checks.append('phase-B recorded plan/command selects hashed snapshot without a live source plan')
        report['plan_sha256'] = '0' * 64
        write_text(path, json.dumps(report))
        a = Auditor(root)
        a.report(path)
        assert a.finish()['status'] == 'FAIL'
        checks.append('phase-B snapshot plan hash mismatch is rejected')
    return dict(status='PASS', scope='CPU-only synthetic auditor checks; no GPU/model/HTTP', checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--input', type=Path, action='append', help='Repeat for selected capacity.json; defaults to all results/*/capacity.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Audit output exists; use a new ID')
    if args.self_test:
        result = self_test()
    else:
        audit = Auditor(args.root)
        for path in args.input or sorted(path_at(args.root, 'results').glob('*/capacity.json')):
            try:
                if not args.input and (path.parent/'command.json').is_file():
                    launch = json.loads((path.parent/'command.json').read_text())
                    if any(Path(arg).name == 'serve_study.py' for arg in launch.get('command', [])):
                        continue  # Scheduler results use their stronger runtime auditor.
                audit.report(path)
            except Exception as exc:
                audit.check(False, str(path), 'Audit could not complete: ' + repr(exc))
        result = audit.finish()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, 'x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'totals', 'errors', 'incomplete', 'checks') if k in result}, ensure_ascii=False))
    return 1 if result['status'] == 'FAIL' else 2 if result['status'] == 'INCOMPLETE' and (not args.allow_incomplete) else 0


if __name__=='__main__':raise SystemExit(main())
