#!/usr/bin/env python3
"""Read-only D metadata adapter over the frozen independent request auditor.

Reuses original numerical/raw/Prometheus validation. New launch/source/class
identity is checked here, with no captured source execution, GPU or HTTP call.
PASS means evidence consistency only, never model quality or capacity approval.
"""

# Locate the shared layout from this file, independent of the caller's cwd.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "project_layout.py").is_file())
if str(_layout_root) not in _layout_sys.path:
    _layout_sys.path.insert(0, str(_layout_root))
from project_layout import path_at, model_path, workspace_path, load_document, write_text, write_bytes, open_output
import argparse
import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent/'scenarios/scripts/capacity_audit.py'
ORIGINAL_FROZEN_SOURCE_SHA = 'f9f5513113a4561f0f524b1c0e6524d7e9398d5325d64d82f76658b4fa4b416d'
FROZEN_SOURCE_SHA = '44842392a40e0898baca404ba00063851e5be313fe9b06a471649fc408f2824f'
if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != FROZEN_SOURCE_SHA:
    raise RuntimeError('Reused historical auditor changed; inspect explicitly before use')
QUALITY_SPEC=importlib.util.spec_from_file_location('d_quality_auditor',Path(__file__).with_name('audit_quality.py'))
quality_audit=importlib.util.module_from_spec(QUALITY_SPEC);QUALITY_SPEC.loader.exec_module(quality_audit)
SPEC=importlib.util.spec_from_file_location('d_frozen_capacity_auditor',SOURCE)
base=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(base)


class Auditor(base.Auditor):
    def __init__(self, root=ROOT):
        super().__init__(root)
        self.runtime_reports=[]
        self.read(SOURCE)
        self.read(Path(__file__))

    def metadata(self,path,report):
        directory=path.parent;where=str(path);snapshot={}
        required=('command.json','status.json','runtime-identity.json','source-snapshot.zip',
                  'config.json','capacity-plan.json','server-command.json','experiment.json')
        missing=[name for name in required if not (directory/name).is_file()]
        if missing:
            self.incomplete.append(where+': missing terminal/runtime metadata '+repr(missing))
            return snapshot,None
        command=self.json(directory/'command.json');status=self.json(directory/'status.json')
        identity_raw=self.read(directory/'runtime-identity.json');identity=json.loads(identity_raw)
        archive_raw=self.read(directory/'source-snapshot.zip')
        self.equal(command.get('runtime_identity_sha256'),base.digest(identity_raw),where+'.runtime_identity_hash')
        self.equal(command.get('source_snapshot_sha256'),base.digest(archive_raw),where+'.source_snapshot_hash')
        hashes=command['source_sha256'];files=identity['files']
        self.equal(set(hashes),set(files),where+'.source_sets')
        with zipfile.ZipFile(io.BytesIO(archive_raw)) as archive:
            names=archive.namelist()
            self.check(len(names)==len(set(names)),where,'Duplicate snapshot members')
            self.equal(set(names),set(hashes),where+'.zip_member_set')
            for info in archive.infolist():
                key=info.filename;parts=PurePosixPath(key)
                self.check(not parts.is_absolute() and '..' not in parts.parts and not info.is_dir()
                           and stat.S_IFMT(info.external_attr>>16) in (0,stat.S_IFREG),where,'Unsafe snapshot member '+key)
                payload=archive.read(info);snapshot[key]=payload
                self.equal(base.digest(payload),hashes[key],where+'.source_hash/'+key)
                self.equal(files[key]['sha256'],hashes[key],where+'.identity_hash/'+key)
                self.equal(len(payload),files[key]['bytes'],where+'.source_bytes/'+key)
        self.totals['source_snapshot_members']+=len(snapshot)
        self.path_sources(snapshot,command,status,where)
        if 'path_context' in command:
            self.check('metadata/scheduler-pair-blocks.json' in snapshot,where,'Registered scheduler pairs missing from launch snapshot')
        for source,field in [('study_client.py','client_sha256'),('serve_study.py','profile_harness_sha256')]:
            key=self.root.name+'/scripts/'+source
            self.equal(report.get(field),base.digest(snapshot[key]),where+'.actual_'+source)
        self.equal(report['base_helper_sha256'],base.digest(snapshot['scripts/benchmark_client.py']),where+'.actual_base_helper')
        record_key=identity['package']['record_key']
        wheel={r[0]:r for r in csv.reader(io.StringIO(snapshot[record_key].decode()))}
        for key,payload in snapshot.items():
            if key.startswith('installed/vllm/'):
                name=key.removeprefix('installed/');entry=wheel.get(name)
                encoded=base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip('=')
                self.check(bool(entry and entry[1]=='sha256='+encoded and entry[2]==str(len(payload))),where,'Installed source not bound to captured RECORD: '+key)
        self.equal(command['experiment_id'],directory.name,where+'.command_id')
        self.equal(status.get('experiment_id'),directory.name,where+'.status_id')
        self.equal(report['plan']['experiment_id'],directory.name,where+'.plan_id')
        for key in ('command','source_sha256','source_snapshot_sha256','runtime_identity_sha256','cwd','environment'):
            self.equal(status.get(key),command[key],where+'.terminal_'+key)
        self.check(bool(status.get('end_utc')),where,'Missing supervisor terminal time')
        self.check(type(status.get('supervisor_exit_code')) is int,where,'Missing actual supervisor exit code')
        self.check(status.get('source_identity_unchanged') is True and status.get('changed_source_members')==[],where,'Runtime source identity was not preserved')
        self.check(status.get('cleanup_error') is None,where,'Supervisor cleanup/source verification error')
        self.equal(command['environment'].get('CUDA_VISIBLE_DEVICES'),'0,1,2,3',where+'.gpus')
        self.equal(command['environment'].get('CUDA_DEVICE_ORDER'),'PCI_BUS_ID',where+'.device_order')
        config=self.json(directory/'config.json');server=self.json(directory/'server-command.json')
        saved_plan=self.json(directory/'capacity-plan.json');self.equal(saved_plan,report['plan'],where+'.saved_plan')
        argv=command['command'];config_arg=argv[argv.index('--config')+1];plan_arg=argv[argv.index('--plan')+1]
        for original,category in ((config_arg,'configs'),(plan_arg,'workloads')):
            key=base.snapshot_source_key(original,self.root,category)
            self.check(key is not None,where,'Launch input outside scheduler '+category)
            self.check(key in snapshot,where,'Launch input missing from snapshot '+str(key))
            if key not in snapshot:
                continue
            if category=='configs':self.equal(self.snapshot_document(snapshot[key],key,command),config,where+'.snapshot_config')
            else:self.equal(base.digest(snapshot[key]),report['plan_sha256'],where+'.snapshot_plan')
        cmd=server['command'];args=config['args'];offset=cmd.index('--port')+2
        self.equal(cmd[offset:],args,where+'.actual_server_args')
        self.equal(server['extra_environment'],config.get('env',{}),where+'.config_env')
        self.equal(cmd[cmd.index('--host')+1],'127.0.0.1',where+'.host')
        self.equal(report['plan']['base_url'],'http://127.0.0.1:'+cmd[cmd.index('--port')+1],where+'.base_url')
        self.equal(cmd[cmd.index('--served-model-name')+1],report['plan']['model'],where+'.model_alias')
        self.equal(cmd[cmd.index('serve')+1],report['plan']['tokenizer_path'],where+'.model_path')
        self.check('--no-enable-prefix-caching' in args and '--enable-prefix-caching' not in args,where,'Initial D study requires APC off')
        owned=server['owned_environment'];is_profile=any(p['role']=='profile' for p in report['plan']['phases'])
        self.equal(owned.get('CUDA_VISIBLE_DEVICES'),'0,1,2,3',where+'.server_gpus')
        self.equal(owned.get('CUDA_DEVICE_ORDER'),'PCI_BUS_ID',where+'.server_device_order')
        context=command.get('path_context',{})
        recorded_root=Path(context['research_root'])/self.root.name if context else self.root
        recorded_run=Path(context['run_dir']) if context else directory
        self.equal(owned.get('PYTHONPATH'),str(recorded_root/'scripts'),where+'.server_module_path')
        self.equal(owned.get('RESEARCH_D_TRACE_DIR'),str(recorded_run) if is_profile else '',where+'.step_trace_mode')
        experiment=self.json(directory/'experiment.json')
        self.equal(experiment.get('profile'),is_profile,where+'.experiment_profile')
        self.equal(experiment.get('excluded_from_performance'),is_profile,where+'.profile_excluded')
        self.equal(report.get('profiling',{}).get('excluded_from_performance'),is_profile,where+'.report_profile_excluded')
        success=status['supervisor_exit_code']==0 and status.get('exit_code')==0 and not any(status.get(k) for k in ('timed_out','interrupted_signal','supervisor_error','cleanup_error'))
        if success:
            self.equal(experiment['status'],'PASS',where+'.successful_experiment')
            self.equal(report['status'],'PASSED',where+'.successful_report')
            self.check(all(c.get('exit_code')==0 for c in experiment['clients']),where,'Successful run contains failed client')
        selected_class=args[args.index('--scheduler-cls')+1] if '--scheduler-cls' in args else None
        scheduler_paths=sorted(directory.glob('scheduler-identity-*.json'))
        if success and selected_class:self.check(len(scheduler_paths)==1,where,'Expected one TP4 scheduler actual identity')
        for identity_path in scheduler_paths:
            scheduler=self.json(identity_path)
            self.equal(scheduler['class_name'],selected_class,where+'.actual_scheduler_class')
            for field,key in [('module',self.root.name+'/scripts/research_scheduler.py'),('parent','installed/vllm/v1/core/sched/async_scheduler.py')]:
                self.equal(scheduler[field+'_sha256'],hashes[key],where+'.scheduler_'+field+'_hash')
                self.equal(scheduler[field+'_file'],files[key]['path'],where+'.scheduler_'+field+'_path')
            self.equal(scheduler.get('step_trace_enabled'),is_profile,where+'.scheduler_trace')
            self.equal(scheduler.get('async_scheduler'),True,where+'.scheduler_async')
        quality=quality_audit.audit(self,directory,report,config,experiment,snapshot,success)
        self.runtime_reports.append(dict(experiment=directory.name,supervisor_success=success,quality_probe=quality,
             selected_scheduler=selected_class,actual_scheduler_identity_files=[str(p) for p in scheduler_paths],
             source_snapshot_members=len(snapshot),profile=is_profile,
             scope='Per-experiment evidence identity; failed/partial execution is retained, never a global capacity verdict.'))
        return snapshot,success

    def row(self,row,item,phase,offset,start,where):
        super().row(row,item,phase,offset,start,where)
        if row['admitted']:
            expected='d-'+row['request_id'].replace('/','-')
            self.equal(row.get('wire_request_id'),expected,where+'.wire_id')
            ids=row.get('server_response_ids')
            self.check(isinstance(ids,list) and all(isinstance(i,str) for i in ids) and len(ids)==len(set(ids)),where,'Malformed response ID evidence')
            if row['ok']:self.equal(ids,['cmpl-'+expected],where+'.response_id')
        else:
            self.check(not row.get('wire_request_id') and not row.get('server_response_ids'),where,'Censored arrival has response identity')

    def run(self,path,run,prepared):
        self.equal(run.get('excluded_from_performance'),run['phase']['role'] in ('warmup','profile'),str(path)+'.phase_exclusion/'+run['phase']['id'])
        return super().run(path,run,prepared)

    def finish(self):
        result=super().finish()
        result.update(adapter_sha256=base.digest(Path(__file__).read_bytes()),
                      reused_auditor_sha256=FROZEN_SOURCE_SHA,
                      original_auditor_sha256=ORIGINAL_FROZEN_SOURCE_SHA,
                      auditor_revision='Portable source paths and recorded placeholder expansion; numerical request metrics unchanged',
                      runtime_reports=self.runtime_reports)
        result['limits']+=['No global all-started capacity decision; every audited run retains its own success/failed/partial status.',
                           'Actual scheduler allocations/kernel traces and model quality need separate analyses.',
                           'Original inherited request metrics remain unchanged; only D launch and wire identity validation is added.']
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new audit output revision')
    auditor = Auditor()
    for path in args.input:
        try:
            auditor.report(path.resolve())
        except Exception as exc:
            auditor.check(False, str(path), 'Audit could not complete: ' + repr(exc))
    result = auditor.finish()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, 'x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'totals', 'errors', 'incomplete')}))
    return 0 if result['status'] == 'PASS' else 2 if result['status'] == 'INCOMPLETE' else 1


if __name__=='__main__':raise SystemExit(main())
