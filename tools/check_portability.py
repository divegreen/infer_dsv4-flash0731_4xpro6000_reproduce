#!/usr/bin/env python3
"""CPU-only migration and orchestration regression; never starts inference."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
from unittest import mock

P=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('portable_entry',P/'tools/reproduce.py')
cli=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(cli)

def numeric(value, path=()):
    if isinstance(value,dict):
        return {k:v for key,x in value.items() for k,v in numeric(x,path+(key,)).items()}
    if isinstance(value,list):
        return {k:v for i,x in enumerate(value) for k,v in numeric(x,path+(i,)).items()}
    return {path:value} if type(value) in (int,float,bool) else {}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=P/'results/validation/portability.json')
    parser.add_argument('--reference-root',type=Path,help='Optional external preserved tree containing archive/ and doc/migration-map.json')
    args=parser.parse_args()
    checks=[]
    def check(name,condition,detail=None):
        checks.append({'name':name,'status':'PASS' if condition else 'FAIL','detail':detail})
        if not condition:raise AssertionError(name+': '+str(detail))
    reference=args.reference_root.expanduser().resolve() if args.reference_root else None
    if reference is not None:
        archive=reference/'archive/initial-project.tar.gz'; manifest_path=reference/'archive/initial-manifest.json'
        if not archive.is_file() or not manifest_path.is_file():
            parser.error('--reference-root requires the original archive and manifest')
        manifest=json.loads(manifest_path.read_text()); mapping=json.loads((reference/'doc/migration-map.json').read_text())['files']
        with tarfile.open(archive) as t:
            check('initial archive manifest covers every file',set(manifest)==set(t.getnames()))
            for name,meta in manifest.items():
                check('original SHA '+name,hashlib.sha256(t.extractfile(name).read()).hexdigest()==meta['sha256'])
            numerical=fixtures=0
            for old,new in mapping.items():
                if old not in manifest or not (P/new).is_file():continue
                original=t.extractfile(old).read(); current=(P/new).read_bytes()
                if '/tasks/' in old or '/workloads/agent/' in old and 'self-test-results' not in old:
                    check('unchanged model fixture '+new,original==current);fixtures+=1
                if '/configs/' in old or '/workloads/' in old and old.endswith('.json') and 'self-test-results' not in old:
                    if old.endswith('.json'):
                        a,b=json.loads(original),json.loads(current)
                        check('unchanged numerical workload '+new,numeric(a)==numeric(b));numerical+=1
            check('compared real input coverage',numerical>80 and fixtures>30,{'numeric_files':numerical,'fixture_files':fixtures})
    with tempfile.TemporaryDirectory(prefix='portable project ') as directory:
        tmp=Path(directory);entry={'id':'synthetic-run','runner':'unused.py','child_argv':['unused.py'],'timeout':1,'gpus':'0', 'reference':'none','executable_definition':True,'depends_on':[],'requires_success':[]}
        fake_cat={'experiments':[entry]};dest=tmp/'results';dest.mkdir()
        def simulated_process(*a,**kw):
            folder=dest/'runs/synthetic-run';folder.mkdir(parents=True)
            cli.save(folder/'status.json',{'experiment_id':'synthetic-run','end_utc':'synthetic-only','exit_code':0})
            cli.save(folder/'measurement.json',{'synthetic':True,'value':1})
            return SimpleNamespace(returncode=0)
        with mock.patch.object(cli,'PROJECT',tmp),mock.patch.object(cli,'RESULTS',dest),mock.patch.object(cli,'catalog',return_value=fake_cat),mock.patch.object(cli,'gpu_reservation',return_value=contextlib.nullcontext({'synthetic':True})),mock.patch.object(cli.subprocess,'run',side_effect=simulated_process) as child:
            args_run=SimpleNamespace(dry_run=False,overwrite=False)
            first=cli.execute_one(entry,args_run,'input-v1')
            check('automatic missing run parent creation',first['status']=='TERMINAL')
            second=cli.execute_one(entry,args_run,'input-v1');check('verified same-input resume avoids duplicate execution',second['status']=='REUSED_TERMINAL' and child.call_count==1)
            receipt={'signature':'input-v1','status':'COMPLETED','runs':[first]};cli.save(dest/'stage-state/topic/process1.json',receipt)
            cli.check_stage('topic/process1','input-v1');check('verified prerequisite accepted',True)
            cli.save(dest/'runs/synthetic-run/measurement.json',{'synthetic':True,'value':2})
            try:cli.check_stage('topic/process1','input-v1')
            except ValueError:check('tampered result rejected even with unchanged terminal status',True)
            else:check('tampered result rejected even with unchanged terminal status',False)
            try:cli.check_stage('missing/process1','input-v1')
            except ValueError:check('missing prerequisite blocked',True)
            else:check('missing prerequisite blocked',False)
            args_run.overwrite=True;third=cli.execute_one(entry,args_run,'input-v2')
            check('overwrite preserves previous bytes',json.loads((dest/'preserved/runs/synthetic-run/attempt-1/measurement.json').read_text())['value']==2)
            check('overwrite starts a fresh attempt',child.call_count==2 and third['status']=='TERMINAL')
        pack=tmp/'source.tar.gz'
        cli.package(SimpleNamespace(output=str(pack)))
        unpack=tmp/'different directory with spaces';unpack.mkdir()
        with tarfile.open(pack) as t:t.extractall(unpack,filter='data')
        copy=unpack/'portable-inference';tool=copy/'tools/reproduce.py'
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',DSV4_MODEL_PATH=str(tmp/'model with spaces'))
        env.pop('VLLM_PYTHON',None)
        dry=subprocess.run([sys.executable,str(tool),'run','--stage','architecture/process1-smoke','--dry-run'],cwd='/',env=env,capture_output=True,text=True)
        check('relocated dry-run works from unrelated cwd without venv',dry.returncode==0,dry.stderr)
        check('dry-run creates no result directory',not (copy/'results').exists())
        prepare=subprocess.run([sys.executable,str(tool),'prepare'],cwd='/',env=env,capture_output=True,text=True)
        check('relocated prepare creates all missing directories',prepare.returncode==0 and (copy/'results/setup/core.requirements.txt').is_file(),prepare.stderr)
        report=json.loads(prepare.stdout);check('model override expands correctly with spaces',report['model']==str(tmp/'model with spaces'))
        validate=subprocess.run([sys.executable,str(tool),'validate'],cwd='/',env=env,capture_output=True,text=True)
        check('native-only package has complete static input closure',validate.returncode==0,validate.stdout+validate.stderr)
    cli.save(args.output,{'generated':True,'status':'PASS','checks':checks,'count':len(checks),'scope':'CPU synthetic lifecycle and real tar/cwd relocation; no inference or performance result'})
    print(json.dumps({'status':'PASS','checks':len(checks),'output':str(args.output)}))

if __name__=='__main__':main()
