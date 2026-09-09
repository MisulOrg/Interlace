"""Run one owned workload under an independent, conservative resource monitor.

Polling is an additional stop mechanism, not an OS-enforced allocation limit.
The algorithm must bound individual allocations before this wrapper is used.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from .resources import process_footprint


def require_guard():
    if os.environ.get('TRANSFORMERMODEL_GUARD_PARENT')!=str(os.getppid()):
        raise RuntimeError('Run training through python -m transformermodel.safe_run --report PATH -- COMMAND')


def swap_used():
    result=subprocess.run(['sysctl','-n','vm.swapusage'],capture_output=True,text=True,check=True,timeout=2)
    match=re.search(r'used = ([\d.]+)([MG])',result.stdout)
    if match is None:raise RuntimeError('cannot read swap usage')
    return float(match[1])*(2**20 if match[2]=='M' else 2**30)


def memory_free_percent():
    result=subprocess.run(['memory_pressure','-Q'],capture_output=True,text=True,check=True,timeout=2)
    match=re.search(r'System-wide memory free percentage: (\d+)%',result.stdout)
    if match is None:raise RuntimeError('cannot read memory pressure')
    return int(match[1])


def run(command,report,*,memory_mib=2048,reserve_mib=20480,swap_growth_mib=128,seconds=180,minimum_free_percent=40,interval=.1):
    report=Path(report);report.parent.mkdir(parents=True,exist_ok=True)
    if report.exists():raise FileExistsError(report)
    free=shutil.disk_usage(report.parent).free
    if free<reserve_mib*2**20:raise RuntimeError('disk reserve preflight failed')
    if memory_free_percent()<minimum_free_percent:raise RuntimeError('system memory preflight failed')
    if not process_footprint()['available']:raise RuntimeError('physical footprint monitor unavailable')
    baseline_swap=swap_used();last_swap=baseline_swap;start=time.monotonic();peak=0;reason=None
    child_env=os.environ.copy();child_env['TRANSFORMERMODEL_GUARD_PARENT']=str(os.getpid())
    child=subprocess.Popen(command,start_new_session=True,env=child_env)
    slow_next=0.;samples=0
    try:
        while child.poll() is None:
            now=time.monotonic();footprint=process_footprint(child.pid)
            if footprint['available']:
                peak=max(peak,footprint['lifetime_max_phys_footprint_bytes']);samples+=1
                if peak>memory_mib*2**20:reason='physical footprint limit'
            elif child.poll() is None:reason='physical footprint monitor unavailable'
            if now-start>seconds:reason='wall time limit'
            if now>=slow_next:
                if shutil.disk_usage(report.parent).free<reserve_mib*2**20:reason='disk reserve'
                last_swap=swap_used()
                if last_swap-baseline_swap>swap_growth_mib*2**20:reason='system swap growth'
                if memory_free_percent()<minimum_free_percent:reason='system memory headroom'
                slow_next=now+1.
            if reason:break
            time.sleep(interval)
    except Exception as error:
        reason=f'monitor failure: {type(error).__name__}'
        raise
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=1.)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
        record={'command':command,'returncode':child.returncode,'stop_reason':reason,
                'elapsed_seconds':time.monotonic()-start,'sampled_peak_phys_bytes':peak,'samples':samples,
                'memory_mib':memory_mib,'disk_reserve_mib':reserve_mib,'maximum_swap_growth_mib':swap_growth_mib,
                'initial_swap_bytes':baseline_swap,'last_sampled_swap_bytes':last_swap,
                'minimum_memory_free_percent':minimum_free_percent,'poll_seconds':interval}
        report.write_text(json.dumps(record,indent=2))
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',required=True)
    parser.add_argument('--memory-mib',type=int,default=2048)
    parser.add_argument('--reserve-mib',type=int,default=20480)
    parser.add_argument('--seconds',type=float,default=180)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args();command=args.command
    if command and command[0]=='--':command=command[1:]
    if not command:parser.error('a command is required after --')
    if min(args.memory_mib,args.reserve_mib,args.seconds)<=0:parser.error('limits must be positive')
    root=Path(__file__).resolve().parents[2]
    with (root/'.training.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('another guarded training run is active')
        result=run(command,args.report,memory_mib=args.memory_mib,reserve_mib=args.reserve_mib,seconds=args.seconds)
    print(json.dumps(result),flush=True)
    sys.exit(0 if result['returncode']==0 and result['stop_reason'] is None else 1)


if __name__=='__main__':main()
