"""V17 serial queue manager: manifest order, DONE admission, retry<=2,
repeat-pool consumption to 36h, telemetry, watchdog hooks.

Run inside the persistent tmux session (single-GPU discipline):
  python scripts/v17/queue.py --manifest exp/v17/manifest_36h.json
State: runs/v17/queue_state.json (atomic).  A resume is a restart of this
process; DONE.json presence + hash admission prevents duplicate work.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/v17'
REPO = ROOT


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, default=str),
                   encoding='utf-8')
    tmp.replace(path)


def done_ok(arm_dir: Path) -> bool:
    """Eval arms: valid DONE.json (750/600/150).  Other arms: result.json
    with ok=True written by run.py's verifier path."""
    d = arm_dir / 'DONE.json'
    if d.exists():
        try:
            payload = read(d)
            assert payload.get('n_tasks') == 750
            assert payload.get('n_cross') == 600 and payload.get('n_same') == 150
            return True
        except Exception:
            return False
    r = arm_dir / 'result.json'
    if r.exists():
        try:
            return bool(read(r).get('ok'))
        except Exception:
            return False
    return False


def run_arm(arm_id: str, cfg: dict, seed: int, out_dir: Path, log_fh,
            dry=False) -> int:
    cfg_path = out_dir / 'config_resolved.json'
    cfg['arm_id'] = arm_id
    save(cfg_path, cfg)
    cmd = [sys.executable, '-u', '-m', 'exp.v17.run', '--config',
           str(cfg_path), '--seed', str(seed), '--resume', 'auto',
           '--out', str(out_dir)]
    if dry:
        cmd.append('--dry-run')
    log_fh.write(f'[{utc()}] START {arm_id} seed={seed}\n')
    log_fh.flush()
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding='utf-8', errors='replace')
    t0 = time.time()
    for line in proc.stdout:
        log_fh.write(line)
        log_fh.flush()
    code = proc.wait()
    log_fh.write(f'[{utc()}] END {arm_id} rc={code} '
                 f'{(time.time()-t0)/60:.1f}min\n')
    log_fh.flush()
    return code


def elapsed_valid_hours(state) -> float:
    return state.get('valid_task_minutes', 0.) / 60.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--dry-run-first', action='store_true')
    a = ap.parse_args()
    man = read(a.manifest)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'logs').mkdir(exist_ok=True)
    state_path = OUT / 'queue_state.json'
    state = read(state_path) if state_path.exists() else {
        'started_utc': utc(), 'valid_task_minutes': 0., 'arms': {}}
    if a.check:
        n = len(man['main_arms'])
        print(f'manifest OK: {n} main arms, repeat pool '
              f'{man["repeat_pool"]["arms"]} x seeds '
              f'{len(man["repeat_pool"]["seeds"])}')
        return

    log_fh = (OUT / 'logs' / 'queue.log').open('a', encoding='utf-8')
    log_fh.write(f'[{utc()}] queue start pid={os.getpid()}\n')
    log_fh.flush()

    def mark(arm_id, **kv):
        state['arms'][arm_id] = {**state['arms'].get(arm_id, {}), **kv}
        state['valid_task_minutes'] = sum(
            v.get('minutes', 0.) for v in state['arms'].values())
        save(state_path, state)

    def queue_iter():
        """Main arms in order, then round-robin repeat pool, then long-run."""
        for a_ in man['main_arms']:
            yield a_['arm_id'], a_['config'], a_['seed']
        pool = man['repeat_pool']
        for s in pool['seeds']:
            for arm_name in pool['arms']:
                rid = f'R_{arm_name}_s{s}'
                cfg = dict(next((m['config'] for m in man['main_arms']
                                 if m['arm_id'].endswith(f'{arm_name}_train')
                                 or m['arm_id'] == f'P2_{arm_name}_train'),
                                {'family': 'student', 'kind': arm_name,
                                 'pool': man['main_arms'][0]['config']['pool']}))
                yield rid, cfg, s
        for lr in man['long_run']['arms']:
            for upd in man['long_run']['updates']:
                for s in man['long_run']['seeds']:
                    rid = f'LR_{lr}_u{upd}_s{s}'
                    base_cfg = next((m['config'] for m in man['main_arms']
                                     if m['arm_id'] == f'P2_{lr}_train'),
                                    {'family': 'student', 'kind': lr})
                    cfg = {**base_cfg, 'updates': upd}
                    yield rid, cfg, s

    target = man['hours']['target']
    floor = man['hours']['min_valid']
    for arm_id, cfg, seed in queue_iter():
        out_dir = OUT / arm_id
        if done_ok(out_dir):
            log_fh.write(f'[{utc()}] SKIP {arm_id} (DONE)\n')
            log_fh.flush()
            continue
        st = state['arms'].get(arm_id, {})
        if st.get('retries', 0) >= 2:
            log_fh.write(f'[{utc()}] SKIP {arm_id} (retries exhausted)\n')
            log_fh.flush()
            continue
        if elapsed_valid_hours(state) >= target:
            log_fh.write(f'[{utc()}] STOP launching: {elapsed_valid_hours(state):.2f}h '
                         f'>= {target}h target\n')
            log_fh.flush()
            break
        mark(arm_id, state='running', utc_start=utc(), seed=seed)
        code = run_arm(arm_id, cfg, seed, out_dir, log_fh,
                       dry=a.dry_run_first and arm_id == state.get(
                           'first_arm', arm_id))
        minutes = 0.
        if 'utc_start' in state['arms'][arm_id]:
            t_start = datetime.datetime.fromisoformat(
                state['arms'][arm_id]['utc_start'])
            minutes = (datetime.datetime.now(datetime.timezone.utc) -
                       t_start).total_seconds() / 60.
        if code == 0 and done_ok(out_dir):
            mark(arm_id, state='complete', exit_code=0, minutes=minutes)
        else:
            mark(arm_id, state='failed', exit_code=code, minutes=minutes,
                 retries=st.get('retries', 0) + 1)
    log_fh.write(f'[{utc()}] queue exit: valid '
                 f'{elapsed_valid_hours(state):.2f}h '
                 f'(floor {floor}h, target {target}h)\n')
    log_fh.flush()


if __name__ == '__main__':
    main()
