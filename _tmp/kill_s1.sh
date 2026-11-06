#!/usr/bin/env bash
pkill -f 'run.py --config configs/v4' && echo stopped || echo no_proc
sleep 2
pgrep -af run.py || echo no_run
