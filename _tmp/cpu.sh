#!/usr/bin/env bash
echo "== nproc"; nproc
echo "== cpuinfo processors"; grep -c ^processor /proc/cpuinfo
echo "== mem"; free -g
echo "== load"; uptime
