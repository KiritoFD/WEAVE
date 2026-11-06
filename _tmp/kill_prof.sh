#!/bin/bash
pkill -f prof_render.py
sleep 2
nvidia-smi --query-gpu=memory.used --format=csv,noheader
