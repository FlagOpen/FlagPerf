#!/bin/bash

TOOL=xpu-smi
LOG=_${TOOL}.log.$$
PERF=/opt/xre/bin/$TOOL

mem=$($PERF -m | head -1 | awk '{print $19}')
echo "[FlagPerf Result] main_memory-capacity=$mem MiB"
