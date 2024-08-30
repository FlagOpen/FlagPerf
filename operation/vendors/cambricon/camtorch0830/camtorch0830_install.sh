#!/bin/bash
mkdir -p $FLAGGEMS_WORK_DIR && cd $FLAGGEMS_WORK_DIR
rm -rf FlagGems
git clone https://mirror.ghproxy.com/https://github.com/FlagOpen/FlagGems.git
cd FlagGems
git checkout v2.0-perf-cambricon 
pip install -e .
/etc/init.d/ssh restart
