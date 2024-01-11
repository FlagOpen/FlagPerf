#!/bin/bash
# using github mirrors to avoid github TTL
git clone https://githubfast.com/FlagOpen/FlagScale
echo 'export PYTHONPATH=$PYTHONPATH:/workspace/FlagScale' >> /root/.bashrc
source /root/.bashrc