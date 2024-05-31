#!/bin/bash
# using github mirrors to avoid github TTL
git clone https://githubfast.com/FlagOpen/FlagScale
cd FlagScale
git checkout 26cd6643c472f853e077779abaa51bb6a1c140bf
echo 'export PYTHONPATH=$PYTHONPATH:/workspace/FlagScale' >> /root/.bashrc
source /root/.bashrc
