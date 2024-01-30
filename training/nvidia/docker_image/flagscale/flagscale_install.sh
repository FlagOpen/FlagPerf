#!/bin/bash
# using github mirrors to avoid github timeout issue
git clone https://githubfast.com/FlagOpen/FlagScale
cd FlagScale
git checkout -b FlagScale ed55532
echo 'export PYTHONPATH=$PYTHONPATH:/workspace/FlagScale' >> /root/.bashrc
source /root/.bashrc
