#!/bin/bash
# using github mirrors to avoid github timeout issue
commit_id="ed55532"
git clone https://githubfast.com/FlagOpen/FlagScale
cd FlagScale
git reset --hard ${commit_id}
echo 'export PYTHONPATH=$PYTHONPATH:/workspace/FlagScale' >> /root/.bashrc
source /root/.bashrc
