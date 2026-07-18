#!/bin/bash

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#SBATCH -p xahdnormal
#SBATCH -N 16
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=dcu:4
#SBATCH -J aquila
#SBATCH -o ./logs/%j.out
#SBATCH -e ./logs/%j.out
#SBATCH --exclusive


echo "START TIME: $(date)"
hostfile=./hostfile/$SLURM_JOB_ID
scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile}
rm `pwd`/hostfile-dl -f
for i in `cat $hostfile`
do
    echo ${i} slots=4 >> `pwd`/hostfile/hostfile-dl-$SLURM_JOB_ID
done
np=$(cat $hostfile|sort|uniq |wc -l)
np=$(($np*4))
nodename=$(cat $hostfile |sed -n "1p")
dist_url=`echo $nodename | awk '{print $1}'`

sleep 8m

mv `pwd`/hostfile/hostfile-dl-$SLURM_JOB_ID /work/home/zhaoying1/work/pr_code/FlagPerf-AI_platform/training/dcu/hosts
sed -i '/hosts = /d' config.py 
sed -i '$a\hosts = ["localhost"]' config.py 
sed -i "s/localhost/$dist_url/g" config.py
source ~/env/megatron.sh
python3 run.py
