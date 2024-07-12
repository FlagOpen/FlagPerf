#!/bin/bash
current_dir=$(pwd)
echo "=====>$current_dir"
script_dir=$(dirname "$(realpath "$0")")
echo "script dir :$script_dir"

cd /workspace/docker_image/FlagGems
#git clone https://mirror.ghproxy.com/https://github.com/FlagOpen/FlagGems.git
#git checkout .
pip3 install .