#!/bin/bash

config_path=$1

echo "config=$config_path"

./tsmPerf -r gemmPerf -c $config_path -d chip_out/ -f 0

