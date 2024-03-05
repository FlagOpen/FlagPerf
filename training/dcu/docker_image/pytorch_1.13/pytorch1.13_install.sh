#!/bin/bash
proxy_server_ip=10.0.35.251
PROXY_URL="http://$proxy_server_ip:3128/"
NO_PROXY_ADDR="127.0.0.1,localhost,.local,.cluster.local,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export no_proxy="$NO_PROXY_ADDR"
export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
export NO_PROXY="$NO_PROXY_ADDR"