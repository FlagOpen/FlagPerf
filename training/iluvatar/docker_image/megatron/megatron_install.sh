#!/bin/bash

SDK_DIR="/workspace/docker_image/sdk_installers"
PKG_DIR="/workspace/docker_image/packages"

search_sdk_results=`find ${SDK_DIR} -name "corex*.run"`
for installer in $search_sdk_results; do
    echo "Install ${installer}"
    sh "${installer}" -- --silent --toolkit
done



