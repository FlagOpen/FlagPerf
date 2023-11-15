#!/bin/bash

SDK_DIR="/workspace/docker_image/sdk_installers"
PKG_DIR="/workspace/docker_image/packages"


search_cuda_results=`find ${SDK_DIR} -name "*cuda*10.2*.run"`
for installer in $search_cuda_results; do
    echo "Install ${installer}"
    sh "${installer}" -- --silent --toolkit
done

search_sdk_results=`find ${SDK_DIR} -name "corex*.run"`
for installer in $search_sdk_results; do
    echo "Install ${installer}"
    sh "${installer}" -- --silent --toolkit
done

torch_packages_results=`find ${PKG_DIR} -name "torch-*.whl"`
if [ -n "$torch_packages_results" ]; then    
    pip3 install "$torch_packages_results"
fi

search_packages_results=`find ${PKG_DIR} -name "*.whl"`
for pkg in $search_packages_results; do
    echo "Install ${pkg}"
    pip3 install "${pkg}"
done



