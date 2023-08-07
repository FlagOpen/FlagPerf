#!/bin/bash

SDK_DIR="/workspace/docker_image/sdk_installers"
PKG_DIR="/workspace/docker_image/packages"

search_cuda_results=`find ${SDK_DIR} -name "partial_install_cuda_header.tar.gz"`
for installer in $search_cuda_results; do
    echo "Install ${installer}"
    tar zxvf ${installer}
    sh "$(echo $(basename ${installer}) | cut -d . -f1)/install-cuda-header.sh" -- --silent --toolkit
    rm -rf "$(echo $(basename ${installer}) | cut -d . -f1)"
done

search_sdk_results=`find ${SDK_DIR} -name "corex*.run"`
for installer in $search_sdk_results; do
    echo "Install ${installer}"
    sh "${installer}" -- --silent --driver --toolkit
done

search_packages_results=`find ${PKG_DIR} -name "*.whl"`
for pkg in $search_packages_results; do
    echo "Install ${pkg}"
    pip3 install "${pkg}"
done

