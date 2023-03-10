#!/bin/bash

SDK_DIR="/installer/sdk_installers"
PKG_DIR="/installer/packages"


search_cuda_results=`find ${SDK_DIR} -name "*cuda*10.2*.run"`
for installer in $search_cuda_results; do
    echo "Install ${installer}"
    sh "${installer}" -- --silent --toolkit
done

search_sdk_results=`find ${SDK_DIR} -name "corex*.run"`
for installer in $search_sdk_results; do
    echo "Install ${installer}"
    sh "${installer}" -- --silent --cudapath=/usr/local/cuda
done

search_packages_results=`find ${PKG_DIR} -name "*.whl"`
for pkg in $search_packages_results; do
    echo "Install ${pkg}"
    pip3 install "${pkg}"
done



