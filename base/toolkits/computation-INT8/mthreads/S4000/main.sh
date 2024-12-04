#!/bin/bash

CXX=g++
MCC=mcc
CXXFLAGS="-std=c++17 -I./include -I/usr/local/musa/include -fPIC"
MCCFLAGS="-std=c++17 --offload-arch=mp_22 -I../include -fPIC -I./include -I/usr/local/musa/include"
LDFLAGS="-lmusart -L/usr/local/musa/lib"

SRC_DIR=src
BUILD_DIR=build
EXECUTABLE=gemm


mkdir -p $BUILD_DIR

$CXX $CXXFLAGS -c $SRC_DIR/common.cpp -o $BUILD_DIR/common.o
$CXX $CXXFLAGS -c $SRC_DIR/logger.cpp -o $BUILD_DIR/logger.o
$CXX $CXXFLAGS -c $SRC_DIR/benchmark_int8.cpp -o $BUILD_DIR/benchmark.o

$MCC $MCCFLAGS -c $SRC_DIR/compute_mma_int8.mu -o $BUILD_DIR/compute_mma_int8.o

$CXX $CXXFLAGS -c $SRC_DIR/main.cpp -o $BUILD_DIR/main.o

$CXX $CXXFLAGS $BUILD_DIR/*.o -o $EXECUTABLE $LDFLAGS

./gemm