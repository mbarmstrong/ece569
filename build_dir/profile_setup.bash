#!/bin/bash
# run this script in Ocelote dekstop or in interactive session

# this script setups the project in the build_dir 
# first cleans all Makefiles
# then loads cuda modules and compiles
# then runs profiler for the vector addition (hw2)
# to generate profile.txt, profiledata and profile.timeline
# launches the nvvp

# observe the content of profile.txt
# load profiledata and profile.timeline into the nvvp

# 
#before executing 
#suggest commenting out line 82: set(CUDA NVCC_FLAGS_DEBUG....
#in labs/CMakefiles.txt

#verified with two versions of the cuda toolkit. 10.0 is commented out. 

#make sure that you use your home directory path (line 26) and 
#input and output files needed by vector addition are in the build_dir. 

cd ~mbarmstrong/ece569/build_dir/
echo "cleaning makefiles"
make clean
rm cmake_install.cmake 
rm Makefile 
rm CMakeCache.txt
rm -rf CMakeFiles

###echo "loading cuda10.0 modules"
###module load cuda10.0/nsight/10.0.130
###module load cuda10.0/profiler/10.0.130
###module load cuda10.0/toolkit/10.0.130

echo "loading cuda9.1 modules"
module load cuda91/toolkit/9.1.85
module load cuda91/nsight/9.1.85
module load cuda91/profiler/9.1.85

echo "setting up CMake"
CC=gcc cmake3 ../labs
make

echo "cleaning profile data"
rm profile.txt
rm profile.timeline
rm profiledata

echo "generating profile1.txt"
# make sure the input0.raw, input1.raw and output.raw files are in the build_dir directory. 
nvprof --log-file profile1.txt ./BasicMatrixMultiplication_Solution -e output.raw -i input0.raw,input1.raw -t vector > vout6.txt

echo "generating profile1.timeline"
nvprof -o profile1.timeline ./BasicMatrixMultiplication_Solution -e output.raw -i input0.raw,input1.raw -t vector > vout7.txt

echo "generating profiledata1 for nvvp"
nvprof --export-profile profiledata1 ./BasicMatrixMultiplication_Solution -e output.raw -i input0.raw,input1.raw -t vector > vout8.txt

### if using 10.0
###echo "starting nvvp, you need to load profiledata and profile.timeline files"
###/cm/shared/apps/cuda10.0/toolkit/10.0.130/libnvvp/nvvp &

###if using 9.1.85
echo "starting nvvp, you need to load profiledata and profile.timeline files"
/cm/shared/apps/cuda91/toolkit/9.1.85/libnvvp/nvvp &





