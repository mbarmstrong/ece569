# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake3

# The command to remove a file.
RM = /usr/bin/cmake3 -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/u30/mbarmstrong/ece569/shadow_removal

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/u30/mbarmstrong/ece569/build_dir

# Include any dependencies generated for this target.
include CMakeFiles/ColorConvertUnitTest_Solution.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ColorConvertUnitTest_Solution.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ColorConvertUnitTest_Solution.dir/flags.make

CMakeFiles/ColorConvertUnitTest_Solution.dir/src/color_conversion/ColorConvertUnitTest_Solution_generated_unit_test.cu.o: CMakeFiles/ColorConvertUnitTest_Solution.dir/src/color_conversion/ColorConvertUnitTest_Solution_generated_unit_test.cu.o.depend
CMakeFiles/ColorConvertUnitTest_Solution.dir/src/color_conversion/ColorConvertUnitTest_Solution_generated_unit_test.cu.o: CMakeFiles/ColorConvertUnitTest_Solution.dir/src/color_conversion/ColorConvertUnitTest_Solution_generated_unit_test.cu.o.Debug.cmake
CMakeFiles/ColorConvertUnitTest_Solution.dir/src/color_conversion/ColorConvertUnitTest_Solution_generated_unit_test.cu.o: /home/u30/mbarmstrong/ece569/shadow_removal/src/color_conversion/unit_test.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/u30/mbarmstrong/ece569/build_dir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/ColorConvertUnitTest_Solution.dir/src/color_conversion/ColorConvertUnitTest_Solution_generated_unit_test.cu.o"
	cd /home/u30/mbarmstrong/ece569/build_dir/CMakeFiles/ColorConvertUnitTest_Solution.dir/src/color_conversion && /usr/bin/cmake3 -E make_directory /home/u30/mbarmstrong/ece569/build_dir/CMakeFiles/ColorConvertUnitTest_Solution.dir/src/color_conversion/.
	cd /home/u30/mbarmstrong/ece569/build_dir/CMakeFiles/ColorConvertUnitTest_Solution.dir/src/color_conversion && /usr/bin/cmake3 -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Debug -D generated_file:STRING=/home/u30/mbarmstrong/ece569/build_dir/CMakeFiles/ColorConvertUnitTest_Solution.dir/src/color_conversion/./ColorConvertUnitTest_Solution_generated_unit_test.cu.o -D generated_cubin_file:STRING=/home/u30/mbarmstrong/ece569/build_dir/CMakeFiles/ColorConvertUnitTest_Solution.dir/src/color_conversion/./ColorConvertUnitTest_Solution_generated_unit_test.cu.o.cubin.txt -P /home/u30/mbarmstrong/ece569/build_dir/CMakeFiles/ColorConvertUnitTest_Solution.dir/src/color_conversion/ColorConvertUnitTest_Solution_generated_unit_test.cu.o.Debug.cmake

# Object files for target ColorConvertUnitTest_Solution
ColorConvertUnitTest_Solution_OBJECTS =

# External object files for target ColorConvertUnitTest_Solution
ColorConvertUnitTest_Solution_EXTERNAL_OBJECTS = \
"/home/u30/mbarmstrong/ece569/build_dir/CMakeFiles/ColorConvertUnitTest_Solution.dir/src/color_conversion/ColorConvertUnitTest_Solution_generated_unit_test.cu.o"

ColorConvertUnitTest_Solution: CMakeFiles/ColorConvertUnitTest_Solution.dir/src/color_conversion/ColorConvertUnitTest_Solution_generated_unit_test.cu.o
ColorConvertUnitTest_Solution: CMakeFiles/ColorConvertUnitTest_Solution.dir/build.make
ColorConvertUnitTest_Solution: /cm/shared/apps/cuda91/toolkit/9.1.85/lib64/libcudart_static.a
ColorConvertUnitTest_Solution: /usr/lib64/librt.so
ColorConvertUnitTest_Solution: libwb.a
ColorConvertUnitTest_Solution: /cm/shared/apps/cuda91/toolkit/9.1.85/lib64/libcudart_static.a
ColorConvertUnitTest_Solution: /usr/lib64/librt.so
ColorConvertUnitTest_Solution: CMakeFiles/ColorConvertUnitTest_Solution.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/u30/mbarmstrong/ece569/build_dir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ColorConvertUnitTest_Solution"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ColorConvertUnitTest_Solution.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ColorConvertUnitTest_Solution.dir/build: ColorConvertUnitTest_Solution

.PHONY : CMakeFiles/ColorConvertUnitTest_Solution.dir/build

CMakeFiles/ColorConvertUnitTest_Solution.dir/requires:

.PHONY : CMakeFiles/ColorConvertUnitTest_Solution.dir/requires

CMakeFiles/ColorConvertUnitTest_Solution.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ColorConvertUnitTest_Solution.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ColorConvertUnitTest_Solution.dir/clean

CMakeFiles/ColorConvertUnitTest_Solution.dir/depend: CMakeFiles/ColorConvertUnitTest_Solution.dir/src/color_conversion/ColorConvertUnitTest_Solution_generated_unit_test.cu.o
	cd /home/u30/mbarmstrong/ece569/build_dir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/u30/mbarmstrong/ece569/shadow_removal /home/u30/mbarmstrong/ece569/shadow_removal /home/u30/mbarmstrong/ece569/build_dir /home/u30/mbarmstrong/ece569/build_dir /home/u30/mbarmstrong/ece569/build_dir/CMakeFiles/ColorConvertUnitTest_Solution.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ColorConvertUnitTest_Solution.dir/depend
