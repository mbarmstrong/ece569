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
CMAKE_SOURCE_DIR = /home/u30/mbarmstrong/ece569/labs

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/u30/mbarmstrong/ece569/build_dir

# Include any dependencies generated for this target.
include CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/flags.make

CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/hw2/ImageColorToGrayscale/dataset_generator.cpp.o: CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/flags.make
CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/hw2/ImageColorToGrayscale/dataset_generator.cpp.o: /home/u30/mbarmstrong/ece569/labs/hw2/ImageColorToGrayscale/dataset_generator.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/u30/mbarmstrong/ece569/build_dir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/hw2/ImageColorToGrayscale/dataset_generator.cpp.o"
	/cm/local/apps/gcc/6.1.0/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/hw2/ImageColorToGrayscale/dataset_generator.cpp.o -c /home/u30/mbarmstrong/ece569/labs/hw2/ImageColorToGrayscale/dataset_generator.cpp

CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/hw2/ImageColorToGrayscale/dataset_generator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/hw2/ImageColorToGrayscale/dataset_generator.cpp.i"
	/cm/local/apps/gcc/6.1.0/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/u30/mbarmstrong/ece569/labs/hw2/ImageColorToGrayscale/dataset_generator.cpp > CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/hw2/ImageColorToGrayscale/dataset_generator.cpp.i

CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/hw2/ImageColorToGrayscale/dataset_generator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/hw2/ImageColorToGrayscale/dataset_generator.cpp.s"
	/cm/local/apps/gcc/6.1.0/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/u30/mbarmstrong/ece569/labs/hw2/ImageColorToGrayscale/dataset_generator.cpp -o CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/hw2/ImageColorToGrayscale/dataset_generator.cpp.s

CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/hw2/ImageColorToGrayscale/dataset_generator.cpp.o.requires:

.PHONY : CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/hw2/ImageColorToGrayscale/dataset_generator.cpp.o.requires

CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/hw2/ImageColorToGrayscale/dataset_generator.cpp.o.provides: CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/hw2/ImageColorToGrayscale/dataset_generator.cpp.o.requires
	$(MAKE) -f CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/build.make CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/hw2/ImageColorToGrayscale/dataset_generator.cpp.o.provides.build
.PHONY : CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/hw2/ImageColorToGrayscale/dataset_generator.cpp.o.provides

CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/hw2/ImageColorToGrayscale/dataset_generator.cpp.o.provides.build: CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/hw2/ImageColorToGrayscale/dataset_generator.cpp.o


# Object files for target ImageColorToGrayscale_DatasetGenerator
ImageColorToGrayscale_DatasetGenerator_OBJECTS = \
"CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/hw2/ImageColorToGrayscale/dataset_generator.cpp.o"

# External object files for target ImageColorToGrayscale_DatasetGenerator
ImageColorToGrayscale_DatasetGenerator_EXTERNAL_OBJECTS =

ImageColorToGrayscale_DatasetGenerator: CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/hw2/ImageColorToGrayscale/dataset_generator.cpp.o
ImageColorToGrayscale_DatasetGenerator: CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/build.make
ImageColorToGrayscale_DatasetGenerator: libwb.a
ImageColorToGrayscale_DatasetGenerator: /cm/shared/apps/cuda91/toolkit/9.1.85/lib64/libcudart_static.a
ImageColorToGrayscale_DatasetGenerator: /usr/lib64/librt.so
ImageColorToGrayscale_DatasetGenerator: CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/u30/mbarmstrong/ece569/build_dir/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ImageColorToGrayscale_DatasetGenerator"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/build: ImageColorToGrayscale_DatasetGenerator

.PHONY : CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/build

CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/requires: CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/hw2/ImageColorToGrayscale/dataset_generator.cpp.o.requires

.PHONY : CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/requires

CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/clean

CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/depend:
	cd /home/u30/mbarmstrong/ece569/build_dir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/u30/mbarmstrong/ece569/labs /home/u30/mbarmstrong/ece569/labs /home/u30/mbarmstrong/ece569/build_dir /home/u30/mbarmstrong/ece569/build_dir /home/u30/mbarmstrong/ece569/build_dir/CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ImageColorToGrayscale_DatasetGenerator.dir/depend

