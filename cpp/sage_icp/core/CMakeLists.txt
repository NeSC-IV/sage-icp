# MIT License

# Copyright (c) 2023 NeSC-IV

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
find_package(PCL REQUIRED COMPONENTS common io registration filters keypoints features visualization range_image surface kdtree segmentation)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
add_compile_options(-std=c++17)
set(CMAKE_CXX_FLAGS "-std=c++17 -pthread -O3 ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") 
add_library(core STATIC)
add_library(sage_icp::core ALIAS core)
# add_library(ikd_Tree STATIC ikd_Tree.cpp)
# target_include_directories(ikd_Tree PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PCL_INCLUDE_DIRS})
target_sources(core PRIVATE Registration.cpp Deskew.cpp VoxelHashMap.cpp Preprocessing.cpp Threshold.cpp ikd_Tree.cpp)
target_link_libraries(core PUBLIC Eigen3::Eigen tsl::robin_map TBB::tbb Sophus::Sophus OpenMP::OpenMP_CXX ${PCL_LIBRARIES})
set_global_target_properties(core)
