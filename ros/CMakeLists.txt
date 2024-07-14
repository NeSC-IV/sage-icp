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

cmake_minimum_required(VERSION 3.16...3.26)
project(sage_icp VERSION 1.0.0 LANGUAGES CXX)

set(ignore ${CATKIN_INSTALL_INTO_PREFIX_ROOT})
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/sage_icp/)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cpp/sage_icp ${CMAKE_CURRENT_BINARY_DIR}/sage_icp)
else()
  message(FATAL_ERROR "SAGE-ICP not found, please clone the repository with --recursive flag")
  # cmake_minimum_required(VERSION 3.18)
  # message(STATUS "Performing out-of-tree build, fetching KISS-ICP v${CMAKE_PROJECT_VERSION} Release from Github")
  # include(FetchContent)
  # FetchContent_Declare(
  #   ext_kiss_icp_core PREFIX kiss_icp_core
  #   URL https://github.com/PRBonn/kiss-icp/archive/refs/tags/v${CMAKE_PROJECT_VERSION}.tar.gz SOURCE_SUBDIR
  #       cpp/sage_icp)
  # FetchContent_MakeAvailable(ext_kiss_icp_core)
endif()

if("$ENV{ROS_VERSION}" STREQUAL "1")
  message(FATAL_ERROR "SAGE-ICP-ROS1 not support!")
  
elseif("$ENV{ROS_VERSION}" STREQUAL "2")
  message(STATUS "SAGE-ICP-ROS2 wrapper will be compiled")

  find_package(ament_cmake REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(tf2_ros REQUIRED)
  find_package(example_interfaces REQUIRED)

  # ROS2 node
  add_executable(odometry_node ros2/OdometryServer.cpp)
  target_include_directories(odometry_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(odometry_node sage_icp::pipeline)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -msse3 -std=c++17") # debug
  install(TARGETS odometry_node RUNTIME DESTINATION lib/${PROJECT_NAME})
  install(DIRECTORY launch rviz DESTINATION share/${PROJECT_NAME}/)
  ament_target_dependencies(odometry_node rclcpp nav_msgs sensor_msgs tf2_ros example_interfaces)
  ament_package()
else()
  message(FATAL_ERROR "catkin or colcon not found SAGE-ICP-ROS disabled")
endif()
