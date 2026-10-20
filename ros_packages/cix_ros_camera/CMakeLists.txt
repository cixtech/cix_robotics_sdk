cmake_minimum_required(VERSION 3.10)
project(cix_ros_camera LANGUAGES C CXX)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -g)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

include_directories(include)

# ---------------------------------------------------------------------------
# V4L2 dmabuf camera - standalone V4L2 capture with dmabuf zero-copy
# ---------------------------------------------------------------------------
add_executable(v4l2_dmabuf_camera
  src/v4l2_dmabuf_camera.cpp)

target_include_directories(v4l2_dmabuf_camera PUBLIC include)

ament_target_dependencies(v4l2_dmabuf_camera
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs
  cix_ros_dmabuf)

target_link_libraries(v4l2_dmabuf_camera
  pthread dl m rt)

install(TARGETS v4l2_dmabuf_camera
  DESTINATION lib/${PROJECT_NAME})

# ---------------------------------------------------------------------------
# Test subscriber component (always built; lightweight)
# cix_ros_dmabuf comes in via ament_auto_find_build_dependencies (package.xml
# <depend>): sub_test_node.cpp includes dmabuf_type_adapter.hpp, whose
# thread_local BrokerClient needs the BrokerClient symbols; without linking
# the dmabuf library the .so has undefined BrokerClient symbols and fails to
# dlopen.
# ---------------------------------------------------------------------------
ament_auto_add_library(cix_ros_camera_sub_test SHARED
  test/src/sub_test_node.cpp)

rclcpp_components_register_nodes(cix_ros_camera_sub_test
  "cix_ros_camera::SubTestNode")

ament_auto_package(INSTALL_TO_SHARE launch)
