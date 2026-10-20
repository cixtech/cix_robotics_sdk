cmake_minimum_required(VERSION 3.10)
project(cix_ros_dmabuf LANGUAGES C CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -g)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# ---------------------------------------------------------------------------
# dmabuf broker library - cross-process zero-copy fd passing (UDS + SCM_RIGHTS)
#
# Pure POSIX C++ (no ROS deps): both publishers (camera capture) and
# subscribers (NPU inference nodes) link this without pulling in rclcpp.
# ---------------------------------------------------------------------------
add_library(${PROJECT_NAME} SHARED src/dmabuf_broker.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(${PROJECT_NAME} pthread)

# ---------------------------------------------------------------------------
# cix_ros_dmabuf_broker - standalone broker executable
#   cix_ros_dmabuf_broker [socket_path]   (default /tmp/cix_ros_dmabuf_broker.sock)
# ---------------------------------------------------------------------------
add_executable(cix_ros_dmabuf_broker src/dmabuf_broker_main.cpp)
target_link_libraries(cix_ros_dmabuf_broker ${PROJECT_NAME})

install(
  DIRECTORY include/
  DESTINATION include)

# Only the library goes into the EXPORT set: consumers link every exported
# target via ament_target_dependencies, and an executable is not linkable.
# The broker binary is installed on its own.
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  INCLUDES DESTINATION include)
install(TARGETS cix_ros_dmabuf_broker
  RUNTIME DESTINATION lib/${PROJECT_NAME})

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)

# Python client (BrokerClient / DmaBuffer counterpart of the C++ headers)
ament_python_install_package(${PROJECT_NAME})

ament_package()
