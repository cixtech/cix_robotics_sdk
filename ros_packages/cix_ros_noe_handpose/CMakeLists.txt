cmake_minimum_required(VERSION 3.10)
project(cix_ros_noe_handpose)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# Message interfaces (HandPoseKeypoints)
# ---------------------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/HandPoseKeypoints.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# ---------------------------------------------------------------------------
# Python side (alternative Python inference node)
# ---------------------------------------------------------------------------
# The package ships a C++ zero-copy node (handpose_dmabuf_node, the primary,
# high-perf path) AND a Python inference stack (handpose_dma_teleop +
# handpose_gesture). rosidl_generate_interfaces already ran
# ament_python_install_package(${PROJECT_NAME}) for the generated msg
# bindings, so a second call here would collide with its CMake targets;
# instead the hand-written module files are installed into the same
# site-packages package dir directly, next to the generated msg/ subpackage.
# __init__.py is not installed - rosidl's generated one already makes the
# directory a package.
install(FILES
  cix_ros_noe_handpose/handpose_dma_teleop.py
  cix_ros_noe_handpose/handpose_gesture.py
  DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME})

# ros2-run-able entry point for the Python node (setup.py entry_points are
# not processed by the ament_cmake build type; a wrapper script is).
install(PROGRAMS scripts/handpose_dma_teleop
  DESTINATION lib/${PROJECT_NAME})

# ---------------------------------------------------------------------------
# C++ side: handpose_dmabuf_node (zero-copy NPU inference)
# ---------------------------------------------------------------------------
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ament_index_cpp REQUIRED)

# cix_ros_dmabuf provides the dmabuf broker client + dmabuf image headers (the
# cross-process zero-copy transport layer, shared with cix_ros_camera).
find_package(cix_ros_dmabuf REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# CIX NPU SDK (board-only). libnoe (cix-noe-umd) provides the low-level
# cix_noe_standard_api.h; libcme (cix-libcme-dev) provides the CME 2D
# accelerator (crop/resize/cvtcolor + normalize) used for zero-copy preprocess
# directly into the NPU input dmabuf. None is present on the host build mount,
# so this package builds on the board. Install libcme dev headers/lib:
#   sudo dpkg -i cix-libcme-dev_1.1.0_arm64.deb
# Note: libcme.pc sets -I/usr/include/cme (for #include <cme.h>), but we mirror
# the cix_yolo_demo convention: -I/usr/include + #include <cme/cme.h>.
set(NOE_ROOT "/usr/share/cix")
include_directories(${NOE_ROOT}/include)
include_directories(${NOE_ROOT}/include/npu)
link_directories(${NOE_ROOT}/lib)
include_directories("/usr/include")
link_directories("/usr/lib/aarch64-linux-gnu")

add_executable(handpose_dmabuf_node
  src/handpose_dmabuf_node.cpp
)

# Generated C++ headers for the interfaces defined in this package.
target_include_directories(handpose_dmabuf_node PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp)

ament_target_dependencies(handpose_dmabuf_node
  rclcpp
  rclcpp_components
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
  OpenCV
  ament_index_cpp
  cix_ros_dmabuf
)

# Link the typesupport library of this package's own interfaces.
rosidl_get_typesupport_target(${PROJECT_NAME}_typesupport_cpp
  ${PROJECT_NAME} rosidl_typesupport_cpp)

target_link_libraries(handpose_dmabuf_node
  ${OpenCV_LIBRARIES}
  cme
  pthread dl m rt
  ${NOE_ROOT}/lib/libnoe.so
  "${${PROJECT_NAME}_typesupport_cpp}"
)

# RPATH so the executable finds the NPU libs and ROS libs at runtime.
# Derive the ROS 2 install prefix from the ament package we build against
# (e.g. /opt/ros/humble) instead of hardcoding the distro path.
get_filename_component(ROS_PREFIX "${rclcpp_DIR}/../../.." ABSOLUTE)
set_target_properties(handpose_dmabuf_node PROPERTIES
  INSTALL_RPATH "${NOE_ROOT}/lib;${ROS_PREFIX}/lib;${ROS_PREFIX}/lib/aarch64-linux-gnu"
  BUILD_RPATH "${NOE_ROOT}/lib;${ROS_PREFIX}/lib;${ROS_PREFIX}/lib/aarch64-linux-gnu"
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Register as a ROS 2 component
rclcpp_components_register_node(handpose_dmabuf_node
  PLUGIN "cix_ros_noe_handpose::HandPoseDmaBufNode"
  EXECUTABLE handpose_dmabuf_node_component
)

install(TARGETS handpose_dmabuf_node
  DESTINATION lib/${PROJECT_NAME}
)

# Data files (launch, config, models). The Python module install is handled
# above; there is no setup.py (ament_cmake build type never runs one).
install(DIRECTORY launch config models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
