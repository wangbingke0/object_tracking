cmake_minimum_required(VERSION 2.8.3)
project(object_tracking)

set(CMAKE_CXX_STANDARD 14)

find_package(yaml-cpp REQUIRED)

# Core ROS dependencies (always required)
find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Check if planning_msgs is available (optional, for original cluster/tracking)
set(HAS_PLANNING_MSGS FALSE)
execute_process(
  COMMAND rospack find planning_msgs
  RESULT_VARIABLE _planning_msgs_result
  OUTPUT_QUIET ERROR_QUIET
)
if(_planning_msgs_result EQUAL 0)
  set(HAS_PLANNING_MSGS TRUE)
  message(STATUS "planning_msgs found -- building all targets")
else()
  message(STATUS "planning_msgs NOT found -- only building nuscenes_tracking target")
endif()

###################################
## catkin specific configuration ##
###################################

catkin_package(
  CATKIN_DEPENDS std_msgs sensor_msgs
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/../../../../devel/include
)

# =============================================
# Original cluster + tracking targets
# (require planning_msgs from the full workspace)
# =============================================
if(HAS_PLANNING_MSGS)

set(SOURCE_FILES2 src/cluster/main.cpp src/cluster/component_clustering.cpp src/cluster/box_fitting.cpp)
add_executable(cluster ${SOURCE_FILES2})
add_dependencies(cluster ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(cluster ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} yaml-cpp)

set(SOURCE_FILES3 src/tracking/main.cpp src/tracking/imm_ukf_jpda.cpp src/tracking/ukf.cpp)
add_executable(tracking ${SOURCE_FILES3})
add_dependencies(tracking ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(tracking ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} yaml-cpp)

endif()

# =============================================
# nuScenes offline tracking target
# (no planning_msgs dependency)
# =============================================
set(SOURCE_FILES_NUSCENES
  src/tracking/nuscenes_main.cpp
  src/tracking/imm_ukf_jpda.cpp
  src/tracking/ukf.cpp
)
add_executable(nuscenes_tracking ${SOURCE_FILES_NUSCENES})
add_dependencies(nuscenes_tracking ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(nuscenes_tracking ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
