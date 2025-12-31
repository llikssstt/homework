cmake_minimum_required(VERSION 3.5)
project(quadrotor_df)

find_package(catkin REQUIRED COMPONENTS
  roscpp
)

find_package(Eigen3 REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(df_quaternion_node src/df_quaternion_node.cpp)
target_link_libraries(df_quaternion_node ${catkin_LIBRARIES})


