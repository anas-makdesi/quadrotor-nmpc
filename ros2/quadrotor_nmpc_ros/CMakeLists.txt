cmake_minimum_required(VERSION 3.16)
project(quadrotor_nmpc_ros C CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(quadrotor_nmpc_msgs REQUIRED)

# ---- acados (built from source; set ACADOS_SOURCE_DIR) ---------------------------
if(NOT ACADOS_SOURCE_DIR)
  set(ACADOS_SOURCE_DIR $ENV{ACADOS_SOURCE_DIR})
endif()
if(NOT ACADOS_SOURCE_DIR)
  message(FATAL_ERROR "Set ACADOS_SOURCE_DIR (cmake -D or environment) to the acados folder")
endif()
find_library(ACADOS_LIB acados PATHS ${ACADOS_SOURCE_DIR}/lib REQUIRED NO_DEFAULT_PATH)
find_library(HPIPM_LIB hpipm PATHS ${ACADOS_SOURCE_DIR}/lib REQUIRED NO_DEFAULT_PATH)
find_library(BLASFEO_LIB blasfeo PATHS ${ACADOS_SOURCE_DIR}/lib REQUIRED NO_DEFAULT_PATH)

# ---- generated solver (python scripts/export_c_code.py) ---------------------------
set(GEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generated/c_generated_code)
if(NOT EXISTS ${GEN_DIR}/acados_solver_quadrotor_nmpc.c)
  message(FATAL_ERROR "Generated solver not found in ${GEN_DIR}: run scripts/export_c_code.py")
endif()
file(GLOB GEN_SOURCES
  ${GEN_DIR}/acados_solver_quadrotor_nmpc.c
  ${GEN_DIR}/quadrotor_nmpc_model/*.c
  ${GEN_DIR}/quadrotor_nmpc_constraints/*.c
  ${GEN_DIR}/quadrotor_nmpc_cost/*.c)

add_library(quadrotor_nmpc_solver STATIC ${GEN_SOURCES} src/nmpc_solver.cpp)
set_target_properties(quadrotor_nmpc_solver PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(quadrotor_nmpc_solver PRIVATE
  $<$<COMPILE_LANGUAGE:C>:-O2 -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable>)
target_include_directories(quadrotor_nmpc_solver PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${CMAKE_CURRENT_SOURCE_DIR}/generated
  ${GEN_DIR}
  ${ACADOS_SOURCE_DIR}/include
  ${ACADOS_SOURCE_DIR}/include/acados
  ${ACADOS_SOURCE_DIR}/include/blasfeo/include
  ${ACADOS_SOURCE_DIR}/include/hpipm/include)
target_link_libraries(quadrotor_nmpc_solver PUBLIC ${ACADOS_LIB} ${HPIPM_LIB} ${BLASFEO_LIB} m)
# libacados pulls in libqpOASES_e/libhpipm/libblasfeo from the same folder: use a classic
# RPATH (searched for indirect dependencies too) rather than RUNPATH
target_link_options(quadrotor_nmpc_solver INTERFACE "-Wl,--disable-new-dtags")

add_executable(nmpc_node src/nmpc_node.cpp)
target_link_libraries(nmpc_node quadrotor_nmpc_solver)
ament_target_dependencies(nmpc_node rclcpp quadrotor_nmpc_msgs)

add_executable(sim_node src/sim_node.cpp)
target_include_directories(sim_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/generated)
ament_target_dependencies(sim_node rclcpp quadrotor_nmpc_msgs)

# the nodes find libacados at run time through this rpath
set_target_properties(nmpc_node sim_node PROPERTIES
  INSTALL_RPATH "${ACADOS_SOURCE_DIR}/lib" BUILD_RPATH "${ACADOS_SOURCE_DIR}/lib")

install(TARGETS nmpc_node sim_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)
  set(FIX ${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures)
  ament_add_gtest(test_model test/test_model.cpp)
  target_include_directories(test_model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/generated)
  target_compile_definitions(test_model PRIVATE FIXTURES="${FIX}")
  ament_add_gtest(test_solver test/test_solver.cpp)
  target_link_libraries(test_solver quadrotor_nmpc_solver)
  target_compile_definitions(test_solver PRIVATE FIXTURES="${FIX}")
  set_target_properties(test_model test_solver PROPERTIES BUILD_RPATH "${ACADOS_SOURCE_DIR}/lib")
  add_launch_test(test/test_closed_loop.py TIMEOUT 90)
endif()

ament_package()
