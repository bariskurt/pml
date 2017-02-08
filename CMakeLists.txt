cmake_minimum_required(VERSION 2.8)

project(pml)

SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin )

SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Werror -std=c++11" )

add_subdirectory(test/)

# testing
enable_testing()
add_test(test_vector ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_vector)
add_test(test_matrix ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_matrix)
add_test(test_random ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_random)
add_test(test_special ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_special)


# Installation
set(INSTALL_DIR /usr/local/include/pml)
install(DIRECTORY DESTINATION ${INSTALL_DIR}) 
install(DIRECTORY include/ DESTINATION ${INSTALL_DIR})