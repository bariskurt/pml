include_directories(../include)
include_directories(../include/ml)

link_libraries(gsl)
link_libraries(gslcblas)

add_executable(test_vector test_vector.cc)

add_executable(test_matrix test_matrix.cc)

add_executable(test_random test_random.cc)

add_executable(test_special test_special.cc)
