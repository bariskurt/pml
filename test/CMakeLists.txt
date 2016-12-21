include_directories(../include)
include_directories(../include/ml)

link_libraries(gsl)
link_libraries(gslcblas)

add_executable(test_memory test_memory.cc)

add_executable(test_vector test_vector.cc)

add_executable(test_vector_view test_vector_view.cc)

add_executable(test_matrix test_matrix.cc)

add_executable(test_random test_random.cc)

add_executable(test_special test_special.cc)

add_executable(test_speed test_speed.cc)

## ML Algorithms

add_executable(test_hmm test_hmm.cc)

add_executable(test_nmf test_nmf.cc)

add_executable(test_bcpm test_bcpm.cc)

add_executable(test_block test_block.cc)


