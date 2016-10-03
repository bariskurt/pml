include_directories(../src)
include_directories(../src/ml)

link_libraries(gsl)
link_libraries(gslcblas)

add_executable(test_vector test_vector.cc)

add_executable(test_matrix test_matrix.cc)

add_executable(test_tensor test_tensor.cc)

add_executable(test_random test_random.cc)

add_executable(test_special test_special.cc)

## ML Algorithms

add_executable(test_bcpm test_bcpm.cc)

add_executable(test_hmm test_hmm.cc)

add_executable(test_nmf test_nmf.cc)
