link_libraries(gsl)
link_libraries(gslcblas)
link_libraries(lapack)

add_executable(test_vector Test_Vector.cc)

add_executable(test_matrix Test_Matrix.cc)

add_executable(test_tensor Test_Tensor.cc)

add_executable(test_block Test_Block.cc)