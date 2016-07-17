include_directories(../src)
link_libraries(gsl)
link_libraries(gslcblas)
link_libraries(lapack)

add_executable(test_array TestArray.cc)

add_executable(test_vector TestVector.cc)

add_executable(test_matrix TestMatrix.cc)

#add_executable(test_tensor ../src/Test_Tensor.cc)
