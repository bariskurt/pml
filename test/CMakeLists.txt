include_directories(../src)
link_libraries(gsl)
link_libraries(gslcblas)

#add_executable(test_objects TestObjects.cc)

add_executable(test_random TestRandom.cc)

add_executable(test_tensor TestTensor.cc)

add_executable(test_hmm TestHMM.cc)

add_executable(test_nmf TestNMF.cc)

<<<<<<< HEAD
add_executable(test_bcpm TestBCPM.cc)

add_executable(test_lds TestLDS.cc)
=======
add_executable(test_bcpm test_bcpm.cc)

add_executable(test_vector test_vector.cc)

add_executable(test_matrix test_matrix.cc)
>>>>>>> a5e6a24ebe3f4f719e6ba06f03a15e94880f764a
