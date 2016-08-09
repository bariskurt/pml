include_directories(../src)
link_libraries(gsl)
link_libraries(gslcblas)
link_libraries(lapack)

# add_executable(test_objects TestObjects.cc)

# add_executable(test_random TestRandom.cc)

# add_executable(test_algebra TestAlgebra.cc)

# add_executable(test_tensor TestTensor.cc)

# add_executable(test_hmm TestHMM.cc)

# add_executable(test_nmf TestNMF.cc)

add_executable(test_bcpm TestBCPM.cc)
