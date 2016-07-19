## PML: Pilab Matrix and Machine Learning Library

Fast prototyping machine learning algorithms on Matlab and Python is joyful, since it takes considerably less time to implement and test our proposed solutions but when the time comes for running batch experiments, we usually find out that those prototypes are not scalable. Keeping this in mind, we developed a C++ machine learning library specially focusing on easy syntax that allows fast development.

Written in C++, PML uses the powerful features of C++11 (such as rvalue optimizations). Furthermore, the library uses GSL functions and CBLAS linear algebra routines so, it is fast.

Disclaimer: This is not a stable version of the library, yet.


## Testing...

You can run the unit tests by running:

./test

this will create directories build and bin and run the unit tests in bin.

## Installing (Optional)

You can install the library under /usr/include/pml by:

sudo ./install
