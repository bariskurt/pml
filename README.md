## PML: Pilab Matrix and Machine Learning Library

Fast prototyping machine learning algorithms on Matlab and Python is joyful, since it takes considerably less time to implement and test our proposed solutions but when the time comes for running batch experiments, we usually find out that those prototypes are not scalable. Keeping this in mind, we developed a C++ machine learning library specially focusing on easy syntax that allows fast development.

Written in C++, PML uses the powerful features of C++11 (such as rvalue optimizations). Furthermore, the library uses GSL functions and CBLAS linear algebra routines so, it is fast.

Disclaimer: This is not a stable version of the library, yet.

## Compiling and Testing
You don't need to compile the library, since it's header only. But it's recommended that you compile the tests. 
The following commands builds and run the tests:

> mkdir build
> cd build
> cmake ..
> make
> make test

## Installing (Optional)

You can just copy the include folder to your project, and use the header files. 

If you've completed the previous step (compiling and testing), you can install the library under /usr/local/include/pml by typing the following command:

> sudo make install
