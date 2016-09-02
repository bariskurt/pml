#include <iostream>

#include <cassert>

#include "pml_hmm.hpp"

using namespace pml;

int main(){

    size_t N = 3;
    size_t T = 100;

    Vector pi = normalize(Vector::ones(N));
    Matrix A = Matrix::identity(N) * 0.9;
    A(1,0) = 0.1;
    A(2,1) = 0.1;
    A(0,2) = 0.1;
    Matrix B = normalize(Matrix::identity(N) * 0.8 + uniform::rand(N,N) * 0.2);

    hmm::DiscreteHMM dhmm(pi, A, B);

    auto seq = dhmm.generateSequence(T);

    auto log_Alpha = dhmm.forwardRecursion(seq.obs);
    auto log_Beta = dhmm.backwardRecursion(seq.obs);
    Matrix log_Gamma = log_Alpha + log_Beta;

    seq.save("/tmp/seq.txt");
    dhmm.save("/tmp");
    normalizeExp(log_Alpha, 0).saveTxt("/tmp/alpha.txt");
    normalizeExp(log_Beta, 0).saveTxt("/tmp/beta.txt");
    normalizeExp(log_Gamma, 0).saveTxt("/tmp/gamma.txt");

    return 0;
}