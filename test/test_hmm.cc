#include <iostream>

#include <cassert>

#include "pml_hmm.hpp"

using namespace pml;

int main(){

  size_t N = 3;
  size_t T = 100;

  Vector p1 = normalize(Vector::ones(N));
  Matrix A = Matrix::identity(N) * 0.9;
  A(1,0) = 0.1;
  A(2,1) = 0.1;
  A(0,2) = 0.1;
  Matrix B = normalize(Matrix::identity(N) * 0.8 + Uniform().rand(N,N) * 0.2);


  DiscreteHMM dhmm(p1, A, B);

  Vector states, obs;
  std::tie(states, obs) = dhmm.generateData(T);

  auto alpha = dhmm.forward(obs);
  auto beta = dhmm.backward(obs);
  Matrix gamma = alpha + beta;



  dhmm.save("/tmp");
  states.saveTxt("/tmp/states.txt");
  obs.saveTxt("/tmp/obs.txt");
  normalizeExp(alpha, 0).saveTxt("/tmp/alpha.txt");
  normalizeExp(beta, 0).saveTxt("/tmp/beta.txt");
  normalizeExp(gamma, 0).saveTxt("/tmp/gamma.txt");

  return 0;
}