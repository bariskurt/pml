//
// Created by bariskurt on 26.03.2016.
//

#include "pml_rand.hpp"
#include <cassert>

using namespace pml;


void test_dirichlet() {
  std::cout << "test_dirichlet...\n";

  Vector alpha = {1, 2, 3, 4, 5};

  Dirichlet dir(alpha);

  Matrix data = dir.rand(1000);

  Dirichlet dir_est = Dirichlet::fit(data);

  std::cout << dir_est.alpha << std::endl;

  std::cout << "OK.\n";
}

void test_gamma() {
  std::cout << "test_gamma...\n";

  double a = 10;
  double b = 2;

  Gamma gamma(a,b);

  Vector data = gamma.rand(1000);

  Gamma gamma_est = Gamma::fit(data);

  std::cout << gamma_est.a << " " << gamma_est.b << std::endl;

  std::cout << "OK.\n";
}


int main(){

  test_dirichlet();
  test_gamma();
  return 0;
}

