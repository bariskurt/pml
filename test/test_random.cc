//
// Created by bariskurt on 26.03.2016.
//

#include "pml_rand.hpp"
#include <cassert>

using namespace pml;


void test_dirichlet() {
  std::cout << "test_dirichlet...";

  Vector alpha = {1, 2, 3, 4, 5};

  Dirichlet dir(alpha);

  Matrix data = dir.rand(100);

  Dirichlet dir_est = Dirichlet::fit(data);

  std::cout << dir_est << std::endl;

  std::cout << "OK.\n";
}


int main(){

  test_dirichlet();
  return 0;
}

