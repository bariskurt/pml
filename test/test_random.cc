//
// Created by bariskurt on 26.03.2016.
//

#include "pml_random.hpp"
#include <cassert>

using namespace pml;


void test_dirichlet() {
  std::cout << "test_dirichlet...\n";

  Vector alpha = {1, 2, 3, 4, 5};

  Dirichlet dir(alpha);

  Matrix data = dir.rand(1000);

  Dirichlet dir_est = Dirichlet::fit(data);

  std::cout << "Original parameters : " << dir.alpha << std::endl;

  std::cout << "Estimated parameters: " << dir_est.alpha << std::endl;

  std::cout << "OK.\n\n";
}

void test_gamma() {
  std::cout << "test_gamma...\n";

  double a = 20;
  double b = 3;

  Gamma gamma(a,b);

  Vector data = gamma.rand(1000);

  Gamma gamma_est = Gamma::fit(data);

  std::cout << "Original parameters : a = " << a << ", b = " << b << std::endl;

  std::cout << "Estimated parameters: a = "
            << gamma_est.a << ", b = " << gamma_est.b << std::endl;

  std::cout << "OK.\n\n";
}

void test_categorical(){

  std::cout << "test_categorical...\n";

  Categorical cat(Vector({1,2,3,4}));

  Vector data = cat.rand(1000);

  Categorical cat_est = Categorical::fit(data, 4);

  std::cout << "Original parameters : a = " << cat.p << std::endl;

  std::cout << "Estimated parameters : a = " << cat_est.p << std::endl;

  std::cout << "OK.\n\n";

}

void test_gaussian(){
  std::cout << "test_gaussian...\n";
  Gaussian g(5, 2);

  Vector data = g.rand(1000);

  Gaussian g_est = Gaussian::fit(data);

  std::cout << "Original parameters : mu = " << g.mu
            << ", sigma = " << g.sigma << "\n";

  std::cout << "Estimated parameters : mu = " << g_est.mu
            << ", sigma = " << g_est.sigma << "\n";

  std::cout << "OK.\n\n";
}

int main(){

  test_dirichlet();
  test_gamma();
  test_categorical();
  test_gaussian();


  return 0;
}

