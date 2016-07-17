//
// Created by bariskurt on 26.03.2016.
//

#include "pml.hpp"
#include "pml_rand.hpp"
#include <cassert>

using namespace pml;


void test_random() {
  std::cout << "test_random...";

  double d = uniform::rand();
  assert((d >=0) && (d <= 1));
  Vector v = uniform::rand(10);
  Matrix m = uniform::rand(10,5);

  std::cout << "OK.\n";
}


int main(){

  test_random();
  return 0;
}

