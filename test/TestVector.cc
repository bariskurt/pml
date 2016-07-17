//
// Created by bariskurt on 26.03.2016.
//

#include <cassert>

#include "pml.hpp"
//#include "../src/pml_rand.hpp"

using namespace pml;

std::string test_dir = "/tmp/";

void test_vector(){

  std::cout << "test_vector...";

  // Constructors
  Vector v1({1,1,1,1,1});
  Vector v2(5, 1);
  assert(v1 == v2);

  Array a({2,3}, {1,2,3,4,5,6});
  Vector v3 = Vector::flatten(a);
  assert(v3 != a); // due to shape

  Vector v4;
  v4 = Vector::flatten(a);
  assert(v3 == v4);

  Vector v5 = Vector::ones(5);
  assert(v5.size() == 5);
  assert(v5 == 1);

  Vector v6 = Vector::zeros(6);
  assert(v6.size() == 6);
  assert(v6 == 0);

  // Dot Product:
  Vector x({1,2,3,4,5});
  Vector y({5,4,3,2,1});
  assert(dot(x,y) == 5+8+9+8+5);

  // Save & Load
  x.save("/tmp/x.txt");
  Vector z = Array::load("/tmp/x.txt");
  assert(x == z);

  // Misc
  z = x + y;
  assert(z == 6);

  std::cout << "OK.\n";
}


int main(){

  test_vector();

  return 0;
}