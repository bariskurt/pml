#include <cassert>

#include "pml_new.hpp"

using namespace pml;


void test_vector_algebra(){
  std::cout << "test_vector_algebra...";
  Vector x(5, 3);
  Vector y(5, 5);

  // A = A op b
  x += 1; assert(x == 4);
  x -= 1; assert(x == 3);
  x *= 2; assert(x == 6);
  x /= 2; assert(x == 3);


  // A = A op B
  x += y; assert(x == 8);
  x -= y; assert(x == 3);
  x *= y; assert(x == 15);
  x /= y; assert(x == 3);


  // C = A op b
  // C = b op A
  Vector z;
  z = x + 1; assert(z == 4);
  z = 1 + x; assert(z == 4);
  z = x - 1; assert(z == 2);
  z = 1 - x; assert(z == -2);
  z = x * 2; assert(z == 6);
  z = 2 * x; assert(z == 6);
  z = x / 2; assert(z == 1.5);
  z = 2 / x; assert(z == 2.0/3.0);

  // C = A op B
  z = x + y; assert(z == 8);
  z = x - y; assert(z == -2);
  z = x * y; assert(z == 15);
  z = x / y; assert(z == 3.0/5.0);

  std::cout << "OK.\n";
}

void test_matrix_algebra(){
  std::cout << "test_matrix_algebra...";
  Matrix x(3, 4, 3);
  Matrix y(3, 4, 5);

  // A = A op b
  x += 1; assert(x == 4);
  x -= 1; assert(x == 3);
  x *= 2; assert(x == 6);
  x /= 2; assert(x == 3);


  // A = A op B
  x += y; assert(x == 8);
  x -= y; assert(x == 3);
  x *= y; assert(x == 15);
  x /= y; assert(x == 3);

  Matrix z;
  z = x + 1; assert(z == 4);
  z = 1 + x; assert(z == 4);
  z = x - 1; assert(z == 2);
  z = 1 - x; assert(z == -2);
  z = x * 2; assert(z == 6);
  z = 2 * x; assert(z == 6);
  z = x / 2; assert(z == 1.5);
  z = 2 / x; assert(z == 2.0/3.0);

  // C = A op B
  z = x + y; assert(z == 8);
  z = x - y; assert(z == -2);
  z = x * y; assert(z == 15);
  z = x / y; assert(z == 3.0/5.0);

  std::cout << "OK.\n";
}

int main(){

  test_vector_algebra();
  test_matrix_algebra();
  return 0;
}