#include <cassert>

#include "pml_vector.hpp"

using namespace pml;

std::string test_dir = "/tmp/";


void test_vector(){
  std::cout << "test_vector...\n";
  // Constructor 1
  Vector v1(5, 3);
  assert(v1.size() == 5);
  assert(v1 == 3);
  assert(!v1.empty());

  // Constructor 2
  double d[4] = {1, 2, 3, 4};
  Vector v2(4, d);
  assert(v2.size() == 4);
  assert(v2.first() == 1);
  assert(v2[1] == 2);
  assert(v2(2) == 3);
  assert(v2.last() == 4);

  // Constructor 3
  Vector v3({5,6,7});
  assert(v3.size() == 3);

  // Zeros & Ones
  Vector v4 = Vector::zeros(5);
  assert(v4.size() == 5);
  assert(v4 == 0);

  Vector v5 = Vector::ones(7);
  assert(v5.size() == 7);
  assert(v5 == 1);

  // Test append, push_back
  Vector v6({1,2,3,4});
  v6.append(5);
  v6.push_back(6);  assert(v6.size() == 6);
  v6.pop_back();    assert(v6.size() == 5);

  // Assign, Copy
  Vector v7(v6);  assert(v6 == v7);
  Vector v8; v8 = v6;  assert(v6 == v8);

  // Load and Save
  v6.saveTxt("/tmp/dummy.txt");
  Vector v9 = Vector::loadTxt("/tmp/dummy.txt");
  assert(v6 == v9);

  std::cout << "OK.\n";
}

void test_vector_functions() {

  std::cout << "test_vector_functions...\n";

  Vector v1 = {1, 2, 3, 4, 5};
  assert(fequal(min(v1), 1));
  assert(fequal(max(v1), 5));
  assert(fequal(sum(v1), 15));
  assert(fequal(mean(v1), 3));
  assert(fequal(var(v1), 2.5));
  assert(fequal(stdev(v1), 1.581138));

  assert(lgamma(v1) == Vector({0, 0, 0.693147, 1.791759, 3.178053}));

  assert(exp(v1) == Vector({2.718281, 7.389056, 20.085536,
                            54.598150, 148.413159}));

  assert(log(v1) == Vector({0, 0.693147, 1.098612, 1.386294, 1.609437}));

  assert(normalize(v1) == Vector({0.066667, 0.133333, 0.2,
                                  0.266667, 0.333333}));

  Vector z = log(v1);
  assert(normalizeExp(z) == normalize(v1));
  assert(logSumExp(z) == std::log(sum(v1)));

  Vector v2 = {-1,-2,-3,-4,-5};
  assert(v1 == abs(v2));

  Vector v3 = {1.1, 2.4, 2.9, 4, 4.8};
  assert(v1 == round(v3));

  std::cout << "OK.\n";
}

void test_vector_algebra(){
  std::cout << "test_vector_algebra...\n";
  Vector x(5, 3);
  Vector y(5, 5);

  assert(!(x==y));

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

  // Dot Product
  Vector a({1,2,3});
  Vector b({3,2,1});
  assert(dot(a,b) == 10);

  std::cout << "OK.\n";
}

int main(){
  test_vector();
  test_vector_functions();
  test_vector_algebra();
  return 0;
}