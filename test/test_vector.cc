#include <cassert>

#include "pml_vector.hpp"

using namespace pml;

std::string test_dir = "/tmp/";


void test_vector(){
  std::cout << "test_vector...\n";

  // Constructor 1
  Vector v1(5, 3);
  assert(v1.size() == 5);
  assert(all(v1 == 3));
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
  assert(all(v4 == 0));

  Vector v5 = Vector::ones(7);
  assert(v5.size() == 7);
  assert(all(v5 == 1));

  // Test append, push_back
  Vector v6({1,2,3,4});
  v6.append(5);
  v6.push_back(6);  assert(v6.size() == 6);
  v6.pop_back();    assert(v6.size() == 5);

  // Test append 2 Vectors
  v6.append(Vector()); assert(v6.size() == 5);
  v6.append(Vector({10,11})); assert(v6.size() == 7);

  // Assign, Copy
  Vector v7(v6);  assert(v6.equals(v7));
  Vector v8; v8 = v6;  assert(v6.equals(v8));

  std::cout << "OK.\n";
}


void test_load_save(){
  std::cout << "test_load_save...\n";

  Vector x({1,2,3,4,5,6,7,8});

  // Load and Save in Binary
  x.save("/tmp/test_vector.pml");
  Vector y = Vector::load("/tmp/test_vector.pml");
  assert(x.equals(y));

  // Load and Save in Text
  x.saveTxt("/tmp/test_vector.txt");
  Vector z = Vector::loadTxt("/tmp/test_vector.txt");
  assert(x.equals(z));

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


  Vector exp_v1 = {2.718281, 7.389056, 20.085536, 54.598150, 148.413159};
  assert(exp(v1).equals(exp_v1));

  Vector log_v1 = {0, 0.693147, 1.098612, 1.386294, 1.609437};
  assert(log(v1).equals(log_v1));

  Vector normalize_v1 = {0.066667, 0.133333, 0.2, 0.266667, 0.333333};
  assert(normalize(v1).equals(normalize_v1));

  Vector z = log(v1);
  assert(normalizeExp(z).equals(normalize(v1)));
  assert(fequal(logSumExp(z), std::log(sum(v1))));

  Vector v2 = {-1,-2,-3,-4,-5};
  assert(v1.equals(abs(v2)));

  Vector v3 = {1.1, 2.4, 2.9, 4, 4.8};
  assert(v1.equals(round(v3)));

  // Test slice
  Vector v4 = {0,1,2,3,4,5,6,7,8};
  assert(v4.getSlice(0, v4.size()).equals(v4));
  assert(v4.getSlice(0, 4).equals(Vector({0,1,2,3})));
  assert(v4.getSlice(0, 0).equals(Vector()));
  assert(v4.getSlice(0, v4.size(), 2).equals(Vector({0,2,4,6,8})));
  assert(v4.getSlice(1, v4.size(), 2).equals(Vector({1,3,5,7})));

  std::cout << "OK.\n";
}

void test_vector_algebra(){
  std::cout << "test_vector_algebra...\n";
  Vector x(5, 3);
  Vector y(5, 5);

  // A = A op b
  x += 1; assert(all(x == 4));
  x -= 1; assert(all(x == 3));
  x *= 2; assert(all(x == 6));
  x /= 2; assert(all(x == 3));


  // A = A op B
  x += y; assert(all(x == 8));
  x -= y; assert(all(x == 3));
  x *= y; assert(all(x == 15));
  x /= y; assert(all(x == 3));


  // C = A op b
  // C = b op A
  Vector z;
  z = x + 1; assert(all(z == 4));
  z = 1 + x; assert(all(z == 4));
  z = x - 1; assert(all(z == 2));
  z = 1 - x; assert(all(z == -2));
  z = x * 2; assert(all(z == 6));
  z = 2 * x; assert(all(z == 6));
  z = x / 2; assert(all(z == 1.5));
  z = 2 / x; assert(all(z == 2.0/3.0));

  // C = A op B
  z = x + y; assert(all(z == 8));
  z = x - y; assert(all(z == -2));
  z = x * y; assert(all(z == 15));
  z = x / y; assert(all(z == 3.0/5.0));

  // Dot Product
  Vector a({1,2,3});
  Vector b({3,2,1});
  assert(dot(a,b) == 10);

  std::cout << "OK.\n";
}



void test_vector_comparison() {
  std::cout << "test_vector_comparison...\n";

  Vector v({1,2,3});
  assert(sum(v == v) == 3);
  assert(sum(v == 2) == 1);
  assert(sum(v < 2) == 1);
  assert(sum(v > 2) == 1);

  assert(any(v == 2));
  assert(any(v < 2));
  assert(any(v > 2));

  assert(!all(v == 2));
  assert(!all(v < 2));
  assert(!all(v > 2));

  Vector v2({4,5,6});
  assert(sum(v == v2) == 0);
  assert(sum(v < v2) == 3);
  assert(sum(v > v2) == 0);

  Vector result = find(v==2);
  assert(result.size() == 1);
  assert(result.first() == 1);

  result = find(v < v2 );
  assert(result.size() == 3);

  std::cout << "OK.\n";
}


int main(){
  test_vector();
  test_vector_functions();
  test_vector_algebra();
  test_vector_comparison();
  test_load_save();
  return 0;

}
