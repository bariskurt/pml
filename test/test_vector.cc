#include <cassert>

#include "pml_vector.hpp"

using namespace pml;

std::string test_dir = "/tmp/";

void test_vector(){

  std::cout << "test_vector...\n";

  // Constructor 1
  Vector v1(5, 3);
  assert(v1.size() == 5);
  for(double value : v1)
    assert(value == 3);
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
  assert(v3[0] == 5);
  assert(v3[1] == 6);
  assert(v3(2) == 7);

  // Zeros
  Vector v4 = Vector::zeros(5);
  assert(v4.size() == 5);
  for(double value : v4)
    assert(value == 0);

  // Ones
  Vector v5 = Vector::ones(7);
  assert(v5.size() == 7);
  for(double value : v5)
    assert(value == 1);

  // Test append
  Vector v6({1,2,3,4});
  v6.append(5); assert(v6.size() == 5);
  v6.append(Vector()); assert(v6.size() == 5);
  v6.append(Vector({10,11})); assert(v6.size() == 7);

  // Assign, Copy
  Vector v7(v6);  assert(v6 == v7);
  Vector v8; v8 = v6;  assert(v6 == v8);

  std::cout << "OK.\n";
}

void test_const_view(){
  std::cout << "test_const_view...\n";

  Vector v({0,1,2,3,4,5,6,7});
  Vector::const_view cv(v);
  assert(cv.size() == v.size());
  size_t i = 0;
  for(double value : cv)
    assert(value == v[i++]);

  // Apply and create new
  Vector v2 = cv + 1;
  assert(v2[0] == 1);

  Vector v3 = cv - 1;
  assert(v3[0] == -1);

  Vector v4 = cv * 2;
  assert(v4[1] == 2);

  Vector v5 = cv / 2;
  assert(v5[1] == 0.5);

  Vector a({0,1,2});
  Vector::const_view cva(a);
  Vector b({3,4,5});
  Vector::const_view cvb(b);
  Vector result = cva + cvb;
  assert(result.size() == 3);
  assert(result[0] == 3);
  assert(result[1] == 5);
  assert(result[2] == 7);
  assert(cva == a);
  assert(cvb == b);

  std::cout << "OK.\n";
}

void test_view(){
  std::cout << "test_view...\n";

  Vector v({0,1,2,3,4,5,6,7});
  Vector::view vw(v);
  assert(vw.size() == v.size());
  size_t i = 0;
  for(double &value : vw)
    assert(value == v[i++]);

  // Algebra
  vw += 5;
  assert(v[0] == 5);

  vw -= 5;
  assert(v[0] == 0);

  vw *= 2;
  assert(v[1] == 2);

  vw /= 2;
  assert(v[1] == 1);

  Vector v2 = Vector(v.size(), 2);
  Vector::view vw2(v2);
  vw += vw2;
  assert(v[0] == 2);

  vw -= vw2;
  assert(v[0] == 0);

  vw *= vw2;
  assert(v[1] == 2);

  vw /= vw2;
  assert(v[1] == 1);



  std::cout << "OK.\n";
}

void test_load_save(){
  std::cout << "test_load_save...\n";

  Vector x({1,2,3,4,5,6,7,8});

  // Load and Save in Binary
  x.save("/tmp/test_vector.pml");
  Vector y = Vector::load("/tmp/test_vector.pml");
  assert(x == y);

  // Load and Save in Text
  x.saveTxt("/tmp/test_vector.txt");
  Vector z = Vector::loadTxt("/tmp/test_vector.txt");
  assert(x == z);

  std::cout << "OK.\n";
}

void test_vector_algebra(){
  std::cout << "test_vector_algebra...\n";
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

  // Dot Product
  Vector a({1,2,3});
  Vector b({3,2,1});
  assert(dot(a,b) == 10);

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
  assert(exp(v1) == exp_v1);

  Vector log_v1 = {0, 0.693147, 1.098612, 1.386294, 1.609437};
  assert(log(v1) == log_v1);

  Vector normalize_v1 = {0.066667, 0.133333, 0.2, 0.266667, 0.333333};
  assert(normalize(v1) == normalize_v1);

  Vector z = log(v1);
  assert(normalizeExp(z) == normalize(v1));
  assert(logSumExp(z) == std::log(sum(v1)));

  Vector v2 = {-1,-2,-3,-4,-5};
  assert(abs(v2) == v1);

  Vector v3 = {1.1, 2.4, 2.9, 4, 4.8};
  assert(round(v3) == Vector({1,2,3,4,5}));
  assert(ceil(v3) == Vector({2,3,3,4,5}));
  assert(floor(v3) == Vector({1,2,2,4,4}));

  Vector a({0,1,2}), b({3,4,5});
  assert(cat(a, b) == Vector({0,1,2,3,4,5}));

  std::cout << "OK.\n";
}


void test_slice(){
  std::cout << "test_slice...\n";

  // Test slice
  Vector v4 = {0,1,2,3,4,5,6,7,8};
  assert(v4.slice(0, v4.size()) == v4);
  assert(v4.slice(0, 4) == Vector({0,1,2,3}));
  assert(v4.slice(0, 0) == Vector());
  assert(v4.slice(0, v4.size(), 2) == Vector({0,2,4,6,8}));
  assert(v4.slice(1, v4.size(), 2) == Vector({1,3,5,7}));

  std::cout << "OK.\n";
}


int main(){

  test_vector();
  test_const_view();
  test_view();
  test_load_save();
  test_vector_algebra();
  test_slice();
  test_vector_functions();

  return 0;

}
