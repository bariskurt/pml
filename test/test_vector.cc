#include <cassert>

#include "pml_vector.hpp"
#include "pml_time.hpp"

using namespace pml;

std::string test_dir = "/tmp/";

void assert_equal(const Vector&x, const Vector&x2){
  assert(x.size() == x2.size());
  for(size_t i=0; i < x.size(); ++i)
    assert(fequal(x[i], x2[i]));
}

void assert_not_the_same(const Vector&x, const Vector&x2){
  assert(x.data() != x2.data());
}

void test_vector_constructors() {
  std::cout << "test_vector_constructors...\n";

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
  Vector v3({5, 6, 7});
  assert(v3.size() == 3);

  // Zeros
  Vector v4 = Vector::zeros(5);
  assert(v4.size() == 5);
  assert(all(v4 == 0));

  // Ones
  Vector v5 = Vector::ones(7);
  assert(v5.size() == 7);
  assert(all(v5 == 1));

  std::cout << "OK.\n";
}

void test_vector_copy_constructors(){
  std::cout << "test_vector_copy_constructors...\n";

  Vector x({1,2,3,4,5,6,7,8});

  // Copy Constructor
  {
    Vector x2(x);
    assert_not_the_same(x, x2);
    assert_equal(x, x2);
  }

  // Assignment
  {
    Vector x2 = x;
    assert(fequal(x, x2));
    assert_not_the_same(x, x2);
    assert_equal(x, x2);
  }

  // Assignment 2
  {
    Vector x2;
    x2 = x;
    assert(fequal(x, x2));
    assert_not_the_same(x, x2);
    assert_equal(x, x2);
  }

  std::cout << "OK.\n";
}

/*
  // Test append, push_back
  Vector v6({1,2,3,4});
  v6.append(5);
  v6.append(6);  assert(v6.size() == 6);

  // Test append 2 Vectors
  v6.append(Vector()); assert(v6.size() == 6);
  v6.append(Vector({10,11})); assert(v6.size() == 8);

  // Assign, Copy
  Vector v7(v6);  assert(fequal(v6,v7));
  Vector v8; v8 = v6;  assert(fequal(v6,v8));

  std::cout << "OK.\n";
}
*/

void test_load_save(){
  std::cout << "test_load_save...\n";

  Vector x({1,2,3,4,5,6,7,8});

  // Load and Save in Binary
  x.save("/tmp/test_vector.pml");
  Vector y = Vector::load("/tmp/test_vector.pml");
  assert(fequal(x,y));

  // Load and Save in Text
  x.saveTxt("/tmp/test_vector.txt");
  Vector z = Vector::loadTxt("/tmp/test_vector.txt");
  assert(fequal(x, z));

  std::cout << "OK.\n";
}

void test_vector_view(){
  std::cout << "test_vector_view...\n";

  // Part 1: ConstVectorView -> Vector
  {
    Vector v = {0, 1, 2, 3, 4, 5, 6, 7};
    ConstVectorView cvw(v);

    Vector v2(cvw);  assert(fequal(v, v2));

    Vector v3 = v;  assert(fequal(v, v3));

    Vector v4; v4 = cvw; assert(fequal(v, v4));
  }

  // Part 2: VectorView -> Vector

  // Part 3: ConstVectorView -> VectorView

  // Part 4: VectorView -> VectorView

  // Part5 : ConstVectorView -> X



  Vector v = {0, 1, 2, 3, 4, 5, 6, 7};
  VectorView vw(v);

  // VectorView -> Vector



  Vector cv = {7, 6, 5, 4, 3, 2, 1, 0};
  ConstVectorView cvw(cv);

  // ConstVectorView -> Vector
  //Vector a1(cvw);  assert(fequal(v, a1));
  //Vector a2(cv);  assert(fequal(v, a2));

  std::cout << "OK.\n";
}

/*
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
  assert(fequal(exp(v1), exp_v1));

  Vector log_v1 = {0, 0.693147, 1.098612, 1.386294, 1.609437};
  assert(fequal(log(v1), log_v1));

  Vector normalize_v1 = {0.066667, 0.133333, 0.2, 0.266667, 0.333333};
  assert(fequal(normalize(v1), normalize_v1));

  Vector z = log(v1);
  assert(fequal(normalizeExp(z), normalize(v1)));
  assert(fequal(logSumExp(z), std::log(sum(v1))));

  Vector v2 = {-1,-2,-3,-4,-5};
  assert(fequal(v1, abs(v2)));

  Vector v3 = {1.1, 2.4, 2.9, 4, 4.8};
  assert(fequal(v1, round(v3)));

  // Test slice
  Vector v4 = {0,1,2,3,4,5,6,7,8};
  assert(fequal(v4.getSlice(0, v4.size()), v4));
  assert(fequal(v4.getSlice(0, 4), Vector({0,1,2,3})));
  assert(fequal(v4.getSlice(0, 0), Vector()));
  assert(fequal(v4.getSlice(0, v4.size(), 2), Vector({0,2,4,6,8})));
  assert(fequal(v4.getSlice(1, v4.size(), 2), Vector({1,3,5,7})));

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

void test_range(){
  std::cout << "test_range.\n";

  Range r1(0,10);
  assert(r1.size() == 10);

  Range r2(0,10, 2);
  assert(r2.size() == 5);

  Range r3(0,9, 2);
  assert(r3.size() == 5);

  Range r4(0,9, 3);
  assert(r4.size() == 3);

  Range r5(0,9, 4);
  assert(r5.size() == 3);

  std::cout << "OK.\n";
}
*/
int main(){

  test_vector_constructors();
  test_vector_copy_constructors();

//  test_vector_view();
//  test_load_save();
//  test_vector_algebra();
//  test_vector_comparison();
//  test_vector_functions();
//  test_range();

  return 0;

}
