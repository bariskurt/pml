#include <cassert>

#include "pml_matrix.hpp"

using namespace pml;

void test_matrix(){
  std::cout << "test_matrix...\n";

  Matrix M1 = Matrix(3,2, {1,2,3,4,5,6});

  Matrix M2(M1);  assert(M1 == M2);

  Matrix M3; M3 = M2; assert(M1 == M3);

  // Zeros and Ones
  Matrix M4 = Matrix::ones(4,5);
  assert(M4.nrows() == 4);
  assert(M4.ncols() == 5);
  assert(M4 == 1);

  Matrix M5 = Matrix::zeros(4,5);
  assert(M5 == 0);

  // Identity
  Matrix I = Matrix::identity(4);
  for(size_t i = 0; i < I.nrows(); ++i){
    for(size_t j = 0; j < I.ncols(); ++j){
      if( i == j ){
        assert(I(i,j) == 1);
      } else{
        assert(I(i,j) == 0);
      }
    }
  }

  // Set & Get Columns
  Matrix m(3,4, {0,1,2,3,4,5,6,7,8,9,10,11});
  Vector v1 = m.getColumn(3);
  Vector v2(m.nrows());
  v2(0) = m(0,3);
  v2(1) = m(1,3);
  v2(2) = m(2,3);
  assert(v1 == v2);

  m.setColumn(2, Vector::ones(3));
  assert(m(0,2) == 1);
  assert(m(1,2) == 1);
  assert(m(2,2) == 1);

  // Set & Get Rows
  m = Matrix(3,4, {0,1,2,3,4,5,6,7,8,9,10,11});
  v1 = m.getRow(2);
  v2 = Vector({m(2,0), m(2,1), m(2,2), m(2,3)});
  assert(v1 == v2);

  m.setRow(1, Vector::ones(4));
  assert(m(1,0) == 1);
  assert(m(1,1) == 1);
  assert(m(1,2) == 1);
  assert(m(1,3) == 1);

  // Test File operations
  m.saveTxt("/tmp/dummy.txt");
  Matrix m2 = Matrix::loadTxt("/tmp/dummy.txt");
  assert(m == m2);


  std::cout << "OK\n";
}

void test_matrix_functions(){
  std::cout << "test_matrix_functions...\n";



  std::cout << "OK\n";
}

void test_matrix_algebra(){
  std::cout << "test_matrix_algebra...\n";

  Matrix x({3, 4}, 3);
  Matrix y({3, 4}, 5);

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


  // Multiply Matrix columns..
  Matrix a({3, 4}, 3);
  Vector b({1, 2, 3});
  a = a * b;
  assert(a.getRow(0) == 3);
  assert(a.getRow(1) == 6);
  assert(a.getRow(2) == 9);

  a = a + b;
  a = a - b;
  a = a / b;

  // Dot Product
  Vector v({1,2,3});
  Matrix m(2,3,{1,2,3,4,5,6});
  assert(dot(m, v) == Vector({22, 28}));
  assert(dot(m,transpose(m)) == Matrix(2,2, {35, 44, 44, 56}));

  std::cout << "OK\n";
}


int main(){
  test_matrix();
  test_matrix_functions();
  test_matrix_algebra();
  return 0;
}

