#include <cassert>

#include "pml_matrix.hpp"

using namespace pml;

void test_matrix(){
  std::cout << "test_matrix...\n";

  Matrix M1 = Matrix(3,2, {1,2,3,4,5,6});

  Matrix M2(M1);  assert(M1.equals(M2));

  Matrix M3; M3 = M2; assert(M1.equals(M3));

  // Zeros and Ones
  Matrix M4 = Matrix::ones(4,5);
  assert(M4.nrows() == 4);
  assert(M4.ncols() == 5);
  assert(all(M4 == 1));

  Matrix M5 = Matrix::zeros(4,5);
  assert(all(M5 == 0));

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
  assert(v1.equals(v2));

  m.setColumn(2, Vector::ones(3));
  assert(m(0,2) == 1);
  assert(m(1,2) == 1);
  assert(m(2,2) == 1);

  // Set & Get Rows
  m = Matrix(3,4, {0,1,2,3,4,5,6,7,8,9,10,11});
  v1 = m.getRow(2);
  v2 = Vector({m(2,0), m(2,1), m(2,2), m(2,3)});
  assert(v1.equals(v2));

  m.setRow(1, Vector::ones(4));
  assert(m(1,0) == 1);
  assert(m(1,1) == 1);
  assert(m(1,2) == 1);
  assert(m(1,3) == 1);

  // Test File operations
  m.saveTxt("/tmp/dummy.txt");
  Matrix m2 = Matrix::loadTxt("/tmp/dummy.txt");
  assert(m.equals(m2));


  std::cout << "OK\n";
}

void test_matrix_functions(){
  std::cout << "test_matrix_functions...\n";

  // Sum, Min, Max
  Matrix m(2, 3, {0, 1, 2, 3, 4, 5});
  assert(sum(m) == 15);
  assert(sum(m,0).equals(Vector({1, 5, 9})));
  assert(sum(m,1).equals(Vector({6, 9})));

  assert(min(m) == 0);
  assert(min(m,0).equals(Vector({0, 2, 4})));
  assert(min(m,1).equals(Vector({0, 1})));

  assert(max(m) == 5);
  assert(max(m,0).equals(Vector({1, 3, 5})));
  assert(max(m,1).equals(Vector({4, 5})));

  // Round and Abs
  Matrix m2(3, 5, -2.3);
  assert(all(abs(m2) == 2.3));
  assert(all(round(abs(m2)) == 2));

  // Log, Exp, Psi
  Matrix m3 = Matrix(3, 4, 0.5);
  assert(all(log(m3) == -0.6931471));
  assert(all(exp(m3) == 1.64872127));
  assert(all(psi(m3) == -1.96351002));

  // Normalizations
  Matrix m4(2,2, {1, 2, 3, 4});
  assert(normalize(m4).equals(Matrix(2,2, {0.1, 0.2, 0.3, 0.4})));
  assert(normalize(m4,0).equals(Matrix(2,2, {1.0/3, 2.0/3, 3.0/7, 4.0/7})));
  assert(normalize(m4,1).equals(Matrix(2,2, {1.0/4, 2.0/6, 3.0/4, 4.0/6})));

  // Normalize Exp
  Matrix m5 = log(m4);
  assert(normalizeExp(m5).equals(Matrix(2,2, {0.1, 0.2, 0.3, 0.4})));
  assert(normalizeExp(m5,0).equals(Matrix(2,2, {1.0/3, 2.0/3, 3.0/7, 4.0/7})));
  assert(normalizeExp(m5,1).equals(Matrix(2,2, {1.0/4, 2.0/6, 3.0/4, 4.0/6})));

  // LogSumExp
  assert(logSumExp(m5) == std::log(10));
  assert(logSumExp(m5,0).equals(log(sum(m4,0))));
  assert(logSumExp(m5,1).equals(log(sum(m4,1))));

  // Tile
  Vector v = {1,2};
  assert(tileRows(v, 2).equals(Matrix(2,2, {1,1,2,2})));
  assert(tileCols(v, 2).equals(Matrix(2,2, {1,2,1,2})));
  assert(repmat(v, 2, 2).equals(Matrix(4,2, {1,2,1,2,1,2,1,2})));

   std::cout << "OK\n";
}

void test_matrix_algebra(){
  std::cout << "test_matrix_algebra...\n";

  Matrix x({3, 4}, 3);
  Matrix y({3, 4}, 5);

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

  Matrix z;
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


  // Multiply Matrix columns..
  Matrix a({3, 4}, 3);
  Vector b({1, 2, 3});
  a = a * b;
  assert(all(a.getRow(0) == 3));
  assert(all(a.getRow(1) == 6));
  assert(all(a.getRow(2) == 9));

  a = a + b;
  a = a - b;
  a = a / b;

  // Dot Product
  Vector v({1,2,3});
  Matrix m(2,3,{1,2,3,4,5,6});
  assert(dot(m, v).equals(Vector({22, 28})));
  assert(dot(m,transpose(m)).equals(Matrix(2,2, {35, 44, 44, 56})));

  std::cout << "OK\n";
}

void test_matrix_append(){

  std::cout << "test_matrix_append...\n";

  Matrix m1;
  m1.appendColumn(Vector({1,2,3,4}));
  m1.appendColumn(Vector({1,2,3,4}));
  assert(m1.nrows()==4);
  assert(m1.ncols()==2);
  assert(m1.getColumn(0).equals(Vector({1,2,3,4})));
  assert(m1.getColumn(1).equals(Vector({1,2,3,4})));

  Matrix m2;
  m2.appendRow(Vector({1,2,3,4}));
  m2.appendRow(Vector({1,2,3,4}));
  assert(m2.nrows()==2);
  assert(m2.ncols()==4);
  assert(m2.getRow(0).equals(Vector({1,2,3,4})));
  assert(m2.getRow(1).equals(Vector({1,2,3,4})));

  std::cout << "OK\n";
}


int main(){
  test_matrix();
  test_matrix_functions();
  test_matrix_algebra();
  test_matrix_append();
  return 0;
}

