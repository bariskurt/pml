#include <cassert>

#include "pml_matrix.hpp"

using namespace pml;


void assert_equal(const Matrix&x, const Matrix&x2){
  assert(x.size() == x2.size());
  assert(x.nrows() == x2.nrows());
  assert(x.ncols() == x2.ncols());
  for(size_t i=0; i < x.size(); ++i)
    assert(fequal(x[i], x2[i]));
}

void assert_not_the_same(const Matrix&x, const Matrix&x2){
  assert(x.data() != x2.data());
}


void test_matrix(){
  std::cout << "test_matrix...\n";

  Matrix M1 = Matrix(3,2, {1,2,3,4,5,6});

  Matrix M2(M1);  assert(fequal(M1 ,M2));

  Matrix M3; M3 = M2; assert(fequal(M1 ,M3));

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

  std::cout << "OK\n";
}

void test_matrix_copy_constructors(){
  std::cout << "test_matrix_copy_constructors...\n";

  // Copy Constructor
  {
    Matrix x = Matrix(3,2, {1,2,3,4,5,6});
    Matrix x2(x);
    assert_not_the_same(x, x2);
    assert_equal(x, x2);
  }

  // Assignment
  {
    Matrix x = Matrix(3,2, {1,2,3,4,5,6});
    Matrix x2 = x;
    assert(fequal(x, x2));
    assert_not_the_same(x, x2);
    assert_equal(x, x2);
  }

  // Assignment 2
  {
    Matrix x = Matrix(3,2, {1,2,3,4,5,6});
    Matrix x2;
    x2 = x;
    assert(fequal(x, x2));
    assert_not_the_same(x, x2);
    assert_equal(x, x2);
  }

  // Move constructor
  {
    Matrix x = Matrix(3,2, {1,2,3,4,5,6});
    const double *data_ = x.data();
    const size_t x_size = x.size();
    const size_t x_nrows = x.nrows();
    const size_t x_ncols = x.ncols();

    Matrix x2(std::move(x));
    assert(x.size() == 0);
    assert(x.nrows() == 0);
    assert(x.ncols() == 0);
    assert(x.data() != data_);

    assert(x2.data() == data_);
    assert(x2.size() == x_size);
    assert(x2.nrows() == x_nrows);
    assert(x2.ncols() == x_ncols);
    for(size_t i = 0; i < x_size; ++i)
      assert( x2[i] == i+1);

  }

  // Move Assignment
  {
    Matrix x = Matrix(3,2, {1,2,3,4,5,6});
    const double *data_ = x.data();
    const size_t x_size = x.size();
    const size_t x_nrows = x.nrows();
    const size_t x_ncols = x.ncols();

    Matrix x2; x2 = std::move(x);

    assert(x.size() == 0);
    assert(x.nrows() == 0);
    assert(x.ncols() == 0);
    assert(x.data() != data_);

    assert(x2.data() == data_);
    assert(x2.size() == x_size);
    assert(x2.nrows() == x_nrows);
    assert(x2.ncols() == x_ncols);
    for(size_t i = 0; i < x_size; ++i)
      assert( x2[i] == i+1);
  }

  std::cout << "OK.\n";
}

void test_load_save(){
  std::cout << "test_load_save...\n";

  Matrix m(3,4, {0,1,2,3,4,5,6,7,8,9,10,11});
  // Save and load in binary
  m.save("/tmp/test_matrix.pml");
  Matrix m2 = Matrix::load("/tmp/test_matrix.pml");
  assert(fequal(m, m2));

  // Save and load in text
  m.saveTxt("/tmp/test_matrix.txt");
  Matrix m3 = Matrix::loadTxt("/tmp/test_matrix.txt");
  assert(fequal(m, m3));

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
  assert(all(a.row(0) == 3));
  assert(all(a.row(1) == 6));
  assert(all(a.row(2) == 9));

  a = a + b;
  a = a - b;
  a = a / b;

  // Dot Product
  Vector v({1,2,3});
  Matrix m(2,3,{1,2,3,4,5,6});
  assert(fequal(dot(m, v), Vector({22, 28})));
  assert(fequal(dot(m,tr(m)), Matrix(2,2, {35, 44, 44, 56})));

  std::cout << "OK\n";
}


void test_rows_cols() {

  std::cout << "test_rows_cols...\n";
  {
    Matrix m(3, 4, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

    // Get Cols
    assert(fequal(m.col(0), Vector({0, 1, 2})));
    assert(fequal(m.col(1), Vector({3, 4, 5})));
    assert(fequal(m.col(2), Vector({6, 7, 8})));
    assert(fequal(m.col(3), Vector({9, 10, 11})));

    // Get Rows
    assert(fequal(m.row(0), Vector({0, 3, 6, 9})));
    assert(fequal(m.row(1), Vector({1, 4, 7, 10})));
    assert(fequal(m.row(2), Vector({2, 5, 8, 11})));
  }

  // Set Cols
  {
    Matrix x(3, 4, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    Matrix y(3, 4, {-1, -1, -1, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    Matrix z(3, 4, {-1, -1, -1, 3, 4, 5, -7, -8, -9, 9, 10, 11});

    x.col(0) = -1;
    assert(fequal(x ,y));

    x.col(2) = Vector({-7, -8, -9});
    assert(fequal(x ,z));

  }

  // Set Rows
  {
    Matrix x(3, 4, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    Matrix y(3, 4, {-1, 1, 2, -1, 4, 5, -1, 7, 8, -1, 10, 11});
    Matrix z(3, 4, {-1, 1, -6, -1, 4, -7, -1, 7, -8, -1, 10, -9});

    x.row(0) = -1;
    assert(fequal(x, y));

    x.row(2) = Vector({-6,-7,-8,-9});
    assert(fequal(x, z));
  }

  std::cout << "OK\n";
}


void test_matrix_functions(){
  std::cout << "test_matrix_functions...\n";

  // Sum, Min, Max
  Matrix m(2, 3, {0, 1, 2, 3, 4, 5});
  assert(sum(m) == 15);
  assert(fequal(sum(m,0), Vector({1, 5, 9})));
  assert(fequal(sum(m,1), Vector({6, 9})));

  assert(min(m) == 0);
  assert(fequal(min(m,0), Vector({0, 2, 4})));
  assert(fequal(min(m,1), Vector({0, 1})));

  assert(max(m) == 5);
  assert(fequal(max(m,0), Vector({1, 3, 5})));
  assert(fequal(max(m,1), Vector({4, 5})));

  // Round and Abs
  Matrix m2(3, 5, -2.3);
  assert(all(abs(m2) == 2.3));
  assert(all(round(abs(m2)) == 2));

  // Log, Exp, Psi
  Matrix m3 = Matrix(3, 4, 0.5);
  assert(all(log(m3) == -0.6931471));
  assert(all(exp(m3) == 1.64872127));

  // Normalizations
/*
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
  assert(fequal(logSumExp(m5), std::log(10)));
  assert(logSumExp(m5,0).equals(log(sum(m4,0))));
  assert(logSumExp(m5,1).equals(log(sum(m4,1))));

  // Tile
  //Vector v = {1,2};
  //assert(tile(v, 2).equals(Matrix(2,2, {1,1,2,2})));
  //assert(tile(v, 2, 1).equals(Matrix(2,2, {1,2,1,2})));
  //assert(repmat(v, 2, 2).equals(Matrix(4,2, {1,2,1,2,1,2,1,2})));
*/
   std::cout << "OK\n";
}

/*
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

  // append column wise
  Matrix m3 = cat(Matrix(), m1);
  assert(m3.equals(m1));
  m3 = cat(m1, m1);
  for(int i=0; i < 4; ++i)
    assert(m3.getColumn(i).equals(Vector({1,2,3,4})));

  // append row wise
  Matrix m4 = cat(Matrix(), m2, 0);
  assert(m4.equals(m2));
  m4 = cat(m2, m2, 0);
  for(int i=0; i < 4; ++i)
    assert(m4.getRow(i).equals(Vector({1,2,3,4})));

  std::cout << "OK\n";
}
*/

int main(){
  test_matrix();
  test_matrix_copy_constructors();
  test_load_save();
  test_matrix_algebra();
  test_rows_cols();
  test_matrix_functions();
  //test_matrix_append();
  return 0;
}

