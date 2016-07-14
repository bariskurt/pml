//
// Created by bariskurt on 26.03.2016.
//

#include "pml.hpp"
#include "pml_linalg.hpp"
#include "pml_rand.hpp"

#include <cassert>

using namespace pml;

std::string test_dir = "/tmp/";

void Test_Object() {
  std::cout << "Test_Matrix::Object    ";

  Matrix M1 = uniform::rand(100, 1000);
  Matrix M2(M1);
  Matrix M3;
  M3 = M2;

  assert(M1 == M2);
  assert(M1 == M3);
  assert(uniform::rand(3,5) != uniform::rand(30,50));

  std::cout << "OK.\n";
}

void Test_Operators(){
  std::cout << "Test_Matrix::Operators ";
  Matrix M1(2, 3, {1, 4, 2, 5, 3, 6});
  Matrix M2(2, 3, {1, 4, 2, 5, 3, 6});
  Matrix M3(2, 3, {1, 4, 2, 5, 3, 7});
  Matrix M4(2, 2, {1, 3, 2, 4});

  assert(M1 == M2);
  assert(M1 != M3);
  assert(M1 != M4);

  M1 = 7;
  assert(M1 == 7);


  std::cout << "OK.\n";
}

void Test_SpecialMatrices(){
  std::cout << "Test_Matrix::Specials  ";
  Matrix m1 = Matrix::Zeros(3,5);
  for(size_t i = 0; i < m1.length(); ++i){
    assert(m1(i) == 0);
  }

  Matrix m2 = Matrix::Ones(3,5);
  for(size_t i = 0; i < m2.length(); ++i){
    assert(m2(i) == 1);
  }

  Matrix m3 = Matrix::Identity(5);
  for(size_t i = 0; i < m3.num_rows(); ++i){
    for(size_t j = 0; j < m3.num_cols(); ++j) {
      if(i ==j ){
        assert(m3(i,j) == 1);
      } else {
        assert(m3(i,j) == 0);
      }
    }
  }
  std::cout << "OK.\n";
}

void Test_SaveLoad(){
  std::cout << "Test_Matrix::SaveLoad  ";

  Matrix M1 = uniform::rand(100, 1000);
  M1.Save(test_dir + "matrix.txt");

  Matrix M2 = Matrix::Load(test_dir + "matrix.txt");

  assert(M1 == M2);

  std::cout << "OK.\n";

}


void Test_Sum(){
  std::cout << "Test_Matrix::Sum       ";
  /*
     M = [ 1  4  7  10 ]
         [ 2  5  8  11 ]
         [ 3  6  9  12 ]

     cs = [6 15 24 33]
     rs = [22 26 30]
     sum = 78
  */

  Matrix M(3,4);
  for(unsigned i=0; i<M.length(); ++i){
    M(i) = i+1;
  }

  // Column Sums
  assert(SumCols(M) == Vector({6, 15, 24, 33}));

  // Row Sums
  assert(SumRows(M) == Vector({22,26,30}));

  // Sum
  assert(sum(M) == 78);

  // Min
  assert(MinCols(M) == Vector({1,4,7,10}));

  assert(MinRows(M) == Vector({1,2,3}));

  assert(min(M) == 1);

  // Max
  assert(MaxCols(M) == Vector({3, 6, 9, 12}));

  assert(MaxRows(M) == Vector({10,11,12}));

  assert(max(M) == 12);

  std::cout << "OK.\n";
}



void Test_LogExp(){
  std::cout << "Test_Matrix::LogExp    ";

  Matrix M(3,4);
  for(unsigned i=0; i<M.length(); ++i){
    M(i) = i+1;
  }

  Matrix M2 = Log(M);
  for(unsigned i=0; i<M.length(); ++i){
    assert(M2(i) == log(M(i)));
  }

  Matrix M3 = Exp(M);
  for(unsigned i=0; i<M.length(); ++i){
    assert(M3(i) == exp(M(i)));
  }
  std::cout << "OK.\n";
}

void Test_Transpose(){
  std::cout << "Test_Matrix::Transpose ";
  Matrix m = uniform::rand(100,1000);
  assert(m == Transpose(Transpose(m)));
  std::cout << "OK.\n";
}


void Test_Normalize(){
  std::cout << "Test_Matrix::Normalize ";

  // Normalize Columns
  Matrix M = Normalize(uniform::rand(100,1000), Matrix::COLS);
  assert(SumCols(M) == Vector::Ones(1000));

  // Normalize Rows
  M = Normalize(uniform::rand(100,1000), Matrix::ROWS);
  assert(SumRows(M) == Vector::Ones(100));

  // Normalize All
  M = Normalize(uniform::rand(100,1000));
  assert( fabs(sum(M)-1) < 1e-6);

  std::cout << "OK.\n";
}

void Test_Column() {
  std::cout << "Test_Matrix::Column    ";
  Matrix m = uniform::rand(3,4);
  Vector v1 = m.GetColumn(3);
  Vector v2(m.num_rows());
  v2(0) = m(0,3);
  v2(1) = m(1,3);
  v2(2) = m(2,3);
  assert(v1 == v2);

  m.SetColumn(2, Vector::Ones(3));
  assert(m(0,2) == 1);
  assert(m(1,2) == 1);
  assert(m(2,2) == 1);

  std::cout << "OK.\n";
}


void Test_Row(){
  std::cout << "Test_Matrix::Row       ";
  Matrix m = uniform::rand(3,4);
  Vector v1 = m.GetRow(2);
  Vector v2({m(2,0), m(2,1), m(2,2), m(2,3)});
  assert(v1 == v2);

  m.SetRow(1, Vector::Ones(4));
  assert(m(1,0) == 1);
  assert(m(1,1) == 1);
  assert(m(1,2) == 1);
  assert(m(1,3) == 1);

  std::cout << "OK.\n";
}


void Test_Algebra(){
  std::cout << "Test_Matrix::Algebra   ";

  Matrix M(2, 3, {1,2,3,4,5,6});

  assert(M + 1 == Matrix(2, 3, {2,3,4,5,6,7}));
  assert(M - 1 == Matrix(2, 3, {0,1,2,3,4,5}));
  assert(M * 2 == Matrix(2, 3, {2,4,6,8,10,12}));
  assert(M / 2 == Matrix(2, 3, {0.5, 1, 1.5, 2, 2.5, 3}));

  assert(1 + M == Matrix(2, 3, {2,3,4,5,6,7}));
  assert(1 - M == Matrix(2, 3, {0,-1,-2,-3,-4,-5}));
  assert(2 * M == Matrix(2, 3, {2,4,6,8,10,12}));
  assert(60 /M == Matrix(2, 3, {60, 30, 20, 15, 12, 10}));

  M += 1; assert(M == Matrix(2, 3, {2,3,4,5,6,7}));
  M -= 1; assert(M == Matrix(2, 3, {1,2,3,4,5,6}));
  M *= 2; assert(M == Matrix(2, 3, {2,4,6,8,10,12}));
  M /= 2; assert(M == Matrix(2, 3, {1,2,3,4,5,6}));


  Matrix M2(M);

  assert(M + M2 == Matrix(2, 3, {2,4,6,8,10,12}));
  assert(M - M2 == Matrix(2, 3, {0,0,0,0,0,0}));
  assert(M * M2 == Matrix(2, 3, {1,4,9,16,25,36}));
  assert(M / M2 == Matrix(2, 3, {1, 1, 1, 1, 1, 1}));

  M += M2; assert(M == Matrix(2, 3, {2,4,6,8,10,12}));
  M -= M2; assert(M == Matrix(2, 3, {1,2,3,4,5,6}));
  M *= M2; assert(M == Matrix(2, 3, {1,4,9,16,25,36}));
  M /= M2; assert(M == Matrix(2, 3, {1,2,3,4,5,6}));

  Matrix M3(M);
  Vector row({10,20,30});
  Vector col({10,20});
  assert(M3.rows() + row == Matrix(2,3,{11,12,23,24,35,36}));
  assert(col + M3.cols() == Matrix(2,3,{11,22,13,24,15,26}));

  std::cout << "OK.\n";
}

void Test_Dot(){
  Matrix M (3,4);
  for(size_t i=0; i< M.length(); ++i){
    M(i) = i;
  }
  std::cout << M;

  Matrix M2  = Matrix::Ones(4,3);
  std::cout << Dot(M, M2);

  Vector v  = Vector::Ones(4);
  std::cout << Dot(M, v);
}

void Test_Inverse(){
  Matrix M = uniform::rand(2,2);
  Matrix M2 = Inv(M);
  std::cout << M ;
  std::cout << "-----------\n" ;
  std::cout << M2 ;
}

int main(){

  Test_Object();
  Test_Operators();
  Test_SpecialMatrices();
  Test_SaveLoad();
  Test_Algebra();
  Test_Sum();
  Test_LogExp();
  Test_Transpose();
  Test_Normalize();
  Test_Column();
  Test_Row();
  Test_Dot();
  Test_Inverse();

  return 0;
}

