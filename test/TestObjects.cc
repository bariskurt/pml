//
// Created by baris on 01.04.2016.
//

//
// Created by bariskurt on 26.03.2016.
//

#include <cassert>

#include "pml_new.hpp"

using namespace pml;

std::string test_dir = "/tmp/";

void test_vector(){

  std::cout << "test_array...";

  // First constructor
  Vector v({1,2,3,4,5,6});
  assert(v.size() == 6);
  assert(v.first() == 1);
  assert(v.last() == 6);


  // Second constructor
  Vector v2(10, 5);
  assert(v2.size() == 10);
  assert(v2 == 5);

  // Copy constructor
  Vector v3(v2);
  assert( v2 == v3 );

  // Move-Copy constructor
  Vector v4(std::move(v2));
  assert( v3 == v4 );
  assert( v2.empty() );

  // Assignment
  Vector y;
  y = v;
  assert( y == v );

  // Move-Assignment
  Vector z;
  z = std::move(y);
  assert( z == v );
  assert( y.empty() );

  std::cout << "OK.\n";
}


void test_matrix() {
  std::cout << "test_matrix...";

  // Constructors
  Matrix M1 = Matrix(3,2, {1,2,3,4,5,6});
  Matrix M2(M1);
  Matrix M3;
  M3 = M2;

  assert(M1 == M2);
  assert(M1 == M3);

  M1 = Matrix::ones(4,5);
  assert(M1 == 1);

  M1 = Matrix::zeros(4,5);
  assert(M1 == 0);

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
  std::cout << "OK.\n";
}

void test_sum_min_max() {
  std::cout << "test_sum_min_max...";
  Vector x({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

  // Sum, Min, Max
  assert(sum(x) == 45);
  assert(min(x) == 0);
  assert(max(x) == 9);

  Matrix M(2, 5, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  assert(sum(M) == 45);
  assert(min(M) == 0);
  assert(max(M) == 9);

  std::cout << "OK.\n";
}

void test_round_log_exp_psi(){

  std::cout << "test_round_log_exp_psi...";
  // Abs, Round
  Vector x(3, -2.3);
  assert(abs(x) == 2.3);
  assert(round(abs(x)) == 2);

  Matrix m(3, 5, -2.3);
  assert(abs(m) == 2.3);
  assert(round(abs(m)) == 2);

  // Exp, Log
  Vector y;
  x = Vector(3, 0.5);
  y = log(x); assert(similar(x,y)); assert(y == -0.6931471);
  assert(exp(x) == 1.64872127);
  assert(psi(x) == -1.96351002);

  Matrix n;
  m = Matrix(3, 4, 0.5);
  n = log(m); assert(similar(m,n)); assert(n == -0.6931471);
  assert(exp(m) == 1.64872127);
  assert(psi(m) == -1.96351002);

  std::cout << "OK.\n";
}

void test_vector_normalize(){

  std::cout << "test_vector_normalize...";

  // normalize, normalizeExp, logSumExp
  Vector x({1, 2, 3, 4});
  assert(normalize(x) == Vector({0.1, 0.2, 0.3, 0.4}));
  assert(normalizeExp(log(x)) == Vector({0.1, 0.2, 0.3, 0.4}));
  assert(logSumExp(log(x)) == std::log(10));

  //KL Divergence
  Vector y = normalize(Vector({1, 2, 3, 4}));
  Vector z = normalize(Vector({2, 2, 2, 2}));
  assert(fequal(klDiv(y,z), 0.1064401));

  std::cout << "OK.\n";
}

/*
void test_save_load(){

  std::cout << "test_save_load...";
  Array A({5,2}, {0,1,2,3,4,5,6,7,8,9});
  A.save("/tmp/a.txt");
  Array B = Array::load("/tmp/a.txt");
  assert(A == B);
  std::cout << "OK.\n";
}
 */

int main(){

  test_vector();
  test_matrix();
  test_sum_min_max();
  test_vector_normalize();
  //test_algebra();
  //test_friends();
  //test_save_load();

  return 0;
}