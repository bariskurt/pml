//
// Created by baris on 01.04.2016.
//

//
// Created by bariskurt on 26.03.2016.
//

#include <cassert>

#include "pml.hpp"

using namespace pml;

std::string test_dir = "/tmp/";

void test_array(){

  std::cout << "test_array...";

  // First constructor
  Array a1({4,5,6}, 5);
  assert(a1.size() == 4*5*6);
  assert(a1.first() == 5);
  assert(a1.last() == 5);
  assert(a1 == 5); // checks all elements
  assert(a1.ndims() == 3);
  assert(a1.dim(0) == 4);
  assert(a1.dim(1) == 5);
  assert(a1.dim(2) == 6);

  // Second constructor
  Array a2({2, 3}, {1,2,3,4,5,6});
  for(size_t i = 0; i < a2.size(); ++i){
    assert(a2(i) == i+1);
  };
  assert(a1 != a2);

  // Copy constructor
  Array a3(a2);
  assert( a2 == a3 );

  // Move-Copy constructor
  Array a4(std::move(a2));
  assert( a3 == a4 );
  assert( a2.empty() );

  // Assignment
  Array b3;
  b3 = a3;
  assert( a3 == b3 );

  // Move-Assignment
  Array b4;
  b4 = std::move(a4);
  assert( b4 == a3 );
  assert( a4.empty() );

  std::cout << "OK.\n";
}

void test_algebra(){
  std::cout << "test_algebra...";
  Array A({2,2,3}, 3);
  Array B({2,2,3}, 5);

  // A = A op b
  A += 1; assert(A == 4);
  A -= 1; assert(A == 3);
  A *= 2; assert(A == 6);
  A /= 2; assert(A == 3);


  // A = A op B
  A += B; assert(A == 8);
  A -= B; assert(A == 3);
  A *= B; assert(A == 15);
  A /= B; assert(A == 3);

  // C = A op b
  // C = b op A
  Array C;
  C = A + 1; assert(C == 4);
  C = 1 + A; assert(C == 4);
  C = A - 1; assert(C == 2);
  C = 1 - A; assert(C == -2);
  C = A * 2; assert(C == 6);
  C = 2 * A; assert(C == 6);
  C = A / 2; assert(C == 1.5);
  C = 2 / A; assert(C == 2.0/3.0);

  // C = A op B
  C = A + B; assert(C == 8);
  C = A - B; assert(C == -2);
  C = A * B; assert(C == 15);
  C = A / B; assert(C == 3.0/5.0);

  std::cout << "OK.\n";
}

void test_friends(){
  std::cout << "test_friends...";
  Array A({10}, {0,1,2,3,4,5,6,7,8,9});

  // Sum, Min, Max
  assert(sum(A) == 45);
  assert(min(A) == 0);
  assert(max(A) == 9);

  // Abs, Round
  A = Array({3}, -2.3);
  assert(abs(A) == 2.3);
  assert(round(abs(A)) == 2);

  // Exp, Log
  A = Array({3}, 0.5);
  assert(log(A) == -0.6931471);
  assert(exp(A) == 1.64872127);
  assert(psi(A) == -1.96351002);

  // normalize, normalizeExp, logSumExp
  A = Array({4}, {1, 2, 3, 4});
  assert(normalize(A) == Array({4}, {0.1, 0.2, 0.3, 0.4}));
  assert(normalizeExp(log(A)) == Array({4}, {0.1, 0.2, 0.3, 0.4}));
  assert(logSumExp(log(A)) == std::log(10));

  //KL Divergence
  Array A1 = normalize(Array({4}, {1, 2, 3, 4}));
  Array A2 = normalize(Array({4}, {2, 2, 2, 2}));
  assert(fequal(klDiv(A1,A2), 0.1064401));

  std::cout << "OK.\n";
}

void test_save_load(){

  std::cout << "test_save_load...";
  Array A({5,2}, {0,1,2,3,4,5,6,7,8,9});
  A.save("/tmp/a.txt");
  Array B = Array::load("/tmp/a.txt");
  assert(A == B);
  std::cout << "OK.\n";
}

int main(){

  test_array();
  test_algebra();
  test_friends();
  test_save_load();

  return 0;
}