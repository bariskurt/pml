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

void Test_Object(){

  std::cout << "Test_Block::Object        ";
  Block b1(3); b1(0) = 2; b1(1) = 3; b1(2) = 4;
  //Vector v1 = random::Rand(100);
  Block b2(b1);
  Block b3;
  b3 = b2;

  assert(b1 == Block({2,3,4}));
  assert(b1 == b2);
  assert(b1 == b3);

  std::cout << "OK.\n";

}

void Test_Sum(){
  std::cout << "Test_Block::Sum           ";
  Block b( {1, 2, 3, 4, 5});
  assert(b.sum() == 15);
  std::cout << "OK.\n";
}

/*
void Test_LogExp(){
  std::cout << "Test_Block::LogExp        ";
  Block b1({1, 2, 3, 4, 5});
  Block b2({ 0.0, 0.693147, 1.098612, 1.386294, 1.609437});
  Block b3({2.718281, 7.389056, 20.085536, 54.598150, 148.413159});
  assert(Log(b1) == b2);
  assert(Exp(b1) == b3);
  b2.exp();
  b3.log();
  assert(b1 == b2);
  assert(b1 == b3);
  std::cout << "OK.\n";
}
*/


void Test_Algebra(){
  std::cout << "Test_Block::Algebra       ";
  Block b({1,2,3,});
  Block b2({4,5,6,});

  b += 1; assert(b == Block({2,3,4}));
  b -= 1; assert(b == Block({1,2,3}));
  b *= 2; assert(b == Block({2,4,6}));
  b /= 2; assert(b == Block({1,2,3}));


  b += b2; assert(b == Block({5,7,9}));
  b -= b2; assert(b == Block({1,2,3}));
  b *= b2; assert(b == Block({4,10,18}));
  b /= b2; assert(b == Block({1,2,3}));

  b = 5;   assert(b == Block({5,5,5}));

  std::cout << "OK.\n";
}

void Test_Nan_Inf() {
  std::cout << "Test_Block::Nan_Inf       ";
  Block b({1,2,3,}); assert(!b.ContainsNan()); assert(!b.ContainsInf());
  b(0) =  sqrt(-1);   assert(b.ContainsNan());
  b(1) = INFINITY; assert(b.ContainsInf());

  std::cout << "OK.\n";
}


int main(){

  Test_Object();
  Test_Algebra();
  Test_Nan_Inf();

  return 0;
}