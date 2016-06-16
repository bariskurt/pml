//
// Created by bariskurt on 26.03.2016.
//

#include <cassert>

#include "pml.hpp"
#include "pml_rand.hpp"

using namespace pml;

std::string test_dir = "/tmp/";

void Test_Object(){

  std::cout << "Test_Vector::Object       ";
  Vector v(3); v(0) = 2; v(1) = 3; v(2) = 4;
  Vector v1 = uniform::rand(100);
  Vector v2(v1);
  Vector v3;
  v3 = v2;

  assert(v == Vector({2,3,4}));
  assert(v1 == v2);
  assert(v1 == v3);
  assert(uniform::rand(100) != uniform::rand(1000));

  std::cout << "OK.\n";

}


void Test_SaveLoad(){
  std::cout << "Test_Vector::SaveLoad     ";

  Vector v1 = uniform::rand(1000);
  v1.Save(test_dir + "vector.txt");

  Vector v2 = Vector::Load(test_dir + "vector.txt");

  v2.Save(test_dir + "vector2.txt");

  assert( v1 == v2);

  std::cout << "OK.\n";
}


void Test_Sum(){
  std::cout << "Test_Vector::Sum          ";
  Vector v1({1, 2, 3, 4, 5});
  assert(v1.sum() == 15);
  std::cout << "OK.\n";
}


void Test_LogExp(){
  std::cout << "Test_Vector::LogExp       ";
  Vector v1({1, 2, 3, 4, 5});
  Vector v2({ 0.0, 0.693147, 1.098612, 1.386294, 1.609437});
  Vector v3({2.718281, 7.389056, 20.085536, 54.598150, 148.413159});
  assert(Log(v1) == v2);
  assert(Exp(v1) == v3);
  std::cout << "OK.\n";
}


void Test_Normalize(){
  std::cout << "Test_Vector::Normalize    ";
  Vector v1 = uniform::rand(1000);
  v1 = Normalize(v1);
  assert(fequal(v1.sum(), 1));
  std::cout << "OK.\n";
}

void Test_NormalizeExp(){
  std::cout << "Test_Vector::NormalizeExp ";
  Vector v1({5/10.0, 3/10.0, 2/10.0});
  assert( Normalize(v1) == NormalizeExp(Log(v1)));
  std::cout << "OK.\n";
}

void Test_LogSumExp(){
  std::cout << "Test_Vector::LogSumExp    ";
  Vector v({0.1, 0.2, 0.3});
  assert( fequal(LogSumExp(Log(v)), log(v.sum())));
  std::cout << "OK.\n";
}

void Test_Algebra(){
  std::cout << "Test_Vector::Algebra      ";
  Vector v({1,2,3,});
  Vector v2({4,5,6,});

  // Scalars Part 1
  assert(v + 1 == Vector({2,3,4}));
  assert(v - 1 == Vector({0,1,2}));
  assert(v * 2 == Vector({2,4,6}));
  assert(v / 2 == Vector({0.5,1,1.5}));

  // Scalars Part 2
  assert(1 + v == Vector({2,3,4}));
  assert(1 - v == Vector({0,-1,-2}));
  assert(5 * v == Vector({5,10,15}));
  assert(12 / v == Vector({12,6,4}));

  // Scalars Part 3
  v += 1; assert(v == Vector({2,3,4}));
  v -= 1; assert(v == Vector({1,2,3}));
  v *= 2; assert(v == Vector({2,4,6}));
  v /= 2; assert(v == Vector({1,2,3}));



  // Vectors Part 1
  assert(v + v2 == Vector({5,7,9}));
  assert(v - v2 == Vector({-3,-3,-3}));
  assert(v * v2 == Vector({4,10,18}));
  assert(v / v2 == Vector({1.0/4.0, 2.0/5.0, 3.0/6.0}));

  // Vectors Part 2
  v += v2; assert(v == Vector({5,7,9}));
  v -= v2; assert(v == Vector({1,2,3}));
  v *= v2; assert(v == Vector({4,10,18}));
  v /= v2; assert(v == Vector({1,2,3}));

  std::cout << "OK.\n";
}

void Test_Resize(){
  std::cout << "Test_Vector::Resize       ";
  Vector v1( {1, 2, 3, 4, 5, 6, 7, 8, 9});

  // First 5:
  Vector v2 = v1.slice(0, 5, 1);
  assert(v2.length() == 5);
  for(size_t i=0; i< v2.length(); ++i){
    assert(v2(i) == v1(i));
  }

  // Odds:
  Vector v3 = v1.slice(0, 5, 2);
  assert(v3.length() == 5);
  for(size_t i=0; i< v3.length(); ++i){
    assert(v3(i) == v1(2*i));
  }

  // Shrink vector itself:
  v1 = v1.slice(0, 3, 1);
  assert(v1.length() == 3);
  assert(v1(0) == 1);
  assert(v1(1) == 2);
  assert(v1(2) == 3);

  std::cout << "OK.\n";
}


int main(){

  Test_Object();
  Test_SaveLoad();
  Test_Sum();
  Test_Algebra();
  Test_LogExp();
  Test_Normalize();
  Test_NormalizeExp();
  Test_LogSumExp();
  Test_Resize();

  return 0;
}