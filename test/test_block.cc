#include <cassert>

#include "pml_block.hpp"

using namespace pml;

std::string test_dir = "/tmp/";

void test_constructors(){

  std::cout << "test_constructors...\n";

  // Empty block
  Block b;
  assert(b.size() == 0);
  assert(b.stride() == 1);
  assert(b.is_owner());
  assert(b.data() == nullptr);

  // NonEmpty block
  size_t size = 10;
  Block b2(size);
  for(size_t i = 0; i < size; ++i){
    b2[i] = i;
  }
  assert(b2.size() == 10);  
  for(size_t i = 0; i < size; ++i){
    assert(b2[i] == i);
  }

  // Copy constructor (deep copy)
  Block b3(b2);
  assert(b3.size() == 10);
  assert(b3.stride() == 1);
  assert(b3.is_owner());
  assert(b3.data() != b2.data());
  for(size_t i = 0; i < size; ++i){
    assert(b3[i] == i);
  }

  // Assigment operator
  Block b4;
  b4 = b2;
  assert(b4.size() == 10);
  assert(b4.stride() == 1);
  assert(b4.is_owner());
  assert(b4.data() != b2.data());
  for(size_t i = 0; i < size; ++i){
    assert(b4[i] == i);
  }

  std::cout << "OK.\n";
}

void test_size(){

  std::cout << "test_size...\n";

  Block b(5);
  for(size_t i = 0; i < 5; ++i){
    b[i] = i;
  }
  b.append(5);
  b.append(6);
  b.append(7);
  b.append(8);
  b.append(9);

  assert(b.size() == 10);
  assert(b.stride() == 1);
  assert(b.is_owner());
  for(size_t i = 0; i < 10; ++i){
    assert(b[i] == i);
  }

  b.resize(4);
  assert(b.size() == 4);
  for(size_t i = 0; i < 4; ++i){
    assert(b[i] == i);
  }

  b.resize(20);
  assert(b.size() == 20);
  for(size_t i = 0; i < 4; ++i){
    assert(b[i] == i);
  }

  std::cout << "OK.\n";
}

void test_stride(){
  std::cout << "test_stride...\n";
  size_t size = 10;
  Block b(size);
  for(size_t i = 0; i < size; ++i){
    b[i] = i;
  }
  double *b_data_old = b.data();

  Block b2(b.data(), b.size() / 2, 2);
  assert(b2.size() == size / 2);
  assert(b2.stride() == 2);
  assert(!b2.is_owner());
  assert(b2.data() == b.data());
  for(size_t i = 0; i < size / 2; ++i){
    assert(b2[i] == 2*i);
  }

  Block b3(size);
  for(size_t i = 0; i < size; ++i){
    b3[i] = 5*i;
  }
  b2 = b3;
  assert(b2.size() == size);
  assert(b2.stride() == 1);
  assert(b2.is_owner());
  assert(b2.data() != b.data());
  assert(b.data() == b_data_old);
  for(size_t i = 0; i < size; ++i){
    assert(b2[i] == (5*i));
  }

  std::cout << "OK.\n";
}


int main(){
  test_constructors();
  test_size();
  test_stride();
}