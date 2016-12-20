#include <cassert>

#include "pml_block.hpp"
#include "pml_memory.hpp"
#include "pml_time.hpp"

#include <vector>

using namespace pml;

/*

template <class Alloc>
void test_speed(){
  std::cout << "test_speed...\n";
  pml::TicTocTimer ticTocTimer;
  ticTocTimer.tic();
  for(int i=0; i < 10000; ++i){
    std::vector<double> v(4096);
  }
  std::cout << ticTocTimer.toc().to_string() << std::endl;
  std::cout << "OK.\n";
}

void test_speed2(){
  std::cout << "test_speed2...\n";
  pml::TicTocTimer ticTocTimer;
  ticTocTimer.tic();
  for(int i=0; i < 10000; ++i){
    double *data = (double*) Memory::malloc(sizeof(double) * 4096);
    Memory::free(data);
  }
  std::cout << ticTocTimer.toc().to_string() << std::endl;
  std::cout << "OK.\n";
}

void test_speed3(){
  std::cout << "test_speed2...\n";
  pml::TicTocTimer ticTocTimer;
  ticTocTimer.tic();
  for(int i=0; i < 10000; ++i){
    double *data = (double*) std::malloc(sizeof(double) * 4096);
    std::free(data);
  }
  std::cout << ticTocTimer.toc().to_string() << std::endl;
  std::cout << "OK.\n";
}

void test_speed4(){
  std::cout << "test_speed4...\n";
  pml::TicTocTimer ticTocTimer;
  ticTocTimer.tic();
  for(int i=0; i < 10000; ++i){
    Block b(4096);
  }
  std::cout << ticTocTimer.toc().to_string() << std::endl;
  std::cout << "OK.\n";
}
*/

void test_speed(){
  std::cout << "Creating std::vector of size 4096...\n";
  const size_t num_repeat = 10000;
  const size_t vec_size = 4096;
  pml::TicTocTimer ticTocTimer;
  ticTocTimer.tic();
  for(size_t i=0; i < num_repeat; ++i){
    std::vector<double> v(vec_size);
  }
  std::cout << ticTocTimer.toc().to_string() << std::endl;

  std::cout << "Creating block of size 4096...\n";
  ticTocTimer.tic();
  for(size_t i=0; i < num_repeat; ++i){
    Block b(vec_size);
  }
  std::cout << ticTocTimer.toc().to_string() << std::endl;
  std::cout << "OK.\n";
}


void test_push_back(){
  std::cout << "push_back std::vector of size 4096...\n";
  const size_t num_repeat = 10000;
  const size_t vec_size = 4096;
  pml::TicTocTimer ticTocTimer;
  ticTocTimer.tic();
  for(size_t i=0; i < num_repeat; ++i){
    std::vector<double> v;
    for(size_t j=0; j < vec_size; ++j)
      v.push_back(j);
  }
  std::cout << ticTocTimer.toc().to_string() << std::endl;

  std::cout << "push_back block of size 4096...\n";
  ticTocTimer.tic();
  for(size_t i=0; i < num_repeat; ++i){
    Block b;
    for(size_t j=0; j < vec_size; ++j)
      b.push_back(j);
  }
  std::cout << ticTocTimer.toc().to_string() << std::endl;
  std::cout << "OK.\n";
}



int main(){

  Memory::init(Memory::DEFAULT_SIZE);
  /*test_speed<std::allocator<double>>();
  //test_speed<Allocator<double>>();
  test_speed2();
  test_speed3();
  test_speed4();
   */

  test_speed();
  test_push_back();

  return 0;
}

