#include <cassert>

#include "pml_memory.hpp"
#include "pml_time.hpp"

#include <vector>

using namespace pml;

template <class Alloc>
void test_speed(){
  std::cout << "test_speed...\n";
  pml::TicTocTimer ticTocTimer;
  ticTocTimer.tic();
  int counter = 0;
  for(int i=0; i < 10000; ++i){
    ++counter;
    std::vector<double> v;
    v.reserve(4096);
    v[4095] = 6;
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

int main(){

  Memory::init(Memory::DEFAULT_SIZE);
  test_speed<std::allocator<double>>();
  //test_speed<Allocator<double>>();
  test_speed2();
  test_speed3();

  return 0;
}

