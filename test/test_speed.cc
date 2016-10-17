#include <cassert>
#include <chrono>

#include "pml.hpp"

using namespace pml;

void test_column_row_ops(){
  std::cout << "test_column_row_ops...\n";

  auto t_start = std::chrono::system_clock::now();

  Matrix m = Uniform(0, 10).rand(1000,1000);

  for(int i=0; i < 10; ++i){
    Vector v1 = logSumExp(m, 0);
    Vector v2 = logSumExp(m, 1);
  }

  auto t_end = std::chrono::system_clock::now();
  std::chrono::duration<double> t_elapsed = t_end-t_start;
  std::cout << "time elapsed: " <<  t_elapsed.count() <<" seconds.\n";
}


int main(){
  test_column_row_ops();
  return 0;
}