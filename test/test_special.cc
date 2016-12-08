#include <cassert>

#include "pml_special.hpp"

using namespace pml;

void test_special(){

  std::cout << "test_special()...\n";

  Vector v1 = {1, 2, 3, 4, 5};

  Vector lgamma_v1 = {0, 0, 0.693147, 1.791759, 3.178053};
  assert(gammaln(v1).equals(lgamma_v1));

  Vector psi_v1 = {-0.577215, 0.422784, 0.922784, 1.256117, 1.506117};
  assert(psi(v1).equals(psi_v1));

  Vector psi_v11 = {1.644934, 0.644934 ,0.394934 ,0.283822 ,0.221322};
  assert(psi(v1,1).equals(psi_v11));

  std::cout << "OK.\n";

}


int main(){
  test_special();
  return 0;
}