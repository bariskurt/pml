#include "pml_nmf.hpp"

using namespace pml;


void test_nmf(){

  std::cout << "test_nmf()...\n";

  size_t dim1 = 10;
  size_t dim2 = 40;
  size_t rank = 3;

  Matrix X = NMF::randgen(dim1, dim2, rank);


  auto solution = NMF::ml(X, rank);

  // Save experiment
  X.saveTxt("/tmp/x.txt");
  solution.save("/tmp");

  std::cout << "done\n";
}


int main(){

  test_nmf();

  return 0;
}