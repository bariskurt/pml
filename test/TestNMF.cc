#include "pml_nmf.hpp"

using namespace pml;


void test_nmf_ml(){

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


void test_nmf_vb(){

  size_t dim1 = 10;
  size_t dim2 = 40;
  size_t rank = 3;

  double at = 0.2;
  double bt = 1;

  double av = 10;
  double bv = 1;

  NMF nmf(dim1, dim2, rank, at, bt, av, bv);

  std::cout << "hello\n";
  Matrix X = nmf.randgen();


  auto solution = nmf.vb(X);

  // Save experiment
  X.saveTxt("/tmp/x.txt");
  solution.save("/tmp");
}

int main(){

  // test_nmf_ml();
  test_nmf_vb();

  return 0;
}