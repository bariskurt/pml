#include "pml_utils.hpp"
#include "pml_nmf.hpp"

using namespace pml;


void test_nmf_ml(){

  std::cout << "test_nmf()...\n";

  NMF nmf(10, 40, 3);
  Matrix X = nmf.randgen();
  X.saveTxt("/tmp/x.txt");
  nmf.T.saveTxt("/tmp/t.txt");
  nmf.V.saveTxt("/tmp/v.txt");

  find_or_create("/tmp/sol");
  auto solution = nmf.ml(X);
  solution.save("/tmp/sol");

  if( system("~/Apps/anaconda3/bin/python3 ../test/python/test_nmf.py") )
    std::cout <<"plotting error...\n";
  std::cout << "OK.\n";
}


void test_nmf_vb(){

  std::cout << "test_nmf_vb()...\n";

  NMF nmf(10, 40, 3);
  nmf.At *= 10;
  nmf.Bt *= 1;
  nmf.Av *= 1;
  nmf.Bv *= 100;

  Matrix X = nmf.randgen();
  X.saveTxt("/tmp/x.txt");
  nmf.T.saveTxt("/tmp/t.txt");
  nmf.V.saveTxt("/tmp/v.txt");

  find_or_create("/tmp/sol");
  NMF nmf2(10, 40, 3);
  nmf2.At *= 10;
  auto solution = nmf2.vb(X, "tie_all");
  solution.save("/tmp/sol");

  std::cout << nmf2.Bt(0) << std::endl;
  std::cout << nmf2.Bv(0) << std::endl;

  if( system("~/Apps/anaconda3/bin/python3 ../test/python/test_nmf.py") )
    std::cout <<"plotting error...\n";
  std::cout << "OK.\n";
}

int main(){

//  test_nmf_ml();
  test_nmf_vb();

  return 0;
}