#include "pml_lds.hpp"

using namespace pml;

void saveSequence(const std::pair<Vector, Vector> &sequence){
  Matrix temp;
  temp.appendColumn(sequence.first);
  temp.appendColumn(sequence.second);
  temp.saveTxt("/tmp/prm.txt");
}

void testPoissonResetModel(){
  PoissonResetModel m(0.05, 10, 1);

  //auto seq = m.generateSequence(250);
  //Vector &obs = seq.second;
  //saveSequence(seq);

  Matrix temp = Matrix::loadTxt("/home/bariskurt/Desktop/test_data/prm.txt");
  Vector obs = temp.getColumn(1);

  Vector mean, cpp;
  std::tie(mean, cpp) = m.forward_filter(obs);
  mean.saveTxt("/tmp/mean.txt");
  cpp.saveTxt("/tmp/cpp.txt");

  std::cout << system("python3 ../test/python/visualizePoissonReset.py");
}

int main(){
  testPoissonResetModel();
  return 0;
}