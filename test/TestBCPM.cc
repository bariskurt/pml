#include <pml_bcpm.hpp>

using namespace std;
using namespace pml;

// constants for bcpm
const int K = 10;
const int T = 100;
const int LAG = 10;

// path to python3 binaries
const string PYTHON_PATH = "/home/cagatay/anaconda3/bin/python";

int main() {
  // data generation
  double c = 0.05;
  Vector alpha = Vector::ones(K);
  pair<Matrix,Vector> data =  DirichletModel(c, alpha).generateData(T);
  Matrix obs = data.first;
  Vector cps = data.second;

  // running the model
  DirichletModel dirichletModel(0.01, uniform::rand(K));
  ForwardBackward fb(&dirichletModel, LAG);
  fb.smoothing(obs);

  // visualization
  obs.saveTxt("/tmp/obs.txt");
  fb.cpp.saveTxt("/tmp/cpp.txt");
  fb.mean.saveTxt("/tmp/mean.txt");
  cps.saveTxt("/tmp/real_cps.txt");
  cout << system( (PYTHON_PATH + " ../etc/hist_plot.py").c_str() );

  return 0;
}