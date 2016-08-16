#include <pml_bcpm.hpp>

using namespace std;
using namespace pml;

// constants for bcpm
const int K = 10;
const int T = 100;
const int LAG = 10;

// path to python3 binaries
const string PYTHON_PATH = "/home/cagatay/anaconda3/bin/python";

pair<Matrix,Vector> genData() {
  double c = 0.05;
  Vector alpha = Vector::ones(K);
  return  DirichletModel(c, alpha).generateData(T);
}

void visualize(const Matrix& obs, const Vector& cps, ForwardBackward& fb) {
  obs.saveTxt("/tmp/obs.txt");
  fb.cpp.saveTxt("/tmp/cpp.txt");
  fb.mean.saveTxt("/tmp/mean.txt");
  cps.saveTxt("/tmp/real_cps.txt");
  cout << system( (PYTHON_PATH + " ../etc/hist_plot.py").c_str() );
}

void offline(const Matrix& obs, ForwardBackward& fb) {
  fb.smoothing(obs);
}

void streaming(const Matrix& obs, ForwardBackward& fb)  {
  for(size_t i=0; i<obs.ncols(); i++) {
    fb.processObs(obs.getColumn(i));
  }
}

int main() {
  // data generation
  // feature vectors stay in columns
  pair<Matrix,Vector> data = genData();
  Matrix obs = data.first;
  Vector cps = data.second;

  // model
  DirichletModel dirichletModel(0.01, uniform::rand(K));
  ForwardBackward fb(&dirichletModel, LAG);

  /*
   * you can analyze the data either in offline mode
   * or on streaming data. if you use the latter,
   * you need to set the LAG field (to an integer)
   * in ForwardBackward object for fixed lag smoothing.
   */

  // offline
  offline(obs, fb);

  // on streaming data
  // streaming(obs,fb);

  // visualization
  visualize(obs, cps, fb);

  return 0;
}