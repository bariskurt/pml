#include <pml_bcpm.hpp>

using namespace std;
using namespace pml;

// constants for bcpm
const int K = 10;
const int T = 100;
const int LAG = 10;

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
  cout << system("python3 ../etc/hist_plot.py");
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

  // You can analyze the data either in offline mode or on streaming data.
  // If you use the latter, you need to set the LAG field (to an integer)
  // in ForwardBackward object for fixed lag smoothing.

  // offline
  offline(obs, fb);      // Runs forward - backward

  // on streaming data
  // streaming(obs,fb);  // Runs forward - fixed lag smoothing

  // visualization
  visualize(obs, cps, fb);

  return 0;
}