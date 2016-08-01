
#include "pml.hpp"
#include "pml_bcpm.hpp"

using namespace std;
using namespace pml;

int DIM = 10;
int T = 200;

int main() {

  DirichletModel dirichletModel(0.05, Vector::ones(DIM));
  ForwardBackward fb(&dirichletModel, DIM);

  time_t beg = time(nullptr);


  pair<Matrix, Vector> data = dirichletModel.generateData(T);
  Matrix obs = data.first;
  Vector cps = data.second;
  obs.saveTxt("/tmp/obs.txt");
  cps.saveTxt("/tmp/real_cps.txt");
  // fb.smoothing(obs);

  dirichletModel.alpha = Vector::ones(DIM)*10;

  for (size_t t=0; t<obs.ncols(); t++) {
    fb.processObs(obs.getColumn(t));
    if (t % 100 == 0) {
      cout << t << "obs processed in " << time(nullptr) - beg << " seconds" << endl;
    }
  }

  cout << "Total elapsed time: " << time(nullptr) - beg << endl;

  fb.cpp.saveTxt("/tmp/cpp.txt");
  fb.mean.saveTxt("/tmp/mean.txt");

  cout << system("/home/cagatay/anaconda3/bin/python ../etc/hist_plot.py");

  return 0;
}