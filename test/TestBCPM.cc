
#include "pml.hpp"
#include "pml_bcpm.hpp"

using namespace std;
using namespace pml;

int DIM = 10;
int LAG = 0;
int T = 200;


int main() {

  DirichletModel dirichletModel(0.05, Vector::ones(DIM));
  ForwardBackward fb(&dirichletModel, LAG);

  pair<Matrix, Vector> data = dirichletModel.generateData(T);
  Matrix obs = data.first;
  Vector cps = data.second;

  time_t beg = time(nullptr);
  for (size_t t=0; t<obs.ncols(); t++) {
    fb.processObs(obs.getColumn(t));
    if (t % 100 == 0) {
      cout << t << "obs processed in " << time(nullptr) - beg << " seconds" << endl;
    }
  }

  vector<int> diffs;
  for (int t=0; t<T; t++) {
    if ( abs(cps(t)-fb.cpp(t)) > 1e-1 ) {
      diffs.push_back(t);
    }
  }

  cout << "Total elapsed time: " << time(nullptr) - beg << ". Number of difference: " << diffs.size() << endl;


  obs.saveTxt("/tmp/obs.txt");
  cps.saveTxt("/tmp/real_cps.txt");
  fb.cpp.saveTxt("/tmp/cpp.txt");
  fb.mean.saveTxt("/tmp/mean.txt");

  cout << system("/home/cagatay/anaconda3/bin/python ../etc/hist_plot.py");


  return 0;
}