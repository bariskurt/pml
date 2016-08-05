
#include "pml.hpp"
#include "pml_bcpm.hpp"

using namespace std;
using namespace pml;

const int DIM = 10;
const int LAG = 0;
const int T = 20;

const int EM_MAX_ITER = 20;
const int INV_DIG_ITER = 1000;
const double EM_EPS = 1e-5;

const double DIGAMMA_LOWER = 1e-3;
const double POLYGAMMA_LOWER = 1e-2;
const double INVERSE_GAMMA_UPPER = 10;


void EM(const Matrix& data, double c=0.2, Vector alpha=uniform::rand(DIM)) {
  // initial model
  DirichletModel dirichletModel(c, alpha);
  ForwardBackward fb(&dirichletModel, LAG);

  Vector loglhoods;
  for(size_t epoch=0; epoch<EM_MAX_ITER; epoch++) {
    ///////////// log-likelihood calculation /////////////
    double ll = ForwardBackward::loglhood(fb.model, data);
    loglhoods.append(ll);

    //////////////// check convergence //////////////////
    if (epoch>0) {
      if (loglhoods(epoch) - loglhoods(epoch-1) < 0) {
        cout << "*** LIKELIHOOD DECREASED. CHECKING SMOOTHED MESSAGES... ***" << endl;
        for (size_t i=0; i<data.ncols(); i++) {
          bool found = false;
          for (Component* comp : fb.smoothed_msgs[i]->components) {
            DirichletComponent* d = static_cast<DirichletComponent*>(comp);
            for(size_t j=0; j<d->alpha.size(); j++) {
              if (d->alpha(j) < DIGAMMA_LOWER) {
                found = true;
                cout << "Dirichlet potential with negative support!" << endl;
                break;
              }
            }
            if (found) { break; }
          }
          if (found) { break; }
        }
      }
      else if (loglhoods(epoch) - loglhoods(epoch-1) < EM_EPS) {
        cout << "*** CONVERGED ***" << endl;
        break;
      }
    }
  }
}


int main() {

  DirichletModel dirichletModel(0.05, Vector::ones(DIM));
  ForwardBackward fb(&dirichletModel, LAG);

  pair<Matrix, Vector> data = dirichletModel.generateData(T);
  Matrix obs = data.first;
  Vector cps = data.second;

  fb.smoothing(obs);


  for (int i=1; i<100; i+=10) {
    DirichletModel mod(0.05, Vector::ones(DIM)*i);
    cout << "likelihood for i=" << i << " is " << ForwardBackward::loglhood(&mod, obs) << endl;
  }

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

  cout << fb.cpp << endl;

  obs.saveTxt("/tmp/obs.txt");
  cps.saveTxt("/tmp/real_cps.txt");
  fb.cpp.saveTxt("/tmp/cpp.txt");
  fb.mean.saveTxt("/tmp/mean.txt");

  // cout << system("/home/cagatay/anaconda3/bin/python ../etc/hist_plot.py");


  return 0;
}