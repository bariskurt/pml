
#include "pml.hpp"
#include "pml_bcpm.hpp"

using namespace std;
using namespace pml;

const int T = 50;
const int K = 10;
const int LAG = 0;

const int EM_MAX_ITER = 15;
const int INV_DIG_ITER = 100;
const double EM_EPS = 1e-5;

const double DIGAMMA_LOWER = 1e-3;
const double POLYGAMMA_LOWER = 1e-2;
const double INVERSE_GAMMA_UPPER = 20;


/*
 * checks whether any mean vector in asmoothed message contains
 * a value that is less than DIGAMMA_LOWER
 */
bool smtMsgIsSmall(const vector<Message*>& smoothed_messages) {
  for (size_t i=0; i<smoothed_messages.size(); i++) {
    for (Component* comp : smoothed_messages[i]->components) {
      DirichletComponent* d = static_cast<DirichletComponent*>(comp);
      for(size_t j=0; j<d->alpha.size(); j++) {
        if (d->alpha(j) < DIGAMMA_LOWER) {
          cout << "Dirichlet potential with negative support!" << endl;
          return true;
        }
      }
    }
  }
  return false;
}

double my_digamma(double x, double THRESHOLD=DIGAMMA_LOWER) {
  if (abs(x) <= THRESHOLD) {
    throw invalid_argument( "too small for digamma" );
  }
  return gsl_sf_psi(x);
}

Vector my_digamma(Vector vec, double THRESHOLD=DIGAMMA_LOWER) {
  Vector y;
  for(size_t i=0; i<vec.size(); i++) {
    y.append(my_digamma(vec(i),THRESHOLD));
  }
  return y;
}

double my_polygamma(double x, double THRESHOLD=POLYGAMMA_LOWER) {
  if (x <= THRESHOLD) {
    throw invalid_argument( "too small for polygamma" );
  }
  return gsl_sf_psi_1(x);
}

Vector my_polygamma(Vector vec, double THRESHOLD=POLYGAMMA_LOWER) {
  Vector y;
  for(size_t i=0; i<vec.size(); i++) {
    y.append(my_polygamma(vec(i),THRESHOLD));
  }
  return y;
}

Vector inv_digamma(Vector y, double THRESHOLD=INVERSE_GAMMA_UPPER, double eps_=1e-5) {
  // check if params are valid
  for(size_t i=0; i<y.size(); i++) {
    if (y(i) >= THRESHOLD) {
      throw invalid_argument( "too big for inv_gamma" );
    }
  }
  // find the initial x
  Vector x;
  for(size_t i=0; i<y.size(); i++) {
    if (y(i) > -2.22) {
      x.append(exp(y(i))+0.5);
    }
    else {
      x.append(-1/(y(i)-my_digamma(1)));
    }
  }
  // newton iterations
  while ( sum(abs(my_digamma(x)-y)) >  eps_) {
    try {
      x -= (my_digamma(x)-y) / my_polygamma(x);
    }
    catch (const string msg) {
      cout << "inv_gamma: newton interations terminated. " << msg << endl;
      break;
    }
  }
  return x;
}

/*
 * takes a smoothed message, p(\pi_t | x_{1:T},\theta), and
 * returns the sufficient statistics
 */
Vector compute_ss(Message*& msg) {
  DirichletComponent* d = static_cast<DirichletComponent*>(msg->components[0]);
  size_t K = d->alpha.size();
  size_t M = msg->components.size();
  // put params into Vector & Matrix
  Matrix tmp;
  Vector log_norm_consts;
  for(size_t i=0; i<M; i++) {
    d = static_cast<DirichletComponent*>(msg->components[i]);
    log_norm_consts.append(d->log_c);
    tmp.appendColumn( my_digamma(d->alpha) - my_digamma(sum(d->alpha)) );
  }
  log_norm_consts -= max(log_norm_consts);
  Vector norm_consts = exp(log_norm_consts);
  Vector ss(K);
  for(size_t i=0; i<M; i++) {
    ss += tmp.getColumn(i)*norm_consts(i);
  }
  return ss / sum(norm_consts);
}

void EM(const Matrix& data, double c=0.2, Vector alpha=uniform::rand(K)*10) {
  Vector loglhoods;
  size_t T = data.ncols();

  for(size_t epoch=0; epoch<EM_MAX_ITER; epoch++) {
    // init model
    DirichletModel dirichletModel(c, alpha);
    ForwardBackward fb(&dirichletModel, LAG);

    ///////////// log-likelihood calculation /////////////
    fb.smoothing(data);
    // double ll = ForwardBackward::loglhood(fb.model, data);
    Vector consts;
    vector<Component*> comps = fb.alpha_update.back()->components;
    for (size_t i=0; i<comps.size(); i++) {
      consts.append(comps[i]->log_c);
    }
    double ll = logSumExp(consts);
    loglhoods.append(ll);
    cout << "ll is " << ll << endl;
    cout << "\t\tmax(a) is " << max(alpha) << endl;

    //////////////// check convergence //////////////////
    if (epoch>0) {
      if (loglhoods(epoch) - loglhoods(epoch-1) < 0) {
        cout << "*** LIKELIHOOD DECREASED ***" << endl;
        if (smtMsgIsSmall(fb.smoothed_msgs)) {
          break;
        }
      }
      else if (loglhoods(epoch) - loglhoods(epoch-1) < EM_EPS) {
        cout << "*** CONVERGED ***" << endl;
        break;
      }
    }

    ////////////////////// E step ///////////////////////
    Matrix E_log_pi_weighted;
    for (size_t j=0; j<T; j++) {
      Vector ss_j = compute_ss(fb.smoothed_msgs[j])*fb.cpp(j);
      E_log_pi_weighted.appendColumn(ss_j);
    }
    Vector ss = sum(E_log_pi_weighted,1) / sum(fb.cpp);


    ////////////////////// M step ///////////////////////
    int iter=0;
    for(; iter<INV_DIG_ITER; iter++) {
      try {
        my_digamma(sum(alpha));
      }
      catch (const string msg) {
        cout << "digamma: Terminated at M step. num steps is " << iter << endl;
        break;
      }
      try {
        alpha = inv_digamma( ss + my_digamma(sum(alpha)) );
      }
      catch (const string msg) {
        cout << "inv_digamma: Terminated at M step. num steps is " << iter << endl;
        cout<< "ss: " << ss << endl;
        cout << "my_digamma(a.sum()): " << my_digamma(sum(alpha)) << endl;
        break;
      }
    }
    if (iter>0) {
      c = sum(fb.cpp)/T;
    }
  }
}

void exampleRun() {
  DirichletModel dirichletModel(0.05, Vector::ones(K));
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
  for (size_t t=0; t<T; t++) {
    if ( abs(cps(t)-fb.cpp(t)) > 1e-1 ) {
      diffs.push_back(t);
    }
  }

  cout << "Total elapsed time: " << time(nullptr) - beg << ". Number of incorrect labels: " << diffs.size() << endl;

  cout << fb.cpp << endl;

  obs.saveTxt("/tmp/obs.txt");
  cps.saveTxt("/tmp/real_cps.txt");
  fb.cpp.saveTxt("/tmp/cpp.txt");
  fb.mean.saveTxt("/tmp/mean.txt");

  // cout << system("/home/cagatay/anaconda3/bin/python ../etc/hist_plot.py");
}

int main() {

  /*
  DirichletModel dirichletModel(0.05, Vector::ones(K));
  ForwardBackward fb(&dirichletModel, LAG);

  pair<Matrix, Vector> data = dirichletModel.generateData(T);
  Matrix obs = data.first;
  Vector cps = data.second;

  */
  Matrix obs = Matrix::loadTxt("/tmp/data.txt");
  Vector alpha = Vector::loadTxt("/tmp/alpha.txt");
  double c = 0.2;

  EM(obs, c, alpha);

  return 0;
}