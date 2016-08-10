#include <pml_bcpm.hpp>
#include "pml.hpp"
#include "pml_bcpm.hpp"

using namespace std;
using namespace pml;

const int LAG = 10;

const int EM_MAX_ITER = 15;
const int INV_DIG_ITER = 100;
const double EM_EPS = 1e-5;

const double DIGAMMA_LOWER = 1e-3;
const double POLYGAMMA_LOWER = 1e-5;
const double INVERSE_GAMMA_UPPER = 10;

const bool DEBUG = false;

////////////////////// HELPERS //////////////////////
void visualize(const ForwardBackward& fb, const Matrix& obs, Vector cps=Vector::zeros(5), bool draw=true) {
  obs.saveTxt("/tmp/obs.txt");
  fb.cpp.saveTxt("/tmp/cpp.txt");
  fb.mean.saveTxt("/tmp/mean.txt");
  cps.saveTxt("/tmp/real_cps.txt");
  if(draw) { cout << system("/home/cagatay/anaconda3/bin/python ../etc/hist_plot.py"); }
}
pair<Matrix, Vector> genData(size_t T=100, Vector alpha=Vector::ones(10)*5, double c=0.05) {
  return DirichletModel(c, alpha).generateData(T);
}
pair<Matrix, Vector> readData(const string& obs_path="../etc/simulator_logs/log_low_250.txt",
                              const string& cps_path="../etc/simulator_logs/log_low_250_cps.txt") {
  Matrix obs = Matrix::loadTxt(obs_path);
  // if path contains the word "simulator", take the transpose
  if (obs_path.find("simulator") != string::npos) { obs = transpose(obs); }
  Vector cps;
  if(cps_path=="") { cps = Vector::zeros(obs.ncols()); }
  else { cps = Vector::loadTxt(cps_path); }
  return make_pair(obs, cps);
}
pair<Matrix, Vector> crop(const pair<Matrix, Vector>& data, size_t start, size_t end) {
  Matrix obs = data.first;
  Vector cps = data.second;
  Matrix retMat;
  Vector retVec;
  for(size_t i=start; i<end; i++) {
    retMat.appendColumn(obs.getColumn(i));
    retVec.append(cps(i));
  }
  return make_pair(retMat,retVec);
}
////////////////////// HELPERS //////////////////////

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


////////////////////// FUNCTIONS WITH NUMERICAL THRESHOLDS //////////////////////
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
      throw invalid_argument( "too big for inv_digamma" );
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
    catch(exception& ex) {
      cout << "inv_digamma: newton interations terminated. " << ex.what() << endl;
      break;
    }
  }
  return x;
}
////////////////////// FUNCTIONS WITH NUMERICAL THRESHOLDS //////////////////////

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

// not being used
Vector compute_p0(Message* beta_0_1, const Vector& alpha) {
  Message* smoothed_msg = new Message();
  DirichletComponent* prior = new DirichletComponent(alpha, 0);
  for (Component* comp: beta_0_1->components) {
    DirichletComponent* d = static_cast<DirichletComponent*>(comp);
    smoothed_msg->components.push_back(DirichletComponent::multiply(d, prior));
  }
  Vector ss = compute_ss(smoothed_msg);
  delete prior;
  delete smoothed_msg;
  return ss;
}

ForwardBackward EM(ForwardBackward fb, const pair<Matrix, Vector>& data,
                   double c=0.2, Vector alpha=uniform::rand(10)*10, bool visualize_=false) {
  Matrix obs = data.first;
  Vector cps = data.second;

  Vector loglhoods;
  size_t T = obs.ncols();

  size_t epoch=0;
  for(; epoch<EM_MAX_ITER; epoch++) {
    // init model
    DirichletModel dirichletModel = DirichletModel(c, alpha);
    fb = ForwardBackward(&dirichletModel, LAG, 50);

    ///////////// log-likelihood calculation /////////////
    fb.smoothing(obs);
    // double ll = ForwardBackward::loglhood(fb.model, data);
    Vector consts;
    vector<Component*> comps = fb.alpha_update.back()->components;
    for (size_t i=0; i<comps.size(); i++) {
      consts.append(comps[i]->log_c);
    }
    double ll = logSumExp(consts);
    loglhoods.append(ll);
    cout << "ll is " << ll << endl;

    //////////////// check convergence //////////////////
    if (epoch>0) {
      if (loglhoods(epoch) - loglhoods(epoch-1) < 0) {
        cout << "*** LIKELIHOOD DECREASED ***" << endl;
        if (smtMsgIsSmall(fb.smoothed_msgs)) {
          return fb;
        }
        return fb;
      }
      else if (loglhoods(epoch) - loglhoods(epoch-1) < EM_EPS) {
        cout << "*** CONVERGED ***" << endl;
        return fb;
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
      catch (exception& ex) {
        cout << "digamma: Terminated at M step. num steps is " << iter << endl;
        break;
      }
      try {
        inv_digamma( ss + my_digamma(sum(alpha)) );
      }
      catch (exception& ex) {
        cout << "inv_digamma: Terminated at M step. num steps is " << iter << endl;
        if (DEBUG) cout << "ss: " << ss << endl;
        if (DEBUG) cout << "my_digamma(a.sum()): " << my_digamma(sum(alpha)) << endl;
        break;
      }
      alpha = inv_digamma( ss + my_digamma(sum(alpha)) );
    }
    if(visualize_) { visualize(fb, obs, cps); }
    if (iter>0) {
      c = sum(fb.cpp)/T;
    }
  }
  return fb;
}

// can run filtering, smoothing or em. see main function for usage
void runExperiment(ForwardBackward& fb, const string& experiment, const pair<Matrix, Vector>& data, bool visualize_=false) {
  // get data
  Matrix obs = data.first;
  size_t K = obs.nrows();
  size_t T = obs.ncols();
  Vector cps = data.second;
  // run the experiment
  if (experiment=="filtering" || experiment=="smoothing") {
    if (experiment=="filtering" ) {
      for (size_t t=0; t<obs.ncols(); t++) { fb.processObs(obs.getColumn(t)); }
    }
    else {
      fb.smoothing(obs);
      cout << fb.cpp << endl;
    }
    // update results
    for(size_t i=5; i<fb.cpp.size(); i++) {
      for(int j=1; j<5; j++) {
        if (fb.cpp(i-j)>0.2) {
          fb.cpp(i) = 0;
          break;
        }
      }
    }
    // evaluate and display performance
    vector<size_t> incorrect;
    vector<size_t> missed;
    for (size_t t=0; t<T; t++) {
      if (fb.cpp(t)>0.9 && cps(t)==0) { incorrect.push_back(t); }
      else if (fb.cpp(t)<0.9 && cps(t)==1) { missed.push_back(t); }
    }
    cout << "Number of incorrect/missed labels: " << incorrect.size() << "/" << missed.size() << endl;
    if(visualize_) { visualize(fb, obs, cps); }
  }
  else if (experiment == "em") {
    double c = 0.2;
    Vector alpha = Vector::ones(K)*10;
    ForwardBackward res = EM(fb, data, c, alpha, visualize_);
  }
}


int main() {

  // pair<Matrix,Vector> data = genData(50,Vector::ones(10),0.05);
  pair<Matrix, Vector> data = readData();

  data = crop(data, 500, 900);
  size_t K = data.first.nrows();

  DirichletModel dirichletModel(0.01, Vector::ones(K)*10+uniform::rand(K)*0);
  ForwardBackward fb(&dirichletModel, LAG);
  runExperiment(fb, "em", data, true);

  /*
  dirichletModel = DirichletModel(0.2, Vector::ones(K)*10+uniform::rand(K)*0);
  fb = ForwardBackward(&dirichletModel, LAG);
  runExperiment(fb, "smoothing", data, false);
   */

  return 0;
}