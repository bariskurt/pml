#include <pml_bcpm.hpp>

using namespace std;
using namespace pml;

// constants for bcpm
const int K = 10;
const int T = 100;
const int LAG = 10;

/*
void test_dm(){
  // Generate Model
  DM_Model model(DirichletPotential(Vector::ones(10)), 0.05);

  // Generate Sequence
  Matrix states, obs;
  std::tie(states, obs) = model.generateData(100);
  states.saveTxt("/tmp/states.txt");
  obs.saveTxt("/tmp/obs.txt");

  // Filtering and Smoothing
  Matrix mean;
  Vector cpp;
  int lag = 10;
  int max_components = 20;
  DM_ForwardBackward fb(model, lag, max_components);

  // Filtering
  std::tie(mean, cpp) = fb.filtering(obs);
  mean.saveTxt("/tmp/mean_filtering.txt");
  cpp.saveTxt("/tmp/cpp_filtering.txt");

  // Smoothing
  std::tie(mean, cpp) = fb.smoothing(obs);
  mean.saveTxt("/tmp/mean_smoothing.txt");
  cpp.saveTxt("/tmp/cpp_smoothing.txt");

  // Fixed Lag
  std::tie(mean, cpp) = fb.online_smoothing(obs);
  mean.saveTxt("/tmp/mean_fixed_lag.txt");
  cpp.saveTxt("/tmp/cpp_fixed_lag.txt");

  cout << system("anaconda3 ../test/python/visualizeMultinomialReset.py");
}
*/

/*
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
*/

/*
void test_dm_em(){
  // Load Data
  pair<Matrix, Vector> data = readData();
  data = crop(data, 500, 900);
  Matrix &obs = data.first;
//  Vector &cps = data.second;
  size_t K = obs.nrows();

  DM_Model model(Vector::ones(K)*10, 0.01);
  DM_ForwardBackward fb(model);

  fb.learn_params(obs);

}

void test_dm_em2(){

  cout << "test_dm_em2...\n";

  size_t K = 5;
  size_t length = 2;
  double c = 0.5;
  Vector alpha = Vector::ones(K)*1;
  // size_t max_iter = 10;

  // Generate data:
  Matrix states, obs;
  DM_Model model(DirichletPotential(alpha), c);
  std::tie(states, obs) = model.generateData(length);
  // Real Change Points
  Vector cps = Vector::zeros(length);
  for(size_t t = 1; t < length; ++t) {
    cps[t] = (states.getColumn(t) != states.getColumn(t-1));
  }
  obs.saveTxt("/tmp/obs.txt");
  cps.saveTxt("/tmp/cps.txt");

  // Estimate with true parameters
  DM_ForwardBackward fb(model);
  auto result = fb.smoothing(obs);
  result.first.saveTxt("/tmp/mean.txt");
  result.second.saveTxt("/tmp/cpp.txt");
  
  // Learn parameters
  double c_init = 0.00001;
  Vector alpha_init = alpha;
  DM_Model init_model(DirichletPotential(alpha_init), c_init);
  DM_ForwardBackward fb2(init_model);
  result = fb2.learn_params(obs);
  result.first.saveTxt("/tmp/mean2.txt");
  result.second.saveTxt("/tmp/cpp2.txt");


  cout << "done.\n";
}
  */


void test_pg(){

  cout << "test_pg_em...\n";
  size_t length = 1000;
  double c = 0.01;
  double a = 10;

  // Generate data:
  Matrix states, obs;
  PG_Model model(GammaPotential(a, 1), c);
  std::tie(states, obs) = model.generateData(length);

  // Save data:
  obs.saveTxt("/tmp/obs.txt");
  states.saveTxt("/tmp/states.txt");

  // Estimate with true parameters
  PG_ForwardBackward fb(model);


  std::cout << "Filtering...\n";
  auto result = fb.filtering(obs);
  result.first.saveTxt("/tmp/mean.txt");
  result.second.saveTxt("/tmp/cpp.txt");

  std::cout << "Smoothing...\n";
  result = fb.smoothing(obs);
  result.first.saveTxt("/tmp/mean2.txt");
  result.second.saveTxt("/tmp/cpp2.txt");


  std::cout << "Online smoothing...\n";
  result = fb.online_smoothing(obs, 100);
  result.first.saveTxt("/tmp/mean3.txt");
  result.second.saveTxt("/tmp/cpp3.txt");

  cout << "done.\n";
}

void test_pg_em(){
  cout << "test_pg_em...\n";
  size_t length = 999;
  double c = 0.1;
  double a = 10;

  // Generate data:
  Matrix states, obs;
  PG_Model model(GammaPotential(a, 1), c);
  std::tie(states, obs) = model.generateData(length);

  double init_c = 0.0001;
  double init_a = a;
  PG_Model init_model(GammaPotential(init_a, 1), init_c);
  PG_ForwardBackward fb(init_model);
  fb.learn_parameters(obs);

  cout << "done.\n";
}

int main() {

  // test_gp();

  // test_dm();

  // test_dm_em2();

  test_pg_em();

  return 0;
}
