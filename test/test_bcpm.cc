#include <ml/pml_bcpm.hpp>

using namespace std;
using namespace pml;

// constants for bcpm
const int K = 10;
const int T = 100;
const int LAG = 10;



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

void test_dm(){
  cout << "test_dm()...\n";

  // Generate Model
  DM_Model model(DirichletPotential(Vector::ones(10)*5), 0.01);

  // Generate Sequence
  Matrix states, obs;
  size_t length = 99;
  std::tie(states, obs) = model.generateData(length);
  states.saveTxt("/tmp/states.txt");
  obs.saveTxt("/tmp/obs.txt");

  // Filtering and Smoothing
  Matrix mean;
  Vector cpp;
  int max_components = 100;
  DM_ForwardBackward fb(model, max_components);

  // Filtering
  std::cout << "Filtering...\n";
  std::tie(mean, cpp) = fb.filtering(obs);
  mean.saveTxt("/tmp/mean.txt");
  cpp.saveTxt("/tmp/cpp.txt");

  // Smoothing
  std::cout << "Smoothing...\n";
  std::tie(mean, cpp) = fb.smoothing(obs);
  mean.saveTxt("/tmp/mean2.txt");
  cpp.saveTxt("/tmp/cpp2.txt");

  // Fixed Lag
  int lag = 10;
  std::cout << "Online smoothing...\n";
  std::tie(mean, cpp) = fb.online_smoothing(obs, lag);
  mean.saveTxt("/tmp/mean3.txt");
  cpp.saveTxt("/tmp/cpp3.txt");

  if(system("anaconda3 ../test/python/visualizeMultinomialReset.py")){
    std::cout <<"plotting error...\n";
  }
  cout << "OK.\n";
}

void test_dm_em(){

  cout << "test_dm_em()...\n";

  size_t K = 25;
  size_t length = 200;
  double c = 0.1;
  Vector alpha = Uniform(0, 3).rand(K);

  // Generate data:
  Matrix states, obs;
  DM_Model model(DirichletPotential(alpha), c);
  std::tie(states, obs) = model.generateData(length);

  // Real Change Points
  Vector cps = Vector::zeros(length);
  for(size_t t = 1; t < length; ++t) {
    Vector diff = (states.getColumn(t) - states.getColumn(t-1)) > 0;
    cps[t] = any(diff);
  }
  obs.saveTxt("/tmp/obs.txt");
  states.saveTxt("/tmp/states.txt");
  cps.saveTxt("/tmp/cps.txt");
  std::cout << "Num. change points: " << sum(cps) << std::endl;

  // Estimate with true parameters
  DM_ForwardBackward fb(model);
  auto result = fb.smoothing(obs);
  result.first.saveTxt("/tmp/mean.txt");
  result.second.saveTxt("/tmp/cpp.txt");

  // Learn parameters
  double c_init = 0.0001;
  Vector alpha_init = Uniform(0, 10).rand(K);
  DM_Model init_model(DirichletPotential(alpha_init), c_init);
  DM_ForwardBackward fb2(init_model);
  result = fb2.learn_parameters(obs);
  result.first.saveTxt("/tmp/mean2.txt");
  result.second.saveTxt("/tmp/cpp2.txt");

  std::cout << "-----------\n";
  std::cout << alpha << std::endl;
  fb2.model.prior.print();
  std::cout << fb2.model.p1 << std::endl;
  std::cout << "-----------\n";

  cout << "OK.\n";
}


void test_pg(){

  cout << "test_pg()...\n";
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

  if(system("anaconda3 ../test/python/visualizePoissonReset.py")){
    std::cout <<"plotting error...\n";
  }
  cout << "OK.\n";
}

void test_pg_em(){
  cout << "test_pg_em...\n";
  size_t length = 100;
  double c = 0.1;
  double a = Uniform(0, 10).rand();
  double b = 1;

  // Generate data:
  Matrix states, obs;
  PG_Model model(GammaPotential(a, b), c);
  std::tie(states, obs) = model.generateData(length);

  double init_c = 0.0001;
  double init_a = Uniform(0, 10).rand();
  double init_b = 1;

  PG_Model init_model(GammaPotential(init_a, init_b), init_c);
  PG_ForwardBackward fb(init_model);

  auto result = fb.learn_parameters(obs);
  result.first.saveTxt("/tmp/mean2.txt");
  result.second.saveTxt("/tmp/cpp2.txt");

  std::cout << "Original parameters: a = " << a << ", b = " << b << std::endl;
  std::cout << "Estimated parameters: a = " << fb.model.prior.a
            << ", b = " << fb.model.prior.b <<  std::endl;

  cout << "done.\n";
}

int main() {

  // test_dm();
  // test_dm_em();

  // test_pg();
  test_pg_em();

  return 0;
}
