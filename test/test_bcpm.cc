#include <ml/pml_bcpm.hpp>

using namespace std;
using namespace pml;

void test_dm(){
  cout << "test_dm()...\n";

  size_t K = 5;
  double precision = K;
  bool fixed_precision = false;
  Vector alpha = normalize(Vector::ones(K)) * precision;
  double p1 = 0.01;

  size_t lag = 10;
  size_t length = 500;

  // Generate Model
  DM_Model model(alpha, p1, fixed_precision);

  // Generate Sequence
  Matrix states, obs;
  std::tie(states, obs) = model.generateData(length);
  states.saveTxt("/tmp/states.txt");
  obs.saveTxt("/tmp/obs.txt");

  // Generate Forward-Backward
  Matrix mean;
  Vector cpp;
  DM_ForwardBackward fb(&model);

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
  std::cout << "Online smoothing...\n";
  std::tie(mean, cpp) = fb.online_smoothing(obs, lag);
  mean.saveTxt("/tmp/mean3.txt");
  cpp.saveTxt("/tmp/cpp3.txt");

  if(system("anaconda3 ../test/python/test_bcpm_dm.py False")){
    std::cout <<"plotting error...\n";
  }
  cout << "OK.\n";
}

void test_dm_em(){

  cout << "test_dm_em()...\n";

  size_t K = 5;
  double precision = K;
  bool fixed_precision = true;
  Vector alpha = normalize(Vector::ones(K)) * precision;
  double p1 = 0.01;

  size_t length = 100;

  // Generate data:
  Matrix states, obs;
  DM_Model model(alpha, p1, fixed_precision);
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

  // Estimate with true parameters
  DM_ForwardBackward fb(&model);
  auto result = fb.smoothing(obs);
  result.first.saveTxt("/tmp/mean.txt");
  result.second.saveTxt("/tmp/cpp.txt");

  // Learn parameters
  double c_init = 0.0001;
  DM_Model em_model(DirichletPotential::rand_gen(K, precision).alpha,
                    c_init, fixed_precision);
  DM_Model em_init_model = em_model;
  DM_ForwardBackward fb_em(&em_model);

  // Run with EM inital
  result = fb_em.smoothing(obs);
  result.first.saveTxt("/tmp/mean2.txt");
  result.second.saveTxt("/tmp/cpp2.txt");

  // Learn parameters
  result = fb_em.learn_parameters(obs);
  result.first.saveTxt("/tmp/mean3.txt");
  result.second.saveTxt("/tmp/cpp3.txt");

  std::cout << "-----------\n";
  std::cout << "True model:\n";
  model.print();
  std::cout << "-----------\n";
  std::cout << "EM(initial) model:\n";
  em_init_model.print();
  std::cout << "-----------\n";
  std::cout << "EM(final) model:\n";
  em_model.print();
  std::cout << "-----------\n";

  if(system("anaconda3 ../test/python/test_bcpm_dm.py True")){
    std::cout <<"plotting error...\n";
  }

  cout << "OK.\n";
}


void test_pg(){

  cout << "test_pg()...\n";
  size_t length = 1000;
  double c = 0.01;
  double a = 10;
  double b = 1;

  // Generate data:
  Matrix states, obs;
  PG_Model model(a, b, c);
  std::tie(states, obs) = model.generateData(length);

  // Save data:
  obs.saveTxt("/tmp/obs.txt");
  states.saveTxt("/tmp/states.txt");

  // Estimate with true parameters
  PG_ForwardBackward fb(&model);

  std::cout << "Filtering...\n";
  auto result = fb.filtering(obs);
  result.first.saveTxt("/tmp/mean.txt");
  result.second.saveTxt("/tmp/cpp.txt");

  std::cout << "Smoothing...\n";
  result = fb.smoothing(obs);
  result.first.saveTxt("/tmp/mean2.txt");
  result.second.saveTxt("/tmp/cpp2.txt");


  std::cout << "Online smoothing...\n";
  size_t lag = 10;
  result = fb.online_smoothing(obs, lag);
  result.first.saveTxt("/tmp/mean3.txt");
  result.second.saveTxt("/tmp/cpp3.txt");

  if(system("anaconda3 ../test/python/test_bcpm_pg.py False")){
    std::cout <<"plotting error...\n";
  }
  cout << "OK.\n";
}

void test_pg_em(){
  cout << "test_pg_em...\n";
  size_t length = 200;
  double p1 = 0.05;
  double a = 10;
  double b = 1;
  bool fixed_scale = true;

  // Generate data:
  Matrix states, obs;
  PG_Model model(a, b, p1, fixed_scale);
  std::tie(states, obs) = model.generateData(length);

  // Save data:
  obs.saveTxt("/tmp/obs.txt");
  states.saveTxt("/tmp/states.txt");

  // Filtering with true parameters
  PG_ForwardBackward fb(&model);
  auto result = fb.smoothing(obs);
  result.first.saveTxt("/tmp/mean.txt");
  result.second.saveTxt("/tmp/cpp.txt");

  // Generate random model for EM
  double init_p1 = 0.001;
  double init_a = Uniform(0, 10).rand();
  double init_b = 1;

  PG_Model init_model(init_a, init_b, init_p1, fixed_scale);
  PG_Model em_model = init_model;
  PG_ForwardBackward fb_em(&em_model);

  // Run initial model:
  auto result_em = fb_em.smoothing(obs);
  result_em.first.saveTxt("/tmp/mean2.txt");
  result_em.second.saveTxt("/tmp/cpp2.txt");

  // Run EM:
  auto result_dummy = fb_em.learn_parameters(obs);
  result_dummy.first.saveTxt("/tmp/mean3.txt");
  result_dummy.second.saveTxt("/tmp/cpp3.txt");

  std::cout << "-----------\n";
  std::cout << "True model:\n";
  model.print();
  std::cout << "-----------\n";
  std::cout << "EM(initial) model:\n";
  init_model.print();
  std::cout << "-----------\n";
  std::cout << "EM(final) model:\n";
  em_model.print();
  std::cout << "-----------\n";

  if(system("anaconda3 ../test/python/test_bcpm_pg.py True")){
    std::cout <<"plotting error...\n";
  }
  cout << "OK.\n";
}

void test_g(){

  double p1 = 0.01;
  double mu = 3;
  double sigma = 2;
  size_t length = 200;
  size_t lag = 10;

  // Generate data:
  Matrix states, obs;
  G_Model model(mu, sigma, p1);
  std::tie(states, obs) = model.generateData(length);

  // Save data:
  obs.saveTxt("/tmp/obs.txt");
  states.saveTxt("/tmp/states.txt");

  // Estimate with true parameters
  G_ForwardBackward fb(&model);

  std::cout << "Filtering...\n";
  auto result = fb.filtering(obs);
  result.first.saveTxt("/tmp/mean.txt");
  result.second.saveTxt("/tmp/cpp.txt");

  std::cout << "Smoothing...\n";
  result = fb.smoothing(obs);
  result.first.saveTxt("/tmp/mean2.txt");
  result.second.saveTxt("/tmp/cpp2.txt");

  std::cout << "Online smoothing...\n";
  result = fb.online_smoothing(obs, lag);
  result.first.saveTxt("/tmp/mean3.txt");
  result.second.saveTxt("/tmp/cpp3.txt");

  std::cout << "Visualizing...\n";
  if(system("anaconda3 ../test/python/test_bcpm_pg.py False")){
    std::cout <<"plotting error...\n";
  }
  cout << "OK.\n";

}

int main() {

  // test_dm();
  // test_dm_em();

  //test_pg();
  //test_pg_em();

  test_g();

  return 0;
}
