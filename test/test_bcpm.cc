#include <ml/pml_bcpm.hpp>

using namespace std;
using namespace pml;

void test_dm(){
  cout << "test_dm()...\n";

  size_t K = 5;
  int lag = 10;
  double p1 = 0.1;
  Vector alpha = Vector::ones(K);
  size_t length = 100;

  // Generate Model
  DM_Model model(DirichletPotential(alpha), p1);

  // Generate Sequence
  Matrix states, obs;
  std::tie(states, obs) = model.generateData(length);
  states.saveTxt("/tmp/states.txt");
  obs.saveTxt("/tmp/obs.txt");

  // Filtering and Smoothing
  Matrix mean;
  Vector cpp;
  DM_ForwardBackward fb(model);

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

  size_t K = 8;
  size_t length = 200;
  double c = 0.01;
  Vector alpha = Uniform(0, 10).rand(K);

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

  // Smoothing with dummy parameters
  DM_ForwardBackward fb_dummy(init_model);
  auto result_dummy = fb_dummy.smoothing(obs);
  result_dummy.first.saveTxt("/tmp/mean3.txt");
  result_dummy.second.saveTxt("/tmp/cpp3.txt");

  std::cout << "-----------\n";
  std::cout << alpha << std::endl;
  fb2.model.prior.print();
  std::cout << fb2.model.p1 << std::endl;
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
  size_t length = 99;
  double c = 0.05;
  double a = 20; //Uniform(0, 10).rand();
  double b = 16;

  // Generate data:
  Matrix states, obs;
  PG_Model model(GammaPotential(a, b), c);
  std::tie(states, obs) = model.generateData(length);

  // Save data:
  obs.saveTxt("/tmp/obs.txt");
  states.saveTxt("/tmp/states.txt");

  // Filtering with true parameters
  PG_ForwardBackward fb(model);
  auto result = fb.smoothing(obs);
  result.first.saveTxt("/tmp/mean.txt");
  result.second.saveTxt("/tmp/cpp.txt");


  double init_c = 0.001;
  double init_a = Uniform(0, 10).rand();
  double init_b = 1;

  PG_Model init_model(GammaPotential(init_a, init_b), init_c);
  PG_ForwardBackward fb_em(model);

  auto result_em = fb_em.learn_parameters(obs);
  result_em.first.saveTxt("/tmp/mean2.txt");
  result_em.second.saveTxt("/tmp/cpp2.txt");

  std::cout << "Original parameters: a = " << a << ", b = " << b
            << ", c = " << c << std::endl;
  std::cout << "Estimated parameters: a = " << fb_em.model.prior.a
            << ", b = " << fb_em.model.prior.b
            << ", c = " << fb_em.model.p1 << std::endl;


  // Smoothing with dummy parameters
  PG_ForwardBackward fb_dummy(init_model);
  auto result_dummy = fb_dummy.smoothing(obs);
  result_dummy.first.saveTxt("/tmp/mean3.txt");
  result_dummy.second.saveTxt("/tmp/cpp3.txt");

  if(system("anaconda3 ../test/python/test_bcpm_pg.py True")){
    std::cout <<"plotting error...\n";
  }
  cout << "OK.\n";
}

void test_matlab(){

  Matrix states = Matrix::loadTxt("/tmp/states.txt");
  Matrix obs = Matrix::loadTxt("/tmp/obs.txt");

  double a = 10;
  double b = 1;
  double p1 = 0.2;

  PG_Model model(GammaPotential(a, b), p1);

  PG_ForwardBackward fb(model);
  auto result = fb.learn_parameters(obs);

  cout << result.second << endl;
  cout << result.first << endl;

  result.first.saveTxt("/tmp/mean.txt");
  result.second.saveTxt("/tmp/cpp.txt");


}

int main() {

  //test_dm();
  // test_dm_em();

  //test_pg();
  test_pg_em();

  //test_matlab();

  return 0;
}
