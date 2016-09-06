#include <pml_bcpm.hpp>

using namespace std;
using namespace pml;

// constants for bcpm
const int K = 10;
const int T = 100;
const int LAG = 10;

/*
void save(const Matrix& obs, const Vector& cps, ForwardBackward& fb) {
  obs.saveTxt("/tmp/obs.txt");
  fb.cpp.saveTxt("/tmp/cpp.txt");
  fb.mean.saveTxt("/tmp/mean.txt");
  cps.saveTxt("/tmp/real_cps.txt");

  Matrix temp;
  temp.appendColumn(cps);
  temp.appendColumn(obs);
  temp.saveTxt("/tmp/prm.txt");
}

void offline(const Matrix& obs, ForwardBackward& fb) {
  fb.smoothing(obs);
}

void streaming(const Matrix& obs, ForwardBackward& fb)  {
  for(size_t i=0; i<obs.ncols(); i++) {
    fb.processObs(obs.getColumn(i));
  }
}

void test_dm_model(){

  // Generate data:
  Matrix obs;
  Vector cps;
  double c = 0.05;
  Vector alpha = Vector::ones(K);
  std::tie(obs, cps) =  DirichletModel(c, alpha).generateData(T);

  // Estimate
  DirichletModel model(c, uniform::rand(K));
  ForwardBackward fb(&model, LAG);
  streaming(obs, fb);  // Runs forward - fixed lag smoothing

  // Visualize
  save(obs, cps, fb);
  cout << system("python3 ../etc/plot_dm_bcpm.py");
}


void test_gp_model(){

  double c = 0.05;
  double a = 10.0;
  double b = 1.0;

  // Generate data:
  Matrix obs;
  Vector cps;
  //Vector alpha = Vector::ones(K);
  //std::tie(obs, cps) =  GammaModel(c, a, b).generateData(T);

  Matrix temp = Matrix::loadTxt("/home/bariskurt/Desktop/test_data/prm.txt");
  cps = temp.getColumn(0);
  obs.appendColumn(temp.getColumn(1));
  obs = transpose(obs);

  // Estimate
  GammaModel model(c, a, b);
  ForwardBackward fb(&model, 0);
  streaming(obs,fb);  // Runs forward - fixed lag smoothing

  // Visualize
  save(obs, cps, fb);
  cout << system("python3 ../etc/plot_gp_bcpm.py");
}

 */

void test_dm_new(){
  // Generate Model
  Model<DirichletPotential, MultinomialRandom>
          model(DirichletPotential(Vector::ones(10)), 0.05);

  // Generate Sequence
  Matrix states, obs;
  std::tie(states, obs) = model.generateData(100);

  // Filtering and Smoothing
  Matrix mean1, mean2;
  Vector cpp1, cpp2;
  ForwardBackward<DirichletPotential, MultinomialRandom> fb(model);
  std::tie(mean1, cpp1) = fb.filtering(obs);
  std::tie(mean2, cpp2) = fb.smoothing(obs);

  // Save Results
  states.saveTxt("/tmp/states.txt");
  obs.saveTxt("/tmp/obs.txt");
  // Filtered
  mean1.saveTxt("/tmp/mean_filtering.txt");
  cpp1.saveTxt("/tmp/cpp_filtering.txt");
  // Smoothed
  mean2.saveTxt("/tmp/mean_smoothing.txt");
  cpp2.saveTxt("/tmp/cpp_smoothing.txt");

  cout << system("python3 ../test/python/visualizeMultinomialReset.py");
}



void test_gp_new(){

  // Generate Model
  Model<GammaPotential, PoissonRandom> model(GammaPotential(10, 1), 0.02);

  // Generate Sequence
  Matrix states, obs;
  std::tie(states, obs) = model.generateData(100);
  states.saveTxt("/tmp/states.txt");
  obs.saveTxt("/tmp/obs.txt");

  // Filtering and Smoothing
  Matrix mean;
  Vector cpp;
  int lag = 10;
  int max_components = 10;
  ForwardBackward<GammaPotential, PoissonRandom> fb(model, lag, max_components);

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

  //Visualize
  cout << system("anaconda3 ../test/python/visualizePoissonReset.py");
}

int main() {

  test_gp_new();

  //test_dm_new();

  return 0;
}