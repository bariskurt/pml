#include <pml_bcpm.hpp>

using namespace std;
using namespace pml;

// constants for bcpm
const int K = 10;
const int T = 100;
const int LAG = 10;

void test_dm_new(){
  // Generate Model
  Model<DirichletPotential, MultinomialRandom>
          model(DirichletPotential(Vector::ones(10)), 0.05);

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
  ForwardBackward<DirichletPotential, MultinomialRandom> fb(model, lag,
                                                            max_components);

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
  int max_components = 20;
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

//  test_dm_new();

  return 0;
}