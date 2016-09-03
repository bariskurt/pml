#include <pml_bcpm.hpp>

using namespace std;
using namespace pml;

// constants for bcpm
const int K = 10;
const int T = 100;
const int LAG = 10;

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


int main() {

  //test_dm_model();

  test_gp_model();

  return 0;
}