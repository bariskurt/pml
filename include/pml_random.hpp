#ifndef PML_RAND_H_
#define PML_RAND_H_

#include "pml_matrix.hpp"
#include "pml_special.hpp"

#include <ctime>

#include <gsl/gsl_randist.h>

namespace pml {

  inline gsl_rng *rnd_get_rng() {
    static gsl_rng *rng = NULL;
    if (rng == NULL) {
      gsl_rng_env_setup();
      rng = gsl_rng_alloc(gsl_rng_default);
      gsl_rng_set(rng, (unsigned long) time(0));
    }
    return rng;
  }

  // Abstract Class for Uni-variate Distributions.
  class Distribution1D{
    public:
      virtual double randgen() const = 0;

      double rand() const{
        return randgen();
      };

      Vector rand(size_t length) const{
        Vector result(length);
        for(auto &d : result)
          d = rand();
        return result;
      }

      Matrix rand(size_t nrows, size_t ncols) const {
        Matrix result(nrows, ncols);
        for(auto &d : result)
          d = rand();
        return result;
      }
  };

  // Abstract Class for Multi-variate Distributions.
  class DistributionND{
    public:
      virtual Vector randgen() = 0;

      Vector rand() {
        return randgen();
      }

      Matrix rand(size_t ncols) {
        Matrix result;
        for(size_t i = 0; i < ncols; ++i)
          result.appendColumn(rand());
        return result;
      }
  };

  class Uniform : public Distribution1D {
    public:
      Uniform(double low_ = 0, double high_ = 1) : low(low_), high(high_) {
        range = high - low;
      }

      double randgen() const override{
        return low + gsl_rng_uniform(rnd_get_rng()) * range;
      }

      // Integer versions:
      int randi(){
        return randgen();
      }

      Vector randi(size_t length) {
        return ceil(rand(length));
      }

      Matrix randi(size_t nrows, size_t ncols) {
        return ceil(rand(nrows, ncols));
      }

    private:
      double low, high, range;
  };

  class Bernoulli : public Distribution1D{
    public:
      Bernoulli(double p_) : p(p_){}

      double randgen() const override {
        return gsl_rng_uniform(rnd_get_rng()) < p;
      }

    public:
      double p;
  };

  class Categorical : public Distribution1D{
    public:
      Categorical(const Vector &p_) {
        p = normalize(p_);
        ptable = gsl_ran_discrete_preproc (p.size(), p.data());
      }

      double randgen() const override {
        return gsl_ran_discrete(rnd_get_rng(), ptable);
      }

      static Categorical fit(const Vector &data, size_t K){
        Vector h = Vector::zeros(K);
        for(auto d : data) ++h[d];
        return Categorical(h);
      }

    public:
      Vector p;
      gsl_ran_discrete_t *ptable;
  };

  class Binomial : public Distribution1D{
    public:
      Binomial(double pi_, size_t trials_) : pi(pi_), trials(trials_) {}

      double randgen() const override {
        return gsl_ran_binomial(rnd_get_rng(), pi, trials);
      }

      double pmf(unsigned i, unsigned j) {
        return gsl_ran_binomial_pdf(j, pi, i);
      }

      double log_pmf(unsigned i, unsigned j) {
        return std::log(pmf(i, j));
      }

    public:
      double pi;
      size_t trials;
  };

  class Multinomial : public DistributionND{

    public:
      Multinomial(const Vector &p_, size_t trials_){
        p = normalize(p_);
        trials = trials_;
      }

      Vector randgen() override {
        Vector samples(p.size());
        unsigned buf[p.size()];
        gsl_ran_multinomial(rnd_get_rng(), p.size(), trials, p.data(), buf);
        for (size_t i = 0; i < samples.size(); ++i) {
          samples(i) = buf[i];
        }
        return samples;
      }
/*
      double log_pmf(const Vector &x) {
        unsigned counts[x.size()];
        for (size_t i = 0; i < x.size(); ++i) {
          counts[i] = (size_t) x(i);
        }
        return gsl_ran_multinomial_lnpdf(p.size(), p.data(), counts);
      }

      double pmf(Vector &x, Vector &p) {
        return std::exp(log_pmf(x, p));
      }
*/
    public:
      Vector p;
      size_t trials;
  };


  class Gaussian : public Distribution1D{
    public:
      Gaussian(double mu_ = 0, double sigma_ = 1) : mu(mu_), sigma(sigma_) {}

      double randgen() const override {
        return mu + gsl_ran_gaussian(rnd_get_rng(), sigma);
      }

      static Gaussian fit(const Vector &data){
        double mean_x = mean(data);
        double var_x = sum(pow(data - mean_x, 2)) / data.size();
        return Gaussian::fit(mean_x, var_x);
      }

      static Gaussian fit(double mean_x, double var_x){
        return Gaussian(mean_x, std::sqrt(var_x));
      }

    public:
      double mu, sigma;
  };

  class Poisson : public Distribution1D{
    public:
      Poisson(double lambda_) : lambda(lambda_) {}

      double randgen() const override {
        return gsl_ran_poisson(rnd_get_rng(), lambda);
      }

    public:
      double lambda;
  };


  // a = shape parameter
  // b = scale parameter
  // log p(x|a,b) = (a-1) log(x) - x/b - log(Gamma(a) - a log(b)
  class Gamma : public Distribution1D{
    private:
      static const size_t MAX_ITER = 5;

    public:
      Gamma(double a_, double b_) : a(a_), b(b_) {}

      double randgen() const override {
        return gsl_ran_gamma_knuth(rnd_get_rng(), a, b);
      }

      static Gamma fit(const Vector &data){
        return Gamma::fit(mean(data), mean(log(data)));
      }

      static Gamma fit(double mean_x, double mean_log_x){
        //std::cout << "ss : " << mean_x << ", " << mean_log_x << std::endl;
        double log_mean_x = std::log(mean_x);
        double a = 0.5 / (log_mean_x - mean_log_x);
        for(size_t iter = 0; iter < MAX_ITER; ++iter){
          double temp = mean_log_x - log_mean_x + std::log(a) - psi(a);
          temp /= a * a *(1/a - psi(a,1));
          a = 1/(1/a + temp);
        }
        double b = mean_x / a;
        return Gamma(a, b);
      }

    public:
      double a, b;
  };


  class Dirichlet : public DistributionND {
    private:
      static const size_t MIN_ITER = 100;
      static const size_t MAX_ITER = 1000;

    public:
      Dirichlet(const Vector &alpha_) : alpha(alpha_) { }

      Vector randgen() override {
        Vector result(alpha.size());
        gsl_ran_dirichlet(rnd_get_rng(), alpha.size(),
                          alpha.data(), result.data());
        return result;
      }

      static Dirichlet fit(const Matrix &data){
        Vector ss = mean(log(data),1); // sufficient statistics
        return fit(ss);
      }

      static Dirichlet fit(const Vector &ss){
        Vector alpha = normalize(ss);
        Vector alpha_new;
        for(size_t iter=0; iter < MAX_ITER; iter++) {
          alpha_new = inv_psi( ss + psi(sum(alpha)) );
          // Break if converged.
          if( iter > MIN_ITER && sum(abs(alpha-alpha_new)) < 1e-6 )
            break;
          alpha = alpha_new;
        }
        return Dirichlet(alpha);
      }

    public:
      Vector alpha;
  };

} // pml

#endif // MATLIB_PML_RAND_H