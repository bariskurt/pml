#ifndef MATLIB_PML_RAND_H
#define MATLIB_PML_RAND_H

#include "pml_matrix.hpp"

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

  class Distribution1D{
    public:
      virtual double rand() = 0;

      Vector rand(size_t length){
        Vector result(length);
        for(auto &d : result)
          d = rand();
        return result;
      }

      Matrix rand(size_t nrows, size_t ncols) {
        Matrix result(nrows, ncols);
        for(auto &d : result)
          d = rand();
        return result;
      }
  };

  class DistributionND{
    public:
      virtual Vector rand() = 0;

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

      double rand() override{
        return low + gsl_rng_uniform(rnd_get_rng()) * range;
      }

      // Integer versions:
      int randi(){
        return rand();
      }

      Vector randi(size_t length) {
        return ceil(Distribution1D::rand(length));
      }

      Matrix randi(size_t nrows, size_t ncols) {
        return ceil(Distribution1D::rand(nrows, ncols));
      }

    private:
      double low, high, range;
  };

  class Bernoulli : public Distribution1D{
    public:
      Bernoulli(double pi_) : pi(pi_){}

      double rand() override {
        return gsl_rng_uniform(rnd_get_rng()) < pi;
      }

    public:
      double pi;
  };

  class Binomial : public Distribution1D{
    public:
      Binomial(double pi_, size_t trials_) : pi(pi_), trials(trials_) {}

      double rand() override {
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

      Vector rand() override {
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
      Gaussian(double mu_, double sigma_) : mu(mu_), sigma(sigma_) {}

      double rand() override {
        return mu + gsl_ran_gaussian(rnd_get_rng(), sigma);
      }

    public:
      double mu, sigma;
  };

  class Poisson : public Distribution1D{
    public:
      Poisson(double lambda_) : lambda(lambda_) {}

      double rand() override {
        return gsl_ran_poisson(rnd_get_rng(), lambda);
      }

    public:
      double lambda;
  };


  // a = shape parameter
  // b = scale parameter
  // log p(x|a,b) = (a-1) log(x) - x/b - log(Gamma(a) - a log(b)
  class Gamma : public Distribution1D{
    public:
      Gamma(double a_, double b_) : a(a_), b(b_) {}

      double rand() override {
        return gsl_ran_gamma_knuth(rnd_get_rng(), a, b);
      }
    public:
      double a, b;
  };


  class Dirichlet : public DistributionND {
    public:
      Dirichlet(const Vector &alpha_) : alpha(alpha_) { }

      Vector rand() override {
        Vector result(alpha.size());
        gsl_ran_dirichlet(rnd_get_rng(), alpha.size(),
                          alpha.data(), result.data());
        return result;
      }

      static Dirichlet fit(const Matrix &data){
        Vector ss =  sum(log(data),
      }

    public:
      Vector alpha;
  };

} // pml

#endif // MATLIB_PML_RAND_H