#ifndef MATLIB_PML_RAND_H
#define MATLIB_PML_RAND_H

#include "pml.hpp"

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

  namespace uniform {

    // Returns a double precision random number in [0,1).
    inline double rand() {
      return gsl_rng_uniform(rnd_get_rng());
    }

    // Returns random integers in [low. high]
    inline uint64_t randi(int low, int high) {
      return gsl_rng_uniform_int(rnd_get_rng(), high-low+1) + low;
    }

    inline Vector rand(size_t length) {
      Vector result(length);
      for (unsigned i = 0; i < result.size(); ++i) {
        result(i) = rand();
      }
      return result;
    }

    inline Matrix rand(size_t num_rows, size_t num_cols) {
      Matrix result(num_rows, num_cols);
      for(auto &value : result){
        value = rand();
      }
      return result;
    }

  } // uniform

  namespace gaussian{
    inline double rand(double mu, double sigma) {
      return mu + gsl_ran_gaussian(rnd_get_rng(), sigma);
    }

    inline Vector rand(double mu, double sigma, size_t length) {
      Vector result(length);
      for(auto &item : result){
        item = gaussian::rand(mu, sigma);
      }
      return result;
    }

    inline Matrix rand(double mu, double sigma, size_t dim1, size_t dim2) {
      Matrix result(dim1, dim2);
      for(auto &item : result){
        item = gaussian::rand(mu, sigma);
      }
      return result;
    }

    // ToDo: implement
    inline Vector rand(const Vector &mu, const Matrix &sigma){
      return Vector();
    }

    // ToDo: implement
    inline Matrix rand(const Vector &mu, const Matrix &sigma, size_t ncols){
      return Matrix();
    }

  }

  namespace poisson {

    inline unsigned rand(double mu) {
      return gsl_ran_poisson(rnd_get_rng(), mu);
    }

    inline Vector rand(double mu, size_t length) {
      Vector result(length);
      for(auto &item : result){
        item = poisson::rand(mu);
      }
      return result;
    }

    inline Vector rand(const Vector &mu) {
      Vector result;
      for(auto &m : mu){
        result.append(poisson::rand(m));
      }
      return result;
    }

    inline Matrix rand(double mu, size_t dim1, size_t dim2) {
      Matrix result(dim1, dim2);
      for(auto &item : result){
        item = poisson::rand(mu);
      }
      return result;
    }

    inline Matrix rand(const Matrix &mu) {
      Matrix result(mu.shape());
      for(size_t i=0; i<mu.size(); ++i){
        result(i) = poisson::rand(mu(i));
      }
      return result;
    }

  } // Poisson

  namespace categorial {

    inline unsigned rand(const Vector &v) {
      Vector tmp = normalize(v);
      double rnd = uniform::rand();
      double cum_sum = 0;
      unsigned i = 0;
      for (; i < tmp.size(); i++) {
        cum_sum += tmp(i);
        if (rnd < cum_sum) {
          break;
        }
      }
      return i;
    }

  } // Categorial

  namespace gamma {
    /*
    * a = shape parameter
    * b = scale parameter
    * log p(x|a,b) = (a-1) log(x) - x/b - log(Gamma(a) - a log(b)
    */

    inline double rand(double a, double b) {
      return gsl_ran_gamma_knuth(rnd_get_rng(), a, b);
    }

    inline Vector rand(double a, double b, size_t length) {
      Vector result(length);
      for(auto &item : result){
        item = gamma::rand(a,b);
      }
      return result;
    }

    inline Vector rand(const Vector &a, const Vector &b){
      assert(a.size() == b.size());
      Vector result;
      for(size_t i=0; i < a.size(); ++i){
        result.append(gamma::rand(a(i), b(i)));
      }
    }

    inline Matrix rand(double a, double b, size_t nrows, size_t ncols) {
      Matrix result(nrows, ncols);
      for(auto &item : result){
        item = gamma::rand(a,b);
      }
      return result;
    }

    inline Matrix rand(const Matrix &a, const Matrix &b) {
      assert(a.shape() == b.shape());
      Matrix result(a.shape());
      for (size_t i = 0; i < a.size(); ++i) {
        result(i) = gamma::rand(a(i), b(i));
      }
      return result;
    }
  } // Gamma

  namespace binomial {
    inline unsigned rand(unsigned N, double rate) {
      return gsl_ran_binomial(rnd_get_rng(), rate, N);
    }

    inline double pmf(unsigned i, unsigned j, double p) {
      return gsl_ran_binomial_pdf(j, p, i);
    }

    inline double log_pmf(unsigned i, unsigned j, double p) {
      return std::log(pmf(i, j, p));
    }
  } // Binomial

  namespace multinomial {

    inline Vector rand(const Vector &p, unsigned N) {
      Vector samples(p.size());
      unsigned buf[p.size()];
      gsl_ran_multinomial(rnd_get_rng(), p.size(), N, p.data(),
                          buf);
      for (size_t i = 0; i < samples.size(); ++i) {
        samples(i) = buf[i];
      }
      return samples;
    }

    inline double log_pmf(const Vector &x, const Vector &p) {
      unsigned counts[x.size()];
      for (size_t i = 0; i < x.size(); ++i) {
        counts[i] = (size_t) x(i);
      }
      return gsl_ran_multinomial_lnpdf(p.size(), p.data(), counts);
    }

    inline double pmf(Vector &x, Vector &p) {
      return std::exp(log_pmf(x, p));
    }

  } // Multinomial

  namespace dirichlet {

    inline Vector rand(const Vector &alpha) {
      Vector result(alpha.size());
      gsl_ran_dirichlet(rnd_get_rng(), alpha.size(),
                        alpha.data(), result.data());
      return result;
    }

    inline Matrix rand(const Vector &alpha, unsigned num_cols) {
      Matrix result(alpha.size(), num_cols);
      Vector buf(alpha.size());
      for (size_t j = 0; j < num_cols; ++j) {
        gsl_ran_dirichlet(rnd_get_rng(), alpha.size(),
                          alpha.data(), buf.data());
        result.setColumn(j, buf);
      }
      return result;
    }

  } // Dirichlet

} // pml

#endif // MATLIB_PML_RAND_H