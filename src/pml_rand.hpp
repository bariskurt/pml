#ifndef MATLIB_PML_RAND_H
#define MATLIB_PML_RAND_H

#include "pml.hpp"

#include <ctime>

#include <gsl/gsl_randist.h>

namespace pml {

  namespace uniform {

    inline gsl_rng *rnd_get_rng() {
      static gsl_rng *rng = NULL;
      if (rng == NULL) {
        gsl_rng_env_setup();
        rng = gsl_rng_alloc(gsl_rng_default);
        gsl_rng_set(rng, (unsigned long) time(0));
      }
      return rng;
    }

    // Returns a double precision random number in [0,1).
    inline double rand() {
      return gsl_rng_uniform(rnd_get_rng());
    }

    // Returns random integers in [low. high]
    inline uint64_t randi(int low, int high) {
      return gsl_rng_uniform_int(rnd_get_rng(), high-low+1) + low;
    }

    inline Vector rand(unsigned length) {
      Vector result(length);
      for (unsigned i = 0; i < result.length(); ++i) {
        result(i) = rand();
      }
      return result;
    }

    inline Matrix rand(unsigned num_rows, unsigned num_cols) {
      Matrix result(num_rows, num_cols);
      for (unsigned i = 0; i < result.length(); ++i) {
        result(i) = rand();
      }
      return result;
    }

  } // uniform

  namespace poisson {

    inline unsigned rand(double mu) {
      return gsl_ran_poisson(uniform::rnd_get_rng(), mu);
    }

  } // Poisson

  namespace categorial {

    inline unsigned rand(const Vector &v) {
      Vector tmp = Normalize(v);
      double rnd = uniform::rand();
      double cum_sum = 0;
      unsigned i = 0;
      for (; i < tmp.length(); i++) {
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
      return gsl_ran_gamma_knuth(uniform::rnd_get_rng(), a, b);
    }

    inline Vector rand(double a, double b, unsigned N) {
      Vector v(N);
      for (unsigned i = 0; i < N; ++i) {
        v(i) = gsl_ran_gamma_knuth(uniform::rnd_get_rng(), a, b);
      }
      return v;
    }

    inline Matrix rand(double a, double b,
                       unsigned num_rows, unsigned num_cols) {
      Matrix M(num_rows, num_cols);
      for (unsigned i = 0; i < M.num_rows(); ++i) {
        for (unsigned j = 0; j < M.num_cols(); ++j) {
          M(i, j) = gsl_ran_gamma_knuth(uniform::rnd_get_rng(), a, b);
        }
      }
      return M;
    }
  } // Gamma

  namespace binomial {
    inline unsigned rand(unsigned N, double rate) {
      return gsl_ran_binomial(uniform::rnd_get_rng(), rate, N);
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
      Vector samples(p.length());
      unsigned buf[p.length()];
      gsl_ran_multinomial(uniform::rnd_get_rng(), p.length(), N, p.data(),
                          buf);
      for (unsigned i = 0; i < samples.length(); ++i) {
        samples(i) = buf[i];
      }
      return samples;
    }

    inline double log_pmf(const Vector &x, const Vector &p) {
      unsigned counts[x.length()];
      for (unsigned i = 0; i < x.length(); ++i) {
        counts[i] = (unsigned) x(i);
      }
      return gsl_ran_multinomial_lnpdf(p.length(), p.data(), counts);
    }

    inline double pmf(Vector &x, Vector &p) {
      return std::exp(log_pmf(x, p));
    }

  } // Multinomial

  namespace dirichlet {

    inline Vector rand(const Vector &alpha) {
      Vector result(alpha.length());
      gsl_ran_dirichlet(uniform::rnd_get_rng(), alpha.length(),
                        alpha.data(), result.data());
      return result;
    }

    inline Matrix rand(const Vector &alpha, unsigned num_cols) {
      Matrix result(alpha.length(), num_cols);
      Vector buf(alpha.length());
      for (unsigned j = 0; j < num_cols; ++j) {
        gsl_ran_dirichlet(uniform::rnd_get_rng(), alpha.length(),
                          alpha.data(), buf.data());
        result.SetColumn(j, buf);
      }
      return result;
    }

  } // Dirichlet

} // pml

#endif // MATLIB_PML_RAND_H