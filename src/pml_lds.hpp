#ifndef MATLIB_PML_LDS_H
#define MATLIB_PML_LDS_H

#include <pml.hpp>
#include <pml_rand.hpp>

namespace pml {

  class PoissonResetModel {

    struct GammaPotential{
        GammaPotential(double a_, double b_, double l_ = 0)
                : a(a_), b(b_), l(l_) {}

        void operator*=(const GammaPotential &other){
          *this = this->operator*(other);
        }

        GammaPotential operator*(const GammaPotential &other){
          double l2 = std::lgamma(a + other.a - 1)
                      - std::lgamma(a) - std::lgamma(other.a)
                      + std::log(b + other.b)
                      + a * std::log(b/(b + other.b))
                      + other.a * std::log(other.b/(b + other.b));
          return GammaPotential(a + other.a - 1,
                                b + other.b,
                                l + other.l + l2);
        }
        double a;
        double b;
        double l;   //log of normalizing constant
    };

    public:
      PoissonResetModel(double p1_ = 0.01, double a0_ = 1, double b0_ = 1)
              : p1(p1_), a0(a0_), b0(b0_){}

      // Generate a sequence of length T
      std::pair<Vector, Vector> generateSequence(size_t T){
        Vector states, obs;
        double lambda = gamma::rand(a0, b0);
        for (size_t t=0; t<T; t++) {
          if (t > 0 && uniform::rand() < p1) {
            lambda = gamma::rand(a0, b0);
          }
          states.append(lambda);
          obs.append(poisson::rand(lambda));
        }
        return {states, obs};
      }

      std::vector<std::vector<GammaPotential>> forwardRecursion(const Vector& obs){
        std::vector<std::vector<GammaPotential>> estimates(obs.size());
        double log_p0 = std::log(1-p1);
        double log_p1 = std::log(p1);

        for(size_t t=0; t < obs.size(); ++t){
          // predict
          if(t == 0) {
            estimates[t].emplace_back(a0, b0, log_p0);
            estimates[t].emplace_back(a0, b0, log_p1);
          } else {
            Vector consts;
            // no change components
            for(auto &gp : estimates[t-1]) {
              consts.append(gp.l);
              estimates[t].emplace_back(gp.a, gp.b, gp.l + log_p0);
            }
            // change component
            estimates[t].emplace_back(a0, b0, log_p1 + logSumExp(consts));
          }
          // update
          GammaPotential gp_obs = GammaPotential(obs(t)+1, 1);
          for(auto &gp : estimates[t]){
            gp *= gp_obs;
          }
        }
        return estimates;
      }

      std::pair<double, double> calc_mean_and_cpp(
              const std::vector<GammaPotential> &potentials){
        Vector consts;
        Vector params;
        for(auto &gp: potentials){
          consts.append(gp.l);
          params.append(gp.a / gp.b);
        }
        consts = normalizeExp(consts);
        return {dot(params, consts), consts.last()};
      }

      std::pair<Vector, Vector> forward_filter(const Vector &obs){
        auto estimates = forwardRecursion(obs);
        Vector mean, cpp;
        for(auto &potentials: estimates){
          auto state = calc_mean_and_cpp(potentials);
          mean.append(state.first);
          cpp.append(state.second);
        }
        return {mean, cpp};
      }

      std::vector<std::vector<GammaPotential>> backwardRecursion(const Vector&obs){
        std::vector<std::vector<GammaPotential>> estimates(obs.size());
        //double log_p1 = std::log(p1);
        //double log_p0 = std::log(1-p1);
        return estimates;
      }

    private:
      double p1;  // probability of change
      double a0;  // gamma prior
      double b0;  // gamma prior
  };

}
#endif