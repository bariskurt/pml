//
// Created by cagatay on 16.07.2016.
//


#ifndef MATLIB_PML_HMM_H
#define MATLIB_PML_HMM_H

#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <pml.hpp>
#include <pml_random.hpp>

//
// We use the notation of the book "Bayesian Machine Learning" from David Barber
//
namespace pml {

  class DiscreteHMM {

    public:
      // Constructs DiscreteHMM with the given prior probabilities,
      // transition matrix and observation matrix.
      DiscreteHMM(const Vector& p1_, const Matrix& A_, const Matrix& B_) {
        p1 = normalize(p1_);
        A = normalize(A_, 0);
        B = normalize(B_, 0);
      }

      // Randomly generates a DiscreteHMM with given cardinalities.
      //   H  = number of hidden states
      //   V  = number of discrete observations
      DiscreteHMM(unsigned H, unsigned V) {
        p1 = Dirichlet(Vector::ones(H)).rand();
        A = Dirichlet(Vector::ones(H)).rand(H);
        B = Dirichlet(Vector::ones(V)).rand(H);
      }

      // Getters:
      const Vector& get_pi() {
        return p1;
      }

      const Matrix& get_A() {
        return A;
      }

      const Matrix& get_B() {
        return B;
      }

      // Setters:
      void set_pi(const Vector& p1_) {
        p1 = normalize(p1_);
      }

      void set_A(const Matrix& A_) {
        A = normalize(A_, 0);
      }

      void set_B(const Matrix& B_){
        B = normalize(B_, 0);
      }

      // Saves HMM matrices seperately.
      void save(const std::string &dir){
        p1.saveTxt(dir + "/p1.txt");
        A.saveTxt(dir + "/A.txt");
        B.saveTxt(dir + "/B.txt");
      }

      // Generates a sequence of length T.
      std::pair<Vector, Vector> generateData(size_t T) {
        Vector states, obs;
        // Generate Hidden Sequence
        std::vector<Categorical> a_dist;
        for(size_t i=0; i < A.ncols(); ++i){
          a_dist.emplace_back(A.getColumn(i));
        }
        states.append(Categorical(p1).rand());
        for (size_t t=1; t<T; t++)
          states.append(a_dist[states.last()].rand());

        // Generate Observations:
        std::vector<Categorical> b_dist;
        for(size_t i=0; i < B.ncols(); ++i){
          b_dist.emplace_back(B.getColumn(i));
        }
        for (size_t t=0; t<T; t++)
          obs.append(b_dist[states[t]].rand());

        return {states, obs};
      }

      // Returns log of alpha messages
      //  alpha(t) = log p(h_t, v_{1:t}) as in David Barber's book, page 455
      Matrix forward(const Vector& obs){
        ASSERT_TRUE(!obs.empty(),
                    "DiscreteHMM::forwardRecursion Error: no observations");
        Matrix alpha;
        Vector alpha_last;
        Matrix logA = log(A), logB = log(B);
        for(size_t t = 0; t < obs.size(); ++t){
          if( t == 0){
            alpha_last = log(p1) + logB.getRow(obs(t));
          } else {
            Matrix temp = tileRows(alpha_last, alpha_last.size());
            alpha_last = logSumExp(logA + temp, 1) + logB.getRow(obs(t));
          }
          alpha.appendColumn(alpha_last);
        }
        return alpha;
      }

      // Returns log of beta messages
      //  beta(t) = log p(v_{t+1:T} | h_t) as in David Barber's book, page 456
      Matrix backward(const Vector& obs){
        ASSERT_TRUE(!obs.empty(),
                    "DiscreteHMM::backwardRecursion Error: no observations");
        Matrix beta;
        Vector beta_last;
        Matrix logA = log(A), logB = log(B);
        for(size_t t = obs.size(); t > 0; --t){
          if( t == obs.size() ){
            beta_last = Vector::zeros(p1.size());
          } else {
            Matrix temp = tile(logB.getRow(obs(t)), beta_last.size());
            beta_last = logSumExp(logA + beta_last + temp, 0);
          }
          beta.appendColumn(beta_last);
        }
        return fliplr(beta);
      }

      // Runs forward and bacward recursion and calculates the gamma messages.
      // gamma(t) = log p(h_t , v_{1:T})
      //          = log p(h_t, v_{1:t}) + log  p(v_{t+1:T} | h_t)
      //          = alpha(t) * beta(t)
      Matrix FB_Smoother(const Vector &obs){
        auto alpha = forward(obs);
        auto beta = backward(obs);
        return alpha + beta;
      }

      // Log Likelihood of observing the given sequence under the model
      // p(v_{1:T}) = \sum_{h_t} p(h_t, v_{1:T})
      //            = \sum_{h_t} gamma(t)
      double log_likelihood(const Vector &obs){
        Matrix gamma = FB_Smoother(obs);
        return logSumExp(gamma.getColumn(0));
      }

    protected:
      Vector p1;      // [1 x H] initial state distribution
      Matrix A;       // [H x H] state transition matrix
      Matrix B;       // [V x H] observation matrix
  };

}

#endif