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
#include <pml_rand.hpp>

using namespace pml;

//
// We use the notation of the book "Bayesian Machine Learning" from David Barber
//
namespace hmm {

  // Discrete HMM Sequence
  //  - Contains both hidden states vector and observations vector
  struct DiscreteHMMSequence {
    Vector states;  // h_{1:T}
    Vector obs;     // v_{1:T}

    // Default constructor.
    DiscreteHMMSequence(size_t T = 0){
      states.resize(T);
      obs.resize(T);
    }

    // Constructs sequence from a saved sequence file.
    DiscreteHMMSequence(const std::string& filename){
      Matrix temp = Matrix::loadTxt(filename);
      states = temp.getColumn(0);
      obs = temp.getColumn(1);
    }

    // Saves sequence in a single text file
    void save(const std::string& filename){
      Matrix temp;
      temp.appendColumn(states);
      temp.appendColumn(obs);
      temp.saveTxt(filename);
    }
  };

  class DiscreteHMM {

    public:
      // Constructs DiscreteHMM with the given prior probabilities,
      // transition matrix and observation matrix.
      DiscreteHMM(const Vector& pi_, const Matrix& A_, const Matrix& B_) {
        pi = normalize(pi_);
        A = normalize(A_, Matrix::COLS);
        B = normalize(B_, Matrix::COLS);
      }

      // Randomly generates a DiscreteHMM with given cardinalities.
      //   H  = number of hidden states
      //   V  = number of discrete observations
      DiscreteHMM(unsigned H, unsigned V) {
        pi = dirichlet::rand(Vector::ones(H));
        A = dirichlet::rand(Vector::ones(H),H);
        B = dirichlet::rand(Vector::ones(V),H);
      }

      // Getters:
      const Vector& get_pi() {
        return pi;
      }

      const Matrix& get_A() {
        return A;
      }

      const Matrix& get_B() {
        return B;
      }

      // Setters:
      void set_pi(const Vector& pi_) {
        pi = normalize(pi_);
      }

      void set_A(const Matrix& A_) {
        A = normalize(A_, Matrix::COLS);
      }

      void set_B(const Matrix& B_){
        B = normalize(B_, Matrix::COLS);
      }

      // Saves HMM matrices seperately.
      void save(const std::string &dir){
        pi.saveTxt(dir + "/pi.txt");
        A.saveTxt(dir + "/A.txt");
        B.saveTxt(dir + "/B.txt");
      }

      // Generates a sequence of length T.
      DiscreteHMMSequence generateSequence(size_t T) {
        DiscreteHMMSequence seq(T);
        unsigned current_state;
        for (size_t t=0; t<T; t++) {
          if( t == 0){
            current_state = categorial::rand(pi);
          } else {
            current_state = categorial::rand(A.getColumn(current_state));
          }
          seq.states(t) = current_state;
          seq.obs(t) = categorial::rand(B.getColumn(current_state));
        }
        return seq;
      }

      // Returns log of alpha messages
      //  alpha(t) = p(h_t, v_{1:t}) as in David Barber's book, page 455
      Matrix forwardRecursion(const Vector& obs){
        Matrix log_Alpha;
        Vector log_alpha;
        Matrix logA = log(A), logB = log(B);
        for(size_t t = 0; t < obs.size(); ++t){
          if(t == 0){
            log_alpha = log(pi) + logB.getRow(obs(t));
          } else {
            Matrix temp = tile(log_alpha, log_alpha.size(), Matrix::ROWS);
            log_alpha = logSumExp(logA + temp, 1) + logB.getRow(obs(t));
          }
          log_Alpha.appendColumn(log_alpha);
        }
        return log_Alpha;
      }

      // Returns log of beta messages
      //  beta(t) = p(v_{t+1:T} | h_t) as in David Barber's book, page 456
      Matrix backwardRecursion(const Vector& obs){
        Matrix log_Beta;
        Vector log_beta;
        Matrix logA = log(A), logB = log(B);
        for(size_t t = obs.size(); t > 0; --t){
          if(t == obs.size()){
            log_beta = Vector::zeros(pi.size());
          } else {
            Matrix temp = tile(logB.getRow(obs(t)), log_beta.size());
            log_beta = logSumExp(logA + log_beta + temp, 0);
          }
          log_Beta.appendColumn(log_beta);
        }
        return fliplr(log_Beta);
      }

      // Runs forward and bacward recursion and calculates the gamma messages.
      // gamma(t) = p(h_t , v_{1:T})
      //          = p(h_t, v_{1:t}) * p(v_{t+1:T} | h_t)
      //          = alpha(t) * beta(t)
      Matrix FB_Smoother(const Vector &obs){
        auto log_Alpha = forwardRecursion(obs);
        auto log_Beta = backwardRecursion(obs);
        return log_Alpha + log_Beta;
      }

      // Log Likelihood of observing the given sequence under the model
      // p(v_{1:T}) = \sum_{h_t} p(h_t, v_{1:T})
      //            = \sum_{h_t} gamma(t)
      double log_likelihood(const Vector &obs){
        Matrix log_gamma = FB_Smoother(obs);
        return logSumExp(log_gamma.getColumn(0));
      }

  protected:
      Vector pi;      // [1 x H] initial state distribution
      Matrix A;       // [H x H] state transition matrix
      Matrix B;       // [V x H] observation matrix
  };

}

#endif