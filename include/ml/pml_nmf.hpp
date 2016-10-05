//
// Created by baris on 17.03.2016.
//

#ifndef PML_NMF_H
#define PML_NMF_H

#include "pml.hpp"
#include "pml_random.hpp"

namespace pml {

  class NMF {

    public:
      static const size_t MAX_ITER = 10000;
      static const size_t MIN_ITER = 1000;

    public:
      class Solution{
        public:
          void save(const std::string &work_dir){
            t.saveTxt(work_dir + "/t.txt");
            v.saveTxt(work_dir + "/v.txt");
            kl.saveTxt(work_dir + "/kl.txt");
          }
        public:
          Matrix t;
          Matrix v;
          Vector kl;
      };

    NMF(double at_=1, double bt_=1, double av_=1, double bv_=1)
        : at(at_), bt(bt_), av(av_), bv(bv_) {}

    Matrix randgen(size_t dim1, size_t dim2, size_t rank){
      Matrix T = Gamma(at, bt).rand(dim1, rank);
      Matrix V = Gamma(av, bv).rand(rank, dim2);
      return dot(T,V);
    }

  public:
    // Maximum Likelihood Solution
    Solution ml(const Matrix &X, size_t rank, size_t max_iter = MAX_ITER){

      // Initialize T,V
      //Matrix X = X_ + std::numeric_limits<double>::epsilon();
      size_t dim1 = X.nrows();
      size_t dim2 = X.ncols();

      Vector kl;
      Matrix T = Dirichlet(Vector::ones(dim1)).rand(rank);
      Matrix V = Dirichlet(Vector::ones(rank)).rand(dim2);
      Matrix M = Matrix::ones(dim1, dim2);

      for(unsigned i = 0; i < max_iter; ++i){

        Matrix TV = dot(T, V);
        kl.append(kl_div(X, TV));

        // Early stop ?
        if(i > MIN_ITER && kl(i) - kl(i-1) < 1e-5){
          break;
        }

        // Update T and normalize its columns to avoid degeneracy
        T = (T * dot((X / TV), V, false, true)) / dot(M, V, false, true);
        T = normalize(T, 0);

        // Update V
        V = (V * dot(T, (X / TV ), true)) / (dot(T, M, true));
      }

      return {T, V, kl};
    }

  public:
    // Variational Bayes Solution
    Solution vb(const Matrix &X, size_t rank, size_t max_iter = MAX_ITER){

      size_t dim1 = X.nrows();
      size_t dim2 = X.ncols();

      // Initialize T,V
      Matrix Lt = Gamma(at, bt).rand(dim1, rank);
      Matrix Et = Lt;

      Matrix Lv = Gamma(av, bv).rand(rank, dim2);
      Matrix Ev = Lv;

      Matrix M = Matrix::ones(dim1, dim2);
      Matrix Mt = Matrix::ones(dim1, rank);
      Matrix Mv = Matrix::ones(rank, dim2);

      Vector kl;


      for(unsigned i = 0; i < max_iter; ++i){

        // 1. Source sufficient statistics
        Matrix Z = X / dot(Lt,Lv);

        // 2. Means
        Matrix alpha_t = at + (Lt * dot( Z, Lv, false, true));
        Matrix beta_t = Mt / ((at/bt) + dot(M, Ev, false, true));
        Et = alpha_t * beta_t;

        Matrix alpha_v = av + (Lv * dot( Lt, Z, true, false));
        Matrix beta_v = Mv / ((av/bv) + dot(Et, M, true, false));
        Ev = alpha_v * beta_v;

        // 3. Means of Logs
        Lt = exp(psi(alpha_t)) * beta_t;
        Lv = exp(psi(alpha_v)) * beta_v;

        // 3a. Update Hyper parameters
        //Bt = (At * T_).Sum() / At.Sum();
        //Bv = (Av * V_).Sum() / Av.Sum();

        // 4. Track KL
        kl.append(kl_div(X, dot(Et, Ev)));
        // Early stop ?
        if(i > MIN_ITER && kl(i) - kl(i-1) < 1e-5){
          break;
        }
      }

      // Normalize T_ columns to get basis vectors:
      Matrix V = Ev * sum(Et, 0);
      Matrix T = normalize(Et, 0);
      return {T, V, kl};
    }

    Solution icm(const Matrix &X, size_t rank, size_t max_iter = MAX_ITER){

      size_t dim1 = X.nrows();
      size_t dim2 = X.ncols();

      // Initialize T,V
      Matrix T = Gamma(at, bt).rand(dim1, rank);
      Matrix V = Gamma(av, bv).rand(rank, dim2);

      Vector kl;
      Matrix M = Matrix::ones(X.nrows(), X.ncols());

      for(unsigned i = 0; i < max_iter; ++i){
        Matrix TV = dot(T, V);
        kl.append(kl_div(X, TV));

        // Early stop ?
        if(i > MIN_ITER && kl(i) - kl(i-1) < 1e-5){
          break;
        }

        T = (at + (T * dot((X / TV), V, false, true)))
            / ((at / bt) + dot(M, V, false, true));
        T = normalize(T, 0);

        V = (av + (V * dot(T, (X / TV ), true)))
             / ((av / bv) + dot(T, M, true));
      }
      return {T, V, kl};
    }

  public:
      double at, bt, av, bv;
  };

}
#endif //THINNMF_NMF_H
