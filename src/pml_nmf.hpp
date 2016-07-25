//
// Created by baris on 17.03.2016.
//

#ifndef PML_NMF_H
#define PML_NMF_H

#include "pml.hpp"
#include "pml_rand.hpp"

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

    static Matrix randgen(size_t dim1, size_t dim2, size_t rank){
      Matrix T = dirichlet::rand(Vector::ones(dim1), rank);
      Matrix V = dirichlet::rand(Vector::ones(rank), dim2);
      Matrix X = dot(T,V);
      return  X + abs(gaussian::rand(0,0.1,dim1,dim2));
    }

    // Maximum Likelihood Solution
    static Solution ml(const Matrix &X, size_t rank,
                      size_t max_iter = MAX_ITER){

      // Initialize T,V
      //Matrix X = X_ + std::numeric_limits<double>::epsilon();
      Vector kl;
      Matrix T = dirichlet::rand(Vector::ones(X.nrows()), rank);
      Matrix V = dirichlet::rand(Vector::ones(rank), X.ncols());

      Matrix M = Matrix::ones(X.nrows(), X.ncols());

      for(unsigned i = 0; i < max_iter; ++i){

        Matrix TV = dot(T, V);
        kl.append(kl_div(X, TV));

        // Early stop ?
        if(i > MIN_ITER && kl(i) - kl(i-1) < 1e-5){
          break;
        }

        // Update T and normalize its columns to avoid degeneracy
        T = (T * dot((X / TV), V, false, true)) / dot(M, V, false, true);
        T = normalizeCols(T);

        // Update V
        V = (V * dot(T, (X / TV ), true)) / (dot(T, M, true));
      }

      return {T, V, kl};
    }

  public:
    NMF(size_t dim1, size_t dim2, size_t rank,
        double at=1, double bt=1, double av=1, double bv=1) {
      At = Matrix (dim1, rank, at);
      Bt = Matrix (dim1, rank, bt);
      At = Matrix (rank, dim2, av);
      Bt = Matrix (rank, dim2, bv);
    }

    NMF(const Matrix &At_, const Matrix &Bt_,
        const Matrix &Av_, const Matrix &Bv_)
        : At(At_), Bt(Bt_), Av(Av_), Bv(Bv_) {}

    Matrix randgen(){
      Matrix T = gamma::rand(At, Bt);
      Matrix V = gamma::rand(Av, Bv);
      return dot(T,V);
    }

    // Variational Bayes Solution
    Solution vb(const Matrix &X, size_t max_iter = MAX_ITER){

      // Initialize T,V
      Matrix Lt = gamma::rand(At, Bt);
      Matrix Et = Lt;

      Matrix Lv = gamma::rand(Av, Bv);
      Matrix Ev = Lv;

      Matrix M = Matrix::ones(X.nrows(), X.ncols());
      Matrix Mt = Matrix::ones(At.nrows(), At.ncols());
      Matrix Mv = Matrix::ones(Av.nrows(), Av.ncols());

      Vector kl;


      for(unsigned i = 0; i < max_iter; ++i){

        // 1. Source sufficient statistics
        Matrix Z = X / dot(Lt,Lv);

        // 2. Means
        Matrix alpha_t = At + (Lt * dot( Z, Lv, false, true));
        Matrix beta_t = Mt / ((At/Bt) + dot(M, Ev, false, true));
        Et = alpha_t * beta_t;

        Matrix alpha_v = Av + (Lv * dot( Lt, Z, true, false));
        Matrix beta_v = Mv / ((Av/Bv) + dot(Et, M, true, false));
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
      Matrix V = Ev * sumCols(Et);
      Matrix T = normalizeCols(Et);
      return {T, V, kl};
    }

    Solution icm(const Matrix &X, size_t max_iter = MAX_ITER){

      // Initialize T,V
      Matrix T = gamma::rand(At, Bt);
      Matrix V = gamma::rand(Av, Bv);

      Vector kl;
      Matrix M = Matrix::ones(X.nrows(), X.ncols());

      for(unsigned i = 0; i < max_iter; ++i){
        Matrix TV = dot(T, V);
        kl.append(kl_div(X, TV));

        // Early stop ?
        if(i > MIN_ITER && kl(i) - kl(i-1) < 1e-5){
          break;
        }

        T = (At + (T * dot((X / TV), V, false, true)))
            / ((At / Bt) + dot(M, V, false, true));
        T = normalizeCols(T);

        V = (Av + (V * dot(T, (X / TV ), true)))
             / ((Av / Bv) + dot(T, M, true));
      }

      return {T, V, kl};
    }

  public:
      Matrix At, Bt, Av, Bv;
  };

}
#endif //THINNMF_NMF_H
