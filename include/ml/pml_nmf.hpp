//
// Created by baris on 17.03.2016.
//

#ifndef PML_NMF_H
#define PML_NMF_H

#include "../pml.hpp"

/*
 * T ~ Gamma(x; a_t, b_t/a_t)
 * V ~ Gamma(x; a_v, b_v/a_v)
 * X ~ Sum_k Poisson(S[i,k,j]; T[i,k] * V[k,j])
 *
 * mean(T) = b_t
 * var(T) = b_t^2 / a_t
 *
 *  Large a_t results in low variance, similar values
 *  Small a_t results in high variance, hence favors sparsity
 */

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
            b.saveTxt(work_dir + "/b.txt");
          }
        public:
          Matrix t;
          Matrix v;
          Vector kl;
          Vector b;
      };

    public:
      NMF(size_t dim1, size_t dim2, size_t rank) {
        At = Matrix::ones(dim1, rank);
        Bt = Matrix::ones(dim1, rank);
        Av = Matrix::ones(rank, dim2);
        Bv = Matrix::ones(rank, dim2);
      }

      NMF(const Matrix &At_, const Matrix &Bt_,
          const Matrix &Av_, const Matrix &Bv_)
          : At(At_), Bt(Bt_), Av(Av_), Bv(Bv_) {
      }

    public:
      Matrix randgen() {
        initialize_T_V();
        Matrix X = Matrix::zeros(At.nrows(), Av.ncols());
        for(size_t i = 0; i < X.nrows(); ++i)
          for(size_t j = 0; j < X.ncols(); ++j)
            for(size_t k = 0; k < At.ncols(); ++k)
              X(i,j) += Poisson(T(i,k) * V(k, j)).rand();

        return X;
      }

      size_t rank() const {
        return At.ncols();
      }

    public:
      // Maximum Likelihood Solution
      Solution ml(const Matrix &X){

        // Initialize T,V
        initialize_T_V();
        Matrix M = Matrix::ones(X.nrows(), X.ncols());
        Vector kl;

        for(unsigned i = 0; i < MAX_ITER; ++i){

          Matrix TV = dot(T, V);
          kl.append(kl_div(X, TV));

          // Early stop ?
          if(i > MIN_ITER && kl(i) - kl(i-1) < 1e-5)
            break;

          // Update T and normalize its columns to avoid degeneracy
          T = (T * dot((X / TV), V, false, true)) / dot(M, V, false, true);
          T = normalize(T, 0);

          // Update V
          V = (V * dot(T, (X / TV ), true)) / (dot(T, M, true));
        }

        return {T, V, kl};
      }

    public:
      // Iterative Conditional Modes
      Solution icm(const Matrix &X){

        initialize_T_V();
        Matrix M = Matrix::ones(X.nrows(), X.ncols());
        Vector kl;

        for(unsigned i = 0; i < MAX_ITER; ++i){
          Matrix TV = dot(T, V);
          kl.append(kl_div(X, TV));

          // Early stop ?
          if(i > MIN_ITER && kl(i) - kl(i-1) < 1e-5)
            break;

          T = (At + (T * dot((X / TV), V, false, true)))
              / ((At / Bt) + dot(M, V, false, true));
          T = normalize(T, 0);

          V = (Av + (V * dot(T, (X / TV ), true)))
              / ((Av / Bv) + dot(T, M, true));
        }
        return {T, V, kl};
      }

    public:
      // Variational Bayes Solution
      Solution vb(const Matrix &X, const std::string &opt_params_update= ""){

        // Initialize
        initialize_T_V();

        Matrix Lt = T;
        Matrix Et = T;

        Matrix Lv = V;
        Matrix Ev = V;

        Matrix M = Matrix::ones(X.nrows(), X.ncols());
        Matrix Mt = Matrix::ones(T.nrows(), T.ncols());
        Matrix Mv = Matrix::ones(V.nrows(), V.ncols());

        double gammalnX = sum(gammaln(X+1));

        Vector kl;
        Vector b;

        for(unsigned i = 0; i < MAX_ITER; ++i){

          // 1. Source sufficient statistics
          Matrix Z = X / dot(Lt,Lv);
          Matrix Sigma_t = Lt * dot( Z, Lv, false, true);
          Matrix Sigma_v = Lv * dot( Lt, Z, true, false);

          // 1a. Precompute for fast calculation
          Matrix AtBt = At / Bt;
          Matrix AvBv = Av / Bv;
          Matrix LtLv = dot(Lt, Lv);

          // 2. Means
          Matrix alpha_t = At + Sigma_t;
          Matrix beta_t = Mt / (AtBt + dot(M, Ev, false, true));
          Et = alpha_t * beta_t;

          Matrix alpha_v = Av + Sigma_v;
          Matrix beta_v = Mv / (AvBv + dot(Et, M, true, false));
          Ev = alpha_v * beta_v;

          // Calculate Bound
          b.append(0);
          b.last() -= sum(dot(Et, Ev)) + gammalnX;
          b.last() -= sum(((dot(Lt * log(Lt), Lv) + dot(Lt, Lv*log(Lv))) /
                                 LtLv - log(LtLv)) * X);
          b.last() -= sum(AtBt * Et) + sum(gammaln(At)) - sum(At * log(AtBt));
          b.last() += sum(alpha_t * (log(beta_t) + 1) + gammaln(alpha_t));
          b.last() -= sum(AvBv * Ev) + sum(gammaln(Av)) - sum(Av * log(AvBv));
          b.last() += sum(alpha_v * (log(beta_v) + 1) + gammaln(alpha_v));

          // 3. Means of Logs
          Lt = exp(psi(alpha_t)) * beta_t;
          Lv = exp(psi(alpha_v)) * beta_v;


          // Update parameters if necessary (B only, all tied)
          if(!opt_params_update.empty()){
            update_parameters(Et, Ev, opt_params_update);
          }

          // 4. Track KL
          kl.append(kl_div(X, dot(Et, Ev)));

          // Early stop ?
          if(i > MIN_ITER && kl(i) - kl(i-1) < 1e-5)
            break;
        }

        // Normalize T_ columns to get basis vectors:
        V = Ev * sum(Et, 0);
        T = normalize(Et, 0);

        return {T, V, kl, b};
      }

    private:

      void initialize_T_V() {
        T = Matrix(At.shape());
        for(size_t i=0; i < T.size(); ++i)
          T[i] = Gamma(At[i], Bt[i] / At[i]).rand();

        V = Matrix(Av.shape());
        for(size_t i=0; i < V.size(); ++i)
          V[i] = Gamma(Av[i], Bv[i] / Av[i]).rand();
      }

      void update_parameters(const Matrix &Et, const Matrix &Ev,
                             const std::string &opt_params_update){
        if (opt_params_update == "tie_none"){
          Bt = Et;
          Bv = Ev;
        } else if (opt_params_update == "tie_columns"){
          Bt = tile(sum(At * Et, 1) / sum(At, 1), Bt.ncols(), 1);
          Bv = tile(sum(Av * Ev, 1) / sum(Av, 1), Bv.ncols(), 1);
        } else if (opt_params_update == "tie_rows"){
          Bt = tile(sum(At * Et, 0) / sum(At, 0), Bt.nrows(), 0);
          Bv = tile(sum(Av * Ev, 0) / sum(Av, 0), Bv.nrows(), 0);
        } else if (opt_params_update == "tie_all"){
          Bt = sum(At * Et) / sum(At);
          Bv = sum(Av * Ev) / sum(Av);
        }
      }

    public:
      Matrix At, Bt, Av, Bv;
      Matrix T, V;
  };

}
#endif //THINNMF_NMF_H
