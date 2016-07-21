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

using namespace std;
using namespace pml;

namespace hmm {

    class HMM {

        public:
            // SS = sufficient statistics
            enum SS_METHOD {
                CORRECTION_SMOOTHER,
                RECURSIVE_SMOOTHER
            };


        public:
            HMM(const Vector& pi_, const Matrix& A_, const Matrix& B_) {
                N = pi_.size();
                M = B_.nrows();
                pi = normalize(pi_);
                A = normalize(A_, Matrix::COLS);
                B = normalize(B_, Matrix::COLS);
                logpi = log(pi);
                logA = log(A);
                logB = log(B);
            }
    
            HMM(unsigned N_, unsigned M_) {
                N = N_;
                M = M_;
                pi = dirichlet::rand(Vector::ones(N_));
                A = dirichlet::rand(Vector::ones(N_),N_);
                B = dirichlet::rand(Vector::ones(M_),N_);
                setpi(pi);
                setA(A);
                setB(B);
            }
    
            HMM(const std::string &dump_file) {
                std::ifstream infile(dump_file);
                ASSERT_TRUE(infile.is_open(), "HMM is not loaded from " + dump_file);
                infile >> N >> M;
                Vector pi_ = Vector::zeros(N);
                Matrix A_= Matrix::zeros(N,N);
                Matrix B_= Matrix::zeros(M,N);
                for (size_t t=0; t<N; t++) {
                    infile >> pi_(t);
                }
                setpi(pi_);
                for (size_t i=0; i<N; i++) {
                    for (size_t j=0; j<N; j++) {
                        infile >> A_(j,i);
                    }
                }
                setA(A_);
                for (size_t i=0; i<N; i++) {
                    for (size_t j=0; j<M; j++) {
                        infile >> B_(j,i);
                    }
                }
                setB(B_);
                infile.close();
            }
    
            HMM(const HMM &other) {
                N = other.N;
                M = other.M;
                setpi(other.pi);
                setA(other.A);
                setB(other.B);
            }
    
            HMM& operator=(const HMM &other) {
                if (this!=&other) {
                    N = other.N;
                    M = other.M;
                    setpi(other.pi);
                    setA(other.A);
                    setB(other.B);
                }
                return *this;
            }

            Vector getpi() { return pi;}
            Matrix getA() { return A;}
            Matrix getB() { return B;}
    
            void setpi(const Vector& pi_) {
                pi = normalize(pi_);
                logpi = log(pi);
            }
            void setA(const Matrix& A_) {
                A = normalize(A_, Matrix::COLS);
                logA = log(A);
            }
            void setB(const Matrix& B_){
                B = normalize(B_, Matrix::COLS);
                logB = log(B);
            }
    
            tuple<vector<unsigned>,vector<unsigned>> generateSequence(unsigned T) {
                vector<unsigned> states(T);
                vector<unsigned> obs(T);
    
                states[0] = randgen(pi);
                obs[0] = randgen(B.getColumn(states[0]));
                for (size_t i=1; i<T; i++) {
                    states[i] = randgen(A.getColumn(states[i-1]));
                    obs[i] = randgen(B.getColumn(states[i]));
                }
                return make_tuple(states, obs);
            }
    
            Vector predict(const Vector &pot) {
                double mx = max(pot);
                return log(dot(A,exp(pot-mx))) + mx;
            }
            Vector postdict(const Vector &pot) {
                double mx = max(pot);
                return log(dot(A,exp(pot-mx),true)) + mx;
            }
            Vector update(unsigned obs, const Vector &pot) {
                return logB.getRow(obs) + pot;
            }
    
    
            tuple<Matrix,Matrix> forwardRecursion(const vector<unsigned> &obs) {
                unsigned long T = obs.size();
                Matrix log_alpha_pred = Matrix::zeros(N,T);     // p(x_t|y_{1:t-1})
                Matrix log_alpha = Matrix::zeros(N,T);          // p(x_t|y_{1:t})
                size_t i = 0;
    
                log_alpha_pred.setColumn(i,logpi);
                log_alpha.setColumn(i, update(obs[i], log_alpha_pred.getColumn(i)));
                for(i=1; i<T; ++i) {
                    log_alpha_pred.setColumn(i, predict(log_alpha.getColumn(i - 1)));
                    log_alpha.setColumn(i, update(obs[i], log_alpha_pred.getColumn(i)));
                }
                return make_tuple(log_alpha_pred,log_alpha);
            }
            tuple<Matrix,Matrix> backwardRecursion(const vector<unsigned> &obs) {
                size_t T = obs.size();
                Matrix log_beta_postdict = Matrix::zeros(N,T);      // p(x_t|y_{t-1:T})
                Matrix log_beta = Matrix::zeros(N,T);               // p(x_t|y_{t:T})
                log_beta.setColumn(T-1, update(obs[T - 1], log_beta_postdict.getColumn(T - 1)));
    
                for(int i=T-2; i>=0; i--) {
                    log_beta_postdict.setColumn(i, postdict(log_beta.getColumn(i + 1)));
                    log_beta.setColumn(i, update(obs[i], log_beta_postdict.getColumn(i)));
                }
    
                return make_tuple(log_beta_postdict,log_beta);
    
            }
            Matrix FB_Smoother(const vector<unsigned> &obs) {
                tuple<Matrix,Matrix> alphas = forwardRecursion(obs);
                tuple<Matrix,Matrix> betas = backwardRecursion(obs);
    
                Matrix log_gamma = get<1>(alphas) + get<0>(betas);
                return log_gamma;
    
            }
            double evaluateLogLHood(const vector<unsigned> &obs) {
                Matrix log_gamma = FB_Smoother(obs);
                return logSumExp(log_gamma.getColumn(0));
            }
    
            tuple<Vector,Matrix, Matrix, Matrix> correctionSmoother(const vector<unsigned> &obs) {
                size_t T = obs.size();
                tuple<Matrix,Matrix> tupl = forwardRecursion(obs);
                Matrix log_alpha = get<1>(tupl);
                if (ContainsInf(log_alpha) || ContainsNan(log_alpha)) {
                    return make_tuple(Vector::zeros(1),Matrix::zeros(10,10),Matrix::zeros(10,10),Matrix::zeros(10,10));
                }
    
                Matrix log_gamma_corr = Matrix::zeros(N,T);
                log_gamma_corr.setColumn(T-1,log_alpha.getColumn(T-1));
                if (ContainsInf(log_gamma_corr) || ContainsNan(log_gamma_corr)) {
                    return make_tuple(Vector::zeros(1),Matrix::zeros(10,10),Matrix::zeros(10,10),Matrix::zeros(10,10));
                }
    
                Matrix C2 = Matrix::zeros(N,N);     // sufficient stats for trans. matr
                Matrix C3 = Matrix::zeros(M,N);     // sufficient stats for obs. matr
                C3.setRow(obs[T-1], normalizeExp(log_gamma_corr.getColumn(T-1)) );
    
                Matrix log_old_pairwise_marginal, numerator, log_new_pairwise_marginal;
                Vector log_old_marginal;
                for(int k=T-2; k>=0; --k) {
                    log_old_pairwise_marginal = tile(log_alpha.getColumn(k),logA.nrows(),Matrix::ROWS) + logA;
                    log_old_marginal = predict(log_alpha.getColumn(k));
                    // next two lines: log_new_pairwise_marginal = log_old_pairwise_marginal + log_gamma_corr[:,k+1].reshape(self.S,1) - log_old_marginal.reshape(self.S,1)
                    numerator = log_old_pairwise_marginal + tile(log_gamma_corr.getColumn(k+1),log_old_pairwise_marginal.ncols(),Matrix::COLS);
                    log_new_pairwise_marginal = numerator - tile(log_old_marginal,numerator.ncols(),Matrix::COLS);
                    log_gamma_corr.setColumn(k,logSumExp(log_new_pairwise_marginal, Matrix::COLS));
    
                    C2 = C2 + normalizeExp(log_new_pairwise_marginal);
                    C3.setRow(obs[k],C3.getRow(obs[k])+normalizeExp(log_gamma_corr.getColumn(k)));

                }
                Vector C1 = normalizeExp(log_gamma_corr.getColumn(0));
                return make_tuple(C1,C2,C3,log_gamma_corr);
            }
    
            /*
             * returns sufficient statistics, log-likelihood and three intermediate variables
             */
            tuple<Vector,Matrix, Matrix, double, Matrix, Tensor3D, Tensor3D> forwardOnlySS(const vector<unsigned> &obs) {
                size_t T = obs.size();
                Matrix V1 = Matrix::identity(N);
                Tensor3D V2 = Tensor3D::zeros(N,N,N);
                Tensor3D V3 = Tensor3D::zeros(N,N,M);
    
                Vector log_alpha_pred = logpi;
                Vector log_alpha = update(obs[0], log_alpha_pred);
    
                Matrix P, lp;
                Vector p_xT;
                Matrix eye = Matrix::identity(M);
                for(size_t k=1; k<T; k++) {
                    log_alpha_pred = predict(log_alpha);
                    // Calculate p(x_{k-1}|y_{1:k-1}, x_k)
                    lp = tile(log(normalizeExp(log_alpha)),logA.nrows(),Matrix::COLS) + transpose(logA);
                    P = normalizeExp(lp,Matrix::COLS);
    
                    // V update
                    V1 = dot(V1,P);
                    V2 = tm_prod(V2,P) + broaden(P);
                    V3 = tm_prod(V3,P) + weight_tensor(eye.getColumn(obs[k-1]),P);

                    log_alpha = update(obs[k], log_alpha_pred);
                    p_xT = normalizeExp(log_alpha);
                }
    
                double ll = logSumExp(log_alpha);
    
                Vector C1 = dot(V1,p_xT);
                Matrix C2 = tv_prod(V2,p_xT);
                Matrix C3 = tv_prod(V3,p_xT);
                C3.setRow(obs[T-1],C3.getRow(obs[T-1])+p_xT);
    
                return make_tuple(C1,C2,C3,ll,V1,V2,V3);
            }
    
            /*
             * n_min                        number of training samples processed before updating model parameters
             * log_period                   the period in which model parameters are recorded
             * learning_rate
             * learning_rate_update_period
             * C                            the learning constant (see the code)
             *
             * returns recorded model parameters
             */
            vector<tuple<Vector,Matrix,Matrix>> runOnlineEM(const vector<unsigned> &obs, unsigned n_min,
                                                            long log_period,
                                                            double learning_rate, int learning_rate_update_period,
                                                            double C) {
                double gamma = 0.2*C;
                size_t T = obs.size();
                vector<tuple<Vector,Matrix,Matrix>> parameters;
                Matrix V1 = Matrix::identity(N);
                Tensor3D V2 = Tensor3D::zeros(N,N,N);
                Tensor3D V3 = Tensor3D::zeros(N,N,M);
    
                Vector log_alpha_pred = logpi;
                Vector log_alpha = update(obs[0], log_alpha_pred);
    
                // intermediate variables
                Vector p_xT;
                Matrix P, lp;
                Matrix eye = Matrix::identity(M);
    
                // cumulative sufficient statistics
                Vector C1;
                Matrix C2;
                Matrix C3;
                for(size_t k=1; k<T; k++) {
                    // logging data
                    if ( fmod(k,log_period) == 1) {
                        parameters.push_back(make_tuple(pi,A,B));
                    }
    
                    log_alpha_pred = predict(log_alpha);
                    // Calculate p(x_{k-1}|y_{1:k-1}, x_k)
                    lp = tile(log(normalizeExp(log_alpha)),logA.nrows(),Matrix::COLS) + transpose(logA);
                    P = normalizeExp(lp,Matrix::COLS);
    
                    // V update
                    V1 = dot(V1,P);
                    V2 = (1-gamma)*tm_prod(V2,P) + gamma*broaden(P);
                    V3 = (1-gamma)*tm_prod(V3,P) + gamma*weight_tensor(eye.getColumn(obs[k-1]),P);
    
                    log_alpha = update(obs[k], log_alpha_pred);
                    p_xT = normalizeExp(log_alpha);
    
                    // M step
                    if (k > n_min) {
                        C1 = dot(V1,p_xT);
                        C2 = tv_prod(V2,p_xT);
                        C3 = tv_prod(V3,p_xT);
                        C3.setRow(obs[T-1],C3.getRow(obs[T-1])+p_xT);
    
                        setpi(C1+1e-15);
                        setA(C2+1e-15);
                        setB(C3+1e-15);
    
                        if (fmod(k,learning_rate_update_period) == 0) {
                            gamma = pow(k,learning_rate)*C;
                        }
                    }
                }
                return parameters;
    
            }
    
            vector<double> learnParameters(const vector<unsigned> &obs, unsigned EPOCH,
                                           SS_METHOD ss_method = SS_METHOD::CORRECTION_SMOOTHER) {
                vector<double> ll(EPOCH);
                Vector pi_ss;
                Matrix A_ss;
                Matrix B_ss;
                for (unsigned i=0; i<EPOCH; i++) {
                    // E step
                    if (ss_method == CORRECTION_SMOOTHER) {
                        tuple<Vector,Matrix, Matrix, Matrix> stats = correctionSmoother(obs);
                        pi_ss = get<0>(stats)+1e-15;
                        A_ss = get<1>(stats)+1e-15;
                        B_ss = get<2>(stats)+1e-15;
                        ll[i] = logSumExp(get<3>(stats).getColumn(0));
                    }
                    else if (ss_method == RECURSIVE_SMOOTHER) {
                        tuple<Vector,Matrix, Matrix, double, Matrix, Tensor3D, Tensor3D> stats = forwardOnlySS(obs);
                        pi_ss = get<0>(stats)+1e-15;
                        A_ss = get<1>(stats)+1e-15;
                        B_ss = get<2>(stats)+1e-15;
                        ll[i] = get<3>(stats);
                    }
    
                    // M step
                    setpi(pi_ss);
                    setA(A_ss);
                    setB(B_ss);
    
                }
                return ll;
            }

            friend ostream& operator<<(ostream& out, const HMM& hmm) {
                out << hmm.pi << endl << hmm.A << endl << hmm.B << endl;
                return out;
            }

        /*
         * Helpers
         */
        private:
            unsigned randgen(const Vector& v) {
                unsigned i=0;
                double rnd = uniform::rand();
                double cumsum = 0;
                for (; i<v.size(); i++) {
                    cumsum += v(i);
                    if (cumsum > rnd) {
                        break;
                    }
                }
                return i;
            }
            bool ContainsNan(const Matrix& m) {
                for (vector<double>::const_iterator iter=m.begin(); iter!=m.end(); iter++) {
                    if (std::isnan(*iter)) {
                        return true;
                    }
                }
                return false;
            }
            bool ContainsInf(const Matrix& m) {
                for (vector<double>::const_iterator iter=m.begin(); iter!=m.end(); iter++) {
                    if (std::isinf(*iter)) {
                        return true;
                    }
                }
                return false;
            }
            Matrix matVecSum(const Matrix& mat, const Vector& vec, Matrix::Axes ax) {
                Matrix ret(mat.nrows(), mat.ncols());
                if (ax == Matrix::ROWS && mat.ncols() == vec.size()) {
                    for (size_t i=0; i<mat.nrows(); i++) {
                        ret.setRow(i,ret.getRow(i)+vec);
                    }
                    return ret;
                }
                else if (ax == Matrix::COLS && mat.nrows() == vec.size()) {
                    for (size_t i=0; i<mat.ncols(); i++) {
                        ret.setColumn(i,ret.getColumn(i)+vec);
                    }
                    return ret;
                }
                cout << "hay allah!" << endl;
                throw std::invalid_argument("Axes must be either Matrix::ROWS or Matrix::COLS");
            }
            // special tensor operations
            Tensor3D tm_prod(const Tensor3D& tensor, const Matrix& rhs) {
                Tensor3D tens = tensor;
                Tensor3D result(tensor.dim0(),rhs.ncols(),tensor.dim2());
                for (size_t i=0; i<tensor.dim2(); i++) {
                    result.setSlice(i,dot(tens.getSlice(i),rhs));
                }
                return result;
            }
            Matrix tv_prod(const Tensor3D& tensor, const Vector& vec) {
                Matrix mat(tensor.dim2(),tensor.dim0());
                for (size_t i=0; i<tensor.dim2(); i++) {
                    Matrix tmp = tensor.getSlice(i);
                    mat.setRow(i,dot(tmp,vec));
                }
                return mat;
            }
            Tensor3D broaden(const Matrix &m) {
                Tensor3D tens = Tensor3D::zeros(m.nrows(),m.ncols(),m.ncols());
                for (size_t i=0; i<m.ncols(); i++) {
                    Matrix tmp = Matrix::zeros(m.ncols(),m.ncols());
                    tmp.setColumn(i,m.getColumn(i));
                    tens.setSlice(i,tmp);
                }
                return tens;
            }
            Tensor3D weight_tensor(const Vector& vec,const Matrix &m) {
                Tensor3D tens = Tensor3D::zeros(m.nrows(),m.ncols(),vec.size());
                for (size_t i=0; i<vec.size(); i++) {
                    tens.setSlice(i,m*vec(i));
                }
                return tens;
            }

    protected:
        size_t N;       // hidden states
        size_t M;       // observations
        Vector pi;      // prior
        Matrix A;       // state transition matrix
        Matrix B;       // observation matrix
        Vector logpi;   // log-prior
        Matrix logA;    // log of state transition matrix
        Matrix logB;    // log of observation matrix
    };





}

#endif