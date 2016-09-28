#ifndef MATLIB_PML_BCPM_H
#define MATLIB_PML_BCPM_H

#include "pml.hpp"
#include "pml_rand.hpp"

#include <algorithm>

namespace pml {
  // ----------- POTENTIALS ----------- //

  class Potential {

    public:
      explicit Potential(double log_c_) : log_c(log_c_) {}

    public:
      bool operator<(const Potential &other) const{
        return this->log_c < other.log_c;
      }

      virtual Vector rand() const = 0;

      virtual Vector mean() const = 0;


    public:
      double log_c;
  };

  class DirichletPotential : public Potential {

    public:
      DirichletPotential(size_t K, double log_c_ = 0) : Potential(log_c_) {
        alpha = Vector::ones(K);
      }

      DirichletPotential(const Vector& alpha_, double log_c_ = 0)
          : Potential(log_c_), alpha(alpha_) {}

    public:
      void operator*=(const DirichletPotential &p){
        *this = this->operator*(p);
      }

      DirichletPotential operator*(const DirichletPotential &p) const{

        double delta = std::lgamma(sum(alpha)) - sum(lgamma(alpha)) +
                       std::lgamma(sum(p.alpha)) - sum(lgamma(p.alpha)) +
                       sum(lgamma(alpha + p.alpha-1)) -
                       std::lgamma(sum(alpha + p.alpha -1));

        return DirichletPotential(alpha + p.alpha - 1,
                                  log_c + p.log_c + delta);
      }

      static DirichletPotential obs2Potential(const Vector& obs){
        return DirichletPotential(obs+1);
      }

      Vector rand() const override {
        return dirichlet::rand(alpha);
      }

      Vector mean() const override {
        return normalize(alpha);
      }

      void print(){
        std::cout << alpha << " log_c:" << log_c << std::endl;
      }

    public:
      Vector alpha;
  };

  class GammaPotential : public Potential {

    public:
      GammaPotential(double a_ = 1, double b_ = 1, double log_c_ = 0)
              : Potential(log_c_), a(a_), b(b_){}

    public:
      void operator*=(const GammaPotential &other){
        *this = this->operator*(other);
      }

      GammaPotential operator*(const GammaPotential &other) const{
        double delta = std::lgamma(a + other.a - 1)
                       - std::lgamma(a) - std::lgamma(other.a)
                       + std::log(b + other.b)
                       + a * std::log(b/(b + other.b))
                       + other.a * std::log(other.b/(b + other.b));
        return GammaPotential(a + other.a - 1,
                              b + other.b,
                              log_c + other.log_c + delta);
      }

      static GammaPotential obs2Potential(const Vector& obs){
        return GammaPotential(obs.first()+1, 1);
      }

      Vector rand() const override {
        return gamma::rand(a, b, 1);
      }

      Vector mean() const override {
        return Vector(1, a / b);
      }

      void print(){
        std::cout << "a:" << a << "  b:" << b
                  << "  log_c: " << log_c << std::endl;
      }

    public:
      double a;
      double b;
  };

  // ----------- MODEL ----------- //

  template <class P>
  class Model{

    public:
      Model(const P &prior_, double p1_) : prior(prior_) {
        set_p1(p1_);
      }

    public:
      void set_p1(double p1_new){
        p1 = p1_new;
        log_p1 = std::log(p1);
        log_p0 = std::log(1-p1);
      }

      virtual Vector rand(const Vector &state) const {
        return Vector();
      }

      std::pair<Matrix, Matrix> generateData(size_t length){
        Matrix states, obs;
        Vector state = prior.rand();
        for (size_t t=0; t<length; t++) {
          if (t == 0 || uniform::rand() < p1) {
            state = prior.rand();
          }
          states.appendColumn(state);
          obs.appendColumn(rand(state));
        }
        return {states, obs};
      }

    public:
      P prior;
      double p1;
      double log_p1, log_p0;
  };

  class PG_Model : public Model<GammaPotential> {

    public:
      PG_Model(const GammaPotential &prior_, double p1_)
          : Model(prior_, p1_) { }

      Vector rand(const Vector &state) const override {
        return poisson::rand(state.first(), 1);
      }
  };

  class DM_Model: public Model<DirichletPotential> {

    public:
      DM_Model(const DirichletPotential &prior_, double p1_)
              : Model(prior_, p1_){ }

      Vector rand(const Vector &state) const override {
        return multinomial::rand(state, 100);
      }
  };

  template <class P>
  class Message {

    public:
      size_t size() const {
        return potentials.size();
      }

      void add_potential(const P &potential){
        potentials.push_back(potential);
      }

      void add_potential(const P &potential, double log_c){
        potentials.push_back(potential);
        potentials.back().log_c = log_c;
      }

      friend Message<P> operator*(const Message<P> &m1, const Message<P> &m2){
        Message<P> msg;
        for(auto &pot1 : m1.potentials)
          for(auto &pot2 : m2.potentials)
            msg.add_potential(pot1 * pot2);
        return msg;
      }

      // Returns mean and cpp (Maybe renamed)
      std::pair<Vector, double> evaluate(int N = 1){
        Vector consts;
        Matrix params;
        for(auto &potential: potentials){
          consts.append(potential.log_c);
          params.appendColumn(potential.mean());
        }
        consts = normalizeExp(consts);
        // Calculate mean
        Vector mean = sumRows(transpose(transpose(params)*consts));
        // Calculate cpp as the sum of last N probabilities
        double cpp = 0;
        for(int i=0; i < N; ++i)
          cpp += consts(consts.size() - i - 1);
        return {mean, cpp};
      };

      void prune(size_t max_components){
        while(size() > max_components){
          // Find mininum no-change element
          auto iter = std::min_element(potentials.begin(), potentials.end()-1);
          // Swap the last two elements to save the order of the change comp.
          std::swap(*(potentials.end()-1), *(potentials.end()-2));
          // Swap last element with the minimum compoment.
          std::swap(*iter, potentials.back());
          // Delete minimum component.
          potentials.pop_back();
        }
      }

      double log_likelihood(){
        Vector consts;
        for(auto &potential: potentials){
          consts.append(potential.log_c);
        }
        return logSumExp(consts);
      }

    public:
      std::vector<P> potentials;

  };


  template <class P>
  class ForwardBackward {
    public:
      ForwardBackward(const Model<P> &model_, int max_components_ = 100)
          :model(model_), max_components(max_components_){
        alpha.clear();
        alpha_predict.clear();
        beta.clear();
      }

    public:
      Message<P> predict(const Message<P>& prev){
        Message<P> message = prev;
        Vector consts;
        for(auto &potential : message.potentials){
          consts.append(potential.log_c);
          potential.log_c += model.log_p0;
        }
        message.add_potential(model.prior, model.log_p1 + logSumExp(consts));
        return message;
      }

      Message<P> update(const Message<P> &prev, const Vector &obs){
        Message<P> message = prev;
        for(auto &potential : message.potentials) {
          potential *= P::obs2Potential(obs);
        }
        return message;
      }

    // ------------- FORWARD ------------- //
    public:
      std::pair<Matrix, Vector> filtering(const Matrix& obs) {
        // Run forward
        forward(obs);

        // Calculate mean and cpp
        Matrix mean;
        Vector cpp;
        for(auto &message : alpha){
          auto result = message.evaluate();
          mean.appendColumn(result.first);
          cpp.append(result.second);
        }
        return {mean, cpp};
      }

      void forward(const Matrix& obs){
        alpha.clear();
        alpha_predict.clear();
        for (size_t i=0; i<obs.ncols(); i++) {
          oneStepForward(obs.getColumn(i));
          alpha.back().prune(max_components);
        }
      }

      void oneStepForward(const Vector& obs) {
        // Predict step
        if (alpha_predict.empty()) {
          Message<P> message;
          message.add_potential(model.prior, model.log_p0);
          message.add_potential(model.prior, model.log_p1);
          alpha_predict.push_back(message);
        }
        else {
          alpha_predict.push_back(predict(alpha.back()));
        }
        // Update step
        alpha.push_back(update(alpha_predict.back(), obs));
      }

      // ------------- BACKWARD ------------- //
      void backward(const Matrix& obs, size_t idx = 0, size_t steps = 0){
        // Start from column "idx" and go back for "steps" steps
        if(steps == 0 ){
          steps = obs.ncols();
          idx = obs.ncols()-1;
        }
        beta.clear();
        for(size_t t = 0; t < steps; ++t, --idx){
          Message<P> message;
          double c = 0;
          if(!beta.empty()){
            // Predict for case s_t = 1, calculate constant only
            message = beta.back();
            for(auto &potential : message.potentials){
              potential *= model.prior;
            }
            c = model.log_p1 + message.log_likelihood();

            // Update :
            message = update(beta.back(), obs.getColumn(idx));
            for(auto &potential : message.potentials){
              potential.log_c += model.log_p0;
            }
          }
          message.add_potential(P::obs2Potential(obs.getColumn(idx)), c);
          beta.push_back(message);
        }
        std::reverse(beta.begin(), beta.end());
      }


      std::pair<Matrix, Vector> smoothing(const Matrix& obs) {
        // Run Forward - Backward
        forward(obs);
        backward(obs);

        // Calculate Smoothed density
        Matrix mean;
        Vector cpp;
        for(size_t i=0; i < obs.ncols(); ++i) {
          Message<P> gamma = alpha_predict[i] * beta[i];
          auto result = gamma.evaluate(beta[i].size());
          mean.appendColumn(result.first);
          cpp.append(result.second);
        }
        return {mean, cpp};
      }

  public:
      Model<P> model;
      int max_components;

    private:
      std::vector<Message<P>> alpha;
      std::vector<Message<P>> alpha_predict;
      std::vector<Message<P>> beta;

  };

  using PG_ForwardBackward = ForwardBackward<GammaPotential>;

} // namespace

#endif //MATLIB_PML_BCPM_H
