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

  // ----------- OBSERVATION MODELS ----------- //

  template <class P>
  class ObservationModel{

    public:
      ObservationModel(const P &prior_) : prior(prior_) {}

    public:
      void reset(){
        lambda = prior.rand();
      }
      virtual Vector rand() const  = 0;

      virtual void update(P &p, const Vector &obs) const = 0;

    public:
      P prior;
      Vector lambda;
  };

  class PoissonRandom : public ObservationModel<GammaPotential> {

    public:
      PoissonRandom(const GammaPotential &prior_) : ObservationModel(prior_){
        reset();
      }

      Vector rand() const override{
        return poisson::rand(lambda.first(), 1);
      }

      GammaPotential transformObs(const Vector &obs){
        return GammaPotential(obs.first()+1, 1);
      }

      void update(GammaPotential &gp, const Vector &obs) const override{
        gp *= GammaPotential(obs.first()+1, 1);
      }
  };

  class MultinomialRandom : public ObservationModel<DirichletPotential> {

    public:
      MultinomialRandom(const DirichletPotential &prior_)
              : ObservationModel(prior_){
        reset();
      }

      Vector rand() const override {
        return multinomial::rand(lambda, 100);
      }

      DirichletPotential transformObs(const Vector &obs){
        return DirichletPotential(obs+1);
      }

      void update(DirichletPotential &dp, const Vector &obs) const override {
        double log_c = std::lgamma(sum(obs)+1) - std::lgamma(sum(obs+1));
        dp *= DirichletPotential(obs+1, log_c);
      }
  };

  // ----------- MESSAGE----------- //

  template <class P>
  class Message {

    public:
      size_t size() const {
        return components.size();
      }

      void add_component(const P &potential){
        components.push_back(potential);
      }

      void add_component(const P &potential, double log_c){
        components.push_back(potential);
        components.back().log_c = log_c;
      }

      friend Message<P> operator*(const Message<P> &m1, const Message<P> &m2){
        Message<P> msg;
        for(auto &component1 : m1.components)
          for(auto &component2 : m2.components)
            msg.add_component(component1 * component2);
        return msg;
      }

      // Returns mean and cpp (Maybe renamed)
      std::pair<Vector, double> evaluate(int N = 1){
        Vector consts;
        Matrix params;
        for(auto &potential: components){
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
        while(components.size() > max_components){
          // Find mininum no-change element
          auto iter = std::min_element(components.begin(), components.end()-1);
          // Swap the last two elements to save the order of the change comp.
          std::swap(*(components.end()-1), *(components.end()-2));
          // Swap last element with the minimum compoment.
          std::swap(*iter, components.back());
          // Delete minimum component.
          components.pop_back();
        }
      }

      double log_likelihood(){
        Vector consts;
        for(auto &potential: components){
          consts.append(potential.log_c);
        }
        return logSumExp(consts);
      }

    public:
      std::vector<P> components;
  };


  // ----------- MODEL----------- //
  template <class P, class Obs>
  class Model{
    public:
      Model(const P &prior_, double p1_)
              : prior(prior_), observation(prior_), p1(p1_){
        log_p1 = std::log(p1);
        log_p0 = std::log(1-p1);
      }

    public:
      // returns  "states" as first matrix and "obervations" as second
      std::pair<Matrix, Matrix> generateData(size_t T){
        Matrix states, obs;

        for (size_t t=0; t<T; t++) {
          if (t > 0 && uniform::rand() < p1) {
            observation.reset();
          }
          states.appendColumn(observation.lambda);
          obs.appendColumn(observation.rand());
        }
        return {states, obs};
      }

      Message<P> initialMessage(){
        Message<P> message;
        message.add_component(prior, log_p0);
        message.add_component(prior, log_p1);
        return message;
      }

      Message<P> predict(const Message<P> &prev){
        Message<P> message = prev;
        Vector consts;
        for(auto &component : message.components){
          consts.append(component.log_c);
          component.log_c += log_p0;
        }
        message.add_component(prior, log_p1 + logSumExp(consts));
        return message;
      }

      void update(Message<P> &message, const Vector &obs){
        for(auto &component : message.components)
          observation.update(component, obs);
      }

      void set_p1(double p1_new){
        p1 = p1_new;
        log_p1 = std::log(p1);
        log_p0 = std::log(1-p1);
      }

      double get_p1(){
        return p1;
      }

      double get_log_p0(){
        return log_p0;
      }

      double get_log_p1(){
        return log_p1;
      }

      void set_prior(const P &prior_new){
        prior = prior_new;
      }

      P get_prior(){
        return prior;
      }

    private:
      P prior;
      Obs observation;
      double p1;                //  probability of change
      double log_p1, log_p0;
  };

  template <class P, class Obs>
  class ForwardBackward {
    public:
      ForwardBackward(const Model<P, Obs> &model_,
                      int lag_ = 0, int max_components_ = 100)
              : model(model_), lag(lag_), max_components(max_components_) {}


    public:
      void oneStepForward(std::vector<Message<P>> &alpha, const Vector& obs) {
        // predict step
        if (alpha.size() == 0) {
          Message<P> message;
          message.add_component(model.get_prior(), model.get_log_p0());
          message.add_component(model.get_prior(), model.get_log_p1());
          alpha.push_back(message);
        }
        else
          alpha.push_back(model.predict(alpha.back()));
        // update step
        model.update(alpha.back(), obs);
      }

      void oneStepBackward(std::vector<Message<P>> &beta, const Vector& obs) {
        // predict step
        if (beta.size()==0) {
          Message<P> message;
          message.add_component(model.get_prior(), 0);
          beta.push_back(message);
        }
        else {
          Message<P> m = beta.back();
          model.update(m, obs);
          beta.push_back(model.predict(m));
        }
      }

      std::vector<Message<P>> forward(const Matrix& obs){
        std::vector<Message<P>> alpha;
        for (size_t i=0; i<obs.ncols(); i++) {
          oneStepForward(alpha, obs.getColumn(i));
          alpha.back().prune(max_components);
        }
        return alpha;
      }

      std::vector<Message<P>> backward(const Matrix& obs){
        std::vector<Message<P>> beta;
        oneStepBackward(beta, Vector());

        for (size_t i=obs.ncols(); i>1; i--) {
          oneStepBackward(beta, obs.getColumn(i-1));
          beta.back().prune(max_components);
        }
        // We calculated Beta backwards, we need to reverse the list.
        std::reverse(beta.begin(), beta.end());
        return beta;
      }

      // Returns mean and cpp
      std::pair<Matrix, Vector> filtering(const Matrix& obs) {
        // Run forward
        auto alpha = forward(obs);

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

      // Returns mean and cpp
      std::pair<Matrix, Vector> smoothing(const Matrix& obs) {
        Matrix mean;
        Vector cpp;
        // Run Forward and Backward
        auto alpha = forward(obs);
        auto beta = backward(obs);

        for(auto &message : alpha){
          std::cout << "alpha:\n";
          for(auto &potential: message.components){
            potential.print();
          }
        }
        std::cout << "-------------\n\n";

        for(auto &message : beta){
          std::cout << "beta:\n";
          for(auto &potential: message.components){
            potential.print();
          }
        }
        std::cout << "-------------\n\n";

        // Calculate Smoothed density
        Message<P> gamma;

        for(size_t i=0; i < obs.ncols(); ++i) {
          gamma = alpha[i] * beta[i];
          std::cout << gamma.log_likelihood() << std::endl;
          std::cout << "smoothed: \n";
          for(auto &potential: gamma.components){
            potential.print();
          }
          auto result = gamma.evaluate(beta[i].size());
          mean.appendColumn(result.first);
          cpp.append(result.second);
        }
        return {mean, cpp};
      }

      // a.k.a. fixed-lag smoothing
      std::pair<Matrix, Vector> online_smoothing(const Matrix& obs) {
        if( lag == 0){
          return filtering(obs);
        }
        Matrix mean;
        Vector cpp;
        std::vector<Message<P>> alpha = forward(obs);
        Message<P> gamma;
        for (size_t t=0; t<obs.ncols()-lag+1; t++) {
          auto beta = backward(obs.getColumns(Range(t, t+lag)));
          gamma = alpha[t] * beta[0];
          auto result = gamma.evaluate(beta[0].size());
          mean.appendColumn(result.first);
          cpp.append(result.second);
          // if T-lag is reached, smooth the rest
          if(t == obs.ncols()-lag){
            for(int i = 1; i < lag; ++i){
              gamma = alpha[t+i] * beta[i];
              auto result = gamma.evaluate(beta[i].size());
              mean.appendColumn(result.first);
              cpp.append(result.second);
            }
          }
        }
        return {mean, cpp};
      }

      std::pair<Matrix, Vector> learn_params(const Matrix& obs){

        std::cout << model.get_p1() << std::endl;
        size_t MAX_ITER = 10;
        Vector ll;

        for(size_t iter = 0; iter < MAX_ITER; ++iter){
          // Forward_backward
          auto alpha = forward(obs);
          auto beta = backward(obs);
          ll.append(alpha.back().log_likelihood());

          // Smooth
          std::vector<Message<P>> gamma;
          Vector cpp;
          for(size_t i=0; i < obs.ncols(); ++i) {
            gamma.push_back(alpha[i] * beta[i]);
            auto result = gamma.back().evaluate(beta[i].size());
            cpp.append(result.second);
          }

          // Log-likelihood
          std::cout << "ll is " <<  ll.last() << std::endl;
          if(iter > 0 && ll[iter] < ll[iter-1]){
            std::cout << "likelihood decreased.\n";
          }

          // E-Step:

          // M-Step:
          model.set_p1(sum(cpp) / cpp.size());
          std::cout << model.get_p1() << std::endl;

        }
        return smoothing(obs);
      }


    public:
      Model<P, Obs> model;
      int lag;
      int max_components;
  };

  using DM_Model = Model<DirichletPotential, MultinomialRandom>;
  using DM_ForwardBackward = ForwardBackward<DirichletPotential, MultinomialRandom>;

  using PG_Model = Model<GammaPotential, PoissonRandom>;
  using PG_ForwardBackward = ForwardBackward<GammaPotential, PoissonRandom>;

}

#endif //MATLIB_PML_BCPM_H
