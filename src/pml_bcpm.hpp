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


/*
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

      Message<P> update(const Message<P> &predict, const Vector &obs){
        Message<P> message = predict;
        for(auto &component : message.components) {
          observation.update(component, obs);
        }
        return message;
      }


  };
    */


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

    // --------- FORWARD ------------- //
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

        for(auto &message : alpha){
          std::cout << "alpha:\n";
          for(auto &potential: message.potentials){
            potential.print();
          }
        }
        std::cout << "-------------\n\n";
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

      // --------- BACKWARD ------------- //
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
        for(auto &message : beta){
          std::cout << "beta:\n";
          for(auto &potential: message.potentials){
            potential.print();
          }
        }
        std::cout << "-------------\n\n";

      }

  public:
      Model<P> model;
      int max_components;

    private:
      std::vector<Message<P>> alpha;
      std::vector<Message<P>> alpha_predict;
      std::vector<Message<P>> beta;

  };


/*

      // --------- BACKWARD RECURSION --------- //

      void oneStepBackward(const Vector& obs) {
        // predict step
        if (beta.size()==0) {
          Message<P> message;
          message.add_component(Obs::transformObs(obs));
          beta.push_back(message);
        }
        else {
          Message<P> m = beta.back();
          m = model.update(m, obs);
          beta.push_back(model.predict(m));
        }
      }

      void backward(const Matrix& obs, int start = -1, int stop = 0){
        beta.clear();
        if(start == -1){
          start = obs.ncols()-1;
        }
        for (int t = start; t >= stop; --t) {
          Message<P> message;
          Vector v = obs.getColumn(t);
          if (beta.size()==0) {
            message.add_component(Obs::transformObs(v));
            beta.push_back(message);
          } else {
            // Predict p1:
            Message<P> temp = model.update(beta.back(), model.prior);
            double c = temp.log_likelihood() + model.log_p1;
            // Predict p0
            Message<P> temp2 = model.update(beta.back(), v);
            for(auto &component: temp2.components){
              component.log_c += model.log_p0;
            }
            // Update
            temp2.add_component(Obs::transformObs(v), c);
            beta.push_back(temp2);
          }
          beta.back().prune(max_components);
        }
        // We calculated Beta backwards, we need to reverse the list.
        std::reverse(beta.begin(), beta.end());
      }

      // Returns mean and cpp
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

      // Returns mean and cpp
      std::pair<Matrix, Vector> smoothing(const Matrix& obs) {

        // Run Forward - Backward
        forward(obs);
        backward(obs);

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
        Matrix mean;
        Vector cpp;
        for(size_t i=0; i < obs.ncols(); ++i) {
          Message<P> gamma = alpha[i] * beta[i];
          std::cout << gamma.log_likelihood() << std::endl;
          std::cout << "smoothed: \n";
          for(auto &potential: gamma.components){
            potential.print();
          }
          auto result = gamma.evaluate(beta[i].size());
          mean.appendColumn(result.first);
          cpp.append(result.second);
        }
        std::cout << "Smoothing out\n";
        return {mean, cpp};
      }

      // a.k.a. fixed-lag smoothing
      std::pair<Matrix, Vector> online_smoothing(const Matrix& obs) {
        if( lag == 0){
          return filtering(obs);
        }
        // Run Forward
        forward(obs);

        // Fixed-Lag Smooting
        Matrix mean;
        Vector cpp;
        for (size_t t=0; t<obs.ncols()-lag+1; t++) {
          backward(obs.getColumns(Range(t,t+lag)));
          Message<P> gamma = alpha[t] * beta[0];
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
          forward(obs);
          backward(obs);
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

    private:
      std::vector<Message<P>> alpha;
      std::vector<Message<P>> alpha_predict;
      std::vector<Message<P>> beta;
  };

  using DM_Model = Model<DirichletPotential, MultinomialRandom>;
  using DM_ForwardBackward = ForwardBackward<DirichletPotential, MultinomialRandom>;


*/
  using PG_ForwardBackward = ForwardBackward<GammaPotential>;

} // namespace

#endif //MATLIB_PML_BCPM_H
