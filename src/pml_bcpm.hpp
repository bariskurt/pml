#ifndef MATLIB_PML_BCPM_H
#define MATLIB_PML_BCPM_H

#include "pml.hpp"
#include "pml_rand.hpp"

#include <algorithm>

namespace pml {
  // ----------- COMPONENTS ----------- //

  class DirichletPotential{

    public:
      DirichletPotential(const Vector& alpha_, double log_c_) :
              alpha(alpha_), log_c(log_c_) {}

    public:
      void operator*=(const DirichletPotential &p){
        *this = this->operator*(p);
      }

      DirichletPotential operator*(const DirichletPotential &p){

        double delta = std::lgamma(sum(alpha)) - sum(lgamma(alpha)) +
                       std::lgamma(sum(p.alpha)) - sum(lgamma(p.alpha)) +
                       sum(lgamma(alpha + p.alpha-1)) -
                       std::lgamma(sum(alpha + p.alpha -1));

        return DirichletPotential(alpha + p.alpha - 1,
                                  log_c + p.log_c + delta);
      }

      Vector rand() const{
        return dirichlet::rand(alpha);
      }

      Vector mean() const{
        return alpha;
      }

    public:
      Vector alpha;
      double log_c;   //log of normalizing constant
  };

  class GammaPotential{
    public:
      GammaPotential(double a_, double b_, double log_c_ = 0)
              : a(a_), b(b_), log_c(log_c_) {}

    public:
      void operator*=(const GammaPotential &other){
        *this = this->operator*(other);
      }

      GammaPotential operator*(const GammaPotential &other){
        double delta = std::lgamma(a + other.a - 1)
                       - std::lgamma(a) - std::lgamma(other.a)
                       + std::log(b + other.b)
                       + a * std::log(b/(b + other.b))
                       + other.a * std::log(other.b/(b + other.b));
        return GammaPotential(a + other.a - 1,
                              b + other.b,
                              log_c + other.log_c + delta);
      }

      Vector rand() const{
        return gamma::rand(a, b, 1);
      }

      Vector mean() const{
        return Vector(1, a / b);
      }

      void update(const Vector &obs){
        this->operator*=(GammaPotential(obs.first()+1, 1));
      }

    public:
      double a;
      double b;
      double log_c;   //log of normalizing constant
  };
  // ----------- MESSAGES----------- //

  template <class P>
  class Message {

    public:
      size_t size() {
        return components.size();
      }

      void add_component(const P &potential){
        components.push_back(potential);
      }

      void add_component(const P &potential, double log_c){
        components.push_back(potential);
        components.back().log_c = log_c;
      }

      Message<P> operator*(const Message<P> m1, const Message<P> m2){
        Message<P> msg;
        // no change particles
        for (size_t i=0; i < m1.size()-1; ++i) {
          for (size_t j=0; j < m2.size(); ++j) {
            msg.add_component(m1.components[i] * m2.components[j]);
          }
        }

        // change particles
        for (size_t j=0; j < m2.size(); ++j) {
          msg.add_component(m1.components.back() * m2.components[j]);
        }

        Vector noChangeNormConstant, changeNormConstant;
        Message* smoothed_msg = new DirichletMessage();
        // no change particles
        for (size_t i=0; i<forward->components.size()-1; i++) {
          DirichletComponent* fc = static_cast<DirichletComponent*>(forward->components[i]);
          for (size_t j=0; j<backward->components.size(); j++) {
            DirichletComponent* bc = static_cast<DirichletComponent*>(backward->components[j]);
            DirichletComponent* newComp = DirichletComponent::multiply(fc, bc);
            noChangeNormConstant.append(newComp->log_c);
            smoothed_msg->components.push_back(newComp);
          }
        }
        // change particles
        DirichletComponent* last = static_cast<DirichletComponent*>(forward->components.back());
        for (size_t i=0; i<backward->components.size(); i++) {
          DirichletComponent *bc = static_cast<DirichletComponent *>(backward->components[i]);
          DirichletComponent* newComp = DirichletComponent::multiply(last, bc);
          changeNormConstant.append(newComp->log_c);
          smoothed_msg->components.push_back(newComp);
        }
        Vector mean = eval_mean_cpp(smoothed_msg).first;
        double logPNoChange = logSumExp(noChangeNormConstant);
        double logPChange = logSumExp(changeNormConstant);
        Vector tmp = normalizeExp(Vector{logPChange, logPNoChange});
        return std::make_tuple(smoothed_msg,mean,tmp(0));
      }

    public:
      std::vector<P> components;
  };


  struct PoissonRandom{
    Vector operator()(const Vector &param) const {
      return poisson::rand(param.first(), 1);
    }
  };

  // ----------- MODELS----------- //
  template <class P, class Obs>
  class Model{
    public:
      Model(const P &prior_, double p1_) : prior(prior_), p1(p1_){
        log_p1 = std::log(p1);
        log_p0 = std::log(1-p1);
      }

    public:
      // returns  "states" as first matrix and "obervations" as second
      std::pair<Matrix, Matrix> generateData(size_t T){
        Matrix states, obs;

        Vector state = prior.rand();
        for (size_t t=0; t<T; t++) {
          if (t > 0 && uniform::rand() < p1) {
            state = prior.rand();
          }
          states.appendColumn(state);
          obs.appendColumn(observation(state));
        }
        return {states, obs};
      }

      Message<P> initForward(){
        Message<P> first_message;
        first_message.add_component(prior, log_p0);
        first_message.add_component(prior, log_p1);
        return first_message;
      }

      Message<P> initBackward(){
        Message<P> first_message;
        first_message.add_component(prior);
        return first_message;
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

      Message<P> update(const Message<P> &prev, const Vector& obs) {
        Message<P> message = prev;
        for(auto &component : message.components){
          component.update(obs);
        }
        return message;
      }

      std::pair<Vector, double> eval_mean_cpp(const Message<P> &message){
        Vector consts;
        Matrix params;
        for(auto &potential: message.components){
          consts.append(potential.log_c);
          params.appendColumn(potential.mean());
        }
        consts = normalizeExp(consts);
        Vector mean = sum(transpose(transpose(params)*consts), 1);
        return {mean, consts.last()};
      }

/*
      virtual std::tuple<Message*,Vector,double> multiply(const Message* forward, const Message* backward) = 0;
*/
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
      void oneStepForward(const Vector& obs) {
        // predict step
        if (alpha_predict.size() == 0) {
          alpha_predict.push_back(model.initForward());
        }
        else {
          // calculate \alpha_{t+1,t}
          alpha_predict.push_back(model.predict(alpha_update.back()));
        }
        // update step, calculate \alpha_{t+1,t+1}
        alpha_update.push_back(model.update(alpha_predict.back(), obs));
      }

      void oneStepBackward(const Vector& obs) {
        if (beta_postdict.size()==0) {
          beta_postdict.push_back(model.initBackward());
        }
        else {
          beta_postdict.push_back(model.predict(beta_update.back()));
        }
        beta_update.push_back(model.update(beta_postdict.back(), obs));
      }

      void forward(const Matrix& obs){
        for (size_t i=0; i<obs.ncols(); i++) {
          oneStepForward(obs.getColumn(i));
        }
      }

      void backward(const Matrix& obs){
        for (size_t i=obs.ncols(); i>0; i--) {
          oneStepBackward(obs.getColumn(i-1));
        }
      }

      std::pair<Matrix, Vector> mean_and_cpp(
              const std::vector<Message<P>> &messages){
        Matrix mean;
        Vector cpp;
        for(auto &message : messages){
          auto result = model.eval_mean_cpp(message);
          mean.appendColumn(result.first);
          cpp.append(result.second);
        }
        return {mean, cpp};
      }

      // Returns mean and cpp
      std::pair<Matrix, Vector> filter(const Matrix& obs) {
        forward(obs);
        return mean_and_cpp(alpha_update);
      }

      // Returns mean and cpp
      std::pair<Matrix, Vector> smooth(const Matrix& obs) {
        forward(obs);
        backward(obs);
        // multiply
        return mean_and_cpp(beta_update);
      }

    public:
      Model<P, Obs> model;
      int lag;
      int max_components;

      // messages
      std::vector<Message<P>> alpha_predict;       // alpha_{t|t-1}
      std::vector<Message<P>> alpha_update;        // alpha_{t|t}
      std::vector<Message<P>> beta_postdict;       // beta_{t|t+1}
      std::vector<Message<P>> beta_update;         // beta_{t|t}
  };
    /*

  class DirichletModel : public Model{
    public:
      DirichletModel(double _p1_, const Vector &_alpha_)
            : Model(_p1_), alpha(_alpha_){}
  
    public:
      std::pair<Matrix, Vector> generateData(size_t T) override{
        Vector cps = Vector::zeros(T);
        Matrix obs;
        Vector theta = dirichlet::rand(alpha);
        for (size_t t=0; t<T; ++t) {
          if (t > 0 && uniform::rand() < p1) {
            cps(t) = 1;
            theta = dirichlet::rand(alpha);
          }
          obs.appendColumn(multinomial::rand(theta,uniform::randi(40,60)));
        }
        return {obs, cps};
      }

      Message* initForward() override{
        Message* msg = new DirichletMessage();
        // Append no change component.
        msg->add_component(new DirichletComponent(alpha, log_p0));
        // Append change component.
        msg->add_component(new DirichletComponent(alpha, log_p1));
        return msg;
      }

      Message* initBackward() override{
        return new DirichletMessage(Vector::ones(alpha.size()));
      }

      Message* predict(const Message* message) override{
        DirichletMessage* dm = copyMessage(message);
        Vector consts(dm->size());
        for (size_t i=0; i<dm->size(); i++) {
          consts(i) = dm->components[i]->log_c;
          dm->components[i]->log_c += log_p0;
        }
        dm->add_component(new DirichletComponent(alpha,
                                                 log_p1 + logSumExp(consts)));
        return dm;
      };


      Message* update(const Message* message, const Vector& data) override{
        DirichletMessage* dm = copyMessage(message);
        for(Component *c : dm->components){
          DirichletComponent* d = static_cast<DirichletComponent*>(c);
          d->log_c += DirichletComponent::dirMultNormConsant(data, d->alpha);
          d->alpha += data;
        }
        return dm;
      }

      std::pair<Vector,double> eval_mean_cpp(const Message* message) override{
        Vector consts;
        Matrix norm_params;
        for(Component *c : message->components) {
          DirichletComponent *d = static_cast<DirichletComponent *>(c);
          consts.append(d->log_c);
          norm_params.appendColumn(normalize(d->alpha));
        }
        Vector exp_consts = exp(consts - max(consts));
        Vector norm_const = normalize(exp_consts);
        Vector mean = sum(transpose(transpose(norm_params)*norm_const), 1);
        return std::make_pair(mean, norm_const.last());
      }

      std::tuple<Message*,Vector,double> multiply(const Message* forward,
                                                  const Message* backward) override {
        Vector noChangeNormConstant, changeNormConstant;
        Message* smoothed_msg = new DirichletMessage();
        // no change particles
        for (size_t i=0; i<forward->components.size()-1; i++) {
          DirichletComponent* fc = static_cast<DirichletComponent*>(forward->components[i]);
          for (size_t j=0; j<backward->components.size(); j++) {
            DirichletComponent* bc = static_cast<DirichletComponent*>(backward->components[j]);
            DirichletComponent* newComp = DirichletComponent::multiply(fc, bc);
            noChangeNormConstant.append(newComp->log_c);
            smoothed_msg->components.push_back(newComp);
          }
        }
        // change particles
        DirichletComponent* last = static_cast<DirichletComponent*>(forward->components.back());
        for (size_t i=0; i<backward->components.size(); i++) {
          DirichletComponent *bc = static_cast<DirichletComponent *>(backward->components[i]);
          DirichletComponent* newComp = DirichletComponent::multiply(last, bc);
          changeNormConstant.append(newComp->log_c);
          smoothed_msg->components.push_back(newComp);
        }
        Vector mean = eval_mean_cpp(smoothed_msg).first;
        double logPNoChange = logSumExp(noChangeNormConstant);
        double logPChange = logSumExp(changeNormConstant);
        Vector tmp = normalizeExp(Vector{logPChange, logPNoChange});
        return std::make_tuple(smoothed_msg,mean,tmp(0));
      };

    private:
      DirichletMessage* copyMessage(const Message* message) {
        DirichletMessage* dm = new DirichletMessage();
        for(Component *c : message->components){
          DirichletComponent* d = static_cast<DirichletComponent*>(c);
          dm->add_component(new DirichletComponent(d->alpha,d->log_c));
        }
        return dm;
      }
  
    public:
      Vector alpha;
  };

  class GammaModel : public Model{
    public:
      GammaModel(double _p1_, double a_, double b_)
            : Model(_p1_), a(a_), b(b_){}

    public:
      std::pair<Matrix, Vector> generateData(size_t T) override{
          Vector cps = Vector::zeros(T);
          Matrix obs;
          double theta = gamma::rand(a, b);
          for (size_t t=0; t<T; ++t) {
              if (t > 0 && uniform::rand() < p1) {
                  cps(t) = 1;
                  theta = gamma::rand(a, b);
              }
              obs.appendColumn(poisson::rand(theta, 1));
          }
          return {obs, cps};
      }

      Message* initForward() override{
          Message* msg = new GammaMessage();
          // Append no change component.
          msg->add_component(new GammaComponent(a, b, log_p0));
          // Append change component.
          msg->add_component(new GammaComponent(a, b, log_p1));
          return msg;
      }

      Message* initBackward() override{
          return new GammaMessage(a, b);
      }

      Message* predict(const Message* message) override{
          GammaMessage* gm = copyMessage(message);
          Vector consts(gm->size());
          for (size_t i=0; i<gm->size(); i++) {
              consts(i) = gm->components[i]->log_c;
              gm->components[i]->log_c += log_p0;
          }
          gm->add_component(new GammaComponent(a, b, log_p1 + logSumExp(consts)));
          return gm;
      };

      Message* update(const Message* message, const Vector& data) override{
          GammaMessage* gm = copyMessage(message);
          for(Component *c : gm->components){
              GammaComponent* g = static_cast<GammaComponent*>(c);
              g->log_c += GammaComponent::gamPoNormConsant(data, g->a, g->b);
              g->a += data.first();
              g->b += 1;
          }
          return gm;
      }

      std::pair<Vector,double> eval_mean_cpp(const Message* message) override{
          Vector consts;
          Vector norm_params;
          for(Component *c : message->components) {
              GammaComponent *g = static_cast<GammaComponent *>(c);
              consts.append(g->log_c);
              norm_params.append(g->a/g->b);
          }
          Vector exp_consts = exp(consts - max(consts));
          Vector norm_const = normalize(exp_consts);
          Vector mean;
          mean.append(dot(norm_const,norm_params));
          return std::make_pair(mean, norm_const.last());
      }

      std::tuple<Message*,Vector,double> multiply(const Message* forward,
                                                  const Message* backward) override {
          Vector noChangeNormConstant, changeNormConstant;
          Message* smoothed_msg = new GammaMessage();
          // no change particles
          for (size_t i=0; i<forward->components.size()-1; i++) {
              GammaComponent* fc = static_cast<GammaComponent*>(forward->components[i]);
              for (size_t j=0; j<backward->components.size(); j++) {
                  GammaComponent* bc = static_cast<GammaComponent*>(backward->components[j]);
                  GammaComponent* newComp = GammaComponent::multiply(fc, bc);
                  noChangeNormConstant.append(newComp->log_c);
                  smoothed_msg->components.push_back(newComp);
              }
          }
          // change particles
          GammaComponent* last = static_cast<GammaComponent*>(forward->components.back());
          for (size_t i=0; i<backward->components.size(); i++) {
              GammaComponent *bc = static_cast<GammaComponent *>(backward->components[i]);
              GammaComponent* newComp = GammaComponent::multiply(last, bc);
              changeNormConstant.append(newComp->log_c);
              smoothed_msg->components.push_back(newComp);
          }
          Vector mean = eval_mean_cpp(smoothed_msg).first;
          double logPNoChange = logSumExp(noChangeNormConstant);
          double logPChange = logSumExp(changeNormConstant);
          Vector tmp = normalizeExp(Vector{logPChange, logPNoChange});
          return std::make_tuple(smoothed_msg,mean,tmp(0));
      };

    private:
      GammaMessage* copyMessage(const Message* message) {
          GammaMessage* gm = new GammaMessage();
          for(Component *c : message->components){
              GammaComponent* g = static_cast<GammaComponent*>(c);
              gm->add_component(new GammaComponent(g->a,g->b,g->log_c));
          }
          return gm;
      }
  
    public:
      double a;  // shape parameter
      double b;  // rate parameter
  };


  // ----------- FORWARD-BACKWARD ----------- //

  class ForwardBackward{
    public:
      ForwardBackward(Model *model_, int lag_=0, int max_components_=100):
              model(model_), lag(lag_), max_components(max_components_) {}

      ~ForwardBackward() {
        freeVec(alpha_predict);
        freeVec(alpha_update);
        freeVec(beta_postdict);
        freeVec(beta_update);
        freeVec(smoothed_msgs);
      }

    // forward-backward in general
    public:
      void oneStepForward(const Vector& obs) {
        // predict step
        if (alpha_predict.size() == 0) {
          // start forward recursion
          alpha_predict.push_back(model->initForward());
        }
        else {
          // calculate \alpha_{t+1,t}
          alpha_predict.push_back(model->predict(alpha_update.back()));
        }
        // update step, calculate \alpha_{t+1,t+1}
        alpha_update.push_back(model->update(alpha_predict.back(), obs));
      }

      void oneStepBackward(const Vector& obs) {
        if (beta_postdict.size()==0) {
          beta_postdict.push_back(model->initBackward());
        }
        else {
          beta_postdict.push_back(model->predict(beta_update.back()));
        }
        beta_update.push_back(model->update(beta_postdict.back(), obs));
      }

      void forwardRecursion(const Matrix& data) {
        for (size_t i=0; i<data.ncols(); i++) {
          oneStepForward(data.getColumn(i));
          if ((int) alpha_update.back()->components.size() > max_components) {
            prun(alpha_update.back());
          }
        }
      }

      void backwardRecursion(const Matrix& data) {
        for (size_t i=data.ncols(); i>0; i--) {
          oneStepBackward(data.getColumn(i-1));
          if ((int) beta_update.back()->components.size() > max_components) {
            prun(beta_update.back());
          }
        }
      }

      static double loglhood(Model* model, const Matrix& data) {
        ForwardBackward fb(model);
        fb.forwardRecursion(data);
        Vector consts;
        for(Component* component: fb.alpha_update.back()->components){
          consts.append(component->log_c);
        }
        return logSumExp(consts);
      }

      void smoothing(const Matrix &data) {
        // run recursions
        forwardRecursion(data);
        backwardRecursion(data);
        size_t T = data.ncols();
        // t_0, t_1, ... t_{T-2}
        for (size_t t=0; t<T-1; t++) {
          auto res = model->multiply(alpha_update[t],beta_postdict[T-t-1]);
          smoothed_msgs.push_back(std::get<0>(res));
          mean.appendColumn(std::get<1>(res));
          cpp.append(std::get<2>(res));
        }
        // t_{T-1}
        std::pair<Vector,double> mean_cpp = model->eval_mean_cpp(alpha_update.back());
        mean.appendColumn(mean_cpp.first);
        cpp.append(mean_cpp.second);
        Message* msg = new Message();

        if(dynamic_cast<DirichletModel*>(model) != 0x0){
          for(Component* comp : alpha_update.back()->components) {
            DirichletComponent* d = static_cast<DirichletComponent*>(comp);
            msg->add_component(new DirichletComponent(d->alpha, d->log_c));
          }
        }else if(dynamic_cast<GammaModel*>(model) != 0x0){
          for(Component* comp : alpha_update.back()->components) {
            GammaComponent* d = static_cast<GammaComponent*>(comp);
            msg->add_component(new GammaComponent(d->a, d->b, d->log_c));
        }
        }else{
            std::cout << "Given Model Type is not known." << std::endl;
        }
        smoothed_msgs.push_back(msg);
      }

    private:
      void freeVec(std::vector<Message*> &vec) {
        for(Message* m : vec) {
          delete m;
        }
        vec.clear();
      }

    // netas-related code
    public:
      size_t getTime() { return data.ncols(); }

      void processObs(const Vector& obs) {
        // predict & update
        oneStepForward(obs);
        // parameter updates
        data.appendColumn(obs);
        std::pair<Vector, double> mean_cpp = model->eval_mean_cpp(alpha_update.back());
        mean.appendColumn(mean_cpp.first);
        cpp.append(mean_cpp.second);
        // smoothing
        if (lag>0 && ((int)getTime()) >= lag) {
          fixedLagSmooth();
        }
        // pruning
        if ((int) alpha_update.back()->components.size() > max_components) {
          prun(alpha_update.back());
        }
      }

      // needs better implementation via heap
      void prun(Message* msg) {
        while (msg->components.size() > (unsigned) max_components) {
          std::vector<Component*> &comps = msg->components;
          double min_c = comps[0]->log_c;
          int min_id = 0;
          for (size_t i=1; i<comps.size(); i++) {
            if (comps[i]->log_c < min_c) {
              min_c = comps[i]->log_c;
              min_id = i;
            }
          }
          comps.erase(comps.begin() + min_id);
        }
      }

      void fixedLagSmooth() {
        size_t T = getTime();
        std::vector<Message*> _beta_postdict;
        std::vector<Message*> _beta_update;
        // lag step backward
        // TODO: implement below loop via ForwardBackward class and oneStepBackward()
        for (int t=0; t<lag; t++){
          if (t==0) {
            _beta_postdict.push_back(model->initBackward());
          }
          else {
            _beta_postdict.push_back(model->predict(_beta_update.back()));
          }
          _beta_update.push_back(model->update(_beta_postdict.back(), data.getColumn(T-1-t)));
        }
        // CHECK HERE AGAIN: posterior is calculated for t = T - Lag
        size_t t = T - lag;
        auto res = model->multiply(alpha_update[t], _beta_postdict.back());
        delete std::get<0>(res);
        mean.setColumn(t, std::get<1>(res));
        cpp(t) = std::get<2>(res);
        freeVec(_beta_postdict);
        freeVec(_beta_update);
      }

    public:
      Model *model;
      int lag;
      int max_components;
      // messages
      std::vector<Message*> alpha_predict;       // alpha_{t|t-1}
      std::vector<Message*> alpha_update;        // alpha_{t|t}
      std::vector<Message*> beta_postdict;       // beta_{t|t+1}
      std::vector<Message*> beta_update;         // beta_{t|t}
      // results
      Vector cpp;                                // p(r_t=1)
      Matrix mean;                               // for saving results
      Matrix data;                               // for saving results
      std::vector<Message*> smoothed_msgs;       // alpha_{t|t}*beta_{t|t+1}
    };
*/
}

#endif //MATLIB_PML_BCPM_H
