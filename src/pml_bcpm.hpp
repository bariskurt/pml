#ifndef MATLIB_PML_BCPM_H
#define MATLIB_PML_BCPM_H

#include "pml.hpp"
#include "pml_rand.hpp"

namespace pml {

  // ----------- COMPONENTS ----------- //

    class Component {
    public:
      Component(double _log_c_) : log_c(_log_c_) {}
      virtual ~Component() {}
    public:
      double log_c;
    };

  class DirichletComponent : public Component {

    public:
      DirichletComponent(const Vector& _alpha_, double _log_c_) :
          Component(_log_c_), alpha(_alpha_) {}

      ~DirichletComponent() {}

    public:
      static DirichletComponent* multiply(const DirichletComponent* comp1,
                                          const DirichletComponent* comp2) {
        double c = comp1->log_c  + comp2->log_c +
                    dirDirNormConsant(comp1->alpha, comp2->alpha);
        Vector alpha = comp1->alpha + comp2->alpha - 1;
        return new DirichletComponent(alpha,c);
      }

      static double dirMultNormConsant(const Vector &data,
                                       const Vector &params) {
        return  std::lgamma(sum(data) + 1) - sum(lgamma(data+1)) +
                std::lgamma(sum(params)) - sum(lgamma(params)) +
                sum(lgamma(data + params)) - std::lgamma(sum(data + params));
      }

      static double dirDirNormConsant(const Vector &params1,
                                      const Vector &params2){
        return  std::lgamma(sum(params1)) - sum(lgamma(params1)) +
                std::lgamma(sum(params2)) - sum(lgamma(params2)) +
                sum(lgamma(params1 + params2 -1)) -
                std::lgamma(sum(params1 + params2 -1));
      }

    public:
      Vector alpha;
  };

  class GammaComponent : public Component {
    public:
      GammaComponent(double _a_, double _b_, double _log_c_)
          : Component(_log_c_), a(_a_), b(_b_) {}
      ~GammaComponent() {}

    public:
      double a;
      double b;
  };

  // ----------- MESSAGES----------- //

  class Message {
    public:
      virtual ~Message() {
        for (Component* c : components) delete c;
      }

      size_t size() {
        return components.size();
      }

      void add_component(Component* component){
        components.push_back(component);
      }

    public:
      std::vector<Component*> components;
  };

  class DirichletMessage : public Message {
    public:
      DirichletMessage() {}

      DirichletMessage(const Vector &param) {
        add_component(new DirichletComponent(param, 0));
      }
  };

  class GammaMessage : public Message {
    public:
      GammaMessage(double a, double b) {
        add_component(new GammaComponent(a,b,0));
      }
  };

  // ----------- MODELS----------- //

  class Model{
    public:
      Model(double _p1_) : p1(_p1_){
        log_p1 = std::log(p1);
        log_p0 = std::log(1-p1);
      }

      virtual ~Model(){}

    public:
      virtual std::pair<Matrix, Vector> generateData(size_t T) = 0;
      virtual Message* initForward() = 0;
      virtual Message* initBackward() = 0;
      virtual Message* predict(const Message* message) = 0;
      virtual Message* update(const Message* message, const Vector& data) = 0;
      virtual std::pair<Vector,double> eval_mean_cpp(const Message* message) = 0;
      virtual std::tuple<Message*,Vector,double> multiply(const Message* forward, const Message* backward) = 0;
  
    public:
      double p1;           //  probability of change
      double log_p1;       //  log probability of change
      double log_p0;       //  log probability of no change
  };

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
      std::pair<Matrix, Vector> generateData(size_t T) {
        return std::make_pair(uniform::rand(20,20),uniform::rand(20));
      }

      Message* initForward() {
        return new GammaMessage(a,b);
      }

      Message* initBackward() {
        return new GammaMessage(a,b);
      }

      Message* predict(const Message* message) {
        return new GammaMessage(a,b);
        // ToDo: Taha'nin ellerinden oper.
      }

      Message* update(const Message* message, const Vector& data) {
        return new GammaMessage(a,b);
        // ToDo: Taha'nin ellerinden oper.
      }

      std::pair<Vector,double> eval_mean_cpp(const Message* message) {
        return std::make_pair(Vector::ones(10),10);
        // ToDo: Taha'nin ellerinden oper.
      }

      std::tuple<Message*, Vector,double> multiply(const Message* forward, const Message* backward) {
        return std::make_tuple(new GammaMessage(1,2), Vector::ones(10),10);
        // ToDo: Taha'nin ellerinden oper.
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
        for (size_t i=data.ncols()-1; i>=0; i--) {
          oneStepBackward(data.getColumn(i));
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
        for(Component* comp : alpha_update.back()->components) {
          DirichletComponent* d = static_cast<DirichletComponent*>(comp);
          msg->add_component(new DirichletComponent(d->alpha, d->log_c));
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
}

#endif //MATLIB_PML_BCPM_H
