#ifndef MATLIB_PML_BCPM_H
#define MATLIB_PML_BCPM_H

#include "../pml.hpp"

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

    public:
      virtual Vector rand() const = 0;
      virtual Vector mean() const = 0;
      virtual Vector get_ss() const = 0;

    public:
      double log_c;
  };

  class DirichletPotential : public Potential {

    public:
      DirichletPotential(size_t K = 0, double log_c_ = 0) : Potential(log_c_) {
        alpha = Vector::ones(K);
      }

      DirichletPotential(const Vector& alpha_, double log_c_ = 0)
          : Potential(log_c_), alpha(alpha_) {}

      static DirichletPotential rand_gen(size_t K, double precision = 1){
        Vector alpha = normalize(Uniform().rand(K)) * precision;
        return DirichletPotential(alpha);
      }

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

      DirichletPotential obs2Potential(const Vector& obs) const{
        double log_c = std::lgamma(sum(obs)+1)
                       - std::lgamma(sum(obs)+obs.size());
        return DirichletPotential(obs+1, log_c);
      }

    public:
      Vector rand() const override {
        return Dirichlet(alpha).rand();
      }

      Vector mean() const override {
        return normalize(alpha);
      }

      Vector get_ss() const override{
        return psi(alpha) - psi(sum(alpha));
      }


      void print() const{
        std::cout << alpha << " log_c:" << log_c << std::endl;
      }

      void fit(const Vector &ss, double precision = 0){
        alpha = Dirichlet::fit(ss, precision).alpha;
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
        *this = *this * other;
      }

      friend GammaPotential operator*(const GammaPotential &g1,
                                      const GammaPotential &g2) {
        double a = g1.a + g2.a - 1;
        double b = (g1.b * g2.b) / (g1.b + g2.b);
        double log_c = std::lgamma(a) + a * std::log(b)
                       - std::lgamma(g1.a) - g1.a * std::log(g1.b)
                       - std::lgamma(g2.a) - g2.a * std::log(g2.b);
        return GammaPotential(a, b, g1.log_c + g2.log_c + log_c);
      }

      GammaPotential obs2Potential(const Vector& obs) const {
        return GammaPotential(obs.first()+1, 1);
      }

    public:
      Vector rand() const override {
        return Gamma(a, b).rand(1);
      }

      Vector mean() const override {
        return Vector(1, a * b);
      }

      Vector get_ss() const override {
        return Vector({a*b, psi(a) + std::log(b)});
      }

      void print() const  {
        std::cout << "a:" << a << "  b:" << b
                  << "  log_c: " << log_c << std::endl;
      }


      void fit(const Vector &ss, double scale = 0){
        Gamma g_est = Gamma::fit(ss[0], ss[1], scale);
        a = g_est.a;
        b = g_est.b;
      }

    public:
      double a;  // shape parameter
      double b;  // scale parameter (!!! NOT THE RATE PARAMETER !!!!)
  };


  class GaussianPotential : public Potential {
    public:
      GaussianPotential(double mu_ = 0, double sigma_ = 1, double log_c = 0) :
          Potential(log_c), mu(mu_), sigma(sigma_){}

    public:
      void operator*=(const GaussianPotential &other){
        *this = *this * other;
      }

      friend GaussianPotential operator*(const GaussianPotential &g1,
                                         const GaussianPotential &g2) {
        double ss1 = std::pow(g1.sigma ,2);
        double ss2 = std::pow(g2.sigma ,2);
        double mu = (g1.mu * ss2 + g2.mu * ss1 ) / (ss1 + ss2);
        double sigma = std::sqrt(( ss1 * ss2 ) / (ss1 + ss2));
        double K = gsl_ran_gaussian_pdf(g1.mu - g2.mu , std::sqrt(ss1 + ss2));
        return GaussianPotential(mu, sigma, K + g1.log_c + g2.log_c);
      }

      GaussianPotential obs2Potential(const Vector& obs) const {
        return GaussianPotential(obs.first());
      }

    public:
      Vector rand() const override {
        return Gaussian(mu, sigma).rand(1);
      }

      Vector mean() const override {
        return Vector(1, mu);
      }

      Vector get_ss() const override{
        return Vector({mu, std::pow(sigma,2)});
      }

      void fit(const Vector &ss) {
        mu = ss(0);
        sigma = std::sqrt(ss(1));
      }

    public:
      double mu, sigma;
  };
  // ----------- MODEL ----------- //

  template <class P>
  class Model{

    public:
      explicit Model(double p1_){
        set_p1(p1_);
      }

    public:
      void set_p1(double p1_new){
        p1 = p1_new;
        log_p1 = std::log(p1);
        log_p0 = std::log(1-p1);
      }

      P obs2Potential(const Vector &obs){
        return prior.obs2Potential(obs);
      }

    public:
      std::pair<Matrix, Matrix> generateData(size_t length){
        Matrix states, obs;
        Vector state = prior.rand();
        Bernoulli bernoulli(p1);
        for (size_t t=0; t<length; t++) {
          if (t == 0 || bernoulli.rand()) {
            state = prior.rand();
          }
          states.appendColumn(state);
          obs.appendColumn(rand(state));
        }
        return {states, obs};
      }

    public:
      virtual Vector rand(const Vector &state) const = 0;
      virtual void fit(const Vector &ss, double p1_new) = 0;
      virtual void saveTxt(const std::string &filename) const = 0;
      virtual void loadTxt(const std::string &filename) = 0;
      virtual void print() const = 0;

    public:
      P prior;
      double p1, log_p1, log_p0;
  };

  class PG_Model : public Model<GammaPotential> {

    public:
      PG_Model(double a, double b, double p1_, bool fixed_scale = false)
          :Model(p1_) {
        prior = GammaPotential(a, b);
        scale = fixed_scale ? b : 0;
      }

      Vector rand(const Vector &state) const override {
        return Poisson(state.first()).rand(1);
      }

      void fit(const Vector &ss, double p1_new) override {
        prior.fit(ss, scale);
        set_p1(p1_new);
      }

      void saveTxt(const std::string &filename) const override {
        const int precision = 10;
        Vector temp;
        temp.append(p1);
        temp.append(prior.a);
        temp.append(prior.b);
        temp.append((int)(scale == 0));
        temp.saveTxt(filename, precision);
      }

      void loadTxt(const std::string &filename) override{
        Vector temp = Vector::loadTxt(filename);
        set_p1(temp(0));
        prior = GammaPotential(temp(1), temp(2));
        scale = temp(3) ? prior.b : 0;
      }

      void print() const override{
        std::cout << "PG_Model:\n";
        std::cout << "a = " << prior.a << "\tb = " << prior.b << "\tp1 = " << p1
                  << "\tfixed_scale = " << (int)(scale == 0) << std::endl;
      }

    public:
      double scale;
  };

  class DM_Model: public Model<DirichletPotential> {

    public:
      DM_Model(const Vector &alpha, double p1_,
               bool fixed_precision = false) : Model( p1_) {
        prior = DirichletPotential(alpha);
        precision = fixed_precision ? sum(alpha) : 0;
      }

      Vector rand(const Vector &state) const override {
        return Multinomial(state, 20).rand();
      }

      void fit(const Vector &ss, double p1_new) override {
        prior.fit(ss, precision);
        set_p1(p1_new);
      }

      void saveTxt(const std::string &filename) const override{
        const int txt_precision = 10;
        Vector temp;
        temp.append(p1);
        temp.append(prior.alpha);
        temp.append(precision == 0);
        temp.saveTxt(filename, txt_precision);
      }

      void loadTxt(const std::string &filename){
        Vector temp = Vector::loadTxt(filename);
        set_p1(temp(0));
        prior = DirichletPotential(temp.getSlice(1, temp.size()-1));
        precision = temp.last() ? sum(prior.alpha) : 0;
      }

      void print() const override{
        std::cout << "DM_Model: \n";
        std::cout << "\talpha = " << prior.alpha << std::endl;
        std::cout << "\tp1 = " << p1 << std::endl;
        std::cout << "\tfixed_precision = " << (int)(precision == 0) << "\n";
      }

    public:
      double precision;
  };

  class G_Model: public Model<GaussianPotential> {

    public:
      G_Model(double mu, double sigma, double p1_)
          : Model(p1_){
        prior = GaussianPotential(mu, sigma);
      }

      Vector rand(const Vector &state) const override {
        return Gaussian(state.first()).rand(1);
      }

      void fit(const Vector &ss, double p1_new) override {
        prior.fit(ss);
        set_p1(p1_new);
      }

      void saveTxt(const std::string &filename) const override {
        const int precision = 10;
        Vector temp;
        temp.append(prior.mu);
        temp.append(prior.sigma);
        temp.saveTxt(filename, precision);
      }

      void loadTxt(const std::string &filename) override{
        Vector temp = Vector::loadTxt(filename);
        prior = GaussianPotential(temp(0), temp(1));
      }

      void print() const override{
        std::cout << "G_Model: \n";
        std::cout << "\tmu = " << prior.mu << "\tsigma = " << prior.sigma
                  << "\tp1 = " << p1 << std::endl;
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

      Vector mean(){
        Matrix params;
        for(auto &potential: potentials){
          params.appendColumn(potential.mean());
        }
        Vector consts = normalizeExp(get_consts());
        return sum(transpose(transpose(params)*consts), 1);
      }

      double cpp(int num_cpp = 1){
        Vector consts = normalizeExp(get_consts());
        if(num_cpp == 1){
          return consts.last();
        }
        double result = 0;
        for(size_t i = consts.size()-num_cpp; i < consts.size(); ++i)
          result += consts[i];
        return result;
      }

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
        return logSumExp(get_consts());
      }

      Vector get_consts(){
        Vector consts(potentials.size());
        for(size_t i = 0; i < potentials.size(); ++i){
          consts[i] = potentials[i].log_c;
        }
        return consts;
      }

    public:
      std::vector<P> potentials;

  };


  template <class P>
  class ForwardBackward {

    using ModelType = Model<P>;
    using MessageType = Message<P>;

    public:
      ForwardBackward(ModelType *model_, int max_components_ = 100)
          :model(model_), max_components(max_components_){
        alpha.clear();
        alpha_predict.clear();
        beta.clear();
      }

    public:
      MessageType predict(const Message<P>& prev){
        MessageType message = prev;
        Vector consts;
        for(auto &potential : message.potentials){
          consts.append(potential.log_c);
          potential.log_c += model->log_p0;
        }
        message.add_potential(model->prior, model->log_p1 + logSumExp(consts));
        return message;
      }

      MessageType update(const MessageType &prev, const Vector &obs){
        MessageType message = prev;
        for(auto &potential : message.potentials) {
          potential *= model->obs2Potential(obs);
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
          mean.appendColumn(message.mean());
          cpp.append(message.cpp());
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
          message.add_potential(model->prior, model->log_p0);
          message.add_potential(model->prior, model->log_p1);
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
        MessageType message;
        for(size_t t = 0; t < steps; ++t, --idx){
          double c = 0;
          if(!beta.empty()){
            // Predict for case s_t = 1, calculate constant only
            MessageType temp = beta.back();
            for(auto &potential : temp.potentials){
              potential *= model->prior;
            }
            c = model->log_p1 + temp.log_likelihood();

            // Update :
            message = update(beta.back(), obs.getColumn(idx));
            for(auto &potential : message.potentials){
              potential.log_c += model->log_p0;
            }
          }
          P pot = model->obs2Potential(obs.getColumn(idx));
          pot.log_c += c;
          message.add_potential(pot);
          message.prune(max_components);
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
          MessageType gamma = alpha_predict[i] * beta[i];
          mean.appendColumn(gamma.mean());
          cpp.append(gamma.cpp(beta[i].size()));
        }
        return {mean, cpp};
      }

      std::pair<Matrix, Vector> online_smoothing(const Matrix& obs, size_t lag){
        if(lag == 0)
          return filtering(obs);

        if(lag >= obs.ncols())
          return smoothing(obs);

        MessageType gamma;
        Matrix mean;
        Vector cpp;

        // Go forward
        forward(obs);

        // Run Fixed-Lags for alpha[0:T-lag]
        for(size_t t=0; t <= obs.ncols()-lag; ++t){
          backward(obs, t+lag-1, lag);
          gamma = alpha[t] * beta.front();
          mean.appendColumn(gamma.mean());
          cpp.append(gamma.cpp(beta.front().size()));
        }

        // Smooth alpha[T-lag+1:T] with last beta.
        for(size_t i = 1; i < lag; ++i){
          gamma = alpha[obs.ncols()-lag+i] * beta[i];
          mean.appendColumn(gamma.mean());
          cpp.append(gamma.cpp(beta[i].size()));

        }

        return {mean, cpp};
      }

      Vector compute_ss(const MessageType &message) {
        Matrix tmp;
        Vector norm_consts;
        for(auto &potential : message.potentials){
          norm_consts.append(potential.log_c);
          tmp.appendColumn( potential.get_ss() );
        }
        norm_consts = normalizeExp(norm_consts);
        tmp = tmp * tile(norm_consts, tmp.nrows());
        return sum(tmp, 1);
      }

      std::pair<Matrix, Vector> learn_parameters(const Matrix& obs,
                                                 size_t max_iter = 100,
                                                 bool verbose = true){
        size_t min_iter = 20;
        Vector ll;

        for(size_t iter = 0; iter < max_iter; ++iter){
          // Forward_backward
          forward(obs);
          backward(obs);

          double cpp=0;
          double cpp_sum=0;
          Vector ss;
          Matrix E_log_pi_weighted;
          for(size_t i=0; i < obs.ncols(); ++i) {
            MessageType gamma = alpha_predict[i] * beta[i];
            cpp = gamma.cpp(beta[i].size());
            cpp_sum += cpp;
            if( i == 0){
              ss = compute_ss(gamma)*cpp;
            } else {
              ss += compute_ss(gamma)*cpp;
            }
          }
          ss /= cpp_sum;

          // Log-likelihood
          ll.append(alpha.back().log_likelihood());
          if(verbose) {
            std::cout << "iter: " << iter
                      << "\tloglikelihood : " << ll.last() << std::endl;
          }

          if(iter > 0 ){
            double ll_diff = ll[iter] - ll[iter-1];
            if( ll_diff < 0 ){
              if(verbose) {
                std::cout << "!!! loglikelihood decreased: "
                          << ll[iter - 1] - ll[iter] << std::endl;
              }
              break;
            }
            if( iter > min_iter && ( ll_diff < 1e-6)){
              if(verbose) {
                std::cout << "converged.\n";
              }
              break;
            }
          }

          // M-Step:
          model->fit(ss, cpp_sum / obs.ncols());
        }

        return smoothing(obs);
      }


    public:
      ModelType *model;
      int max_components;

    private:
      std::vector<MessageType> alpha;
      std::vector<MessageType> alpha_predict;
      std::vector<MessageType> beta;

  };

  using DM_ForwardBackward = ForwardBackward<DirichletPotential>;
  using PG_ForwardBackward = ForwardBackward<GammaPotential>;
  using G_ForwardBackward = ForwardBackward<GaussianPotential>;

} // namespace

#endif //MATLIB_PML_BCPM_H
