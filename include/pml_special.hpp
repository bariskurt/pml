#ifndef PML_SPECIAL_H_
#define PML_SPECIAL_H_

#include "pml_vector.hpp"
#include "pml_matrix.hpp"

namespace pml{

  // ------- Log Gamma function -------
  double gammaln(double x){
    return std::lgamma(x);
  }

  inline Vector gammaln(const Vector &x){
    //return apply(x, std::lgamma);
    return apply(x, [](double d) {return std::lgamma(d);} );
  }

  inline Matrix gammaln(const Matrix &m){
    return apply(m, [](double d) { return std::lgamma(d);} );
  }

  // -------  Polygamma Function -------
  inline double psi(double d, int n = 0){
    if( d == 0){
      std::cout << "n is zero!!\n";
    }

    return gsl_sf_psi_n(n, d);
  }

  inline Vector psi(const Vector &x, int n = 0){
    return apply(x, [n](double d) {return gsl_sf_psi_n(n, d);});
  }

  inline Matrix psi(const Matrix &x, int n = 0){
    return apply(x, [n](double d) {return gsl_sf_psi_n(n, d);} );
  }

  // Inverse Polygamma Function with Newton Method
  double inv_psi(double d){
    double x = d > -2.22 ? std::exp(d)+0.5 : -1/(d + 0.577215);
    // make 5 newton iterations
    for(int i=0; i < 5; ++i)
      x -= (psi(x)-d) / psi(x,1);
    return x;
  }

  inline Vector inv_psi(const Vector &v){
    return apply(v, [](double d) { return inv_psi(d);} );
  }

  inline Matrix inv_psi(const Matrix  &m){
    return apply(m, [](double d) { return inv_psi(d);} );
  }

  // KL Divergence Between Vectors
  inline double kl_div(const Vector &x, const Vector &y) {
    ASSERT_TRUE(x.size() == y.size(), "kl_div:: Size mismatch.");
    double result = 0;
    for (size_t i = 0; i < x.size(); ++i) {
      if(x(i) > 0 && y(i) > 0){
        result += x(i) * (std::log(x(i)) - std::log(y(i))) - x(i) + y(i);
      } else if(x(i) == 0 && y(i) >= 0){
        result += y(i);
      } else {
        result += std::numeric_limits<double>::infinity();
        break;
      }
    }
    return result;
  }

  double kl_div(const Matrix &x, const Matrix &y){
    return kl_div(flatten(x), flatten(y));
  }

} // namespace pml


#endif