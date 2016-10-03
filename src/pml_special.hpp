#ifndef PML_SPECIAL_H_
#define PML_SPECIAL_H_

#include "pml_vector.hpp"
#include "pml_matrix.hpp"

namespace pml{

  // ------- Log Gamma function -------
  double lgamma(double x){
    return std::lgamma(x);
  }

  inline Vector lgamma(const Vector &x){
    return apply(x, std::lgamma);
  }

  inline Matrix lgamma(const Matrix &m){
    return apply(m, std::lgamma);
  }

  // -------  Polygamma Function -------
  inline double psi(double d, int n = 0){
    return gsl_sf_psi_n(n, d);
  }

  inline Vector psi(const Vector &x, int n = 0){
    Vector y = x;
    for(auto &d : y)
      d = gsl_sf_psi_n(n, d);
    return y;
  }

  inline Matrix psi(const Matrix &x, int n = 0){
    Matrix y = x;
    for(auto &d : y)
      d = gsl_sf_psi_n(n, d);
    return y;
  }

  // Inverse Polygamma with Newton Method
  double inv_psi(double d){
    double x = d > -2.22 ? std::exp(d)+0.5 : -1/(d + 0.577215);
    // make 5 newton iterations
    for(int i=0; i < 5; ++i)
      x -= (psi(x)-d) / psi(x,1);
    return x;
  }


  inline Vector inv_psi(const Vector &v){
    return apply(v, inv_psi);
  }

  inline Matrix inv_psi(const Matrix  &m){
    return apply(m, inv_psi);
  }

} // namespace pml


#endif