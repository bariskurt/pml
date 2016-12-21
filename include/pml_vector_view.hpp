#ifndef PML_VECTOR_VIEW_H_
#define PML_VECTOR_VIEW_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "pml_vector.hpp"


namespace pml {

  template <bool is_const>
  class GenericVectorView{

      typedef typename std::conditional<is_const,
              const Vector, Vector>::type ViewType;

      typedef typename std::conditional<is_const,
              const double*, double*>::type PointerType;

      typedef typename std::conditional<is_const,
              const double&, double&>::type ReferenceType;


    public:
      class iterator{
        public:
          iterator(PointerType data, size_t stride)
                  : data_(data), stride_(stride) {}

          bool operator== (const iterator& other) const {
            return data_ == other.data_;
          }

          bool operator!= (const iterator& other) const {
            return !(*this == other);
          }

          ReferenceType operator*() {
            return *data_;
          }

          // prefix: ++it
          iterator& operator++(){
            data_ += stride_;
            return *this;
          }

          // postfix: it++
          iterator operator++(int){
            const iterator old(*this);
            data_ += stride_;
            return old;
          }

          // prefix: --it
          iterator& operator--(){
            data_ -= stride_;
            return *this;
          }

          // postfix: it--
          iterator operator--(int){
            const iterator old(*this);
            data_ -= stride_;
            return old;
          }

        private:
          PointerType data_;
          size_t stride_;
      };

    public:
      GenericVectorView(PointerType data, size_t size, size_t stride = 1)
              : data_(data), size_(size), stride_(stride){}

      GenericVectorView(ViewType &v)
              : data_(v.data()), size_(v.size()), stride_(1){ }

      GenericVectorView(const GenericVectorView<false> &vw)
              : data_(vw.data_), size_(vw.size_), stride_(vw.stride_){ }

      iterator begin(){
        return iterator(data_, stride_);
      }

      iterator end(){
        return iterator(data_ + (stride_ * size_), stride_);
      }

      size_t size() const{
        return size_;
      }

      bool empty() const{
        return size_ == 0;
      }

      Vector copy() {
        // stride 1 is just memcpy
        if(stride_ == 1)
          return Vector(size_, data_);
        // stride > 1 needs individual copies
        Vector v(size_);
        size_t i = 0;
        for(auto it = begin(); it != end(); ++it)
          v[i++] = *it;
        return v;
      }

    public:

      void normalize() {
        double sum_x = std::accumulate(begin(), end(), 1, std::plus<double>());
      }
/*
      void normalizeExp(){
        double x_max = max(*this);
        for (size_t i = 0; i < size(); ++i)
          data_[i] = std::exp(data_[i] - x_max);
        normalize();
      }
*/

    private:
      PointerType data_;
      size_t size_;
      size_t stride_;
  };

  typedef GenericVectorView<false> VectorView;
  typedef GenericVectorView<true>  ConstVectorView;



  // Apply function func to each element of view and return as new Vector
  inline Vector apply(ConstVectorView cvw, double (*func)(double)){
    Vector result(cvw.size());
    size_t i = 0;
    for(const double d : cvw)
      result[i++] = func(d);
    return result;
  }

  inline void operator+=(VectorView vw, const double value){
    for(double &d : vw)
      d += value;
  }

  inline void operator-=(VectorView vw, const double value){
    for(double &d : vw)
      d -= value;
  }

  inline void operator*=(VectorView vw, const double value){
    for(double &d : vw)
      d *= value;
  }

  inline void operator/=(VectorView vw, const double value){
    for(double &d : vw)
      d /= value;
  }

  inline void operator+=(VectorView vw, ConstVectorView cvw){
    ASSERT_TRUE(vw.size() == cvw.size(),
                "VectorView::operator+: size mismatch");
    ConstVectorView::iterator cit = cvw.begin();
    for(VectorView::iterator it = vw.begin(); it != vw.end(); ++it, ++cit)
      *it += *cit;
  }

  inline void operator-=(VectorView vw, ConstVectorView cvw){
    ASSERT_TRUE(vw.size() == cvw.size(),
                "VectorView::operator-: size mismatch");
    ConstVectorView::iterator cit = cvw.begin();
    for(VectorView::iterator it = vw.begin(); it != vw.end(); ++it, ++cit)
      *it -= *cit;
  }

  inline void operator*=(VectorView vw, ConstVectorView cvw){
    ASSERT_TRUE(vw.size() == cvw.size(),
                "VectorView::operator*: size mismatch");
    ConstVectorView::iterator cit = cvw.begin();
    for(VectorView::iterator it = vw.begin(); it != vw.end(); ++it, ++cit)
      *it *= *cit;
  }

  inline void operator/=(VectorView vw, ConstVectorView cvw){
    ASSERT_TRUE(vw.size() == cvw.size(),
                "VectorView::operator/: size mismatch");
    ConstVectorView::iterator cit = cvw.begin();
    for(VectorView::iterator it = vw.begin(); it != vw.end(); ++it, ++cit)
      *it /= *cit;
  }

  // Returns A + b
  inline Vector operator+(ConstVectorView cvw, double value) {
      Vector result(cvw.size());
      size_t i = 0;
      for(const double d : cvw)
        result[i++] = d + value;
      return result;
    }

  // Returns b + A
  inline Vector operator+(const double value, ConstVectorView cvw) {
    return cvw + value;
  }

  // Returns A * b
  inline Vector operator*(ConstVectorView cvw, const double value) {
    Vector result(cvw.size());
    size_t i = 0;
    for(const double d : cvw)
      result[i++] = d * value;
    return result;
  }

  // Returns b + A
  inline Vector operator*(const double value, ConstVectorView cvw) {
    return cvw * value;
  }

  // Returns A - b
  inline Vector operator-(ConstVectorView cvw, const double value) {
    Vector result(cvw.size());
    size_t i = 0;
    for(const double d : cvw)
      result[i++] = d - value;
    return result;
  }

  // Returns b - A
  inline Vector operator-(const double value, ConstVectorView cvw) {
    Vector result(cvw.size());
    size_t i = 0;
    for(const double d : cvw)
      result[i++] = value - d;
    return result;
  }

  // Returns A / b
  inline Vector operator/(ConstVectorView cvw, const double value) {
    Vector result(cvw.size());
    size_t i = 0;
    for(const double d : cvw)
      result[i++] = d / value;
    return result;
  }

  // Returns b / A
  inline Vector operator/(const double value, ConstVectorView cvw) {
    Vector result(cvw.size());
    size_t i = 0;
    for(const double d : cvw)
      result[i++] = value / d;
    return result;
  }


  inline Vector operator+(ConstVectorView cvw1, ConstVectorView cvw2) {
    ASSERT_TRUE(cvw1.size() == cvw2.size(),
                "ConstVectorView::operator+: size mismatch");
    Vector result(cvw1.size());
    ConstVectorView::iterator it1 = cvw1.begin();
    ConstVectorView::iterator it2 = cvw2.begin();
    for(size_t i = 0; i < cvw1.size(); ++i, ++it1, ++it2)
      result[i] = *it1 + *it2;
    return result;
  }

  inline Vector operator-(ConstVectorView cvw1, ConstVectorView cvw2) {
    ASSERT_TRUE(cvw1.size() == cvw2.size(),
                "ConstVectorView::operator-: size mismatch");
    Vector result(cvw1.size());
    ConstVectorView::iterator it1 = cvw1.begin();
    ConstVectorView::iterator it2 = cvw2.begin();
    for(size_t i = 0; i < cvw1.size(); ++i, ++it1, ++it2)
      result[i] = *it1 - *it2;
    return result;
  }

  inline Vector operator*(ConstVectorView cvw1, ConstVectorView cvw2) {
    ASSERT_TRUE(cvw1.size() == cvw2.size(),
                "ConstVectorView::operator*: size mismatch");
    Vector result(cvw1.size());
    ConstVectorView::iterator it1 = cvw1.begin();
    ConstVectorView::iterator it2 = cvw2.begin();
    for(size_t i = 0; i < cvw1.size(); ++i, ++it1, ++it2)
      result[i] = *it1 * *it2;
    return result;
  }

  inline Vector operator/(ConstVectorView cvw1, ConstVectorView cvw2) {
    ASSERT_TRUE(cvw1.size() == cvw2.size(),
                "ConstVectorView::operator/: size mismatch");
    Vector result(cvw1.size());
    ConstVectorView::iterator it1 = cvw1.begin();
    ConstVectorView::iterator it2 = cvw2.begin();
    for(size_t i = 0; i < cvw1.size(); ++i, ++it1, ++it2)
      result[i] = *it1 / *it2;
    return result;
  }

  double sum(ConstVectorView cvw){
    double result = 0;
    for(const double d : cvw)
      result += d;
    return result;
  }

  double mean(ConstVectorView cvw){
    return sum(cvw) / cvw.size();
  }

  double var(ConstVectorView cvw){
    double cvw_mean = mean(cvw);
    double result = 0;
    for(const double d : cvw)
      result += std::pow(d - cvw_mean, 2);
    return result / (cvw.size() - 1);
  }

  inline double min(ConstVectorView cvw) {
    ASSERT_TRUE(!cvw.empty(), "ConstVectorView::min()::error: Block empty");
    double cvw_min = std::numeric_limits<double>::max();
    for(const double d : cvw)
      cvw_min = std::min(cvw_min, d);
    return cvw_min;
  }

  inline double max(ConstVectorView cvw) {
    ASSERT_TRUE(!cvw.empty(), "ConstVectorView::min()::error: Block empty");
    double cvw_max = std::numeric_limits<double>::min();
    for(const double d : cvw)
      cvw_max = std::max(cvw_max, d);
    return cvw_max;
  }

  // Safe log(sum(exp(x)))
  inline double logSumExp(ConstVectorView cvw) {
    double result = 0;
    double cvw_max = max(cvw);
    for(const double d : cvw)
      result += std::exp(d - cvw_max);
    return cvw_max + std::log(result);
  }

  // Standard deviation
  inline double stdev(ConstVectorView cvw){
    return std::sqrt(var(cvw));
  }

  // Absolute value of x
  inline Vector abs(ConstVectorView cvw){
    return apply(cvw, std::fabs);
  }

  // Round to nearest integer
  inline Vector round(ConstVectorView cvw){
    return apply(cvw, std::round);
  }

  // Ceiling
  inline Vector ceil(ConstVectorView cvw) {
    return apply(cvw, std::ceil);
  }

  // Floor
  inline Vector floor(ConstVectorView cvw) {
    return apply(cvw, std::floor);
  }

  // Exponential
  inline Vector exp(ConstVectorView cvw) {
    return apply(cvw, std::exp);
  }

  // Logarithm
  inline Vector log(ConstVectorView cvw) {
    return apply(cvw, std::log);
  }

  // Normalize
  inline Vector normalize(ConstVectorView cvw) {
    Vector result = cvw.copy();
    result /= sum(result);
    return result;
  }

  // Safe normalize(exp(x))
  inline Vector normalizeExp(ConstVectorView cvw) {
    return normalize(exp(cvw - max(cvw)));
  }


}

#endif