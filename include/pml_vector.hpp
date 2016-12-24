#ifndef PML_VECTOR_H_
#define PML_VECTOR_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_sf_psi.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "pml_block.hpp"

#define DEFAULT_PRECISION 6

namespace pml {

  // Defines a series from start to stop EXCLISIVELY with step size step:
  // [start, start+step, start+2*step, ...]
  struct Range {
      Range(size_t start_, size_t stop_, size_t step_ = 1)
              : start(start_), stop(stop_), step(step_) { }
      size_t size(){
        return std::ceil((double)(stop-start) / step);
      }
      size_t start, stop, step;
  };

  inline bool fequal(double a, double b) {
    return fabs(a - b) < 1e-6;
  }

  class ConstVectorView;
  class VectorView;

  class Vector : public Block {

    public:
      // Empty Vector
      Vector() { }

      // Vector of given length and default value.
      explicit Vector(size_t length) : Block(length) { }

      // Vector of given length and default value.
      Vector(size_t length, double value) : Block(length) {
        fill(value);
      }

      // Vector from given array
      Vector(size_t length, const double *values) : Block(length) {
        memcpy(data_, values, sizeof(double) * length);
      }

      // Vector from initializer lsit
      Vector(const std::initializer_list<double> &values) {
        for(const double d : values)
          __push_back__(d);
      }

      // Vector from range
      explicit Vector(Range range) {
        for (double d = range.start; d < range.stop; d += range.step)
          __push_back__(d);
      }

      // Vector of zeros of given length.
      static Vector zeros(size_t length) {
        return Vector(length, 0.0);
      }

      // Vector of ones of given length.
      static Vector ones(size_t length) {
        return Vector(length, 1.0);
      }

    public:
      Vector& operator=(const Vector& that){
        Block::operator=(that);
        return *this;
      }

    public:
      // Generate from views
      explicit Vector(ConstVectorView cvw);

      explicit Vector(VectorView vw);

      Vector& operator=(ConstVectorView v);

      Vector& operator=(VectorView v);


    private:

      void copyFromView(ConstVectorView cvw);

    public:
      // Vector resize. If new size is smaller, the data_ is cropped.
      // If new size is larger, garbage values are appended.
      void resize(size_t new_size) {
        __resize__(new_size);
      }

    public:

      // Append a single value. (same as push_back)
      void append(double value) {
        __push_back__(value);
      }

      // Append a Vector
      void append(const Vector &v) {
        __push_back__(v);
      }

    public:
      // ------- Accessors -------
      double first() const {
        return data_[0];
      }

      double& first() {
        return data_[0];
      }

      double last() const {
        return data_[size_-1];
      }

      double& last() {
        return data_[size_-1];
      }

      // Vector slice
      Vector getSlice(size_t start, size_t stop, size_t step = 1) const {
        Vector result;
        for(size_t i = start; i < stop; i+=step)
          result.append(data_[i]);
        return result;
      }

      // Returns the set of indices i of v, such that v[i] == 1.
      friend Vector find(const Vector &v){
        Vector result;
        for(size_t i = 0; i < v.size_; ++i)
          if( v.data_[i] == 1)
            result.append(i);
        return result;
      }

    public:

      // Load and Save
      friend std::ostream &operator<<(std::ostream &out,
                                      const Vector &x) {
        out << std::setprecision(DEFAULT_PRECISION) << std::fixed;
        for(size_t i=0; i < x.size(); ++i)
          out << x[i]<< "  ";
        return out;
      }

      friend std::istream &operator>>(std::istream &in, Vector &x) {
        for(size_t i=0; i < x.size(); ++i)
          in >> x[i];
        return in;
      }

      void save(const std::string &filename){
        std::ofstream ofs(filename, std::ios::binary | std::ios::out);
        if (ofs.is_open()) {
          double dim = 1;
          double length = size();
          ofs.write(reinterpret_cast<char*>(&dim), sizeof(double));
          ofs.write(reinterpret_cast<char*>(&length), sizeof(double));
          ofs.write(reinterpret_cast<char*>(data()), sizeof(double)*length);
          ofs.close();
        }
      }

      void saveTxt(const std::string &filename,
                   int precision = DEFAULT_PRECISION) const {
        std::ofstream ofs(filename);
        if (ofs.is_open()) {
          ofs << 1 << std::endl;      // dimension
          ofs << size() << std::endl; // size
          ofs << std::setprecision(precision) << std::fixed;
          for(size_t i=0; i < size_; ++i)
            ofs << data_[i] << std::endl;
          ofs.close();
        }
      }

      static Vector load(const std::string &filename){
        Vector result;
        std::ifstream ifs(filename, std::ios::binary | std::ios::in);
        if (ifs.is_open()) {
          double dim, size;
          ifs.read(reinterpret_cast<char*>(&dim), sizeof(double));
          ASSERT_TRUE(dim == 1, "Vector::load:: Dimension mismatch.");
          ifs.read(reinterpret_cast<char*>(&size), sizeof(double));
          result.resize(size);
          ifs.read(reinterpret_cast<char*>(result.data()), sizeof(double)*size);
          ifs.close();
        }
        return result;
      }

      static Vector loadTxt(const std::string &filename) {
        Vector result;
        std::ifstream ifs(filename);
        size_t buffer;
        if (ifs.is_open()) {
          // Read dimension
          ifs >> buffer;
          ASSERT_TRUE(buffer == 1, "Vector::LoadTxt:: Dimension mismatch.");
          ifs >> buffer;
          // Allocate memory
          result.__resize__(buffer);
          ifs >> result;
          ifs.close();
        }
        return result;
      }
  };

  class ConstVectorView {

    friend class Vector;
    friend class VectorView;

    public:
      class iterator{
        public:
          iterator(const double *data, size_t stride)
                  : data_(data), stride_(stride) {}

          bool operator==(const iterator& other) const {
            return data_ == other.data_;
          }

          bool operator!=(const iterator& other) const {
            return !(*this == other);
          }

          const double& operator*() const{
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
          const double* data_;
          size_t stride_;
      };

    public:
      // Create explicitly
      explicit ConstVectorView (const double *data, size_t size, size_t stride = 1)
              : data_(data), size_(size), stride_(stride){}

      // Create implicitly by Vector&
      ConstVectorView (const Vector &v)
              : data_(v.data()), size_(v.size()), stride_(1){}

      ConstVectorView(const ConstVectorView &that)
              : data_(that.data_), size_(that.size_), stride_(that.stride_){}

      ConstVectorView(const VectorView &that);
              //: data_(that.data_), size_(that.size_), stride_(that.stride_){}

      // Delete operator=
      ConstVectorView& operator=(const ConstVectorView& that) = delete;

      iterator begin() const {
        return iterator(data_, stride_);
      }

      iterator end() const {
        return iterator(data_ + (stride_ * size_), stride_);
      }

      size_t size() const{
        return size_;
      }

      size_t stride() const{
        return stride_;
      }

      bool empty() const{
        return size_ == 0;
      }

    public:
      friend std::ostream &operator<<(std::ostream &out, ConstVectorView cvw) {
        out << std::setprecision(DEFAULT_PRECISION) << std::fixed;
        for(auto it = cvw.begin(); it != cvw.end(); ++it)
          out << *it << "  ";
        return out;
      }

    private:
      const double *data_;
      size_t size_;
      size_t stride_;
  };

  class VectorView {

    friend class Vector;
    friend class ConstVectorView;

    public:
      class iterator{
        public:
          iterator(double *data, size_t stride)
                  : data_(data), stride_(stride) {}

          bool operator==(const iterator& other) const {
            return data_ == other.data_;
          }

          bool operator!=(const iterator& other) const {
            return !(*this == other);
          }

          double& operator*() {
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
          double* data_;
          size_t stride_;
      };

    public:
      // Create explicitly
      explicit VectorView(double *data, size_t size, size_t stride = 1)
              : data_(data), size_(size), stride_(stride){}

      // Create implicitly by Vector&
      VectorView(Vector &v)
              : data_(v.data()), size_(v.size()), stride_(1){}

      VectorView(const VectorView &that)
              : data_(that.data_), size_(that.size_), stride_(that.stride_){}

      iterator begin() const{
        return iterator(data_, stride_);
      }

      iterator end() const{
        return iterator(data_ + (stride_ * size_), stride_);
      }

      size_t size() const{
        return size_;
      }

      size_t stride() const{
        return stride_;
      }

      bool empty() const{
        return size_ == 0;
      }

      VectorView operator=(const double d){
        for(iterator it = begin(); it != end(); ++it)
          *it = d;
        return *this;
      }

      VectorView operator=(const ConstVectorView &cvw){
        ASSERT_TRUE(size_ == cvw.size_, "VectorView::operator= size mismatch");
        if(stride_ == 1 && cvw.stride_ == 1){
          memcpy(data_, cvw.data_, sizeof(double)*size_);
        } else {
          ConstVectorView::iterator src = cvw.begin();
          for (iterator dst = begin(); dst != end(); ++dst, ++src)
            *dst = *src;
        }
        return *this;
      }

      VectorView operator=(const VectorView &vw){
        return *this = ConstVectorView(vw);
      }


    public:
      friend std::ostream &operator<<(std::ostream &out, VectorView vw) {
        out << ConstVectorView(vw);
        return out;
      }

    private:
      double *data_;
      size_t size_;
      size_t stride_;
  };

  ConstVectorView::ConstVectorView(const VectorView &that)
          : data_(that.data_), size_(that.size_), stride_(that.stride_){}


  Vector::Vector(ConstVectorView cvw) : Block(cvw.size()) {
    std::cout << "Vector(ConstVectorView vw)\n";
    copyFromView(cvw);
  }

  Vector& Vector::operator=(ConstVectorView cvw){
    std::cout << "Vector::operator=(ConstVectorView vw)\n";
    copyFromView(cvw);
    return *this;
  }

  Vector::Vector(VectorView vw) : Block(vw.size()){
    std::cout << "Vector(VectorView vw)\n";
    copyFromView(vw);
  }

  Vector& Vector::operator=(VectorView vw){
    std::cout << "Vector::operator=(VectorView vw)\n";
    copyFromView(vw);
    return *this;
  }

  void Vector::copyFromView(ConstVectorView cvw){
    if( cvw.size_ != size_)
      __resize__(cvw.size_, true);
    if(cvw.stride_ == 1)
      memcpy(data_, cvw.data_, sizeof(double) * size_);
    else{
      size_t i = 0;
      for(auto it = cvw.begin(); it != cvw.end(); ++it)
        data_[i++] = *it;
    }
  }

  // Apply1 : v[i] = f(cvw[i])
  Vector apply(ConstVectorView cvw, double (*func)(double)){
    Vector result(cvw.size());
    size_t i = 0;
    for(const double d : cvw)
      result[i++] = func(d);
    return result;
  }

  // Apply2 : v[i] = f(cvw[i], value)
  Vector apply(ConstVectorView cvw,
                      double (*func)(double, double), double value){
    Vector result(cvw.size());
    size_t i = 0;
    for(const double d : cvw)
      result[i++] = func(d, value);
    return result;
  }

  // Apply3 : v[i] = f(cvw[i], other[i])
  Vector apply(ConstVectorView cvw,
                      double (*func)(double, double), ConstVectorView other){
    ASSERT_TRUE(cvw.size() == other.size(), "apply: size mismatch");
    Vector result(cvw.size());
    ConstVectorView::iterator it1 = cvw.begin();
    ConstVectorView::iterator it2 = other.begin();
    for(size_t i = 0; i < cvw.size(); ++i, ++it1, ++it2) {
      result[i] = func(*it1, *it2);
    }
    return result;
  }

  // ---------- Any / Or ---------
  bool any(ConstVectorView cvw){
    for(const double d : cvw)
      if( d == 1 )
        return true;
    return false;
  }

  bool all(ConstVectorView cvw){
    for(const double d : cvw)
      if( d == 0 )
        return false;
    return true;
  }

  Vector operator==(ConstVectorView cvw, double d) {
    return apply(cvw, [](double d1, double d2) -> double {
        return fequal(d1, d2);}, d);
  }

  Vector operator==(ConstVectorView cvw, ConstVectorView other) {
    return apply(cvw, [](double d1, double d2) -> double {
        return fequal(d1, d2);}, other);
  }

  Vector operator<(ConstVectorView cvw, double d) {
    return apply(cvw, [](double d1, double d2) -> double { return d1 < d2;}, d);
  }

  Vector operator<(ConstVectorView cvw, ConstVectorView other) {
    return apply(cvw, [](double d1, double d2) -> double {
        return d1 < d2;}, other);
  }

  Vector operator>(ConstVectorView cvw, double d) {
    return apply(cvw, [](double d1, double d2) -> double { return d1 > d2;}, d);
  }


  Vector operator<( double d, ConstVectorView cvw) {
    return cvw > d;
  }

  Vector operator>( double d, ConstVectorView cvw) {
    return cvw < d;
  }

  Vector operator>(ConstVectorView cvw, ConstVectorView other) {
    return apply(cvw, [](double d1, double d2) -> double {
        return d1 > d2;}, other);
  }

  bool fequal(ConstVectorView cvw1, ConstVectorView cvw2){
    if(cvw1.size() != cvw2.size())
      return false;
    ConstVectorView::iterator it1 = cvw1.begin();
    ConstVectorView::iterator it2 = cvw2.begin();
    for(; it1 != cvw1.end(); ++it1, ++it2)
      if( !fequal(*it1, *it2))
        return false;
    return true;
  }

  void operator+=(VectorView vw, const double value){
    for(double &d : vw)
      d += value;
  }

  void operator-=(VectorView vw, const double value){
    for(double &d : vw)
      d -= value;
  }

  void operator*=(VectorView vw, const double value){
    for(double &d : vw)
      d *= value;
  }

  void operator/=(VectorView vw, const double value){
    for(double &d : vw)
      d /= value;
  }

  void operator+=(VectorView vw, ConstVectorView cvw){
    ASSERT_TRUE(vw.size() == cvw.size(),
                "VectorView::operator+: size mismatch");
    ConstVectorView::iterator cit = cvw.begin();
    for(VectorView::iterator it = vw.begin(); it != vw.end(); ++it, ++cit)
      *it += *cit;
  }

  void operator-=(VectorView vw, ConstVectorView cvw){
    ASSERT_TRUE(vw.size() == cvw.size(),
                "VectorView::operator-: size mismatch");
    ConstVectorView::iterator cit = cvw.begin();
    for(VectorView::iterator it = vw.begin(); it != vw.end(); ++it, ++cit)
      *it -= *cit;
  }

  void operator*=(VectorView vw, ConstVectorView cvw){
    ASSERT_TRUE(vw.size() == cvw.size(),
                "VectorView::operator*: size mismatch");
    ConstVectorView::iterator cit = cvw.begin();
    for(VectorView::iterator it = vw.begin(); it != vw.end(); ++it, ++cit)
      *it *= *cit;
  }

  void operator/=(VectorView vw, ConstVectorView cvw){
    ASSERT_TRUE(vw.size() == cvw.size(),
                "VectorView::operator/: size mismatch");
    ConstVectorView::iterator cit = cvw.begin();
    for(VectorView::iterator it = vw.begin(); it != vw.end(); ++it, ++cit)
      *it /= *cit;
  }

  // Returns A + b
  Vector operator+(ConstVectorView cvw, const double d) {
    return apply(cvw, [](double d1, double d2) { return d1 + d2;}, d);
  }

  // Returns b + A
  Vector operator+(const double value, ConstVectorView cvw) {
    return cvw + value;
  }

  // Returns A * b
  Vector operator*(ConstVectorView cvw, const double d) {
    return apply(cvw, [](double d1, double d2) { return d1 * d2;}, d);
  }

  // Returns b + A
  Vector operator*(const double value, ConstVectorView cvw) {
    return cvw * value;
  }

  // Returns A - b
  Vector operator-(ConstVectorView cvw, const double d) {
    return apply(cvw, [](double d1, double d2) { return d1 - d2;}, d);
  }

  // Returns b - A
  Vector operator-(const double d, ConstVectorView cvw) {
    return apply(cvw, [](double d1, double d2) { return d2 - d1;}, d);
  }

  // Returns A / b
  Vector operator/(ConstVectorView cvw, const double d) {
    return apply(cvw, [](double d1, double d2) { return d1 / d2;}, d);
  }

  // Returns b / A
  Vector operator/(const double d, ConstVectorView cvw) {
    return apply(cvw, [](double d1, double d2) { return d2 / d1;}, d);
  }

  // Returns A + B
  Vector operator+(ConstVectorView cvw1, ConstVectorView cvw2) {
    return apply(cvw1, [](double d1, double d2) {return d1 + d2;}, cvw2);
  }

  // Returns A - B
  Vector operator-(ConstVectorView cvw1, ConstVectorView cvw2) {
    return apply(cvw1, [](double d1, double d2) { return d1 - d2;}, cvw2);
  }

  Vector operator*(ConstVectorView cvw1, ConstVectorView cvw2) {
    return apply(cvw1, [](double d1, double d2) { return d1 * d2;}, cvw2);
  }

  Vector operator/(ConstVectorView cvw1, ConstVectorView cvw2) {
    return apply(cvw1, [](double d1, double d2) { return d1 / d2;}, cvw2);
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

  double min(ConstVectorView cvw) {
    ASSERT_TRUE(!cvw.empty(), "ConstVectorView::min()::error: Block empty");
    double cvw_min = std::numeric_limits<double>::max();
    for(const double d : cvw)
      cvw_min = std::min(cvw_min, d);
    return cvw_min;
  }

  double max(ConstVectorView cvw) {
    ASSERT_TRUE(!cvw.empty(), "ConstVectorView::min()::error: Block empty");
    double cvw_max = std::numeric_limits<double>::min();
    for(const double d : cvw)
      cvw_max = std::max(cvw_max, d);
    return cvw_max;
  }

  // Safe log(sum(exp(x)))
  double logSumExp(ConstVectorView cvw) {
    double result = 0;
    double cvw_max = max(cvw);
    for(const double d : cvw)
      result += std::exp(d - cvw_max);
    return cvw_max + std::log(result);
  }

  // Standard deviation
  double stdev(ConstVectorView cvw){
    return std::sqrt(var(cvw));
  }

  // Absolute value of x
  Vector abs(ConstVectorView cvw){
    return apply(cvw, std::fabs);
  }

  // Round to nearest integer
  Vector round(ConstVectorView cvw){
    return apply(cvw, std::round);
  }

  // Ceiling
  Vector ceil(ConstVectorView cvw) {
    return apply(cvw, std::ceil);
  }

  // Floor
  Vector floor(ConstVectorView cvw) {
    return apply(cvw, std::floor);
  }

  // Exponential
  Vector exp(ConstVectorView cvw) {
    return apply(cvw, std::exp);
  }

  // Logarithm
  Vector log(ConstVectorView cvw) {
    return apply(cvw, std::log);
  }

  // Normalize
  Vector normalize(ConstVectorView cvw) {
    Vector result(cvw);
    result /= sum(result);
    return result;
  }

  // Safe normalize(exp(x))
  Vector normalizeExp(ConstVectorView cvw) {
    return normalize(exp(cvw - max(cvw)));
  }

  Vector reverse(ConstVectorView cvw){
    Vector result(cvw.size());
    size_t i = cvw.size();
    for(auto it = cvw.begin(); it != cvw.end(); ++it)
      result[--i] = *it;
    return result;
  }

  // is_nan
  Vector is_nan(ConstVectorView cvw){
    return apply(cvw, [](double d1) -> double { return std::isnan(d1); });
  }

  // is_inf
  Vector is_inf(ConstVectorView cvw){
    return apply(cvw, [](double d1) -> double { return std::isinf(d1); });
  }

  // Power
  Vector pow(ConstVectorView cvw, double p){
    return apply(cvw, [](double d1, double d2) { return std::pow(d1, d2);} ,p);
  }

  // Dot product
  double dot(ConstVectorView cvw1, ConstVectorView cvw2) {
    ASSERT_TRUE(cvw1.size() == cvw2.size(),
                "Vector::dot() Vector sizes mismatch");
    ConstVectorView::iterator it1 = cvw1.begin();
    ConstVectorView::iterator it2 = cvw2.begin();
    double result = 0;
    for(; it1 != cvw1.end(); ++it1, ++it2)
      result += (*it1) * (*it2);
    return result;
  }



} // namespace pml

#endif