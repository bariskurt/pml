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

#include "pml_block.hpp"

#define DEFAULT_PRECISION 6

namespace pml {

  // Defines a series from start to stop exclusively with step size step:
  // [start, start+step, start+2*step, ...]
  struct Range {
      Range(size_t start_, size_t stop_, size_t step_ = 1)
              : start(start_), stop(stop_), step(step_) { }
      size_t start, stop, step;
  };

  inline bool fequal(double a, double b) {
    return fabs(a - b) < 1e-6;
  }

  class Vector {
    public:
      // Empty Vector
      Vector() { }

      // Vector of given length and default value.
      explicit Vector(size_t length, double value = 0) : data_(length) {
        std::fill_n(data_.begin(), length, value);
      }

      // Vector from given array
      Vector(size_t length, const double *values) : data_(length) {
        memcpy(data_.data(), values, sizeof(double) * length);
      }

      // Vector from initializer lsit
      Vector(const std::initializer_list<double> &values) : data_(values) { }


      // Extract vector slice
      Vector(double *data, size_t length, size_t stride,
             bool deep_copy = false) : data_(data, length, stride) {
        if(deep_copy){
          data_ = Block(data, length, stride);
        }
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
      // Returns length of the Vector.
      size_t size() const {
        return data_.size();
      }

      // Checks empty.
      bool empty() const {
        return data_.empty();
      }

      // Vector resize. If new size is smaller, the data_ is cropped.
      // If new size is larger, garbage values are appended.
      void resize(size_t new_size) {
        data_.resize(new_size);
      }

    public:
      // Append a single value. (same as push_back)
      void append(double value) {
        data_.append(value);
      }

      // Append a Vector
      void append(const Vector &v) {
        data_.append(v.data_);
      }

      friend Vector apply(const Vector &x, double (*func)(double)){
        Vector result = x;
        result.apply(func);
        return result;
      }

      void apply(double (*func)(double)){
        for(double &d : data_)
          d = func(d);
      }

    public:

      friend bool any(const Vector &v, double value = 1){
        for(size_t i = 0; i < v.size(); ++i)
          if( v[i] == value )
            return true;
        return false;
      }

      friend bool all(const Vector &v, double value = 1){
        for(size_t i = 0; i < v.size(); ++i)
          if( v[i] != value )
            return false;
        return true;
      }

      friend Vector operator==(const Vector &x, double v) {
        Vector result(x.size());
        Block::iterator in = x.begin();
        Block::iterator out = result.begin();
        for(size_t i = 0; i < x.size(); ++i, ++in, ++out)
          *out = fequal(*in, v);
        return result;
      }

      friend Vector operator==(const Vector &x, const Vector &y) {
        // Check sizes
        ASSERT_TRUE(x.size() == y.size(),
            "Vector::operator== cannot compare vectors of different size" );
        // Check element-wise
        Vector result(x.size());
        Block::iterator in1 = x.begin();
        Block::iterator in2 = y.begin();
        Block::iterator out = result.begin();
        for(size_t i = 0; i < x.size(); ++i, ++in1, ++in2, ++out)
          *out = fequal(*in1, *in2);
        return result;
      }

      friend Vector operator<(const Vector &x, double d) {
        // Check element-wise
        Vector result(x.size());
        for(size_t i = 0; i < x.size(); ++i)
          result[i] = x[i] < d;
        return result;
      }

      friend Vector operator<( double d, const Vector &x) {
        return x > d;
      }

      friend Vector operator<(const Vector &x, const Vector &y) {
        // Check sizes
        ASSERT_TRUE(x.size() == y.size(),
            "Vector::operator== cannot compare vectors of different size" );
        // Check element-wise
        Vector result(x.size());
        for(size_t i = 0; i < x.size(); ++i)
          result[i] = x[i] < y[i];
        return result;
      }

      friend Vector operator>(const Vector &x, double d) {
        Vector result(x.size());
        for(size_t i = 0; i < x.size(); ++i)
          result[i] = x[i] > d;
        return result;
      }

      friend Vector operator>( double d, const Vector &x) {
        return x < d;
      }

      friend Vector operator>(const Vector &x, const Vector &y) {
        // Check sizes
        ASSERT_TRUE(x.size() == y.size(),
            "Vector::operator== cannot compare vectors of different size" );
        // Check element-wise
        Vector result(x.size());
        for(size_t i = 0; i < x.size(); ++i)
          result[i] = x[i] > y[i];
        return result;
      }

      bool equals(const Vector &other){
        if(size() != other.size())
          return false;
        return all(*this == other);
      }

    public:
      //  -------- Iterators--------
      Block::iterator begin() {
        return data_.begin();
      }

      Block::iterator begin() const {
        return data_.begin();
      }

        Block::iterator end() {
        return data_.end();
      }

        Block::iterator end() const {
        return data_.end();
      }

      // ------- Accessors -------

      inline double &operator[](const size_t i0) {
        return data_[i0];
      }

      inline double operator[](const size_t i0) const {
        return data_[i0];
      }

      inline double &operator()(const size_t i0) {
        return data_[i0];
      }

      inline double operator()(const size_t i0) const {
        return data_[i0];
      }

      double* data() {
        return data_.data();
      }

      const double *data() const {
        return data_.data();
      }

      double first() const {
        return data_.front();
      }

      double& first() {
        return data_.front();
      }

      double last() const {
        return data_.back();
      }

      double& last() {
        return data_.back();
      }

      // Vector slice
      Vector slice(size_t start, size_t stop, size_t step = 1,
                   bool deep_copy = false) const {
        double *data_start = data_.data_ + (data_.stride() * start);
        size_t slice_size = std::ceil((double)(stop - start) / step);
        ASSERT_TRUE(slice_size <= size(),
                    "Vector::slice() size exceeds actual size.");
        return Vector(data_start, slice_size, step, deep_copy);
      }

    public:

      // ------ Self Assignment Operators -------

      void operator+=(double value) {
        for (auto &d : data_) { d += value; }
      }

      // A = A - b
      void operator-=(double value) {
        for (auto &d : data_) { d -= value; }
      }

      // A = A * b
      void operator*=(double value) {
        for (auto &d : data_) { d *= value; }
      }

      // A = A / b
      void operator/=(double value) {
        for (auto &d : data_) { d /= value; }
      }

      // A = A + B
      void operator+=(const Vector &other) {
        ASSERT_TRUE(size() == other.size(),
                    "Vector::operator+=:: Size mismatch.");
        for (size_t i = 0; i < data_.size(); ++i) {
          data_[i] += other[i];
        }
      }

      // A = A - B
      void operator-=(const Vector &other) {
        ASSERT_TRUE(size() == other.size(),
                    "Vector::operator-=:: Size mismatch.");
        for (size_t i = 0; i < data_.size(); ++i) {
          data_[i] -= other[i];
        }
      }

      // A = A * B (elementwise)
      void operator*=(const Vector &other) {
        ASSERT_TRUE(size() == other.size(),
                    "Vector::operator*=:: Size mismatch.");
        for (size_t i = 0; i < data_.size(); ++i) {
          data_[i] *= other[i];
        }
      }

      // A = A / B (elementwise)
      void operator/=(const Vector &other) {
        ASSERT_TRUE(size() == other.size(),
                    "Vector::operator/=:: Size mismatch.");
        for (size_t i = 0; i < data_.size(); ++i) {
          data_[i] /= other[i];
        }
      }

      // ------ Vector - Double Operations -------

    public:

      // Returns A + b
      friend Vector operator+(const Vector &x, double value) {
        Vector result(x);
        result += value;
        return result;
      }

      // Returns b + A
      friend Vector operator+(double value, const Vector &x) {
        return x + value;
      }

      // Returns A * b
      friend Vector operator*(const Vector &x, double value) {
        Vector result(x);
        result *= value;
        return result;
      }

      // Returns b * A
      friend Vector operator*(double value, const Vector &x) {
        return x * value;
      }

      // Returns A - b
      friend Vector operator-(const Vector &x, double value) {
        Vector result(x);
        result -= value;
        return result;
      }

      // Returns b - A
      friend Vector operator-(double value, const Vector &x) {
        return (-1 * x) + value;
      }

      // returns A / b
      friend Vector operator/(const Vector &x, double value) {
        Vector result(x);
        result /= value;
        return result;
      }

      // returns b / A
      friend Vector operator/(double value, const Vector &x) {
        Vector result(x);
        for (auto &d : result) { d = value / d; }
        return result;
      }

      // ------ Vector - Vector Operations -------

      // R = A + B
      friend Vector operator+(const Vector &x, const Vector &y) {
        ASSERT_TRUE(x.size() == y.size(), "Vector::operator+:: Size mismatch.");
        Vector result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
          result.data_[i] = x.data_[i] + y.data_[i];
        }
        return result;
      }

      // R = A - B
      friend Vector operator-(const Vector &x, const Vector &y) {
        ASSERT_TRUE(x.size() == y.size(), "Vector::operator-:: Size mismatch.");
        Vector result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
          result.data_[i] = x.data_[i] - y.data_[i];
        }
        return result;
      }

      // R = A * B (elementwise)
      friend Vector operator*(const Vector &x, const Vector &y) {
        ASSERT_TRUE(x.size() == y.size(), "Vector::operator*:: Size mismatch.");
        Vector result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
          result.data_[i] = x.data_[i] * y.data_[i];
        }
        return result;
      }

      // R = A / B (elementwise)
      friend Vector operator/(const Vector &x, const Vector &y) {
        ASSERT_TRUE(x.size() == y.size(), "Vector::operator/:: Size mismatch.");
        Vector result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
          result.data_[i] = x.data_[i] / y.data_[i];
        }
        return result;
      }

      // Load and Save
      friend std::ostream &operator<<(std::ostream &out,
                                      const Vector &x) {
        out << std::setprecision(DEFAULT_PRECISION) << std::fixed;
        for (auto &value : x) {
          out << value << "  ";
        }
        return out;
      }

      friend std::istream &operator>>(std::istream &in, Vector &x) {
        for (auto &value : x) {
          in >> value;
        }
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
          for (auto &value : data_) {
            ofs << value << std::endl;
          }
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
          result.data_.resize(buffer);
          ifs >> result;
          ifs.close();
        }
        return result;
      }

    public:
      Block data_;
  };

  Vector cat(const Vector &v1, const Vector &v2){
    Vector result(v1);
    result.append(v2);
    return result;
  }

  // Returns the set of indices i of v, such that v[i] == 1.
  Vector find(const Vector &v){
    Vector result;
    for(size_t i=0; i < v.size(); ++i){
      if(v[i] == 1)
        result.append(i);
    }
    return result;
  }

  // is_nan
  Vector is_nan(const Vector &v){
    Vector result = Vector::zeros(v.size());
    for(size_t i=0; i < v.size(); ++i){
      if(std::isnan(v[i]))
        result[i] = 1;
    }
    return result;
  }

  // is_inf
  Vector is_inf(const Vector &v){
    Vector result = Vector::zeros(v.size());
    for(size_t i=0; i < v.size(); ++i){
      if(std::isinf(v[i]))
        result[i] = 1;
    }
    return result;
  }

  // Sum
  inline double sum(const Vector &x){
    return std::accumulate(x.begin(), x.end(), 0.0);
  }

  // Power
  inline Vector pow(const Vector &x, double p = 2){
    Vector result(x);
    for(double &d : result)
      d = std::pow(d, p);
    return result;
  }

  //Min
  inline double min(const Vector &x) {
    return *(std::min_element(x.begin(), x.end()));
  }

  // Max
  inline double max(const Vector &x) {
    return *(std::max_element(x.begin(), x.end()));
  }

  // Dot product
  inline double dot(const Vector &x, const Vector &y) {
    return sum(x * y);
  }

  // Mean
  inline double mean(const Vector &x){
    return sum(x) / x.size();
  }

  // Variance
  inline double var(const Vector &x){
    return sum(pow(x - mean(x), 2)) / (x.size() - 1);
  }

  // Standard deviation
  inline double stdev(const Vector &x){
    return std::sqrt(var(x));
  }

  // Absolute value of x
  inline Vector abs(const Vector &x){
    return apply(x, std::fabs);
  }

  // Round to nearest integer
  inline Vector round(const Vector &x){
    return apply(x, std::round);
  }

  // Ceiling
  inline Vector ceil(const Vector &x){
    return apply(x, std::ceil);
  }

  // Floor
  inline Vector floor(const Vector &x){
    return apply(x, std::floor);
  }

  // Exponential
  inline Vector exp(const Vector &x){
    return apply(x, std::exp);
  }

  // Logarithm
  inline Vector log(const Vector &x){
    return apply(x, std::log);
  }

  // Normalize
  inline Vector normalize(const Vector &x) {
    return x / sum(x);
  }

  // Safe normalize(exp(x))
  inline Vector normalizeExp(const Vector &x) {
    return normalize(exp(x - max(x)));
  }

  // Safe log(sum(exp(x)))
  inline double logSumExp(const Vector &x) {
    double x_max = max(x);
    return x_max + std::log(sum(exp(x - x_max)));
  }

  // KL Divergence Between Vectors
  // ToDo: Maybe move to linalgebra
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

} // namespace pml

#endif