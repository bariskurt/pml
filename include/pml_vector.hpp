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

      friend Vector apply(const Vector &x, double (*func)(double)){
        Vector result;
        for(size_t i=0; i < x.size(); ++i)
          result.append(func(x[i]));
        return result;
      }

    public:

      friend Vector operator==(const Vector &x, double v) {
        Vector result(x.size());
        for(size_t i = 0; i < x.size(); ++i)
          result[i] = fequal(x[i], v);
        return result;
      }

      friend Vector operator==(const Vector &x, const Vector &y) {
        // Check sizes
        ASSERT_TRUE(x.size() == y.size(),
            "Vector::operator== cannot compare vectors of different size" );
        // Check element-wise
        Vector result(x.size());
        for(size_t i = 0; i < x.size(); ++i)
          result[i] = fequal(x[i], y[i]);
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

    public:

      // ------ Self Assignment Operators -------

      void operator+=(const double value) {
        for(size_t i=0; i < size_; ++i)
          data_[i] += value;
      }

      // A = A - b
      void operator-=(const double value) {
        for(size_t i=0; i < size_; ++i)
          data_[i] -= value;
      }

      // A = A * b
      void operator*=(const double value) {
        for(size_t i=0; i < size_; ++i)
          data_[i] *= value;
      }

      // A = A / b
      void operator/=(const double value) {
        for(size_t i=0; i < size_; ++i)
          data_[i] /= value;
      }


      // A = A + B
      void operator+=(const Vector &other) {
        ASSERT_TRUE(size() == other.size(),
                    "Vector::operator+=:: Size mismatch.");
        for(size_t i=0; i < size_; ++i)
          data_[i] += other.data_[i];
      }

      // A = A - B
      void operator-=(const Vector &other) {
        ASSERT_TRUE(size() == other.size(),
                    "Vector::operator-=:: Size mismatch.");
        for(size_t i=0; i < size_; ++i)
          data_[i] -= other.data_[i];
      }

      // A = A * B (elementwise)
      void operator*=(const Vector &other) {
        ASSERT_TRUE(size() == other.size(),
                    "Vector::operator*=:: Size mismatch.");
        for(size_t i=0; i < size_; ++i)
          data_[i] *= other.data_[i];
      }

      // A = A / B (elementwise)
      void operator/=(const Vector &other) {
        ASSERT_TRUE(size() == other.size(),
                    "Vector::operator/=:: Size mismatch.");
        for(size_t i=0; i < size_; ++i)
          data_[i] /= other.data_[i];
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
        Vector result(x.size());
        for(size_t i=0; i < x.size(); ++i)
          result.data_[i] = value / x.data_[i];
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

  Vector cat(const Vector &v1, const Vector &v2){
    Vector result(v1);
    result.append(v2);
    return result;
  }

  Vector cat(const std::vector<Vector> &v_list){
    Vector result;
    for(const Vector &v : v_list)
      result.append(v);
    return result;
  }

  Vector reverse(const Vector &v){
    Vector result(v.size());
    if( v.size() > 0) {
      size_t offset = v.size() - 1;
      for (size_t i = 0; i < v.size(); ++i)
        result[i] = v[offset - 1];
    }
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

  // Power
  inline Vector pow(const Vector &x, double p){
    Vector result(x.size());
    for(size_t i = 0; i < x.size(); ++i)
      result[i] = std::pow(x[i], p);
    return result;
  }

  // Dot product
  inline double dot(const Vector &x, const Vector &y) {
    ASSERT_TRUE(x.size() == y.size(), "Vector::dot() Vector sizes mismatch");
    double result = 0;
    for(size_t i = 0; i < x.size(); ++i)
      result += x[i] * y[i];
    return result;
  }



} // namespace pml

#endif