#ifndef MATLIB_PML_H_H
#define MATLIB_PML_H_H

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>
#include <tuple>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_sf_psi.h>


#define DEFAULT_PRECISION 6

namespace pml {

  // Helpers:
  inline bool fequal(double a, double b) {
    return fabs(a - b) < 1e-6;
  }


    class Vector {
      public:
        Vector(){}

        explicit Vector(size_t length, double value = 0)
                : __data__(length, value) {}

        Vector(const std::initializer_list<double> &values)
                : __data__(values) {}

        Vector(const Vector &other) {
          __data__ = other.__data__;
        }

        Vector(Vector &&other) {
          __data__ = std::move(other.__data__);
        }

        Vector &operator=(const Vector &other) {
          __data__ = other.__data__;
          return *this;
        }

        Vector &operator=(Vector &&other) {
          __data__ = std::move(other.__data__);
          return *this;
        }

      // Special Vectors
      public:
        static Vector ones(size_t length) {
          return Vector(length, 1.0);
        }

        static Vector zeros(size_t length) {
          return Vector(length, 0.0);
        }

      public:
        // Total number of values inside the structure.
        size_t size() const {
          return __data__.size();
        }

        bool empty() const {
          return __data__.empty();
        }

        virtual size_t ndims() const {
          return 1;
        }

        friend bool similar(const Vector &x, const Vector &y){
          return (x.ndims() == y.ndims()) && (x.size() == y.size());
        }

      public:
        friend bool operator==(const Vector &x, const Vector &y) {
          if ( x.size() != y.size()) {
            return false;
          }
          for (size_t i = 0; i < x.size(); ++i) {
            if (!fequal(x(i), y(i))) {
              return false;
            }
          }
          return true;
        }

        friend bool operator==(const Vector &x, double d) {
          for (auto &val : x) {
            if (!fequal(val, d)) {
              return false;
            }
          }
          return true;
        }


      public:
        std::vector<double>::iterator begin(){
          return __data__.begin();
        }

        std::vector<double>::const_iterator begin() const{
          return __data__.cbegin();
        }

        std::vector<double>::iterator end(){
          return __data__.end();
        }

        std::vector<double>::const_iterator end() const {
          return __data__.cend();
        }

        inline double &operator()(const size_t i0) {
          return __data__[i0];
        }

        inline double operator()(const size_t i0) const {
          return __data__[i0];
        }

        double *data() {
          return &__data__[0];
        }

        const double *data() const {
          return &__data__[0];
        }

        double first() const {
          return __data__.front();
        }

        double &first() {
          return __data__.front();
        }

        double last() const {
          return __data__.back();
        }

        double &last() {
          return __data__.back();
        }

      public:
        void apply(double (*func)(double)) {
          for (auto &value : __data__) {
            value = func(value);
          }
        }

        void operator+=(double value) {
          for (auto &d : __data__) { d += value; }
        }

        // A = A - b
        void operator-=(double value) {
          for (auto &d : __data__) { d -= value; }
        }

        // A = A * b
        void operator*=(double value) {
          for (auto &d : __data__) { d *= value; }
        }

        // A = A / b
        void operator/=(double value) {
          for (auto &d : __data__) { d /= value; }
        }

        // A = A + B
        void operator+=(const Vector &other) {
          assert(similar(*this, other));
          for (size_t i = 0; i < __data__.size(); ++i) {
            __data__[i] += other.__data__[i];
          }
        }

        // A = A - B
        void operator-=(const Vector &other) {
          assert(similar(*this, other));
          for (size_t i = 0; i < __data__.size(); ++i) {
            __data__[i] -= other.__data__[i];
          }
        }

        // A = A * B (elementwise)
        void operator*=(const Vector &other) {
          assert(similar(*this, other));
          for (size_t i = 0; i < __data__.size(); ++i) {
            __data__[i] *= other.__data__[i];
          }
        }

        // A = A / B (elementwise)
        void operator/=(const Vector &other) {
          assert(similar(*this, other));
          for (size_t i = 0; i < __data__.size(); ++i) {
            __data__[i] /= other.__data__[i];
          }
        }

      // Friend functions for arithmetic:
      public:
        // returns A + b
        friend Vector operator+(const Vector &x, double value) {
          Vector result(x);
          result += value;
          return result;
        }

        // returns b + A
        friend Vector operator+(double value, const Vector &x) {
          return x + value;
        }

        // returns A * b
        friend Vector operator*(const Vector &x, double value) {
          Vector result(x);
          result *= value;
          return result;
        }

        // returns b * A
        friend Vector operator*(double value, const Vector &x) {
          return x * value;
        }

        // returns A - b
        friend Vector operator-(const Vector &x, double value) {
          Vector result(x);
          result -= value;
          return result;
        }

        // returns b - A
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

        // R = A + B
        friend Vector operator+(const Vector &x, const Vector &y) {
          assert(similar(x,y));
          Vector result(x.size());
          for (size_t i = 0; i < x.size(); ++i) {
            result.__data__[i] = x.__data__[i] + y.__data__[i];
          }
          return result;
        }

        // R = A - B
        friend Vector operator-(const Vector &x, const Vector &y) {
          assert(similar(x,y));
          Vector result(x.size());
          for (size_t i = 0; i < x.size(); ++i) {
            result.__data__[i] = x.__data__[i] - y.__data__[i];
          }
          return result;
        }

        // R = A * B (elementwise)
        friend Vector operator*(const Vector &x, const Vector &y) {
          assert(similar(x,y));
          Vector result(x.size());
          for (size_t i = 0; i < x.size(); ++i) {
            result.__data__[i] = x.__data__[i] * y.__data__[i];
          }
          return result;
        }

        // R = A / B (elementwise)
        friend Vector operator/(const Vector &x, const Vector &y) {
          assert(similar(x,y));
          Vector result(x.size());
          for (size_t i = 0; i < x.size(); ++i) {
            result.__data__[i] = x.__data__[i] / y.__data__[i];
          }
          return result;
        }
/*
      public:
        // Matrix - Vector Product
        friend double dot(const Vector &x, const Vector &y) {
          assert(x.size() == y.size());
          return sum(x * y);
        }
*/
      public:
        std::vector<double> __data__;
    };

    class Matrix : public Vector{

      public:
        Matrix(){}

        Matrix(size_t num_rows, size_t num_cols, double value = 0)
                : Vector(num_rows * num_cols, value),
                  __nrows__(num_rows), __ncols__(num_cols) {}

        Matrix(size_t num_rows, size_t num_cols,
               const std::initializer_list<double> &values)
                : Vector(values), __nrows__(num_rows), __ncols__(num_cols) {}

        Matrix(const Matrix &other) {
          __data__ = other.__data__;
          __nrows__ = other.__nrows__;
          __ncols__ = other.__ncols__;
        }

        Matrix(Matrix &&other) {
          __data__ = std::move(other.__data__);
          __nrows__ = other.__nrows__;
          __ncols__ = other.__ncols__;
          other.__nrows__ = other.__ncols__ = 0;
        }

        Matrix &operator=(const Matrix &other) {
          __data__ = other.__data__;
          __nrows__ = other.__nrows__;
          __ncols__ = other.__ncols__;
          return *this;
        }

        Matrix &operator=(Matrix &&other) {
          __data__ = std::move(other.__data__);
          __nrows__ = other.__nrows__;
          __ncols__ = other.__ncols__;
          other.__nrows__ = other.__ncols__ = 0;
          return *this;
        }

      public:
        size_t nrows() const{
          return __nrows__;
        }

        size_t ncols() const{
          return __ncols__;
        }

        virtual size_t ndims() const {
          return 2;
        }

      public:
        static Matrix ones(size_t num_rows, size_t num_cols) {
          return Matrix(num_rows, num_cols, 1.0);
        }

        static Matrix zeros(size_t num_rows, size_t num_cols) {
          return Matrix(num_rows, num_cols, 0.0);
        }

        // ToDo: use slice
        static Matrix identity(size_t size) {
          Matrix X = Matrix::zeros(size, size);
          for (size_t i = 0; i < size; ++i) {
            X(i, i) = 1.0;
          }
          return X;
        }

      public:
        inline double &operator()(const size_t i0, const size_t i1) {
          return __data__[i0 + __nrows__ * i1];
        }

        inline double operator()(const size_t i0, const size_t i1) const {
          return __data__[i0 + __nrows__ * i1];
        }

        friend bool similar(const Matrix &x, const Matrix &y){
          return (x.nrows() == y.nrows()) && (x.ncols() == y.ncols());
        }

      public:
        // returns A + b
        friend Matrix operator+(const Matrix &x, double value) {
          Matrix result(x);
          result += value;
          return result;
        }

        // returns b + A
        friend Matrix operator+(double value, const Matrix &x) {
          return x + value;
        }

        // returns A * b
        friend Matrix operator*(const Matrix &x, double value) {
          Matrix result(x);
          result *= value;
          return result;
        }

        // returns b * A
        friend Matrix operator*(double value, const Matrix &x) {
          return x * value;
        }

        // returns A - b
        friend Matrix operator-(const Matrix &x, double value) {
          Matrix result(x);
          result -= value;
          return result;
        }

        // returns b - A
        friend Matrix operator-(double value, const Matrix &x) {
          return (-1 * x) + value;
        }

        // returns A / b
        friend Matrix operator/(const Matrix &x, double value) {
          Matrix result(x);
          result /= value;
          return result;
        }

        // returns b / A
        friend Matrix operator/(double value, const Matrix &x) {
          Matrix result(x);
          for (auto &d : result) { d = value / d; }
          return result;
        }

        // R = A + B
        friend Matrix operator+(const Matrix &x, const Matrix &y) {
          assert(similar(x,y));
          Matrix result(x.nrows(), x.ncols());
          for (size_t i = 0; i < x.size(); ++i) {
            result.__data__[i] = x.__data__[i] + y.__data__[i];
          }
          return result;
        }

        // R = A - B
        friend Matrix operator-(const Matrix &x, const Matrix &y) {
          assert(similar(x,y));
          Matrix result(x.nrows(), x.ncols());
          for (size_t i = 0; i < x.size(); ++i) {
            result.__data__[i] = x.__data__[i] - y.__data__[i];
          }
          return result;
        }

        // R = A * B (elementwise)
        friend Matrix operator*(const Matrix &x, const Matrix &y) {
          assert(similar(x,y));
          Matrix result(x.nrows(), x.ncols());
          for (size_t i = 0; i < x.size(); ++i) {
            result.__data__[i] = x.__data__[i] * y.__data__[i];
          }
          return result;
        }

        // R = A / B (elementwise)
        friend Matrix operator/(const Matrix &x, const Matrix &y) {
          assert(similar(x,y));
          Matrix result(x.nrows(), x.ncols());
          for (size_t i = 0; i < x.size(); ++i) {
            result.__data__[i] = x.__data__[i] / y.__data__[i];
          }
          return result;
        }

      private:
        size_t __nrows__;
        size_t __ncols__;
    };

    inline double sum(const Vector &x){
      double result = 0;
      for (auto &value: x) {
        result += value;
      }
      return result;
    }

    /*
    inline Vector sum(const Matrix &m, size_t dim) {
      Vector result;
      if (dim == 0) {
        result = Vector(m.num_cols(), 0);
        for (size_t j = 0; j < m.ncols(); ++j) {
          result(j) = sum(array.col(j));
        }
      } else {
        result = Vector(array.num_rows(), 0);
        for (size_t i = 0; i < array.num_rows(); ++i) {
          result(i) = sum(array.row(i));
        }
      }
      return result;
    }
    */


    inline double max(const Vector &x) {
      double result = std::numeric_limits<double>::lowest();
      for (auto &value : x) {
        if (value > result) {
          result = value;
        }
      }
      return result;
    }

    inline double min(const Vector &x) {
      double result = std::numeric_limits<double>::max();
      for (auto &value : x) {
        if (value < result) {
          result = value;
        }
      }
      return result;
    }

    inline Vector abs(const Vector &x){
      Vector y(x);
      y.apply(std::fabs);
      return y;
    }

    inline Matrix abs(const Matrix &x){
      Matrix y(x);
      y.apply(std::fabs);
      return y;
    }

    inline Vector round(const Vector &x){
      Vector y(x);
      y.apply(std::round);
      return y;
    }

    inline Matrix round(const Matrix &x){
      Matrix y(x);
      y.apply(std::round);
      return y;
    }

    inline Vector psi(const Vector &x){
      Vector y(x);
      y.apply(gsl_sf_psi);
      return y;
    }

    inline Matrix psi(const Matrix &x){
      Matrix y(x);
      y.apply(gsl_sf_psi);
      return y;
    }

    inline Vector exp(const Vector &x){
      Vector y(x);
      y.apply(std::exp);
      return y;
    }

    inline Matrix exp(const Matrix &x){
      Matrix y(x);
      y.apply(std::exp);
      return y;
    }

    inline Vector log(const Vector &x){
      Vector y(x);
      y.apply(std::log);
      return y;
    }

    inline Matrix log(const Matrix &x){
      Matrix y(x);
      y.apply(std::log);
      return y;
    }

    inline Vector normalize(const Vector &x) {
      Vector result(x);
      result /= sum(x);
      return result;
    }

    inline Vector normalizeExp(const Vector &x) {
      return normalize(exp(x - max(x)));
    }

    inline double logSumExp(const Vector &x) {
      double x_max = max(x);
      return x_max + std::log(sum(exp(x - x_max)));
    }

    inline double klDiv(const Vector &x, const Vector &y) {
      assert(x.size() == y.size());
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

}

#endif
