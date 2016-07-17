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

  class Array {

/*
    public:
      class Slice{
        public:
          Slice(Array &array_, size_t start_ = 0,
                     size_t step_ = 0, size_t length_ = 0)
                  : start(start_), step(step_), length(length_){
            array = &array_;
          }

          Slice(const Array &array_, size_t start_ = 0,
                     size_t step_ = 0, size_t length_ = 0)
                  : start(start_), step(step_), length(length_){
            array = &array_;
          }

          Slice &operator=(const Array &other) {
            assert(other.size() == length);
            size_t idx = start;
            for(size_t i=0; i < length; ++i){
              array->data_[idx] = other(i);
              idx += step;
            }
            return *this;
          }

        public:
          Array* array;
          size_t start, step, length;
      };
*/

  private:
      static const size_t MAX_DIMENSIONS = 3;

    public:
      Array(){}

      explicit Array(const std::vector<size_t> &dims, double value = 0)
              : dims_(dims) {
        assert(dims_.size() <= MAX_DIMENSIONS);
        // resize data vector:
        size_t new_size = 1;
        for (auto d : dims_) {
          new_size *= d;
        }
        data_ = std::vector<double>(new_size, value);
      }

      Array(const std::vector<size_t> &dims, const std::vector<double> &values)
              : Array(dims) {
        assert(size() == values.size());
        data_ = values;
      }
/*
      Array(const Slice &slice){
        data_.resize(slice.length);
        size_t idx = slice.start;
        for(size_t i=0; i < slice.length; ++i){
          data_[i] = slice.array->data_[idx];
          idx += slice.step;
        }
      }
*/
      Array(const Array &other) : data_(other.data_), dims_(other.dims_) { }

      Array(Array &&other) {
        data_ = std::move(other.data_);
        dims_ = std::move(other.dims_);
      }

      Array &operator=(const Array &other) {
        data_ = other.data_;
        dims_ = other.dims_;
        return *this;
      }

      Array &operator=(Array &&other) {
        data_ = std::move(other.data_);
        dims_ = std::move(other.dims_);
        return *this;
      }

      // Shape & Reshape related stuff
    protected:
      void reshape(const std::vector<size_t> &new_dims) {
        size_t new_size = 1;
        for (auto d : new_dims) {
          new_size *= d;
        }
        assert(size() == new_size);
        dims_ = new_dims;
      }

      bool shape_equal(const Array &other) const {
        return dims_ == other.dims_;
      }

    // Size info
    public:
      // Total number of values inside the structure.
      size_t size() const {
        return data_.size();
      }

      // Number of dimensions.
      size_t ndims() const {
        return dims_.size();
      }

      // Get whole dimensions.
      const std::vector<size_t>& dims() const {
        return dims_;
      }

      // Dimension along idx.
      size_t dim(size_t idx) const {
        assert(idx < ndims());
        return dims_[idx];
      }

      bool empty() const {
        return size() == 0;
      }

    // Accessors
    public:

      std::vector<double>::iterator begin(){
        return data_.begin();
      }

      std::vector<double>::iterator end(){
        return data_.end();
      }

      inline double &operator()(const size_t i0) {
        return data_[i0];
      }

      inline double operator()(const size_t i0) const {
        return data_[i0];
      }

      inline double &operator()(const size_t i0, const size_t i1) {
        assert(ndims() == 2);
        return data_[i0 + dims_[0] * i1];
      }

      inline double operator()(const size_t i0, const size_t i1) const {
        assert(ndims() == 2);
        return data_[i0 + dims_[0] * i1];
      }

      inline double& operator()(const size_t i0, const size_t i1,
                                const size_t i2) {
        assert(ndims() == 3);
        return data_[i0 + dims_[0] * (i1 + dims_[1] * i2)];
      }

      inline double operator()(const size_t i0, const size_t i1,
                               const size_t i2) const{
        assert(ndims() == 3);
        return data_[i0 + dims_[0] * (i1 + dims_[1] * i2)];
      }

      double *data() {
        return &data_[0];
      }

      const double *data() const {
        return &data_[0];
      }

      double first() const {
        return data_[0];
      }

      double &first() {
        return data_[0];
      }

      double last() const {
        return data_.back();
      }

      double &last() {
        return data_.back();
      }

    // Comparison
    public:
      friend bool operator==(const Array &a1, const Array &a2) {
        if (!a1.shape_equal(a2)) {
          return false;
        }
        for (size_t i = 0; i < a1.size(); ++i) {
          if (!fequal(a1(i), a2(i))) {
            return false;
          }
        }
        return true;
      }

      friend bool operator==(const Array &array, double d) {
        for (auto &val : array.data_) {
          if (!fequal(val, d)) {
            return false;
          }
        }
        return true;
      }

      friend bool operator!=(const Array &a1, const Array &a2) {
        return !(a1 == a2);
      };

    // Apply function to manipulate array data
    public:
      void apply(double (*func)(double)) {
        for (auto &value : data_) {
          value = func(value);
        }
      }

      static Array apply(const Array &array, double (*func)(double)) {
        Array result(array);
        for (auto &value : result.data_) {
          value = func(value);
        }
        return result;
      }

    // Unary Operators
    public:

      // A = A + b
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
      void operator+=(const Array &other) {
        assert(shape_equal(other));
        for (size_t i = 0; i < data_.size(); ++i) {
          data_[i] += other.data_[i];
        }
      }

      // A = A - B
      void operator-=(const Array &other) {
        assert(shape_equal(other));
        for (size_t i = 0; i < data_.size(); ++i) {
          data_[i] -= other.data_[i];
        }
      }

      // A = A * B (elementwise)
      void operator*=(const Array &other) {
        assert(shape_equal(other));
        for (size_t i = 0; i < data_.size(); ++i) {
          data_[i] *= other.data_[i];
        }
      }

      // A = A / B (elementwise)
      void operator/=(const Array &other) {
        assert(shape_equal(other));
        for (size_t i = 0; i < data_.size(); ++i) {
          data_[i] /= other.data_[i];
        }
      }

    // Friend functions for arithmetic:
    public:
      // returns A + b
      friend Array operator+(const Array &x, double value) {
        Array result(x);
        result += value;
        return result;
      }

      // returns b + A
      friend Array operator+(double value, const Array &x) {
        return x + value;
      }

      // returns A * b
      friend Array operator*(const Array &x, double value) {
        Array result(x);
        result *= value;
        return result;
      }

      // returns b * A
      friend Array operator*(double value, const Array &x) {
        return x * value;
      }

      // returns A - b
      friend Array operator-(const Array &x, double value) {
        Array result(x);
        result -= value;
        return result;
      }

      // returns b - A
      friend Array operator-(double value, const Array &x) {
        return (-1 * x) + value;
      }

      // returns A / b
      friend Array operator/(const Array &x, double value) {
        Array result(x);
        result /= value;
        return result;
      }

      // returns b / A
      friend Array operator/(double value, const Array &x) {
        Array result(x);
        for (auto &d : result.data_) { d = value / d; }
        return result;
      }

      // R = A + B
      friend Array operator+(const Array &x, const Array &y) {
        assert(x.shape_equal(y));
        Array result(x.dims());
        for (size_t i = 0; i < x.size(); ++i) {
          result.data_[i] = x.data_[i] + y.data_[i];
        }
        return result;
      }

      // R = A - B
      friend Array operator-(const Array &x, const Array &y) {
        assert(x.shape_equal(y));
        Array result(x.dims());
        for (size_t i = 0; i < x.size(); ++i) {
          result.data_[i] = x.data_[i] - y.data_[i];
        }
        return result;
      }

      // R = A * B (elementwise)
      friend Array operator*(const Array &x, const Array &y) {
        assert(x.shape_equal(y));
        Array result(x.dims());
        for (size_t i = 0; i < x.size(); ++i) {
          result.data_[i] = x.data_[i] * y.data_[i];
        }
        return result;
      }

      // R = A / B (elementwise)
      friend Array operator/(const Array &x, const Array &y) {
        assert(x.shape_equal(y));
        Array result(x.dims());
        for (size_t i = 0; i < x.size(); ++i) {
          result.data_[i] = x.data_[i] / y.data_[i];
        }
        return result;
      }

    // Other friend functions
    public:
      friend double sum(const Array &array) {
        double result = 0;
        for (auto &value: array.data_) {
          result += value;
        }
        return result;
      }

      friend double max(const Array &array) {
        double result = std::numeric_limits<double>::lowest();
        for (auto &value : array.data_) {
          if (value > result) {
            result = value;
          }
        }
        return result;
      }

      friend double min(const Array &array) {
        double result = std::numeric_limits<double>::max();
        for (auto &value : array.data_) {
          if (value < result) {
            result = value;
          }
        }
        return result;
      }

      friend Array abs(const Array &array) {
        return apply(array, fabs);
      }

      friend Array round(const Array &array) {
        return apply(array, std::round);
      }

      friend Array exp(const Array &array) {
        return apply(array, std::exp);
      }

      friend Array log(const Array &array) {
        return apply(array, std::log);
      }

      friend Array psi(const Array &array) {
        return apply(array, gsl_sf_psi);
      }

      friend Array normalize(const Array &array) {
        Array result(array);
        result /= sum(result);
        return result;
      }

      friend Array normalizeExp(const Array &array) {
        return normalize(exp(array - max(array)));
      }

      friend double logSumExp(const Array &array) {
        double x_max = max(array);
        return x_max + log(sum(exp(array - x_max)));
      }

      friend double klDiv(const Array &x, const Array &y) {
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

    // Input - Output
    public:

      friend std::ostream &operator<<(std::ostream &out,
                                      const Array &array) {
        out << std::setprecision(DEFAULT_PRECISION) << std::fixed;
        for (auto &value : array.data_) {
          out << value << std::endl;
        }
        return out;
      }

      friend std::istream &operator>>(std::istream &in, Array &array) {
        for (auto &value : array.data_) {
          in >> value;
        }
        return in;
      }

      virtual void save(const std::string &filename) const {
        std::ofstream ofs(filename);
        if (ofs.is_open()) {
          ofs << dims_.size() << std::endl;
          for (auto &d : dims_) {
            ofs << d << std::endl;
          }
          ofs << *this;
          ofs.close();
        }
      }

      static Array load(const std::string &filename) {
        Array result;
        std::ifstream ifs(filename);
        if (ifs.is_open()) {
          size_t __ndims__, temp;
          size_t length = 1;
          // Read dimension
          ifs >> __ndims__;
          for (size_t i = 0; i < __ndims__; ++i) {
            ifs >> temp;
            length *= temp;
            result.dims_.push_back(temp);
          }
          // Allocate memory
          result.data_.resize(length);
          ifs >> result;
          ifs.close();
        }
        return result;
      }

    // Matrix related:
    public:
      size_t num_rows() const { return dims_[0]; }

      size_t num_cols() const {  assert(ndims() > 1); return dims_[1]; }

      friend Array sum(const Array &array, size_t dim) {
        assert(array.ndims() == 2);
        Array result;
        /*
        if (dim == 0) {
          result = Array({array.num_cols()}, 0);
          for (size_t j = 0; j < array.num_cols(); ++j) {
            result(j) = sum(array.col(j));
          }
        } else {
          result = Array({array.num_rows()}, 0);
          for (size_t i = 0; i < array.num_rows(); ++i) {
            result(i) = sum(array.row(i));
          }
        }
         */
        return result;
      }

/*
      Slice row(size_t row_id){
        assert(ndims() == 2);
        return Slice(this, row_id, dim(0), dim(1));
      }


      const Slice row(size_t row_id) const{
        assert(ndims() == 2);
        return Slice(this, row_id, dim(0), dim(1));
      }
*/
/*
      Slice col(size_t  col_id){
        assert(ndims() == 2);
        return Slice(this, col_id * dims_[0], 1, dims_[1]);
      }
*/

    public:
      std::vector<double> data_;
      std::vector<size_t> dims_;
  };

  class Vector : public Array {
    public:
      Vector(){}

      Vector(size_t length, double value = 0) : Array({length}, value) { }

      Vector(const std::initializer_list<double> &values)
              : Array({values.size()}, values) {}

      Vector(const Array &array) {
        assert(array.ndims() == 1);
        data_ = array.data_;
        dims_ = array.dims_;
      }

      Vector(Array &&array) {
        assert(array.ndims() == 1);
        data_ = std::move(array.data_);
        dims_ = std::move(array.dims_);
      }

      static Vector flatten(const Array& array){
        Vector result(array.size());
        result.data_ = array.data_;
        return result;
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
      // Matrix - Vector Product
      friend double dot(const Vector &x, const Vector &y) {
        assert(x.size() == y.size());
        return sum(x * y);
      }
  };

  class Matrix : public Array {
    public:
      Matrix(){}

      Matrix(size_t num_rows, size_t num_cols, double value = 0)
              : Array({num_rows, num_cols}, value) {}

      Matrix(size_t num_rows, size_t num_cols,
             const std::initializer_list<double> &values)
              : Array({num_rows, num_cols}, values) {}

      Matrix(const Array &array) {
        assert(array.ndims() == 2);
        data_ = array.data_;
        dims_ = array.dims_;
      }

      Matrix(Array &&array) {
        assert(array.ndims() == 2);
        data_ = std::move(array.data_);
        dims_ = std::move(array.dims_);
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

      friend Matrix transpose(const Array &M) {
        assert(M.ndims() == 2);
        Matrix result(M.num_cols(), M.num_rows());
        for (unsigned i = 0; i < M.num_rows(); ++i) {
          for (unsigned j = 0; j < M.num_cols(); ++j) {
            result(j, i) = M(i, j);
          }
        }
        return result;
      }

      // Matrix - Matrix Product
      friend Matrix dot(const Matrix &x, const Matrix &y,
                        bool x_transpose = false, bool y_transpose = false){

        assert(x.ndims() == 2);
        assert(y.ndims() == 2);
        CBLAS_TRANSPOSE x_trans = CblasNoTrans;
        size_t M = x.num_rows();
        size_t K = x.num_cols();
        if (x_transpose) {
          x_trans = CblasTrans;
          M = x.num_cols();
          K = x.num_rows();
        }

        CBLAS_TRANSPOSE y_trans = CblasNoTrans;
        size_t N = y.num_cols();
        if (y_transpose) {
          y_trans = CblasTrans;
          N = y.num_rows();
        }

        Matrix result(M, N);

        cblas_dgemm(CblasColMajor, x_trans, y_trans,
                    M, N, K,
                    1.0, x.data(), x.num_rows(),
                    y.data(), y.num_rows(),
                    0.0, result.data(), result.num_rows());

        return result;
      }

      // Matrix - Vector Product
      friend Vector dot(const Matrix &X, const Vector &y,
                        bool x_transpose = false){
        assert(X.ndims() == 2);
        assert(y.ndims() == 1);
        size_t M = X.num_rows();
        CBLAS_TRANSPOSE x_trans = CblasNoTrans;
        if (x_transpose) {
          x_trans = CblasTrans;
          M = X.num_cols();
        }

        Vector result(M);
        cblas_dgemv(CblasColMajor, x_trans,
                    X.num_rows(), X.num_cols(), 1.0, X.data(), X.num_rows(),
                    y.data(), 1,
                    0.0, result.data(), 1);

        return result;
      }

      friend std::ostream &operator<<(std::ostream &out,
                                      const Matrix &mat) {
        out << std::setprecision(DEFAULT_PRECISION) << std::fixed;
        for(size_t i=0; i < mat.num_rows(); ++i) {
          for(size_t j=0; j < mat.num_cols(); ++j){
            out << mat(i,j) << " ";
          }
          out << std::endl;
        }
        return out;
      }
  };

}

#endif