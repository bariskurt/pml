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


  inline void ASSERT_TRUE(bool condition, const std::string &message) {
    if( !condition ) {
      std::cout << "FATAL ERROR: " << message << std::endl;
      exit(-1);
    }
  }


  struct Range{
    Range(int start_, int stop_, int step_=1)
            : start(start_), stop(stop_), step(step_){}
    int start;
    int stop;
    int step;
  };


  class Vector {
    public:
      Vector(){}

      explicit Vector(size_t length, double value = 0)
              : data_(length, value) {}

      Vector(size_t length, double *values)
              : data_(length) {
        memcpy(this->data(), values, sizeof(double)*length);
      }

      Vector(const std::initializer_list<double> &values)
              : data_(values) {}

      explicit Vector(Range range){
        for(double d = range.start; d < range.stop; d+=range.step){
          data_.push_back(d);
        }
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
        return data_.size();
      }

      bool empty() const {
        return data_.empty();
      }

      virtual size_t ndims() const {
        return 1;
      }

      friend bool similar(const Vector &x, const Vector &y){
        return (x.ndims() == y.ndims()) && (x.size() == y.size());
      }

      void resize(size_t new_size){
        data_.resize(new_size);
      }

    public:
      void append(double value){
        data_.push_back(value);
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
        return data_.begin();
      }

      std::vector<double>::const_iterator begin() const{
        return data_.cbegin();
      }

      std::vector<double>::iterator end(){
        return data_.end();
      }

      std::vector<double>::const_iterator end() const {
        return data_.cend();
      }

      inline double &operator()(const size_t i0) {
        return data_[i0];
      }

      inline double operator()(const size_t i0) const {
        return data_[i0];
      }

      double *data() {
        return &data_[0];
      }

      const double *data() const {
        return &data_[0];
      }

      double first() const {
        return data_.front();
      }

      double &first() {
        return data_.front();
      }

      double last() const {
        return data_.back();
      }

      double &last() {
        return data_.back();
      }

    public:
      void apply(double (*func)(double)) {
        for (auto &value : data_) {
          value = func(value);
        }
      }

      friend Vector apply(const Vector &x, double (*func)(double)) {
        Vector result(x);
        result.apply(func);
        return x;
      }

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
        assert(similar(*this, other));
        for (size_t i = 0; i < data_.size(); ++i) {
          data_[i] += other.data_[i];
        }
      }

      // A = A - B
      void operator-=(const Vector &other) {
        assert(similar(*this, other));
        for (size_t i = 0; i < data_.size(); ++i) {
          data_[i] -= other.data_[i];
        }
      }

      // A = A * B (elementwise)
      void operator*=(const Vector &other) {
        assert(similar(*this, other));
        for (size_t i = 0; i < data_.size(); ++i) {
          data_[i] *= other.data_[i];
        }
      }

      // A = A / B (elementwise)
      void operator/=(const Vector &other) {
        assert(similar(*this, other));
        for (size_t i = 0; i < data_.size(); ++i) {
          data_[i] /= other.data_[i];
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
          result.data_[i] = x.data_[i] + y.data_[i];
        }
        return result;
      }

      // R = A - B
      friend Vector operator-(const Vector &x, const Vector &y) {
        assert(similar(x,y));
        Vector result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
          result.data_[i] = x.data_[i] - y.data_[i];
        }
        return result;
      }

      // R = A * B (elementwise)
      friend Vector operator*(const Vector &x, const Vector &y) {
        assert(similar(x,y));
        Vector result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
          result.data_[i] = x.data_[i] * y.data_[i];
        }
        return result;
      }

      // R = A / B (elementwise)
      friend Vector operator/(const Vector &x, const Vector &y) {
        assert(similar(x,y));
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
          out << value << std::endl;
        }
        return out;
      }

      friend std::istream &operator>>(std::istream &in, Vector &x) {
        for (auto &value : x) {
          in >> value;
        }
        return in;
      }

      virtual void saveTxt(const std::string &filename) const {
        std::ofstream ofs(filename);
        if (ofs.is_open()) {
          ofs << 1 << std::endl;
          ofs << size() << std::endl;
          ofs << *this;
          ofs.close();
        }
      }

      static Vector loadTxt(const std::string &filename) {
        Vector result;
        std::ifstream ifs(filename);
        size_t buffer;
        if (ifs.is_open()) {
          // Read dimension
          ifs >> buffer;
          assert(buffer == 1);
          ifs >> buffer;
          // Allocate memory
          result.data_.resize(buffer);
          ifs >> result;
          ifs.close();
        }
        return result;
      }

    public:
      std::vector<double> data_;
  };

  class Matrix : public Vector{

    public:
      enum Axes{
          LINEAR=-1,
          COLS,
          ROWS
      };

    public:
      Matrix() : nrows_(0), ncols_(0) {}

      Matrix(size_t num_rows, size_t num_cols, double value = 0)
              : Vector(num_rows * num_cols, value),
                nrows_(num_rows), ncols_(num_cols) {}

      Matrix(size_t num_rows, size_t num_cols, double *values)
              : Vector(num_rows * num_cols),
                nrows_(num_rows), ncols_(num_cols) {
        memcpy(this->data(), values, sizeof(double)*size());
      }

      Matrix(size_t num_rows, size_t num_cols,
             const std::initializer_list<double> &values)
              : Vector(values), nrows_(num_rows), ncols_(num_cols) {}


      Matrix(std::pair<size_t, size_t> shape, double value = 0):
          Matrix(shape.first, shape.second, value){}

      Matrix(std::pair<size_t, size_t> shape, double* values):
          Matrix(shape.first, shape.second, values){}

      Matrix(std::pair<size_t, size_t> shape,
             const std::initializer_list<double> &values):
          Matrix(shape.first, shape.second, values){}

    public:
      size_t nrows() const{
        return nrows_;
      }

      size_t ncols() const{
        return ncols_;
      }

      size_t ndims() const override {
        return 2;
      }

      void reshape(size_t new_nrows, size_t new_ncols){
        nrows_ = new_nrows;
        ncols_ = new_ncols;
        data_.resize(nrows_ * ncols_);
      }

      std::pair<size_t, size_t> shape() const{
        return {nrows_, ncols_};
      };

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

      using Vector::operator();

      inline double &operator()(const size_t i0, const size_t i1) {
        return data_[i0 + nrows_ * i1];
      }

      inline double operator()(const size_t i0, const size_t i1) const {
        return data_[i0 + nrows_ * i1];
      }

      friend bool similar(const Matrix &x, const Matrix &y){
        return (x.nrows() == y.nrows()) && (x.ncols() == y.ncols());
      }

      Vector getColumn(size_t col_num) const {
        Vector column(nrows_);
        memcpy(column.data(), &data_[col_num * nrows_],
               sizeof(double) * nrows_);
        return column;
      }

      Matrix getColumns(Range range) const {
        Matrix result;
        for(int i = range.start; i < range.stop; i+=range.step){
          result.appendColumn(getColumn(i));
        }
        return result;
      }

      void setColumn(size_t col_num, const Vector &vector) {
        memcpy(&data_[col_num * nrows_], vector.data(),
               sizeof(double) * nrows_);
      }

      Vector getRow(size_t row_num) const {
        Vector row(ncols_);
        size_t idx = row_num;
        for(size_t i=0; i < ncols_; ++i){
          row(i) = data_[idx];
          idx += nrows_;
        }
        return row;
      }

      void setRow(size_t row_num, const Vector &row) {
        size_t idx = row_num;
        for(size_t i=0; i < ncols_; ++i){
          data_[idx] = row(i);
          idx += nrows_;
        }
      }

      void appendColumn(const Vector &v){
        if(empty()){
          nrows_ = v.size();
        } else {
          assert(nrows_ == v.size());
        }
        data_.insert(data_.end(), v.begin(), v.end());
        ncols_++;
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
          result.data_[i] = x.data_[i] + y.data_[i];
        }
        return result;
      }

      // R = A - B
      friend Matrix operator-(const Matrix &x, const Matrix &y) {
        assert(similar(x,y));
        Matrix result(x.nrows(), x.ncols());
        for (size_t i = 0; i < x.size(); ++i) {
          result.data_[i] = x.data_[i] - y.data_[i];
        }
        return result;
      }

      // R = A * B (elementwise)
      friend Matrix operator*(const Matrix &x, const Matrix &y) {
        assert(similar(x,y));
        Matrix result(x.nrows(), x.ncols());
        for (size_t i = 0; i < x.size(); ++i) {
          result.data_[i] = x.data_[i] * y.data_[i];
        }
        return result;
      }

      // R = A / B (elementwise)
      friend Matrix operator/(const Matrix &x, const Matrix &y) {
        assert(similar(x,y));
        Matrix result(x.nrows(), x.ncols());
        for (size_t i = 0; i < x.size(); ++i) {
          result.data_[i] = x.data_[i] / y.data_[i];
        }
        return result;
      }

      // R = A + [b b ... b]
      friend Matrix operator+(const Matrix &x, const Vector &v) {
        assert(x.nrows() == v.size());
        Matrix result(x.shape());
        for (size_t i = 0; i < x.nrows(); ++i) {
          for (size_t j = 0; j < x.ncols(); ++j) {
            result(i,j) = x(i,j) + v(i);
          }
        }
        return result;
      }

      // R = A - [v v ... v]
      friend Matrix operator-(const Matrix &x, const Vector &v) {
        assert(x.nrows() == v.size());
        Matrix result(x.shape());
        for (size_t i = 0; i < x.nrows(); ++i) {
          for (size_t j = 0; j < x.ncols(); ++j) {
            result(i,j) = x(i,j) - v(i);
          }
        }
        return result;
      }

      // R = A * [b b ... b]
      friend Matrix operator*(const Matrix &x, const Vector &v) {
        assert(x.nrows() == v.size());
        Matrix result(x.shape());
        for (size_t i = 0; i < x.nrows(); ++i) {
          for (size_t j = 0; j < x.ncols(); ++j) {
            result(i,j) = x(i,j) * v(i);
          }
        }
        return result;
      }

      // R = A / [v v ... v]
      friend Matrix operator/(const Matrix &x, const Vector &v) {
        assert(x.nrows() == v.size());
        Matrix result(x.shape());
        for (size_t i = 0; i < x.nrows(); ++i) {
          for (size_t j = 0; j < x.ncols(); ++j) {
            result(i,j) = x(i,j) / v(i);
          }
        }
        return result;
      }

    public:
      // Load and Save
      friend std::ostream &operator<<(std::ostream &out,
                                      const Matrix &x) {
        out << std::setprecision(DEFAULT_PRECISION) << std::fixed;
        for(size_t i=0; i < x.nrows(); ++i) {
          for(size_t j=0; j < x.ncols(); ++j){
            out << x(i,j) << " ";
          }
          out << std::endl;
        }
        return out;
      }

      void saveTxt(const std::string &filename) const override  {
        std::ofstream ofs(filename);
        if (ofs.is_open()) {
          ofs << 2 << std::endl;
          ofs << nrows() << std::endl;
          ofs << ncols() << std::endl;
          for(auto &value : data_)
            ofs << value << std::endl;
          ofs.close();
        }
      }

      static Matrix loadTxt(const std::string &filename) {
        Matrix result;
        std::ifstream ifs(filename);
        size_t buffer;
        if (ifs.is_open()) {
          // Read dimension
          ifs >> buffer;
          assert(buffer == 2);
          ifs >> result.nrows_;
          ifs >> result.ncols_;
          // Allocate memory
          result.data_.resize(result.nrows_ * result.ncols_ );
          ifs >> result;
          ifs.close();
        }
        return result;
      }

    private:
      size_t nrows_;
      size_t ncols_;
  };

  // --------------- Tensor3D ---------------
  class Tensor3D : public Vector {

    public:
      Tensor3D() : Vector(), size0_(0), size1_(0), size2_(0) {}

      Tensor3D(size_t s0, size_t s1, size_t s2, double value = 0)
              : Vector(s0 * s1 * s2, value),
                size0_(s0), size1_(s1), size2_(s2) {}

      Tensor3D(size_t s0, size_t s1, size_t s2, double *values)
              : Vector(s0 * s1 * s2, values),
                size0_(s0), size1_(s1), size2_(s2) {}

      static Tensor3D ones(size_t size0_, size_t size1_, size_t size2_) {
        return Tensor3D(size0_, size1_, size2_, 1.0);
      }

      static Tensor3D zeros(size_t size0_, size_t size1_, size_t size2_) {
        return Tensor3D(size0_, size1_, size2_, 0.0);
      }

    public:
      size_t dim0() const { return size0_; }

      size_t dim1() const { return size1_; }

      size_t dim2() const { return size2_; }

      std::tuple<size_t, size_t, size_t> shape() const {
        return std::make_tuple(size0_, size1_, size2_);
      }

    public:
      using Vector::operator();

      inline double& operator()(const size_t i0, const size_t i1,
                                const size_t i2) {
        return data_[i0 + size0_ * (i1 + size1_ * i2)];
      }
      inline double operator()(const size_t i0, const size_t i1,
                               const size_t i2) const{
        return data_[i0 + size0_ * (i1 + size1_ * i2)];
      }

    public:
      void saveTxt(const std::string &filename) const {
        std::ofstream ofs(filename);
        ASSERT_TRUE(ofs.is_open(), "Tensor::Save cannot open file");
        ofs << 3 << std::endl;
        ofs << size0_ << std::endl;
        ofs << size1_ << std::endl;
        ofs << size2_ << std::endl;
        for(auto &value : data_){
          ofs << value << std::endl;
        }
        ofs.close();
      }

      static Tensor3D loadTxt(const std::string &filename) {
        std::ifstream ifs(filename);
        ASSERT_TRUE(ifs.is_open(), "Tensor::Load cannot open file");
        size_t dim, s0, s1, s2;
        ifs >> dim;
        ASSERT_TRUE(dim==3, "Tensor::Load dimension must be 3");
        ifs >> s0 >> s1 >> s2;
        Tensor3D result(s0,s1,s2);
        for(auto &value: result){
          ifs >> value;
        }
        ifs.close();
        return result;
      }

      friend std::ostream &operator<<(std::ostream &out,
                                      const Tensor3D &tensor) {
        out << std::setprecision(DEFAULT_PRECISION) << std::fixed;
        for (size_t k = 0; k < tensor.dim2(); ++k) {
          for (size_t i = 0; i < tensor.dim0(); ++i) {
            for (size_t j = 0; j < tensor.dim1() ; ++j) {
              out << tensor(i, j, k) << " ";
            }
            out << std::endl;
          }
          out << std::endl;
        }
        return out;
      }

    public:

      friend bool operator==(const Tensor3D &t1, const Tensor3D &t2) {
        if(t1.shape() == t2.shape()){
          return operator==((Vector) t1, (Vector) t2);
        }
        return false;
      }


    public:
      friend Tensor3D operator+(const Tensor3D &x, const Tensor3D &y) {
        ASSERT_TRUE(x.shape() == y.shape(),
                    "Tensor3D::operator+ matrices must be of similar shape ");
        Tensor3D result(x.dim0(),x.dim1(),x.dim2());
        for(size_t i=0; i < x.size(); ++i){
          result(i) = x(i) + y(i);
        }
        return result;
      }

      friend Tensor3D operator-(const Tensor3D &x, const Tensor3D &y) {
        ASSERT_TRUE(x.shape() == y.shape(),
                    "Tensor3D::operator+ matrices must be of similar shape ");
        Tensor3D result(x.dim0(),x.dim1(),x.dim2());
        for(size_t i=0; i < x.size(); ++i){
          result(i) = x(i) - y(i);
        }
        return result;
      }

      friend Tensor3D operator*(const Tensor3D &x, const Tensor3D &y) {
        ASSERT_TRUE(x.shape() == y.shape(),
                    "Tensor3D::operator+ matrices must be of similar shape ");
        Tensor3D result(x.dim0(),x.dim1(),x.dim2());
        for(size_t i=0; i < x.size(); ++i){
          result(i) = x(i) * y(i);
        }
        return result;
      }

      friend Tensor3D operator/(const Tensor3D &x, const Tensor3D &y) {
        ASSERT_TRUE(x.shape() == y.shape(),
                    "Tensor3D::operator+ matrices must be of similar shape ");
        Tensor3D result(x.dim0(),x.dim1(),x.dim2());
        for(size_t i=0; i < x.size(); ++i){
          result(i) = x(i) / y(i);
        }
        return result;
      }


      friend Tensor3D operator+(const Tensor3D &x, double value) {
        Tensor3D result(x);
        result += value;
        return result;
      }

      friend Tensor3D operator+(double value, const Tensor3D &x) {
        return x + value;
      }

      friend Tensor3D operator-(const Tensor3D &x, double value) {
        Tensor3D result(x);
        result -= value;
        return result;
      }

      friend Tensor3D operator-(double value, const Tensor3D &x) {
        return -1*x + value;
      }

      friend Tensor3D operator*(const Tensor3D &x, double value) {
        Tensor3D result(x);
        result *= value;
        return result;
      }

      friend Tensor3D operator*(double value, const Tensor3D &x) {
        return x * value;
      }

      friend Tensor3D operator/(const Tensor3D &x, double value) {
        Tensor3D result(x);
        result /= value;
        return result;
      }

      friend Tensor3D operator/(double value, const Tensor3D &x) {
        Tensor3D temp(x.dim0(),x.dim1(),x.dim2(), value);
        return temp / x;
      }

    public:

      Matrix getSlice(size_t index) const {
        Matrix m(size0_,size1_);
        std::memcpy(m.data(), &data_[index*size0_*size1_],
                    sizeof(double)*size0_*size1_);
        return m;
      }

      void setSlice(size_t index, const Vector& x){
        ASSERT_TRUE(x.size() == size0_ * size1_,
                    "Vector length does not match with the matrix length");
        std::memcpy(&data_[index*size0_*size1_], x.data(),
                      sizeof(double)*size0_*size1_);
      }

      void appendSlice(const Matrix &matrix){
        if(empty()){
          size0_ = matrix.nrows();
          size1_ = matrix.ncols();
        } else {
          ASSERT_TRUE((size0_ == matrix.nrows()) && (size1_ == matrix.ncols()),
                      "Tensor3D::appendSlice(): Slice dimensions mismatch");
        }
        data_.insert(data_.end(), matrix.begin(), matrix.end());
        size2_++;
      }

    public:
      size_t size0_, size1_, size2_;
  };


  // Inline functions
  inline Matrix transpose(const Matrix &m){
    Matrix result(m.ncols(), m.nrows());
    for (unsigned i = 0; i < m.nrows(); ++i) {
      for (unsigned j = 0; j < m.ncols(); ++j) {
        result(j, i) = m(i, j);
      }
    }
    return result;
  }

  inline double sum(const Vector &x){
    double result = 0;
    for (auto &value: x) {
      result += value;
    }
    return result;
  }

  inline Vector sum(const Matrix &m, int axis) {
    Vector result;
    if(axis == Matrix::COLS){
      result = Vector::zeros(m.ncols());
      for(size_t i=0; i < m.ncols(); ++i){
        result(i) = sum(m.getColumn(i));
      }
    } else {
      result = Vector::zeros(m.nrows());
      for(size_t i=0; i < m.nrows(); ++i){
        result(i) = sum(m.getRow(i));
      }
    }
    return result;
  }

  Vector sumCols(const Matrix &m){
    return sum(m, Matrix::COLS);
  }

  Vector sumRows(const Matrix &m){
    return sum(m, Matrix::ROWS);
  }

  inline double max(const Vector &x) {
    double result = std::numeric_limits<double>::lowest();
    for (auto &value : x) {
      if (value > result) {
        result = value;
      }
    }
    return result;
  }

  inline Vector max(const Matrix &m, int axis) {
    Vector result;
    if(axis == Matrix::COLS){
      result = Vector::zeros(m.ncols());
      for(size_t i=0; i < m.ncols(); ++i){
        result(i) = max(m.getColumn(i));
      }
    } else {
      result = Vector::zeros(m.nrows());
      for(size_t i=0; i < m.nrows(); ++i){
        result(i) = max(m.getRow(i));
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

  inline Vector min(const Matrix &m, int axis) {
    Vector result;
    if(axis == Matrix::COLS){
      result = Vector::zeros(m.ncols());
      for(size_t i=0; i < m.ncols(); ++i){
        result(i) = min(m.getColumn(i));
      }
    } else {
      result = Vector::zeros(m.nrows());
      for(size_t i=0; i < m.nrows(); ++i){
        result(i) = min(m.getRow(i));
      }
    }
    return result;
  }

  inline double mean(const Vector &x){
    return sum(x) / x.size();
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

  inline Vector psi_n(int n, const Vector &x){
    Vector y(x.size());
    for(size_t i=0; i<y.size(); i++) {
      y(i) = gsl_sf_psi_n(n, x(i));
    }
    return y;
  }

  inline Matrix psi_n(int n, const Matrix &x){
    Matrix y(x.nrows(), x.nrows());
    for(size_t i=0; i<y.size(); i++) {
      y(i) = gsl_sf_psi_n(n, x(i));
    }
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

  inline Matrix normalize(const Matrix &m, int axis = Matrix::LINEAR) {
    Matrix result;
    if( axis == Matrix::LINEAR){
      result = m / sum(m);
    } else if(axis == Matrix::COLS){
      result = Matrix(m.nrows(), m.ncols());
      for(size_t i = 0; i<m.ncols(); i++){
        result.setColumn(i, normalize(m.getColumn(i)));
      }
    } else {
      result = Matrix(m.nrows(), m.ncols());
      for(size_t i = 0; i<m.nrows(); i++){
        result.setRow(i, normalize(m.getRow(i)));
      }
    }
    return result;
  }

  inline Matrix normalizeCols(const Matrix &m){
    return normalize(m, Matrix::COLS);
  }

  inline Matrix normalizeRows(const Matrix &m){
    return normalize(m, Matrix::ROWS);
  }

  inline Vector normalizeExp(const Vector &x) {
    return normalize(exp(x - max(x)));
  }

  inline Matrix normalizeExp(const Matrix &m, int axis = Matrix::LINEAR) {
    Matrix result;
    if( axis == Matrix::LINEAR){
      result = normalize(exp(m - max(m)));
    } else if(axis == 0){
      result = Matrix(m.nrows(), m.ncols());
      for(size_t i = 0; i<m.ncols(); i++){
        result.setColumn(i, normalizeExp(m.getColumn(i)));
      }
    } else {
      result = Matrix(m.nrows(), m.ncols());
      for(size_t i = 0; i<m.nrows(); i++){
        result.setRow(i, normalizeExp(m.getRow(i)));
      }
    }
    return result;
  }

  inline double logSumExp(const Vector &x) {
    double x_max = max(x);
    return x_max + std::log(sum(exp(x - x_max)));
  }

  inline Vector logSumExp(const Matrix &m, size_t dim) {
    Vector result;
    if(dim == 0){
      result = Vector(m.ncols());
      for(size_t i = 0; i<m.ncols(); i++){
        result(i) = logSumExp(m.getColumn(i));
      }
    } else {
      result = Vector(m.nrows());
      for(size_t i = 0; i<m.nrows(); i++){
        result(i) = logSumExp(m.getRow(i));
      }
    }
    return result;
  }

  inline Matrix tile(const Vector &x, size_t n, int axis = Matrix::COLS){
    Matrix result;
    if(axis == Matrix::COLS){
      result = Matrix(x.size(), n);
      for(size_t i = 0; i < result.ncols(); ++i){
        result.setColumn(i, x);
      }
    } else {
      result = Matrix(n, x.size());
      for(size_t i = 0; i < result.nrows(); ++i){
        result.setRow(i, x);
      }
    }
    return result;
  }

  inline double kl_div(const Vector &x, const Vector &y) {
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

  // Matrix - Vector Product
  inline double dot(const Vector &x, const Vector &y) {
    return sum(x * y);
  }

  // Matrix - Vector Product
  inline Vector dot(const Matrix &X, const Vector &y,
                    bool x_transpose = false){
    size_t M = X.nrows();
    CBLAS_TRANSPOSE x_trans = CblasNoTrans;
    if (x_transpose) {
      x_trans = CblasTrans;
      M = X.ncols();
    }

    Vector result(M);
    cblas_dgemv(CblasColMajor, x_trans,
                X.nrows(), X.ncols(), 1.0, X.data(), X.nrows(),
                y.data(), 1,
                0.0, result.data(), 1);

    return result;
  }

  // Matrix - Matrix Product
  inline Matrix dot(const Matrix &x, const Matrix &y,
                    bool x_transpose = false, bool y_transpose = false){
    CBLAS_TRANSPOSE x_trans = CblasNoTrans;
    size_t M = x.nrows();
    size_t K = x.ncols();
    if (x_transpose) {
      x_trans = CblasTrans;
      M = x.ncols();
      K = x.nrows();
    }

    CBLAS_TRANSPOSE y_trans = CblasNoTrans;
    size_t N = y.ncols();
    if (y_transpose) {
      y_trans = CblasTrans;
      N = y.nrows();
    }

    Matrix result(M, N);

    cblas_dgemm(CblasColMajor, x_trans, y_trans,
                M, N, K,
                1.0, x.data(), x.nrows(),
                y.data(), y.nrows(),
                0.0, result.data(), result.nrows());

    return result;
  }

  // Slices
  inline Vector slice(const Vector &v, size_t start,
                      size_t length, size_t step=1 ){
    Vector vslice(length);
    size_t idx = start;
    for(size_t i = 0; i< length; ++i){
      vslice(i) = v(idx);
      idx+=step;
    }
    return vslice;
  }

}

#endif
