//
// Created by baris on 01.04.2016.
//

#ifndef MATLIB_PML_H_H
#define MATLIB_PML_H_H

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <valarray>
#include <tuple>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_sf_psi.h>

#define DEFAULT_PRECISION 6

namespace pml {

  // Helpers:
  inline bool fequal(double a,double b){
    return fabs(a-b) < 1e-6;
  }

  inline void ASSERT_TRUE(bool condition, const std::string &message) {
    if( !condition ) {
      std::cout << "FATAL ERROR: " << message << std::endl;
      exit(-1);
    }
  }

  class Block {

    public:
      Block() {}

      virtual ~Block() {}

      Block(size_t length) : data_(length) {}

      Block(size_t length, const double *initial_values)
              : data_(initial_values, length) {}

      Block(std::initializer_list<double> values) : data_(values) {}

      Block(const std::slice_array<double> &slice) : data_(slice) {}

      Block(const Block &other) : data_(other.data_) {}

      Block(Block &&other) : data_(other.data_) {}


      Block& operator=(const Block &other) {
        data_ = other.data_;
        return *this;
      }

      Block& operator=(Block &&other) {
        data_ = std::move(other.data_);
        return *this;
      }

      virtual double operator=(double d) {
        data_ = d;
        return d;
      }

      void resize(size_t new_size) {
        data_.resize(new_size);
      }

      bool ContainsNan() const{
        for (unsigned i=0; i < data_.size(); ++i) {
          if (isnan(data_[i])) {
            return true;
          }
        }
        return false;
      }

      bool ContainsInf() const{
        for (unsigned i=0; i < data_.size(); ++i) {
          if (isinf(data_[i])) {
            return true;
          }
        }
        return false;
      }

    public:
      double& operator()(const size_t i1) {
        return data_[i1];
      }

      double operator()(const size_t i1) const {
        return data_[i1];
      }

      size_t length() const {
        return data_.size();
      }

      bool empty() const{
        return length() == 0;
      }

      double *data() {
        return &data_[0];
      }

      const double* data() const {
        return &data_[0];
      }

    public:
      void operator+=(double value) {
        this->data_ += value;
      }

      void operator-=(double value) {
        this->data_ -= value;
      }

      void operator*=(double value) {
        this->data_ *= value;
      }

      void operator/=(double value) {
        this->data_ /= value;
      }

      void operator+=(const Block &other) {
        this->data_ += other.data_;
      }

      void operator-=(const Block &other) {
        this->data_ -= other.data_;
      }

      void operator*=(const Block &other) {
        this->data_ *= other.data_;
      }

      void operator/=(const Block &other) {
        this->data_ /= other.data_;
      }

      // Comparison
      friend bool operator==(const Block &b1, const Block &b2) {
        if (b1.length() != b2.length()) {
          return false;
        }
        for (size_t i = 0; i < b1.length(); ++i) {
          if ( !fequal(b1(i), b2(i))) {
            return false;
          }
        }
        return true;
      }

      friend bool operator==(const Block &b1, double d){
        for( auto &val : b1.data_){
          if( !fequal(val, d)){
            return false;
          }
        }
        return true;
      }

      friend bool operator!=(const Block &b1, const Block &b2) {
        return !(b1 == b2);
      };


      // Math
      double min() const{
        return data_.min();
      }

      double max() const{
        return data_.max();
      };

      double sum() const{
        return data_.sum();
      }

      // In-place Operations
      void exp(){
        for( double &val : data_){
          val = std::exp(val);
        }
      }

      void log(){
        for( double &val : data_){
          val = std::log(val);
        }
      }

      void psi(){
        for( double &val : data_){
          val = gsl_sf_psi(val);
        }
      }

      void normalize(){
        data_ /= data_.sum();
      }

      void normalize_exp(){
        data_ -= data_.max();
        for( double &val : data_){
          val = std::exp(val);
        }
        this->normalize();
      }

      // Friend Functions:
      friend double LogSumExp(const Block &x) {
        double x_max = x.max();
        double sum = 0;
        for( auto &val : x.data_){
          sum += std::exp(val - x_max);
        }
        return x_max + std::log(sum);
      }

      friend double KL_Div(const Block &x, const Block &y){
        Block x2(x);
        Block y2(y);
        x2.normalize();
        y2.normalize();
        double result = 0;
        for(unsigned i = 0; i < x2.length(); ++i){
          result += x2(i) * ( std::log(std::fmax(x2(i), 1e-6))
                              - std::log(std::fmax(y2(i), 1e-6)) );
        }
        return result;
      }

      friend double TV_Dist(const Block &x, const Block &y){
        double result = 0;
        for(unsigned i =0; i < x.length(); ++i){
          result += fabs(x(i) - y(i));
        }
        result /= 2;
        return result;
      }

      friend double Sum(const Block &b){
        return b.sum();
      }

      friend double Max(const Block &b){
        return b.max();
      }

      friend double Min(const Block &b){
        return b.min();
      }

    protected:
      std::valarray<double> data_;
    };

  class Vector : public Block {

    friend class Matrix;

    public:
      Vector() : Block(0) {}

      virtual ~Vector() {}

      Vector(size_t length) : Block(length) {}

      Vector(size_t length, const double *initial_values) :
              Block(length, initial_values) {}

      Vector(std::initializer_list<double> values) :
              Block(values) {}

      Vector(const std::slice_array<double> &slice) :
              Block(slice) {}

      Vector(const Vector &other) : Block(other) {}

      Vector(Vector &&other) : Block(std::move(other)) {}

      Vector& operator=(const Vector &other){
        data_ = other.data_;
        return *this;
      }

      Vector& operator=(Vector &&other) {
        data_ = other.data_;
        return *this;
      }

      // Special Vectors
    public:
      static Vector Ones(size_t length){
        Vector result(length);
        result.data_ = 1.0;
        return result;
      }

      static Vector Zeros(size_t length){
        Vector result(length);
        result.data_ = 0.0;
        return result;
      }

      // I/O
    public:
      static Vector Load(const std::string &filename) {
        Vector result;
        std::ifstream ifs(filename);
        if (ifs.is_open()) {
          size_t dim, length;
          ifs >> dim;
          ASSERT_TRUE(dim == 1, "Vector dimension must be 1");
          ifs >> length;
          result.data_.resize(length);
          ifs >> result;
          ifs.close();
        }
        return result;
      }

      // Accessors
      double last() const{
        return data_[length()-1];
      }

      double& last(){
        return data_[length()-1];
      }

      virtual void Save(const std::string &filename) const {
        std::ofstream ofs(filename);
        if (ofs.is_open()) {
          ofs << 1 << std::endl;
          ofs << length() << std::endl;
          ofs << *this;
          ofs.close();
        }
      }

      // Algebra
    public:
      friend Vector operator+(const Vector &x, const Vector &y) {
        Vector result;
        result.data_ = x.data_ + y.data_;
        return result;
      }

      friend Vector operator-(const Vector &x, const Vector &y) {
        Vector result;
        result.data_ = x.data_ - y.data_;
        return result;
      }

      friend Vector operator*(const Vector &x, const Vector &y) {
        Vector result;
        result.data_ = x.data_ * y.data_;
        return result;
      }

      friend Vector operator/(const Vector &x, const Vector &y) {
        Vector result;
        result.data_ = x.data_ / y.data_;
        return result;
      }

      friend Vector operator+(const Vector &x, double d) {
        Vector result;
        result.data_ = x.data_ + d;
        return result;
      }

      friend Vector operator+(double d, const Vector &x) {
        return x+d;
      }

      friend Vector operator-(const Vector &x, double d) {
        Vector result;
        result.data_ = x.data_ - d;
        return result;
      }

      friend Vector operator-(double d, const Vector &x) {
        return -1*x+d;
      }

      friend Vector operator*(const Vector &x, double d) {
        Vector result;
        result.data_ = x.data_ * d;
        return result;
      }

      friend Vector operator*(double d, const Vector &x) {
        return x*d;
      }

      friend Vector operator/(const Vector &x, double d) {
        Vector result;
        result.data_ = x.data_ / d;
        return result;
      }

      friend Vector operator/(double d, const Vector &x) {
        Vector result;
        result.data_ = d/x.data_;
        return result;
      }

      // Matrix - Vector Product
      friend double Dot(const Vector &x, const Vector &y){
        ASSERT_TRUE(x.length() == y.length(),
                    "Dot: Vector lengths must be the same\n");
        double res=0;
        for(size_t i = 0; i < x.length(); ++i){
          res += x(i) * y(i);
        }
        return res;
      }

      // Other Friend Functions
    public:
      friend std::ostream& operator<< (std::ostream &out,
                                       const Vector &vector){
        out << std::setprecision(DEFAULT_PRECISION) << std::fixed;
        for(auto &value : vector.data_){
          out << value << std::endl;
        }
        out << std::endl;
        return out;
      }

      friend std::istream& operator>> (std::istream &in, Vector &vector){
        for(auto &value : vector.data_){
          in >> value;
        }
        return in;
      }

      friend Vector Log(const Vector& v){
        Vector result(v);
        result.log();
        return result;
      }

      friend Vector Exp(const Vector& v){
        Vector result(v);
        result.exp();
        return result;
      }

      friend Vector Psi(const Vector& v){
        Vector result(v);
        result.psi();
        return result;
      }

      friend Vector Normalize(const Vector &v){
        Vector result(v);
        result.normalize();
        return result;
      }

      friend Vector NormalizeExp(const Vector &v){
        Vector result(v);
        result.normalize_exp();
        return result;
      }

  };

  class Matrix : public Block{

    public:
      enum Axes{
          LINEAR,
          COLS,
          ROWS
      };

    public:

      Matrix() : Block(), num_rows_(0), num_cols_(0) {};

      Matrix(size_t num_rows, size_t num_cols)
              : Block(num_rows * num_cols),
                num_rows_(num_rows),
                num_cols_(num_cols) {}

      Matrix(std::pair<size_t, size_t> shape):
              Matrix(shape.first, shape.second){}

      Matrix(size_t num_rows, size_t num_cols, const double *values)
              : Block(num_rows * num_cols, values),
                num_rows_(num_rows),
                num_cols_(num_cols) {}

      Matrix(size_t num_rows, size_t num_cols, double initial_value)
              : Matrix(num_rows, num_cols) {
                data_ = initial_value;
      }

      Matrix(size_t num_rows, size_t num_cols,
                     std::initializer_list<double> values) :
              Block(values),
              num_rows_(num_rows),
              num_cols_(num_cols) {}

      Matrix(const Matrix &matrix)  : Block(matrix),
                                      num_rows_(matrix.num_rows_),
                                      num_cols_(matrix.num_cols_) {}

      Matrix(Matrix &&matrix) : Block(std::move(matrix)),
                                num_rows_(matrix.num_rows_),
                                num_cols_(matrix.num_cols_) {}

      Matrix& operator=(const Matrix &other) {
        if (this != &other) {
          this->data_ = other.data_;
          this->num_cols_ = other.num_cols_;
          this->num_rows_ = other.num_rows_;
        }
        return *this;
      }

      Matrix& operator=(Matrix &&other) {
        if (this != &other) {
          this->data_ = std::move(other.data_);
          this->num_rows_ = other.num_rows_;
          this->num_cols_ = other.num_cols_;
        }
        return *this;
      }

      double operator=(double d) override {
        data_ = d;
        return d;
      }

    public:
      static Matrix Identity(size_t size) {
        Matrix X = Matrix::Zeros(size, size);
        for (size_t i = 0; i < size; ++i) {
          X(i, i) = 1.0;
        }
        return X;
      }

      static Matrix Ones(size_t num_rows, size_t num_cols) {
        return Matrix(num_rows, num_cols, 1.0);
      }

      static Matrix Zeros(size_t num_rows, size_t num_cols) {
        return Matrix(num_rows, num_cols, 0.0);
      }

      // Size
    public:
      size_t num_rows() const { return num_rows_; }

      size_t num_cols() const { return num_cols_; }

      std::pair<size_t, size_t> shape() const {
        return std::make_pair(num_rows_, num_cols_);
      };

      void resize(size_t num_rows, size_t num_cols){
        Block::resize(num_rows * num_cols);
        num_rows_ = num_rows;
        num_cols_ = num_cols;
      }

    public:
      using Block::operator();

      double& operator()(const size_t i1, const size_t i2) {
        return data_[i1 + num_rows_ * i2];
      }

      double operator()(const size_t i1, const size_t i2) const {
        return data_[i1 + num_rows_ * i2];
      }

    public:
      void Save(const std::string &filename) const {
        std::ofstream ofs(filename);
        ASSERT_TRUE(ofs.is_open(),
                    "Matrix::Save cannot open file " + filename);
        ofs << 2 << std::endl;
        ofs << num_rows_ << std::endl;
        ofs << num_cols_ << std::endl;
        for (size_t i = 0; i < length(); ++i) {
          ofs << data_[i] << std::endl;
        }
        ofs.close();
      }

      static Matrix Load(const std::string &filename) {
        std::ifstream ifs(filename);
        ASSERT_TRUE(ifs.is_open(), "Matrix::Load cannot open file " + filename);
        size_t dim, num_rows, num_cols;
        ifs >> dim;
        ASSERT_TRUE(dim == 2, "Matrix::Load dimension must be 2");
        ifs >> num_rows >> num_cols;
        Matrix result(num_rows, num_cols);
        ifs >> result;
        ifs.close();
        return result;
      }


    public:
      using matrix_slices = std::pair<const Matrix* , Axes>;

      matrix_slices rows() const  {
        return std::make_pair(this, Matrix::ROWS);
      }

      matrix_slices cols() const  {
        return std::make_pair(this, Matrix::COLS);
      }

      friend Matrix operator+(matrix_slices list, const Vector& v) {
        const Matrix* mat = list.first;
        Axes axes = list.second;
        Matrix result(mat->shape());
        if (axes == Matrix::ROWS) {
          for (size_t i=0; i< mat->num_rows(); ++i) {
            result.SetRow(i, mat->GetRow(i)+v);
          }
        }
        else if (axes == Matrix::COLS) {
          for (size_t i=0; i< mat->num_cols(); ++i) {
            result.SetColumn(i, mat->GetColumn(i)+v);
          }
        }
        return result;
      }

      friend Matrix operator+(const Vector& v, matrix_slices list){
        return list + v;
      }

      // Other Friend Functions
    public:
      friend std::ostream& operator<< (std::ostream &out, const Matrix &matrix){
        out << std::setprecision(DEFAULT_PRECISION) << std::fixed;
        for(size_t i =0; i < matrix.num_rows(); ++i){
          for(size_t j =0; j < matrix.num_cols(); ++j){
            out << matrix(i,j) << "    ";
          }
          out << std::endl;
        }
        out << std::endl;
        return out;
      }

      friend std::istream& operator>> (std::istream &in, Matrix &matrix){
        for(auto &value : matrix.data_){
          in >> value;
        }
        return in;
      }


    public:
      friend bool operator==(const Matrix &m1, const Matrix &m2) {
        if ( m1.shape() == m2.shape() ) {
          return operator==((Block) m1, (Block) m2);
        }
        return false;
      }

      // Algebra
    public:
      friend Matrix operator+(const Matrix &x, const Matrix &y) {
        Matrix result;
        ASSERT_TRUE(x.shape() == y.shape(),
                    "Matrix::operator+ matrices must be of similar shape ");
        result.data_ = x.data_ + y.data_;
        result.num_rows_ = x.num_rows_;
        result.num_cols_ = x.num_cols_;
        return result;
      }

      friend Matrix operator-(const Matrix &x, const Matrix &y) {
        Matrix result;
        ASSERT_TRUE(x.shape() == y.shape(),
                    "Matrix::operator+ matrices must be of similar shape ");
        result.data_ = x.data_ - y.data_;
        result.num_rows_ = x.num_rows_;
        result.num_cols_ = x.num_cols_;
        return result;
      }

      friend Matrix operator*(const Matrix &x, const Matrix &y) {
        Matrix result;
        ASSERT_TRUE(x.shape() == y.shape(),
                    "Matrix::operator+ matrices must be of similar shape ");
        result.data_ = x.data_ * y.data_;
        result.num_rows_ = x.num_rows_;
        result.num_cols_ = x.num_cols_;
        return result;
      }

      friend Matrix operator/(const Matrix &x, const Matrix &y) {
        Matrix result;
        ASSERT_TRUE(x.shape() == y.shape(),
                    "Matrix::operator+ matrices must be of similar shape ");
        result.data_ = x.data_ / y.data_;
        result.num_rows_ = x.num_rows_;
        result.num_cols_ = x.num_cols_;
        return result;
      }

      friend Matrix operator+(const Matrix &x, double value) {
        Matrix result;
        result.data_ = x.data_ + value;
        result.num_rows_ = x.num_rows_;
        result.num_cols_ = x.num_cols_;
        return result;
      }

      friend Matrix operator+(double value, const Matrix &x) {
        return x + value;
      }

      friend Matrix operator-(const Matrix &x, double value) {
        Matrix result;
        result.data_ = x.data_ - value;
        result.num_rows_ = x.num_rows_;
        result.num_cols_ = x.num_cols_;
        return result;
      }

      friend Matrix operator-(double value, const Matrix &x) {
        return (-1 * x) + value;
      }

      friend Matrix operator*(const Matrix &x, double value) {
        Matrix result;
        result.data_ = x.data_ * value;
        result.num_rows_ = x.num_rows_;
        result.num_cols_ = x.num_cols_;
        return result;
      }
      friend Matrix operator*(double value, const Matrix &x) {
        return x * value;
      }

      friend Matrix operator/(const Matrix &x, double value) {
        Matrix result;
        result.data_ = x.data_ / value;
        result.num_rows_ = x.num_rows_;
        result.num_cols_ = x.num_cols_;
        return result;
      }
      friend Matrix operator/(double value, const Matrix &x) {
        Matrix result;
        result.data_ = value / x.data_;
        result.num_rows_ = x.num_rows_;
        result.num_cols_ = x.num_cols_;
        return result;
      }

    public:
      friend Vector SumRows(const Matrix &M){
        Vector result = Vector::Zeros(M.num_rows());
        for(size_t i = 0; i < M.num_rows(); ++i){
          for(size_t j = 0; j < M.num_cols(); ++j) {
            result(i) += M(i, j);
          }
        }
        return result;
      }

      friend Vector SumCols(const Matrix &M){
        Vector result = Vector::Zeros(M.num_cols());
        for(size_t i = 0; i < M.num_rows(); ++i){
          for(size_t j = 0; j < M.num_cols(); ++j) {
            result(j) += M(i, j);
          }
        }
        return result;
      }

      friend Vector MinCols(const Matrix &M) {
        Vector result(M.num_cols());
        for (unsigned j = 0; j < M.num_cols(); ++j) {
          result(j) = M(0,j);
          for (unsigned i = 1; i < M.num_rows(); ++i) {
            result(j) = std::min(M(i,j), result(j));
          }
        }
        return result;
      }

      friend Vector MaxCols(const Matrix &M) {

        Vector result(M.num_cols());
        for (unsigned j = 0; j < M.num_cols(); ++j) {
          result(j) = M(0,j);
          for (unsigned i = 1; i < M.num_rows(); ++i) {
            result(j) = std::max(M(i,j), result(j));
          }
        }
        return result;
      }

      friend Vector MinRows(const Matrix &M) {
        Vector result(M.num_rows());
        for (unsigned i = 0; i < M.num_rows(); ++i) {
          result(i) = M(i,0);
          for (unsigned j = 1; j < M.num_cols(); ++j) {
            result(i) = std::min(M(i,j), result(i));
          }
        }
        return result;
      }

      friend Vector MaxRows(const Matrix &M) {
        Vector result(M.num_rows());
        for (unsigned i = 0; i < M.num_rows(); ++i) {
          result(i) = M(i,0);
          for (unsigned j = 1; j < M.num_cols(); ++j) {
            result(i) = std::max(M(i,j), result(i));
          }
        }
        return result;
      }

      // Matrix - Matrix Product
      friend Matrix Dot(const Matrix &x, const Matrix &y,
                        bool x_transpose = false, bool y_transpose = false){

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
      friend Vector Dot(const Matrix &X, const Vector &y,
                        bool x_transpose = false){
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

    public:
      friend Matrix Log(const Matrix &M) {
        Matrix result(M);
        result.log();
        return result;
      }

      friend Matrix Exp(const Matrix &M) {
        Matrix result(M);
        result.exp();
        return result;
      }

      friend Matrix Psi(const Matrix &M) {
        Matrix result(M);
        result.psi();
        return result;
      }

      friend Matrix Transpose(const Matrix &M) {
        Matrix result(M.num_cols(), M.num_rows());
        for (unsigned i = 0; i < M.num_rows(); ++i) {
          for (unsigned j = 0; j < M.num_cols(); ++j) {
            result(j, i) = M(i, j);
          }
        }
        return result;
      }

      friend Matrix Multiply(const Matrix &X, const Vector &y, Matrix::Axes axis){
        Matrix result(X.num_rows(), X.num_cols());
        if(axis == Matrix::COLS){
          for (unsigned i = 0; i < X.num_rows(); ++i) {
            for (unsigned j = 0; j < X.num_cols(); ++j) {
              result(i, j) = X(i, j) * y(i);
            }
          }
        } else if(axis == Matrix::ROWS){
          for (unsigned i = 0; i < X.num_rows(); ++i) {
            for (unsigned j = 0; j < X.num_cols(); ++j) {
              result(i, j) = X(i, j) * y(j);
            }
          }
        }
        return result;
      }

      friend Matrix Add(const Matrix &X, const Vector &y, Matrix::Axes axis){
        Matrix result(X.num_rows(), X.num_cols());
        if(axis == Matrix::COLS){
          for (unsigned i = 0; i < X.num_rows(); ++i) {
            for (unsigned j = 0; j < X.num_cols(); ++j) {
              result(i, j) = X(i, j) + y(i);
            }
          }
        } else if(axis == Matrix::ROWS){
          for (unsigned i = 0; i < X.num_rows(); ++i) {
            for (unsigned j = 0; j < X.num_cols(); ++j) {
              result(i, j) = X(i, j) + y(j);
            }
          }
        }
        return result;
      }

      friend Matrix Sub(const Matrix &X, const Vector &y, Matrix::Axes axis){
        Matrix result(X.num_rows(), X.num_cols());
        if(axis == Matrix::COLS){
          for (unsigned i = 0; i < X.num_rows(); ++i) {
            for (unsigned j = 0; j < X.num_cols(); ++j) {
              result(i, j) = X(i, j) - y(i);
            }
          }
        } else if(axis == Matrix::ROWS){
          for (unsigned i = 0; i < X.num_rows(); ++i) {
            for (unsigned j = 0; j < X.num_cols(); ++j) {
              result(i, j) = X(i, j) - y(j);
            }
          }
        }
        return result;
      }

      friend Matrix Divide(const Matrix &X, const Vector &y, Matrix::Axes axis){
        Matrix result(X.num_rows(), X.num_cols());
        if(axis == Matrix::COLS){
          for (unsigned i = 0; i < X.num_rows(); ++i) {
            for (unsigned j = 0; j < X.num_cols(); ++j) {
              result(i, j) = X(i, j) / y(i);
            }
          }
        } else if(axis == Matrix::ROWS){
          for (unsigned i = 0; i < X.num_rows(); ++i) {
            for (unsigned j = 0; j < X.num_cols(); ++j) {
              result(i, j) = X(i, j) / y(j);
            }
          }
        }
        return result;
      }

      friend Matrix Normalize(const Matrix &M,
                              Matrix::Axes axis = Matrix::LINEAR) {
        Matrix result(M);
        if (axis == Matrix::LINEAR) {
          result.data_ = M.data_ / Sum(M);
        } else if (axis == Matrix::COLS) {
          Vector col_sums = SumCols(M);
          for (size_t i = 0; i < M.num_rows(); ++i) {
            for (size_t j = 0; j < M.num_cols(); ++j) {
              result(i,j) /= col_sums(j);
            }
          }
        } else {
          Vector row_sums = SumRows(M);
          for (size_t i = 0; i < M.num_rows(); ++i) {
            for (size_t j = 0; j < M.num_cols(); ++j) {
              result(i,j) /= row_sums(i);
            }
          }
        }
        return result;
      }

      friend Matrix NormalizeExp(const Matrix &M,
                                 Matrix::Axes axis = Matrix::LINEAR) {
        Matrix result(M);
        if (axis == Matrix::LINEAR) {
          result.normalize_exp();
        } else if (axis == Matrix::COLS) {
          for (unsigned j = 0; j < M.num_cols(); ++j) {
            result.SetColumn(j, NormalizeExp(result.GetColumn(j)));
          }
        } else {
          for (unsigned i = 0; i < M.num_rows(); ++i) {
            result.SetRow(i, NormalizeExp(result.GetRow(i)));
          }
        }

        return result;
      }

      friend Vector LogSumExp(const Matrix &M, Matrix::Axes axis) {
        Vector result;
        if (axis == Matrix::COLS) {
          result.resize(M.num_cols());
          for (unsigned j = 0; j < M.num_cols(); ++j) {
            result(j) = LogSumExp(M.GetColumn(j));
          }
        } else if (axis == Matrix::ROWS){
          result.resize(M.num_rows());
          for (unsigned i = 0; i < M.num_rows(); ++i) {
            result(i) = LogSumExp(M.GetRow(i));
          }
        }
        return result;
      }

    public:
      Vector GetColumn(size_t col_num) const {
        Vector column(num_rows_);
        memcpy(column.data(), &data_[col_num * num_rows()], sizeof(double)
                                                            * num_rows());
        return column;
      }

      void SetColumn(size_t col_num, const Vector &vector) {
        memcpy(&data_[col_num * num_rows()], vector.data(),
               sizeof(double) * num_rows());
      }

      Vector GetRow(size_t row_num) const {
        Vector row(num_cols_);
        row.data_ = this->data_[std::slice(row_num, num_cols(), num_rows())];
        return row;
      }

      void SetRow(size_t row_num, const Vector &vector) {
        this->data_[std::slice(row_num, num_cols(), num_rows())] = vector.data_;
      }


  private:
      size_t num_rows_, num_cols_;
  };

  class Tensor3D : public Block {

    public:
      Tensor3D() : Block(), size0_(0), size1_(0), size2_(0) { }

      Tensor3D(size_t s0, size_t s1, size_t s2)
              : Block(s0 * s1 * s2), size0_(s0), size1_(s1), size2_(s2) { }

      Tensor3D(size_t s0, size_t s1, size_t s2, double* values)
              : Block(s0 * s1 * s2, values),
                size0_(s0), size1_(s1), size2_(s2) { }

      Tensor3D(size_t s0, size_t s1, size_t s2, double initial_value)
              : Block(s0 * s1 * s2), size0_(s0), size1_(s1), size2_(s2) {
        data_ = initial_value;
      }

      Tensor3D(const Tensor3D &tensor)
              : Block(tensor), size0_(tensor.size0_),
                size1_(tensor.size1_), size2_(tensor.size2_){ }


      Tensor3D(Tensor3D &&tensor)
              : Block(std::move(tensor)),  size0_(tensor.size0_),
                size1_(tensor.size1_),  size2_(tensor.size2_) { }

      Tensor3D& operator=(const Tensor3D &other) {
        if (this != &other) {
          data_ = other.data_;
          size0_ = other.size0_;
          size1_ = other.size1_;
          size2_ = other.size2_;
        }
        return *this;
      }

      Tensor3D& operator=(Tensor3D &&other) {
        data_ = std::move(other.data_);
        size0_ = other.size0_;
        size1_ = other.size1_;
        size2_ = other.size2_;
        return *this;
      }

    public:

      static Tensor3D Ones(size_t size0_, size_t size1_, size_t size2_) {
        return Tensor3D(size0_, size1_, size2_, 1.0);
      }

      static Tensor3D Zeros(size_t size0_, size_t size1_, size_t size2_) {
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

      using Block::operator();

      inline double& operator()(const size_t i0, const size_t i1,
                                const size_t i2) {
        return data_[i0*size1_*size2_ + i2*size1_ + i1];
      }
      inline const double& operator()(const size_t i0, const size_t i1,
                                      const size_t i2) const{
        return data_[i0*size1_*size2_ + i2*size1_ + i1];
      }

    public:
      void Save(const std::string &filename) const {
        std::ofstream ofs(filename);
        ASSERT_TRUE(ofs.is_open(), "Tensor::Save cannot open file");
        ofs << 3 << std::endl;
        ofs << size0_ << std::endl;
        ofs << size1_ << std::endl;
        ofs << size2_ << std::endl;
        for (size_t i = 0; i < length(); ++i) {
          ofs << data_[i] << std::endl;
        }
        ofs.close();
      }

      static Tensor3D Load(const std::string &filename) {
        std::ifstream ifs(filename);
        ASSERT_TRUE(ifs.is_open(), "Tensor::Load cannot open file");
        size_t dim, s0, s1, s2;
        ifs >> dim;
        ASSERT_TRUE(dim==3, "Tensor::Load dimension must be 3");
        ifs >> s0 >> s1 >> s2;
        Tensor3D result(s0,s1,s2);
        for(size_t i= 0; i < result.length(); ++i){
          ifs >> result(i);
        }
        ifs.close();
        return result;
      }

    public:
      friend std::ostream &operator<<(std::ostream &out,
                                      const Tensor3D &tensor) {
        out << std::setprecision(DEFAULT_PRECISION) << std::fixed;
        for (size_t i = 0; i < tensor.dim0(); ++i) {
          for (size_t j = 0; j < tensor.dim1() ; ++j) {
            for (size_t k = 0; k < tensor.dim2(); ++k) {
              out << tensor(i, j, k) << " ";
            }
            out << std::endl;
          }
          out << std::endl;
        }
        return out;
      }

    public:
      double operator=(double value) override {
        data_ = value;
        return value;
      };

      friend bool operator==(const Tensor3D &t1, const Tensor3D &t2) {
        if(t1.shape() == t2.shape()){
          return operator==((Block) t1, (Block) t2);
        }
        return false;
      }


  public:
      friend Tensor3D operator+(const Tensor3D &x, const Tensor3D &y) {
        Tensor3D result;
        ASSERT_TRUE(x.shape() == y.shape(),
                    "Tensor3D::operator+ matrices must be of similar shape ");
        result.data_ = x.data_ + y.data_;
        result.size0_ = x.size0_;
        result.size1_ = x.size1_;
        result.size2_ = x.size2_;
        return result;
      }

      friend Tensor3D operator-(const Tensor3D &x, const Tensor3D &y) {
        Tensor3D result;
        ASSERT_TRUE(x.shape() == y.shape(),
                    "Tensor3D::operator+ matrices must be of similar shape ");
        result.data_ = x.data_ - y.data_;
        result.size0_ = x.size0_;
        result.size1_ = x.size1_;
        result.size2_ = x.size2_;
        return result;
      }

      friend Tensor3D operator*(const Tensor3D &x, const Tensor3D &y) {
        Tensor3D result;
        ASSERT_TRUE(x.shape() == y.shape(),
                    "Tensor3D::operator+ matrices must be of similar shape ");
        result.data_ = x.data_ * y.data_;
        result.size0_ = x.size0_;
        result.size1_ = x.size1_;
        result.size2_ = x.size2_;
        return result;
      }

      friend Tensor3D operator/(const Tensor3D &x, const Tensor3D &y) {
        Tensor3D result;
        ASSERT_TRUE(x.shape() == y.shape(),
                    "Tensor3D::operator+ matrices must be of similar shape ");
        result.data_ = x.data_ / y.data_;
        result.size0_ = x.size0_;
        result.size1_ = x.size1_;
        result.size2_ = x.size2_;
        return result;
      }


      friend Tensor3D operator+(const Tensor3D &x, double value) {
        Tensor3D result;
        result.data_ = x.data_ * value;
        result.size0_ = x.size0_;
        result.size1_ = x.size1_;
        result.size2_ = x.size2_;
        return result;
      }

      friend Tensor3D operator+(double value, const Tensor3D &x) {
        return x + value;
      }

      friend Tensor3D operator-(const Tensor3D &x, double value) {
        Tensor3D result;
        result.data_ = x.data_ - value;
        result.size0_ = x.size0_;
        result.size1_ = x.size1_;
        result.size2_ = x.size2_;
        return result;
      }

      friend Tensor3D operator-(double value, const Tensor3D &x) {
        return x - value;
      }

      friend Tensor3D operator*(const Tensor3D &x, double value) {
        Tensor3D result;
        result.data_ = x.data_ * value;
        result.size0_ = x.size0_;
        result.size1_ = x.size1_;
        result.size2_ = x.size2_;
        return result;
      }

      friend Tensor3D operator*(double value, const Tensor3D &x) {
        return x * value;
      }

      friend Tensor3D operator/(const Tensor3D &x, double value) {
        Tensor3D result;
        result.data_ = x.data_ / value;
        result.size0_ = x.size0_;
        result.size1_ = x.size1_;
        result.size2_ = x.size2_;
        return result;
      }

      friend Tensor3D operator/(double value, const Tensor3D &x) {
        return x / value;
      }

    public:
      Matrix GetMatrixDir0(size_t index) const {
        double* values = new double[size1_*size2_];
        std::memcpy(values, &data_[index*size1_*size2_],
                    sizeof(double)*size1_*size2_);
        Matrix m(size1_,size2_,values);
        delete[] values;
        return m;
      }
      void SetMatrixDir0(size_t index, const Matrix& matr) {
        std::memcpy(&data_[index*size1_*size2_], matr.data(),
                    sizeof(double)*size1_*size2_);
      }

      void SetMatrixDir0(size_t index, const Vector& vec){
        if (vec.length() == size1_*size2_) {
          std::memcpy(&data_[index*size1_*size2_], vec.data(),
                      sizeof(double)*size1_*size2_);
        }
        else {
          std::cout << "Vector length does not match with the matrix length"
                    << std::endl;
        }
      }

    protected:
      size_t size0_, size1_, size2_;
  };

} // namespace pml

#endif //MATLIB_VECTOR_H_H