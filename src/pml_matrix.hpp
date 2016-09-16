#ifndef PML_MATRIX_H_
#define PML_MATRIX_H_

#include "pml_vector.hpp"

extern "C" {
// LU decomoposition of a general matrix
void dgetrf_(int*, int*, double*, int*, int*, int*);

// generate inverse of a matrix given its LU decomposition
void dgetri_(int*, double*, int*, int*, double*, int*, int* );
}

namespace pml {

  class Matrix {

    public:

      // Empty Matrix
      Matrix() : nrows_(0), ncols_(0) { }

      // Matrix with given size and default value.
      Matrix(size_t num_rows, size_t num_cols, double value = 0)
          : nrows_(num_rows), ncols_(num_cols),
            data_(num_rows * num_cols, value) {}

      // Matrix with given dimensions and array.
      // Matrix is stored in column major order.
      Matrix(size_t num_rows, size_t num_cols, const double *values)
          : nrows_(num_rows), ncols_(num_cols), data_(num_rows * num_cols) {
        memcpy(this->data(), values, sizeof(double) * size());
      }

      // Matrix with given size and array.
      // Matrix is stored in column major order.
      Matrix(size_t num_rows, size_t num_cols,
             const std::initializer_list<double> &values)
          : nrows_(num_rows), ncols_(num_cols), data_(values)  {}

      // Zero Matrix with given shape
      Matrix(std::pair<size_t, size_t> shape, double value = 0) :
          Matrix(shape.first, shape.second, value) { }

      // Matrix with given shape and values
      Matrix(std::pair<size_t, size_t> shape, double *values) :
          Matrix(shape.first, shape.second, values) { }

      // Matrix with given shape and values
      Matrix(std::pair<size_t, size_t> shape,
             const std::initializer_list<double> &values) :
          Matrix(shape.first, shape.second, values) { }

    public:
      // Zeros Matrix
      static Matrix zeros(size_t num_rows, size_t num_cols) {
        return Matrix(num_rows, num_cols, 0.0);
      }

      // Ones Matrix
      static Matrix ones(size_t num_rows, size_t num_cols) {
        return Matrix(num_rows, num_cols, 1.0);
      }

      static Matrix identity(size_t size) {
        Matrix X = Matrix::zeros(size, size);
        for (size_t i = 0; i < size; ++i)
          X(i, i) = 1.0;
        return X;
      }

    public:
      // Get dimensions
      size_t nrows() const{
        return nrows_;
      }

      size_t ncols() const{
        return ncols_;
      }

      size_t size() const{
        return data_.size();
      }

      // Get shape
      std::pair<size_t, size_t> shape() const{
        return {nrows_, ncols_};
      };

      // Reshape. Reshaping invalidates all data.
      void reshape(size_t new_nrows, size_t new_ncols){
        nrows_ = new_nrows;
        ncols_ = new_ncols;
        data_.resize(nrows_ * ncols_);
      }

      // Checks empty.
      bool empty() const {
        return data_.empty();
      }

      friend bool operator==(const Matrix &x, double v) {
        for(double d : x)
          if(!fequal(d, v)) return false;
        return true;
      }

      friend bool operator==(const Matrix &x, const Matrix &y) {
        // Check sizes
        if (x.shape() != y.shape()) return false;
        // Check element-wise
        for (size_t i = 0; i < x.size(); ++i) {
          if (!fequal(x(i), y(i))) return false;
        }
        return true;
      }

    public:
      // -------- Iterators ---------

      std::vector<double>::iterator begin() {
        return data_.begin();
      }

      std::vector<double>::const_iterator begin() const {
        return data_.cbegin();
      }

      std::vector<double>::iterator end() {
        return data_.end();
      }

      std::vector<double>::const_iterator end() const {
        return data_.cend();
      }

    public:
      // Apply x[i] = f(x[i]) to each element.
      // Function signature: double f(double x)
      void apply(double (*func)(double)) {
        for (auto &value : data_) {
          value = func(value);
        }
      }

      // Apply f to a new Vector
      friend Matrix apply(const Matrix &x, double (*func)(double)) {
        Matrix result(x);
        result.apply(func);
        return result;
      }

    public:

      // -------- Accessors ---------
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

      inline double &operator()(const size_t i0, const size_t i1) {
        return data_[i0 + nrows_ * i1];
      }

      inline double operator()(const size_t i0, const size_t i1) const {
        return data_[i0 + nrows_ * i1];
      }

      double* data() {
        return data_.data();
      }

      const double *data() const {
        return data_.data();
      }


    public:

      // ------- Self-Assignment Operations ------

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
      void operator+=(const Matrix &other) {
        ASSERT_TRUE(shape() == other.shape(),
                    "Matrix::operator+=:: Shape mismatch.");
        for (size_t i = 0; i < data_.size(); ++i) {
          data_[i] += other[i];
        }
      }

      // A = A - B
      void operator-=(const Matrix &other) {
        ASSERT_TRUE(shape() == other.shape(),
                    "Matrix::operator-=:: Shape mismatch.");
        for (size_t i = 0; i < data_.size(); ++i) {
          data_[i] -= other[i];
        }
      }

      // A = A * B (elementwise)
      void operator*=(const Matrix &other) {
        ASSERT_TRUE(shape() == other.shape(),
                    "Matrix::operator*=:: Shape mismatch.");
        for (size_t i = 0; i < data_.size(); ++i) {
          data_[i] *= other[i];
        }
      }

      // A = A / B (elementwise)
      void operator/=(const Matrix &other) {
        ASSERT_TRUE(shape() == other.shape(),
                    "Matrix::operator/=:: Shape mismatch.");
        for (size_t i = 0; i < data_.size(); ++i) {
          data_[i] /= other[i];
        }
      }

      // ------- Matrix-Double Operations -------

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

      // ----------- Matrix - Matrix operations --------

      // R = A + B
      friend Matrix operator+(const Matrix &x, const Matrix &y) {
        ASSERT_TRUE(x.shape() == x.shape(),
                    "Matrix::operator+:: Shape mismatch.");
        Matrix result(x.shape());
        for (size_t i = 0; i < x.size(); ++i) {
          result[i] = x[i] + y[i];
        }
        return result;
      }

      // R = A - B
      friend Matrix operator-(const Matrix &x, const Matrix &y) {
        ASSERT_TRUE(x.shape() == x.shape(),
                    "Matrix::operator-:: Shape mismatch.");
        Matrix result(x.shape());
        for (size_t i = 0; i < x.size(); ++i) {
          result[i] = x[i] - y[i];
        }
        return result;
      }

      // R = A * B (elementwise)
      friend Matrix operator*(const Matrix &x, const Matrix &y) {
        ASSERT_TRUE(x.shape() == x.shape(),
                    "Matrix::operator*:: Shape mismatch.");
        Matrix result(x.shape());
        for (size_t i = 0; i < x.size(); ++i) {
          result[i] = x[i] * y[i];
        }
        return result;
      }

      // R = A / B (elementwise)
      friend Matrix operator/(const Matrix &x, const Matrix &y) {
        ASSERT_TRUE(x.shape() == x.shape(),
                    "Matrix::operator/:: Shape mismatch.");
        Matrix result(x.shape());
        for (size_t i = 0; i < x.size(); ++i) {
          result[i] = x[i] / y[i];
        }
        return result;
      }

      // ------- Matrix - Vector Operations --------

      // R = A + [b b ... b]
      friend Matrix operator+(const Matrix &x, const Vector &v) {
        ASSERT_TRUE(x.nrows_ == v.size(),
                    "Matrix::operator+:: Vector size mismatch.");
        Matrix result(x.shape());
        for (size_t i = 0; i < x.nrows_; ++i) {
          for (size_t j = 0; j < x.ncols_; ++j) {
            result(i,j) = x(i,j) + v[i];
          }
        }
        return result;
      }

      // R = A - [v v ... v]
      friend Matrix operator-(const Matrix &x, const Vector &v) {
        ASSERT_TRUE(x.nrows_ == v.size(),
                    "Matrix::operator-:: Vector size mismatch.");
        Matrix result(x.shape());
        for (size_t i = 0; i < x.nrows_; ++i) {
          for (size_t j = 0; j < x.ncols_; ++j) {
            result(i,j) = x(i,j) - v[i];
          }
        }
        return result;
      }

      // R = A * [b b ... b]
      friend Matrix operator*(const Matrix &x, const Vector &v) {
        ASSERT_TRUE(x.nrows_ == v.size(),
                    "Matrix::operator*:: Vector size mismatch.");
        Matrix result(x.shape());
        for (size_t i = 0; i < x.nrows_; ++i) {
          for (size_t j = 0; j < x.ncols_; ++j) {
            result(i,j) = x(i,j) * v[i];
          }
        }
        return result;
      }

      // R = A / [v v ... v]
      friend Matrix operator/(const Matrix &x, const Vector &v) {
        ASSERT_TRUE(x.nrows_ == v.size(),
                    "Matrix::operator/:: Vector size mismatch.");
        Matrix result(x.shape());
        for (size_t i = 0; i < x.nrows_; ++i) {
          for (size_t j = 0; j < x.ncols_; ++j) {
            result(i,j) = x(i,j) / v[i];
          }
        }
        return result;
      }

      // --------- Row and Column Operations -----------

    public:
      // Returns a single column as Vector
      Vector getColumn(size_t col_num) const {
        Vector column(nrows_);
        memcpy(column.data(), &data_[col_num * nrows_],
               sizeof(double) * nrows_);
        return column;
      }

      // Returns several columns as Matrix
      Matrix getColumns(Range range) const {
        Matrix result;
        for(size_t i = range.start; i < range.stop; i+=range.step){
          result.appendColumn(getColumn(i));
        }
        return result;
      }

      // Sets a single column
      void setColumn(size_t col_num, const Vector &v) {
        ASSERT_TRUE(col_num < ncols(),
                    "Matrix::setColumn:: col_num exceeds number of columns");
        ASSERT_TRUE(nrows() == v.size(),
                    "Matrix::setColumn:: Vector size mismatch");
        memcpy(&data_[col_num * nrows_], v.data(), sizeof(double) * nrows_);
      }

      // Returns a single row as vector
      Vector getRow(size_t row_num) const {
        Vector row(ncols_);
        size_t idx = row_num;
        for(size_t i=0; i < ncols_; ++i){
          row(i) = data_[idx];
          idx += nrows_;
        }
        return row;
      }

      // Sets a single row.
      void setRow(size_t row_num, const Vector &row) {
        ASSERT_TRUE(ncols_ == row.size(),
                    "Matrix::setRow:: Vector size mismatch");
        size_t idx = row_num;
        for(size_t i=0; i < ncols_; ++i){
          data_[idx] = row(i);
          idx += nrows_;
        }
      }

      // Appends a column to the right.
      void appendColumn(const Vector &v){
        ASSERT_TRUE( empty() | (nrows_ == v.size()),
                    "Matrix::appendColumn:: Vector size mismatch");
        if(empty()){
          nrows_ = v.size();
        }
        data_.insert(data_.end(), v.begin(), v.end());
        ncols_++;
      }

      // Appends a row to the bottom.
      void appendRow(const Vector &v){
        ASSERT_TRUE(empty() | (ncols_ == v.size()),
                    "Matrix::appendRow:: Vector size mismatch");
        if(empty()) {
          data_.insert(data_.end(), v.begin(), v.end());
          ncols_ = v.size();
        } else {
          Matrix temp(nrows_+1, ncols_);
          double *temp_data = temp.data();
          for(size_t i=0; i < ncols_; ++i){
            memcpy(&temp_data[i * (nrows_+1)], &data_[i * nrows_],
                   sizeof(double) * nrows_);
            temp(nrows_, i) = v(i);
          }
          data_ = std::move(temp.data_);
        }
        nrows_++;
      }

      // ---------- File Operations --------

    public:
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

      friend std::istream &operator>>(std::istream &in, Matrix &x) {
        for (auto &value : x) {
          in >> value;
        }
        return in;
      }

      void saveTxt(const std::string &filename) const {
        std::ofstream ofs(filename);
        if (ofs.is_open()) {
          ofs << 2 << std::endl;          // dimension
          ofs << nrows() << std::endl;    // num. rows
          ofs << ncols() << std::endl;    // num. cols
          for(auto &value : data_)        // values
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
          ASSERT_TRUE(buffer == 2, "Matrix::loadTxt:: dimension mismatch");
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
      std::vector<double> data_;

  };

  // Returns a flat vector from Matrix x
  Vector vectorize(const Matrix &x){
    return Vector(x.size(), x.data());
  }

  // Transpose
  inline Matrix transpose(const Matrix &m){
    Matrix result(m.ncols(), m.nrows());
    for (size_t i = 0; i < m.nrows(); ++i)
      for (size_t j = 0; j < m.ncols(); ++j)
        result(j, i) = m(i, j);
    return result;
  }

  // Inverse : will be added later in linear algebra package
  /*
  Matrix inv(const Matrix &matrix) {
    Matrix result(matrix);
    int N = matrix.nrows();
    int *IPIV = new int[N+1];
    int LWORK = N*N;
    double *WORK = new double[LWORK];
    int INFO;
    dgetrf_(&N,&N,result.data(),&N,IPIV,&INFO);
    dgetri_(&N,result.data(),&N,IPIV,WORK,&LWORK,&INFO);
    delete IPIV;
    delete WORK;
    return result;
  }
  */

  // repmat function of Matlab
  Matrix repmat(const Vector &x, int n, int m ){
    // Prepare initial column.
    Vector initial_column;
    for(int i=0; i < n; ++i)
      initial_column.append(x);
    // Replicate initial_column to form result.
    Matrix result;
    for(int i=0; i < m; ++i)
      result.appendColumn(initial_column);
    return result;
  }

  // Copies column x, n times
  Matrix tileCols(const Vector &x, size_t n){
    Matrix result;
    for(size_t i = 0; i < n; ++i)
      result.appendColumn(x);
    return result;
  }

  // Copies row x, n times
  Matrix tileRows(const Vector &x, size_t n){
    Matrix result = Matrix(n, x.size());
    for(size_t i = 0; i < n; ++i){
      result.setRow(i, x);
    }
    return result;
  }

  Vector for_each_rows(const Matrix& x, double (*func)(const Vector &)){
    Vector result;
    for(size_t i=0; i < x.nrows(); ++i){
      result.append(func(x.getRow(i)));
    }
    return result;
  }

  Vector for_each_cols(const Matrix& x, double (*func)(const Vector &)){
    Vector result;
    for(size_t i=0; i < x.ncols(); ++i){
      result.append(func(x.getColumn(i)));
    }
    return result;
  }

  // Sum
  inline double sum(const Matrix  &x){
    return std::accumulate(x.begin(), x.end(), 0.0);
  }

  inline Vector sumCols(const Matrix &x) {
    return for_each_cols(x, sum);
  }

  inline Vector sumRows(const Matrix &x) {
    return for_each_rows(x, sum);
  }

  //Min
  inline double min(const Matrix &x) {
    return *(std::min_element(x.begin(), x.end()));
  }

  inline Vector minCols(const Matrix &x) {
    return for_each_cols(x, min);
  }

  inline Vector minRows(const Matrix &x) {
    return for_each_rows(x, min);
  }

  // Max
  inline double max(const Matrix &x) {
    return *(std::max_element(x.begin(), x.end()));
  }

  inline Vector maxCols(const Matrix &x) {
    return for_each_cols(x, max);
  }

  inline Vector maxRows(const Matrix &x) {
    return for_each_rows(x, max);
  }

  // Absolute value of x
  inline Matrix abs(const Matrix &x){
    return apply(x, std::fabs);
  }

  // Round to nearest integer
  inline Matrix round(const Matrix &x){
    return apply(x, std::round);
  }

  // Log Gamma function.
  inline Matrix lgamma(const Matrix &x){
    return apply(x, std::lgamma);
  }

  // Polygamma Function.
  inline Matrix psi(const Matrix &x, int n = 0){
    Matrix y(x.shape());
    for(size_t i=0; i < y.size(); i++) {
      y(i) = gsl_sf_psi_n(n, x(i));
    }
    return y;
  }

  // Exponential
  inline Matrix exp(const Matrix &x){
    return apply(x, std::exp);
  }

  // Logarithm
  inline Matrix log(const Matrix &x){
    return apply(x, std::log);
  }

  // Normalize
  inline Matrix normalize(const Matrix  &x) {
    return x / sum(x);
  }

  inline Matrix normalizeCols(const Matrix &x) {
    Matrix col_sums = tileRows(sumCols(x), x.nrows());
    return x / col_sums;
  }

  inline Matrix normalizeRows(const Matrix &x) {
    Matrix row_sums = tileCols(sumRows(x), x.ncols());
    return x / row_sums;
  }

  // Safe  NormalizeExp
  inline Matrix normalizeExp(const Matrix &x) {
    return normalize(exp(x - max(x)));
  }

  inline Matrix normalizeExpCols(const Matrix &x) {
    Matrix max_cols = tileRows(maxCols(x), x.nrows());
    return normalizeCols(exp(x - max_cols));
  }

  inline Matrix normalizeExpRows(const Matrix &x) {
    Matrix max_rows = tileCols(maxRows(x), x.ncols());
    return normalizeRows(exp(x - max_rows));
  }

  // Safe LogSumExp(x)
  inline double logSumExp(const Matrix &x) {
    double x_max = max(x);
    return x_max + std::log(sum(exp(x - x_max)));
  }

  inline Vector logSumExpCols(const Matrix &x) {
    Vector col_max = maxCols(x);
    return col_max + log(sumCols(exp(x-tileRows(col_max, x.nrows()))));
  }

  inline Vector logSumExpRows(const Matrix &x) {
    Vector row_max = maxRows(x);
    return row_max + log(sumRows(exp(x-tileCols(row_max, x.ncols()))));
  }

  double kl_div(const Matrix &x, const Matrix &y){
    return kl_div(vectorize(x), vectorize(y));
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

} // namespace pml

#endif // PML_MATRIX_H_