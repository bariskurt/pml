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

      friend Matrix apply(const Matrix &m, double (*func)(double)){
        Matrix result(m.shape());
        for(size_t i=0; i < m.size(); ++i)
          result[i] = func(m[i]);
        return result;
      }

      void apply(double (*func)(double)){
        for(double &d : data_)
          d = func(d);
      }

    public:

      friend bool operator==(const Matrix &x, double v) {
        for(double value : x)
          if( !fequal(value, v) )
            return false;
        return true;
      }

      friend bool operator!=(const Matrix &x, double v) {
        return !(x == v);
      }

      friend bool operator==(const Matrix &x, const Matrix &y) {
        if(x.shape() != y.shape())
          return false;
        for(size_t i = 0; i < x.size(); ++i)
          if( !fequal(x[i], y[i]) )
            return false;
        return true;
      }

      friend bool operator!=(const Matrix &x, const Matrix &y) {
        return !(x == y);
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
      Vector::view col(size_t col_num){
        return Vector::view(&data_[col_num * nrows_], nrows_);
      }

      Vector::const_view col(size_t col_num) const{
        return Vector::const_view(&data_[col_num * nrows_], nrows_);
      }

      Vector::view row(size_t row_num){
        return Vector::view(&data_[row_num], ncols_, nrows_);
      }

      Vector::const_view row(size_t row_num) const{
        return Vector::const_view(&data_[row_num], ncols_, nrows_);
      }

      // Returns several columns as Matrix
      Matrix cols(Range range) const {
        Matrix result;
        for(size_t i = range.start; i < range.stop; i+=range.step){
          result.append(col(i));
        }
        return result;
      }

      // Append vector
      void append(const Vector::const_view &cv, size_t axis = 1){
        ASSERT_TRUE(axis == 0 || axis == 1,
                    "Matrix::append(const Matrix &):: axis out of bounds");
        if(axis == 1) {
          if (empty()) {
            nrows_ = cv.size();
          } else {
            ASSERT_TRUE(nrows_ == cv.size(),
                        "Matrix::append:: Vector size mismatch");
          }
          for (auto it = cv.begin(); it != cv.end(); ++it)
            data_.push_back(*it);
          ncols_++;
        } else{
          if(empty()) {
            for (auto it = cv.begin(); it != cv.end(); ++it)
              data_.push_back(*it);
            ncols_ = cv.size();
          } else {
            ASSERT_TRUE(ncols_ == cv.size(),
                        "Matrix::append:: Vector size mismatch");
            Matrix temp(nrows_+1, ncols_);
            double *temp_data = temp.data();
            auto it = cv.begin();
            for(size_t i=0; i < ncols_; ++i){
              memcpy(&temp_data[i * (nrows_+1)], &data_[i * nrows_],
                     sizeof(double) * nrows_);
              temp(nrows_, i) = *it++;
            }
            data_ = std::move(temp.data_);
          }
          nrows_++;
        }
      }


      // Appends a column to the right.
      void append(const Matrix &m, size_t axis = 1){
        ASSERT_TRUE(axis == 0 || axis == 1,
                    "Matrix::append(const Matrix &):: axis out of bounds");
        if(m.empty())
          return;
        if(axis == 0){
          // Append rowwise
          ASSERT_TRUE(empty() || ncols() == m.ncols(),
              "Matrix::append(const Matrix &, 1):: column sizes mismatch.");
          std::vector<double> new_data(size() + m.size());
          size_t nrows_new = nrows() + m.nrows();
          for(size_t i=0; i < ncols(); ++i){
            memcpy(&new_data[nrows_new*i], &data_[nrows() * i],
                   sizeof(double) * nrows());
            memcpy(&new_data[nrows_new*i + nrows()], &m.data_[m.nrows() * i],
                   sizeof(double) * m.nrows());
          }
          data_ = std::move(new_data);
          nrows_ = nrows_new;
        } else {
          // Append columnwise
          ASSERT_TRUE(empty() || nrows() == m.nrows(),
              "Matrix::append(const Matrix &, 0):: row sizes mismatch.");
          std::vector<double> new_data(size() + m.size());
          memcpy(new_data.data(), data(), sizeof(double) * size());
          memcpy(new_data.data() + size(), m.data(), sizeof(double) * m.size());
          data_ = std::move(new_data);
          ncols_ += m.ncols();
        }
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

      void save(const std::string &filename){
        std::ofstream ofs(filename, std::ios::binary | std::ios::out);
        if (ofs.is_open()) {
          double dim = 2;
          double dim1 = nrows(), dim2 = ncols();
          ofs.write(reinterpret_cast<char*>(&dim), sizeof(double));
          ofs.write(reinterpret_cast<char*>(&dim1), sizeof(double));
          ofs.write(reinterpret_cast<char*>(&dim2), sizeof(double));
          ofs.write(reinterpret_cast<char*>(data()), sizeof(double)*size());
          ofs.close();
        }
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

      static Matrix load(const std::string &filename){
        Matrix result;
        std::ifstream ifs(filename, std::ios::binary | std::ios::in);
        if (ifs.is_open()) {
          double dim, nrows, ncols;
          ifs.read(reinterpret_cast<char*>(&dim), sizeof(double));
          ASSERT_TRUE(dim == 2, "Matrix::load:: Dimension mismatch.");
          ifs.read(reinterpret_cast<char*>(&nrows), sizeof(double));
          ifs.read(reinterpret_cast<char*>(&ncols), sizeof(double));
          result.reshape(nrows, ncols);
          ifs.read(reinterpret_cast<char*>(result.data()),
                   sizeof(double)*result.size());
          ifs.close();
        }
        return result;
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

  // Concatanate two matrices as in Matlab
  inline Matrix cat(const Matrix &m1, const Matrix &m2, size_t axis = 1){
    Matrix result(m1);
    result.append(m2, axis);
    return result;
  }

  // Flip Left-Right
  inline Matrix fliplr(const Matrix &x){
    Matrix result;
    for(size_t i = x.ncols() ; i > 0; --i){
      result.append(x.col(i-1));
    }
    return result;
  }

  // Flip Up-Down
  inline Matrix flipud(const Matrix &x){
    Matrix result;
    for(size_t i = x.nrows() ; i > 0; --i){
      result.append(x.row(i-1), 0);
    }
    return result;
  }

  // Returns a flat vector from Matrix x
  inline Vector flatten(const Matrix &x){
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
  inline Matrix repmat(const Vector &x, int n, int m ){
    // Prepare initial column.
    Vector initial_column;
    for(int i=0; i < n; ++i)
      initial_column.append(x);
    // Replicate initial_column to form result.
    Matrix result;
    for(int i=0; i < m; ++i)
      result.append(initial_column);
    return result;
  }

  // Copies column x, n times along the axis.
  // tile(x, n, 0)  --> appendRow(x) n times
  // tile(x, n, 1)  --> appendColumn(x) n times
  inline Matrix tile(const Vector &x, size_t n, int axis = 0){
    ASSERT_TRUE(axis==0 || axis==1, "Matrix::tile axis out of bounds.");
    Matrix result;
    for(size_t i = 0; i < n; ++i){
      if ( axis == 0)
        result.append(x, 0);
      else
        result.append(x);
    }
    return result;
  }

  inline Vector for_each_rows(const Matrix& x,
                              double (*func)(const Vector::const_view &)){
    Vector result(x.nrows());
    for(size_t i=0; i < x.nrows(); ++i){
      result[i] = func(x.row(i));
    }
    return result;
  }

  inline Vector for_each_cols(const Matrix& x,
                              double (*func)(const Vector::const_view &)){
    Vector result(x.ncols());
    for(size_t i=0; i < x.ncols(); ++i){
      result[i] = func(x.col(i));
    }
    return result;
  }

  // Sum
  inline double sum(const Matrix  &x){
    return std::accumulate(x.begin(), x.end(), 0.0);
  }

  inline Vector sum(const Matrix &x, int axis) {
    ASSERT_TRUE(axis==0 || axis==1, "Matrix::sum axis out of bounds.");
    Vector result;
    if (axis == 0)
      return for_each_cols(x, sum);
    return for_each_rows(x, sum);
  }

  //Min
  inline double min(const Matrix &x) {
    return *(std::min_element(x.begin(), x.end()));
  }

  inline Vector min(const Matrix &x, int axis) {
    ASSERT_TRUE(axis==0 || axis==1, "Matrix::min axis out of bounds.");
    if (axis == 0)
      return for_each_cols(x, min);
    return for_each_rows(x, min);
  }


  // Max
  inline double max(const Matrix &x) {
    return *(std::max_element(x.begin(), x.end()));
  }

  inline Vector max(const Matrix &x, int axis) {
    ASSERT_TRUE(axis==0 || axis==1, "Matrix::max axis out of bounds.");
    if (axis == 0)
      return for_each_cols(x, max);
    return for_each_rows(x, max);
  }

  inline double mean(const Matrix &x){
    return sum(x) / x.size();
  }

  inline Vector mean(const Matrix &x, int axis){
    ASSERT_TRUE(axis==0 || axis==1, "Matrix::max axis out of bounds.");
    if (axis == 0)
      return sum(x, 0) / x.nrows();
    return sum(x,1) / x.ncols();
  }

  // Absolute value of x
  inline Matrix abs(const Matrix &x){
    return apply(x, std::fabs);
  }

  // Round to nearest integer
  inline Matrix round(const Matrix &x){
    return apply(x, std::round);
  }

  // Ceiling
  inline Matrix ceil(const Matrix &x){
    return apply(x, std::ceil);
  }

  // Floor
  inline Matrix floor(const Matrix &x){
    return apply(x, std::floor);
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
  inline Matrix normalize(const Matrix  &x, int axis = 2) {
    ASSERT_TRUE(axis>=0 && axis<=2, "Matrix::normalize axis out of bounds.");
    if( axis == 0){
      Matrix col_sums = tile(sum(x,0), x.nrows());
      return x / col_sums;
    }
    if( axis == 1){
      Matrix row_sums = tile(sum(x,1), x.ncols(), 1);
      return x / row_sums;
    }
    return x / sum(x);
  }

  // Safe  NormalizeExp
  inline Matrix normalizeExp(const Matrix &x, int axis = 2) {
    ASSERT_TRUE(axis>=0 && axis<=2, "Matrix::normalizeExp axis out of bounds.");
    if( axis == 0){
      Matrix max_cols = tile(max(x,0), x.nrows());
      return normalize(exp(x - max_cols), 0);
    }
    if( axis == 1){
      Matrix max_rows = tile(max(x,1), x.ncols(), 1);
      return normalize(exp(x - max_rows),1);
    }
    return normalize(exp(x - max(x)));
  }


  // Safe LogSumExp(x)
  inline double logSumExp(const Matrix &x) {
    double x_max = max(x);
    return x_max + std::log(sum(exp(x - x_max)));
  }

  inline Vector logSumExp(const Matrix &x, int axis) {
    ASSERT_TRUE(axis==0 || axis==1, "Matrix::logSumExp axis out of bounds.");
    if(axis == 0){
      Vector col_max = max(x,0);
      return col_max + log(sum(exp(x-tile(col_max, x.nrows())),0));
    }
    Vector row_max = max(x,1);
    return row_max + log(sum(exp(x-tile(row_max, x.ncols(), 1)),1));
  }

  double kl_div(const Matrix &x, const Matrix &y){
    return kl_div(flatten(x), flatten(y));
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