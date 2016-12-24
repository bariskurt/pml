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

  class Matrix : public Block {

    public:

      // Empty Matrix
      Matrix() : nrows_(0), ncols_(0) { }

      // Matrix with given size.
      Matrix(size_t num_rows, size_t num_cols)
          : Block(num_rows * num_cols), nrows_(num_rows), ncols_(num_cols) {}

      // Matrix with given size and default value.
      Matrix(size_t num_rows, size_t num_cols, double default_value)
          : Block(num_rows * num_cols), nrows_(num_rows), ncols_(num_cols) {
        fill(default_value);
      }

      // Matrix with given dimensions and array.
      // Matrix is stored in column major order.
      Matrix(size_t num_rows, size_t num_cols, const double *values)
          : Block(num_rows * num_cols), nrows_(num_rows), ncols_(num_cols) {
        std::memcpy(data_, values, sizeof(double) * size_);
      }

      // Matrix with given size and array.
      // Matrix is stored in column major order.
      Matrix(size_t num_rows, size_t num_cols,
             const std::initializer_list<double> &values)
          : nrows_(num_rows), ncols_(num_cols){
        for(const double d : values)
          __push_back__(d);
      }

      // Zero Matrix with given shape
      explicit Matrix(std::pair<size_t, size_t> shape) :
          Matrix(shape.first, shape.second) {}

      // Zero Matrix with given shape
      Matrix(std::pair<size_t, size_t> shape, double value) :
          Matrix(shape.first, shape.second, value) { }

      // Matrix with given shape and values
      Matrix(std::pair<size_t, size_t> shape, double *values) :
          Matrix(shape.first, shape.second, values) { }

      // Matrix with given shape and values
      Matrix(std::pair<size_t, size_t> shape,
             const std::initializer_list<double> &values) :
          Matrix(shape.first, shape.second, values) { }

      // Copy Constructors
      Matrix(const Matrix &other)
          : Block(other), nrows_(other.nrows_), ncols_(other.ncols_){}

      Matrix(Matrix &&other)
        : Block(std::move(other)), nrows_(other.nrows_), ncols_(other.ncols_){
        other.nrows_ = 0;
        other.ncols_ = 0;
      }

      Matrix& operator=(const Matrix &other) {
        Block::operator=(other);
        nrows_ = other.nrows_;
        ncols_ = other.ncols_;
        return *this;
      }

      // Move-Assignment
      Matrix& operator=(Matrix &&other) {
        Block::operator=(std::move(other));
        nrows_ = other.nrows_; other.nrows_ = 0;
        ncols_ = other.ncols_; other.ncols_ = 0;
        return *this;
      }

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
      ConstVectorView const_view() const{
        return ConstVectorView(data_, size_, 1);
      }

      VectorView view() {
        return VectorView(data_, size_, 1);
      }

    public:
      // Get dimensions
      size_t nrows() const{
        return nrows_;
      }

      size_t ncols() const{
        return ncols_;
      }

      // Get shape
      std::pair<size_t, size_t> shape() const{
        return {nrows_, ncols_};
      };

      // Reshape. Reshaping invalidates all data.
      void reshape(size_t new_nrows, size_t new_ncols){
        nrows_ = new_nrows;
        ncols_ = new_ncols;
        __resize__(nrows_ * ncols_);
      }

    public:

      // -------- Accessors ---------
      inline double &operator()(const size_t i0, const size_t i1) {
        return data_[i0 + nrows_ * i1];
      }

      inline double operator()(const size_t i0, const size_t i1) const {
        return data_[i0 + nrows_ * i1];
      }

    public:
      VectorView col(size_t i){
        return VectorView(&data_[i * nrows_], nrows_);
      }

      ConstVectorView col(size_t i) const{
        return ConstVectorView(&data_[i * nrows_], nrows_);
      }

      VectorView row(size_t i){
        return VectorView(&data_[i], ncols_, nrows_);
      }

      ConstVectorView row(size_t i) const{
        return ConstVectorView(&data_[i], ncols_, nrows_);
      }

      // Appends a column to the right.
      void appendColumn(const Vector &v){
        ASSERT_TRUE( empty() | (nrows_ == v.size()),
                     "Matrix::appendColumn:: Vector size mismatch");
        __push_back__(v);
        if(empty())
          nrows_ = size_;
        ++ncols_;
      }

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
        for(size_t i=0; i < x.size(); ++i)
          in >> x[i];
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
          ofs << 2 << std::endl;              // dimension
          ofs << nrows() << std::endl;        // num. rows
          ofs << ncols() << std::endl;        // num. cols
          for(size_t i=0; i < size_; ++i)    // values
            ofs << data_[i] << std::endl;
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
        std::ifstream ifs(filename);
        if (ifs.is_open()) {
          // Read dimensions
          size_t ndims, nrows, ncols;
          ifs >> ndims;
          ASSERT_TRUE(ndims == 2, "Matrix::loadTxt:: dimension mismatch");
          ifs >> nrows;
          ifs >> ncols;
          // Allocate memory
          Matrix result(nrows, ncols);
          ifs >> result;
          ifs.close();
          return result;
        }
        return Matrix(0,0);
      }

    private:
      size_t nrows_;
      size_t ncols_;
  };

  Matrix apply(const Matrix &x, double (*func)(double)){
    Matrix result(x.shape());
    for(size_t i=0; i < x.size(); ++i)
      result[i] = func(x[i]);
    return result;
  }

  Matrix apply(const Matrix &x, double (*func)(double, double), double d){
    Matrix result(x.shape());
    for(size_t i=0; i < x.size(); ++i)
      result[i] = func(x[i], d);
    return result;
  }

  Matrix apply(const Matrix &x, double(*func)(double, double), const Matrix &y){
    ASSERT_TRUE(x.shape() == y.shape(),
           "Matrix::apply == cannot compare matrices of different shape");
    Matrix result(x.shape());
    for(size_t i=0; i < x.size(); ++i)
      result[i] = func(x[i], y[i]);
    return result;
  }

  bool any(const Matrix &m){
    return any(m.const_view());
  }

  bool all(const Matrix &m){
    return all(m.const_view());
  }

  void operator+=(Matrix &m, const double value) {
    m.view() += value;
  }

  // A = A - b
  void operator-=(Matrix &m, const double value) {
    m.view() -= value;
  }

  // A = A * b
  void operator*=(Matrix &m, const double value) {
    m.view() *= value;
  }

  // A = A / b
  void operator/=(Matrix &m, const double value) {
    m.view() /= value;
  }

  // A = A + B
  void operator+=(Matrix &m, const Matrix &other) {
    ASSERT_TRUE(m.shape() == other.shape(),
                "Matrix::operator+=:: Shape mismatch.");
    m.view() += other.const_view();
  }

  // A = A - B
  void operator-=(Matrix &m, const Matrix &other) {
    ASSERT_TRUE(m.shape() == other.shape(),
                "Matrix::operator-=:: Shape mismatch.");
    m.view() -= other.const_view();
  }

  // A = A * B (elementwise)
  void operator*=(Matrix &m, const Matrix &other) {
    ASSERT_TRUE(m.shape() == other.shape(),
                "Matrix::operator*=:: Shape mismatch.");
    m.view() *= other.const_view();
  }

  // A = A / B (elementwise)
  void operator/=(Matrix &m, const Matrix &other) {
    ASSERT_TRUE(m.shape() == other.shape(),
                "Matrix::operator/=:: Shape mismatch.");
    m.view() /= other.const_view();
  }

  // returns A + b
  Matrix operator+(const Matrix &x, double d) {
    return apply(x, [](double d1, double d2) { return d1 + d2;}, d);
  }

  // returns b + A
  Matrix operator+(double d, const Matrix &x) {
    return x + d;
  }

  // returns A * b
  Matrix operator*(const Matrix &x, double d) {
    return apply(x, [](double d1, double d2) { return d1 * d2;}, d);
  }

  // returns b * A
  Matrix operator*(double d, const Matrix &x) {
    return x * d;
  }

  // returns A - b
  Matrix operator-(const Matrix &x, double d) {
    return apply(x, [](double d1, double d2) { return d1 - d2;}, d);
  }

  // returns b - A
  Matrix operator-(double d, const Matrix &x) {
    return apply(x, [](double d1, double d2) { return d2 - d1;}, d);
  }

  // returns A / b
  Matrix operator/(const Matrix &x, double d) {
    return apply(x, [](double d1, double d2) { return d1 / d2;}, d);
  }

  // returns b / A
  Matrix operator/(double d, const Matrix &x) {
    return apply(x, [](double d1, double d2) { return d2 / d1;}, d);
  }

  // R = A + [b b ... b]
  Matrix operator+(const Matrix &x, const Vector &v) {
    ASSERT_TRUE(x.nrows() == v.size(), "Matrix::operator*:: Size mismatch.");
    Matrix result(x);
    for(size_t i=0; i < result.ncols(); ++i)
      result.col(i) += v;
    return result;
  }

  // R = A - [v v ... v]
  Matrix operator-(const Matrix &x, const Vector &v) {
    ASSERT_TRUE(x.nrows() == v.size(), "Matrix::operator*:: Size mismatch.");
    Matrix result(x);
    for(size_t i=0; i < result.ncols(); ++i)
      result.col(i) -= v;
    return result;
  }

  // R = A * [b b ... b]
  Matrix operator*(const Matrix &x, const Vector &v) {
    ASSERT_TRUE(x.nrows() == v.size(), "Matrix::operator*:: Size mismatch.");
    Matrix result(x);
    for(size_t i=0; i < result.ncols(); ++i)
      result.col(i) *= v;
    return result;
  }

  // R = A / [v v ... v]
  Matrix operator/(const Matrix &x, const Vector &v) {
    ASSERT_TRUE(x.nrows() == v.size(), "Matrix::operator/:: Size mismatch.");
    Matrix result(x);
    for(size_t i=0; i < result.ncols(); ++i)
      result.col(i) /= v;
    return result;
  }

  // R = A + B
  Matrix operator+(const Matrix &x, const Matrix &y) {
    ASSERT_TRUE(x.shape() == x.shape(), "Matrix::operator+:: Shape mismatch.");
    return apply(x, [](double d1, double d2) {return d1 + d2;}, y);
  }

  // R = A - B
  Matrix operator-(const Matrix &x, const Matrix &y) {
    ASSERT_TRUE(x.shape() == x.shape(), "Matrix::operator-:: Shape mismatch.");
    return apply(x, [](double d1, double d2) {return d1 - d2;}, y);
  }

  // R = A * B (elementwise)
  Matrix operator*(const Matrix &x, const Matrix &y) {
    ASSERT_TRUE(x.shape() == x.shape(), "Matrix::operator*:: Shape mismatch.");
    return apply(x, [](double d1, double d2) {return d1 * d2;}, y);
  }

  // R = A / B (elementwise)
  Matrix operator/(const Matrix &x, const Matrix &y) {
    ASSERT_TRUE(x.shape() == x.shape(), "Matrix::operator/:: Shape mismatch.");
    return apply(x, [](double d1, double d2) {return d1 / d2;}, y);
  }

  Matrix operator==(const Matrix &x, double v) {
    return apply(x, [](double d1, double d2) -> double {
        return fequal(d1, d2);}, v);
  }

  Matrix operator==(const Matrix &x, const Matrix &y) {
    return apply(x, [](double d1, double d2) -> double {
        return fequal(d1, d2);}, y);
  }

  Matrix operator<(const Matrix &x, double v) {
    return apply(x, [](double d1, double d2) -> double { return d1 < d2;}, v);
  }

  Matrix operator<(const Matrix &x, const Matrix &y) {
    // Check sizes
    ASSERT_TRUE(x.shape() == y.shape(),
                "Matrix::operator== cannot compare matrices of different shape" );
    return apply(x, [](double d1, double d2) -> double { return d1 < d2;}, y);
  }

  Matrix operator>(const Matrix &x, double v) {
    return apply(x, [](double d1, double d2) -> double { return d1 > d2;}, v);
  }

  Matrix operator>(const Matrix &x, const Matrix &y) {
    // Check sizes
    ASSERT_TRUE(x.shape() == y.shape(),
                "Matrix::operator== cannot compare matrices of different shape" );
    return apply(x, [](double d1, double d2) -> double { return d1 > d2;}, y);
  }

  Matrix operator>(double v, const Matrix &x) {
    return x < v;
  }

  Matrix operator<(double v, const Matrix &x) {
    return x > v;
  }

  bool fequal(const Matrix &m1, const Matrix &m2){
    if(m1.shape() != m2.shape())
      return false;
    return fequal(m1.const_view(), m2.const_view());
  }

  // Transpose
  Matrix transpose(const Matrix &m){
    Matrix result(m.ncols(), m.nrows());
    for (size_t i = 0; i < m.nrows(); ++i)
      for (size_t j = 0; j < m.ncols(); ++j)
        result(j, i) = m(i, j);
    return result;
  }

  Matrix tr(const Matrix &m){
    return transpose(m);
  }

  // Returns a flat vector from Matrix x
  Vector flatten(const Matrix &x){
    return Vector(x.size(), x.data());
  }

//-----------------------------------

double sum(const Matrix &x) {
  return sum(x.const_view());
}

Vector sum(const Matrix &x, const size_t axis) {
  // if axis = 0 then sum cols, else sum rows
  ASSERT_TRUE(axis==0 || axis==1, "Matrix::sum axis out of bounds.");
  Vector result;
  for(size_t i=0; i < (axis ? x.nrows() : x.ncols()); ++i)
    result.append( axis ? sum(x.row(i)) : sum(x.col(i)) );
  return result;
}

double min(const Matrix &x) {
  return min(x.const_view());
}

Vector min(const Matrix &x, size_t axis) {
  ASSERT_TRUE(axis==0 || axis==1, "Matrix::max axis out of bounds.");
  Vector result;
  for(size_t i=0; i < (axis ? x.nrows() : x.ncols()); ++i)
    result.append( axis ? min(x.row(i)) : min(x.col(i)) );
  return result;
}

double max(const Matrix &x) {
  return max(x.const_view());
}

Vector max(const Matrix &x, size_t axis) {
  ASSERT_TRUE(axis==0 || axis==1, "Matrix::max axis out of bounds.");
  Vector result;
  for(size_t i=0; i < (axis ? x.nrows() : x.ncols()); ++i)
    result.append( axis ? max(x.row(i)) : max(x.col(i)) );
  return result;
}

// Flip Left-Right
Matrix fliplr(const Matrix &x){
  Matrix result(x.shape());
  size_t offset = x.ncols()-1;
  for(size_t i = 0; i < x.ncols(); ++i) {
    VectorView vw = result.row(i);
    ConstVectorView cvw = x.col(offset - i);
    vw = cvw;
  }
  return result;
}

/*
// Flip Up-Down
Matrix flipud(const Matrix &x){
  Matrix result(x.shape());
  size_t offset = x.nrows()-1;
  //for(size_t i = 0; i < x.ncols(); ++i)
  //  result.row(i) = x.row(offset-i);
  return result;
}
*/

/*
  // Concatenate two matrices as in Matlab
  inline Matrix cat(const Matrix &m1, const Matrix &m2, size_t axis = 1){
    Matrix result(m1);
    result.append(m2, axis);
    return result;
  }
*/

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
/*
  // repmat function of Matlab
  inline Matrix repmat(const Vector &x, int n, int m ){
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

  // Copies column x, n times along the axis.
  // tile(x, n, 0)  --> appendRow(x) n times
  // tile(x, n, 1)  --> appendColumn(x) n times
  inline Matrix tile(const Vector &x, size_t n, int axis = 0){
    ASSERT_TRUE(axis==0 || axis==1, "Matrix::tile axis out of bounds.");
    Matrix result;
    for(size_t i = 0; i < n; ++i){
      if ( axis == 0)
        result.appendRow(x);
      else
        result.appendColumn(x);
    }
    return result;
  }


  inline Vector mean(const Matrix &x, int axis){
    ASSERT_TRUE(axis==0 || axis==1, "Matrix::max axis out of bounds.");
    if (axis == 0)
      return sum(x, 0) / x.nrows();
    return sum(x,1) / x.ncols();
  }
*/
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
/*
  // Normalize
  inline Matrix normalize(const Matrix  &x, int axis = 2) {
    Matrix result(x);
    result.normalize(axis);
    return result;
  }

  // Safe  NormalizeExp
  inline Matrix normalizeExp(const Matrix &x, int axis = 2) {
    Matrix result(x);
    result.normalizeExp(axis);
    return result;
  }


  inline Vector logSumExp(const Matrix &x, int axis) {
    ASSERT_TRUE(axis==0 || axis==1, "Matrix::logSumExp axis out of bounds.");
    Vector result;
    if(axis == 0){
      Vector col_max = max(x,0);
      result = Vector::zeros(x.ncols());
      for(size_t i=0; i < x.nrows(); ++i){
        for(size_t j=0; j < x.ncols(); ++j){
          result[j] += std::exp(x(i,j) - col_max[j]);
        }
      }
      for(size_t j=0; j < x.ncols(); ++j){
        result[j] = std::log(result[j]) + col_max[j];
      }
    } else {
      Vector row_max = max(x,1);
      result = Vector::zeros(x.nrows());
      for(size_t i=0; i < x.nrows(); ++i){
        for(size_t j=0; j < x.ncols(); ++j){
          result[i] += std::exp(x(i,j) - row_max[i]);
        }
      }
      for(size_t i=0; i < x.nrows(); ++i){
        result[i] = std::log(result[i]) + row_max[i];
      }
    }
    return result;
  }
*/
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