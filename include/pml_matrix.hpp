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

      friend bool any(const Matrix &m){
        for(size_t i = 0; i < m.size(); ++i)
          if( m[i] == 1 )
            return true;
        return false;
      }

      friend bool all(const Matrix &m){
        for(size_t i = 0; i < m.size(); ++i)
          if( m[i] == 0 )
            return false;
        return true;
      }

      friend Matrix operator==(const Matrix &x, double v) {
        Matrix result(x.shape());
        for(size_t i = 0; i < x.size(); ++i)
          result[i] = fequal(x[i], v);
        return result;
      }

      friend Matrix operator==(const Matrix &x, const Matrix &y) {
        // Check sizes
        ASSERT_TRUE(x.shape() == y.shape(),
            "Matrix::operator== cannot compare matrices of different shape" );
        // Check element-wise
        Matrix result(x.shape());
        for(size_t i = 0; i < x.size(); ++i)
          result[i] = fequal(x[i], y[i]);
        return result;
      }

      friend Matrix operator<(const Matrix &x, double v) {
        Matrix result(x.shape());
        for(size_t i = 0; i < x.size(); ++i)
          result[i] = x[i] <  v;
        return result;
      }

      friend Matrix operator<(double v, const Matrix &x) {
        return x > v;
      }

      friend Matrix operator<(const Matrix &x, const Matrix &y) {
        // Check sizes
        ASSERT_TRUE(x.shape() == y.shape(),
            "Matrix::operator== cannot compare matrices of different shape" );
        // Check element-wise
        Matrix result(x.shape());
        for(size_t i = 0; i < x.size(); ++i)
          result[i] = x[i] <  y[i];
        return result;
      }

      friend Matrix operator>(const Matrix &x, double v) {
        Matrix result(x.shape());
        for(size_t i = 0; i < x.size(); ++i)
          result[i] = x[i] > v;
        return result;
      }

      friend Matrix operator>(double v, const Matrix &x) {
        return x < v;
      }

      friend Matrix operator>(const Matrix &x, const Matrix &y) {
        // Check sizes
        ASSERT_TRUE(x.shape() == y.shape(),
            "Matrix::operator== cannot compare matrices of different shape" );
        // Check element-wise
        Matrix result(x.shape());
        for(size_t i = 0; i < x.size(); ++i)
          result[i] = x[i] > y[i];
        return result;
      }

      bool equals(const Matrix &other){
        if(shape() != other.shape())
          return false;
        return all(*this == other);
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
        ASSERT_TRUE(row_num < nrows(),
                    "Matrix::setRow:: row_num exceeds number of rows");
        ASSERT_TRUE(ncols_ == row.size(),
                    "Matrix::setRow:: Vector size mismatch");
        size_t idx = row_num;
        for(size_t i=0; i < ncols_; ++i){
          data_[idx] = row(i);
          idx += nrows_;
        }
      }

      // Appends a column to the right.
      void append(const Matrix &m, size_t axis = 1){
        ASSERT_TRUE(axis == 0 || axis == 1,
                    "Matrix::append(const Matrix &):: axis out of bounds");
        if(m.empty())
          return;
        if(axis == 0){
          // Append row-wise
          if(empty()){
            ncols_ = m.ncols();
          } else {
            ASSERT_TRUE(ncols() == m.ncols(),
                  "Matrix::append(const Matrix &, 1):: column sizes mismatch.");
          }
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
          // Append column-wise
          if(empty()){
            nrows_ = m.nrows();
          } else {
            ASSERT_TRUE(nrows() == m.nrows(),
                  "Matrix::append(const Matrix &, 1):: column sizes mismatch.");
          }
          std::vector<double> new_data(size() + m.size());
          memcpy(new_data.data(), data(), sizeof(double) * size());
          memcpy(new_data.data() + size(), m.data(), sizeof(double) * m.size());
          data_ = std::move(new_data);
          ncols_ += m.ncols();
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

      // Sum
      friend double sum(const Matrix &x){
        double result = 0;
        for(size_t i=0; i<x.size(); ++i)
          result += x[i];
        return result;
      }

      friend Vector sum(const Matrix &x, size_t axis) {
        ASSERT_TRUE(axis==0 || axis==1, "Matrix::sum axis out of bounds.");
        Vector result;
        if (axis == 0){
          result = Vector::zeros(x.ncols());
          for(size_t i=0; i < x.nrows(); ++i)
            for(size_t j=0; j < x.ncols(); ++j)
              result[j] += x(i,j);
        } else {
          result = Vector::zeros(x.nrows());
          for(size_t i=0; i < x.nrows(); ++i)
            for(size_t j=0; j < x.ncols(); ++j)
              result[i] += x(i,j);
        }
        return result;
      }

      // Min
      friend double min(const Matrix &x) {
        double min_x = x[0];
        for(size_t i=1; i<x.size(); ++i)
          if( x[i] < min_x )
            min_x = x[i];
        return min_x;
      }

      friend Vector min(const Matrix &x, size_t axis) {
        ASSERT_TRUE(axis==0 || axis==1, "Matrix::max axis out of bounds.");
        Vector result;
        if (axis == 0){
          result = x.getRow(0);
          for(size_t i=1; i < x.nrows(); ++i)
            for(size_t j=0; j < x.ncols(); ++j)
              if( x(i,j) < result[j] )
                result[j] = x(i,j);
        } else {
          result = x.getColumn(0);
          for(size_t i=0; i < x.nrows(); ++i)
            for(size_t j=1; j < x.ncols(); ++j)
              if( x(i,j) < result[i] )
                result[i] = x(i,j);
        }
        return result;
      }

      // Max
      friend double max(const Matrix &x) {
        double max_x = x[0];
        for(size_t i=1; i<x.size(); ++i)
          if( max_x < x[i] )
            max_x = x[i];
        return max_x;
      }

      friend Vector max(const Matrix &x, size_t axis) {
        ASSERT_TRUE(axis==0 || axis==1, "Matrix::max axis out of bounds.");
        Vector result;
        if (axis == 0){
          result = x.getRow(0);
          for(size_t i=1; i < x.nrows(); ++i)
            for(size_t j=0; j < x.ncols(); ++j)
              if( result[j] < x(i,j) )
                result[j] = x(i,j);
        } else {
          result = x.getColumn(0);
          for(size_t i=0; i < x.nrows(); ++i)
            for(size_t j=1; j < x.ncols(); ++j)
              if( result[i] < x(i,j) )
                result[i] = x(i,j);
        }
        return result;
      }

      void normalize(size_t axis = 2){
        ASSERT_TRUE(axis<=2, "Matrix::normalize axis out of bounds.");
        if( axis == 0){
          Vector col_sums = sum(*this, 0);
          for(size_t i=0; i < nrows_; ++i)
            for(size_t j=0; j < ncols_; ++j)
              (*this)(i,j) /= col_sums[j];
        } else if( axis == 1){
          Vector row_sums = sum(*this, 1);
          for(size_t i=0; i < nrows_; ++i)
            for(size_t j=0; j < ncols_; ++j)
              (*this)(i,j) /= row_sums[i];
        } else {
          double sum_x = sum(*this);
          for(size_t i=0; i < size(); ++i)
            data_[i] /= sum_x;
        }
      }

      void normalizeExp(size_t axis = 2){
        ASSERT_TRUE(axis<=2, "Matrix::normalizeExp axis out of bounds.");
        if( axis == 0) {
          Vector col_max = max(*this, 0);
          for(size_t i=0; i < nrows_; ++i)
            for(size_t j=0; j < ncols_; ++j)
              (*this)(i,j) = std::exp((*this)(i,j) - col_max[j]);
          normalize(0);
        } else if( axis == 1) {
          Vector row_max = max(*this, 1);
          for (size_t i = 0; i < nrows_; ++i)
            for (size_t j = 0; j < ncols_; ++j)
              (*this)(i, j) = std::exp((*this)(i, j) - row_max[i]);
          normalize(1);
        } else {
          double x_max = max(*this);
          for (size_t i = 0; i < size(); ++i)
            data_[i] = std::exp(data_[i] - x_max);
          normalize();
        }
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
      result.appendColumn(x.getColumn(i-1));
    }
    return result;
  }

  // Flip Up-Down
  inline Matrix flipud(const Matrix &x){
    Matrix result;
    for(size_t i = x.nrows() ; i > 0; --i){
      result.appendRow(x.getRow(i-1));
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


  // Safe LogSumExp(x)
  inline double logSumExp(const Matrix &x) {
    double result = 0;
    double x_max = max(x);
    for(size_t i=0; i<x.size(); ++i)
      result += std::exp(x(i) - x_max);
    return x_max + std::log(result);
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