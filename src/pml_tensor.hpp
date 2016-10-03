#ifndef MATLIB_PML_LINALG_H
#define MATLIB_PML_LINALG_H

// WARNING: This code is unusable !

namespace pml{

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
              return Vector(t1).equals(Vector(t2));
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

} // pml

#endif